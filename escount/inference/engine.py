"""Inference engine (Task 8): profile -> rotate -> selected ROIs -> projection -> indicators -> model.

InferenceEngine.load(case or profile dir, model='project'|'general') loads project.json, the rotated
mask (background_BW_rotated.png), the model manifest (SHA-256 verified by escount.models.store) and
the selected ROIs. `model='general'` applies the production General model (configs/general.json)
to the project's geometry: every General ROI unit (projection, w, x_norm bin) resolves per image to
the nearest strip of that bin inside the stack, exactly as in training (select_rois.unit_matrix).
infer(image array | path) runs single-process on the CPU and returns N_pred with per-stage
time.perf_counter timings (model load excluded). run(input file | folder, out_dir, OutputOptions)
writes, per image <out>/<image_stem>/: corrected.png, analysis.png, signals.csv, result.json (each
toggleable), and <out>/predictions.csv (image, GT if the file name is a count, N_pred, hit).
The source image SHA-256 is taken before reading and after writing; a change raises an error.
"""
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
from escount.es_counting import file_sha256, list_images, parse_ground_truth, read_gray, write_json
from escount.features.extract import (
    FEATURE_INDICATORS, POSITION_BIN, AugmentGrid, clip_items, config_hash, extent_slots, feature_config,
    indicator_id, projection_sigma,
)
from escount.indicators.counters import IndicatorConfig, count
from escount.inference.benchmark import ProjectRuntime
from escount.inference.plots import save_analysis
from escount.signals.projection import project
from escount.signals.roi_bank import build_roi_bank

MIN_DPI = 64
MODEL_CHOICES = ("project", "general")
PREDICTIONS_NAME = "predictions.csv"
OUTPUT_FILES = {"corrected": "corrected.png", "analysis_plot": "analysis.png", "signals_csv": "signals.csv",
                "result_json": "result.json"}
FALLBACK_MARKS = FEATURE_INDICATORS[12]   # peaks, gate 0.25 (used when the baseline column is period_count)


@dataclass(frozen=True)
class OutputOptions:
    """Per-image output toggles (all on by default) and the analysis figure dpi (>= 64)."""
    corrected: bool = True
    analysis_plot: bool = True
    signals_csv: bool = True
    result_json: bool = True
    dpi: int = 100

    def __post_init__(self):
        if self.dpi < MIN_DPI:
            raise ValueError(f"dpi must be >= {MIN_DPI}, got {self.dpi}")


class GeneralRuntime(ProjectRuntime):
    """General model on one project's geometry: units = (roi dict, candidate items of its x_norm bin)."""

    def __init__(self, profile, units, ext_slots, grid, model, manifest):
        super().__init__(profile, [item for _, candidates in units for item in candidates], ext_slots, grid,
                         model, manifest)
        self.units = units

    def kept_items(self, extent):
        kept = []
        for roi, candidates in self.units:
            clipped = clip_items(candidates, extent)
            if clipped:
                kept.append(min(clipped, key=lambda item: abs(self._x_norm(item[0]) - roi["x_norm_bin"])))
        return kept

    def _x_norm(self, slot):
        return (slot.x - self.valid_x0) / self.valid_width


def _profile_dir(case_or_profile_dir, profiles_root):
    path = Path(case_or_profile_dir)
    if (path / PROFILE_NAME).is_file():
        return path
    candidate = Path(profiles_root) / str(case_or_profile_dir)
    if (candidate / PROFILE_NAME).is_file():
        return candidate
    raise FileNotFoundError(f"no {PROFILE_NAME} for {case_or_profile_dir!r} (looked in {path} and {candidate}); "
                            f"run `escount calibrate` first")


def _general_runtime(profile_dir, model_dir, single_thread):
    from escount.models.store import load_model

    profile = ProjectProfile.load(profile_dir / PROFILE_NAME)
    mask = read_gray(profile_dir / ROTATED_MASK_NAME)
    model, manifest = load_model(model_dir, single_thread=single_thread)
    if not manifest.get("selected_rois"):
        raise ValueError(f"{model_dir}: no selected_rois; run `escount select-rois --general`")
    grid = AugmentGrid.from_dict(manifest["feature_config"]["grid"])
    if config_hash(manifest["feature_config"]) != config_hash(feature_config(grid)):
        raise ValueError(f"{model_dir}: model was trained with a different feature config; re-run prepare/train")
    x0, _, x1, _ = profile.valid_region["bbox"]
    units = []
    for roi in manifest["selected_rois"]:
        candidates = []
        for pitch in grid.pitches:
            for x_offset in grid.x_offsets:
                for slot in build_roi_bank(mask, (roi["w"],), pitch=pitch, side_margin=grid.side_margin,
                                           x_offset=x_offset):
                    x_norm = (slot.x - x0) / (x1 - x0)
                    if f"{round(x_norm / POSITION_BIN) * POSITION_BIN:.2f}" == f"{roi['x_norm_bin']:.2f}":
                        candidates.append((slot, roi["projection"]))
        units.append((roi, candidates))
    return GeneralRuntime(profile, units, extent_slots(mask, grid.side_margin), grid, model, manifest)


def _marks_config(manifest):
    """Indicator whose highs/lows are marked in the plots: the case's baseline count column."""
    column = (manifest.get("baseline") or {}).get("column", "period_count")
    by_id = {indicator_id(config): config for config in FEATURE_INDICATORS}
    name = column[len("cnt_"):].rsplit("_", 1)[0] if column.startswith("cnt_") else None
    config = by_id.get(name, FALLBACK_MARKS)
    return IndicatorConfig(config.name, config.params, config.gate, "both")


def _array_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def collect_inputs(source):
    """A single image file or every analysable PNG of a folder (background files excluded)."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        paths = list_images(path)
        if not paths:
            raise ValueError(f"{path}: no PNG image to infer")
        return paths
    raise FileNotFoundError(f"input {path} not found")


class InferenceEngine:
    def __init__(self, case, model_choice, model_case, runtime):
        self.case, self.model_choice, self.model_case, self.runtime = case, model_choice, model_case, runtime
        self.marks = _marks_config(runtime.manifest)
        size = runtime.profile.image_size
        self.image_shape = (size["height"], size["width"]) if size else None

    @classmethod
    def load(cls, case_or_profile_dir, model="project", profiles_root="outputs/projects",
             general_config="configs/general.json", single_thread=True):
        """Load profile, rotated mask, model (SHA verified) and selected ROIs once (excluded from timings)."""
        if model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {model!r}")
        profile_dir = _profile_dir(case_or_profile_dir, profiles_root)
        profile = ProjectProfile.load(profile_dir / PROFILE_NAME)
        if profile.quality_status != "ok":
            raise ValueError(f"{profile_dir}: project.json quality_status is {profile.quality_status!r}")
        if model == "project":
            runtime = ProjectRuntime.load(profile_dir.name, profile_dir.parent, single_thread=single_thread)
            model_case = profile_dir.name
        else:
            from escount.models.train import load_general_config, production_general

            model_case = production_general(load_general_config(general_config))
            runtime = _general_runtime(profile_dir, profile_dir.parent / model_case / "model", single_thread)
        return cls(profile.case, model, model_case, runtime)

    def infer(self, source, name=None):
        """N_pred of one image (uint8 gray array or file path). Returns a result dict (see module doc)."""
        timings = {}
        if isinstance(source, np.ndarray):
            gray, path = source, None
            sha_before = _array_sha256(gray)
        else:
            path = Path(source)
            sha_before = file_sha256(path)
            start = time.perf_counter()
            gray = read_gray(path)
            timings["read"] = 1000.0 * (time.perf_counter() - start)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError(f"expected a single-channel uint8 image, got {gray.shape} {gray.dtype}")
        if self.image_shape and gray.shape != self.image_shape:
            raise ValueError(f"image size {gray.shape} differs from the project's {self.image_shape}")
        run = self.runtime.run(gray)
        timings.update(run["timings"])
        timings["total"] = sum(timings.values())
        rows, per_roi = run["rows"], run["per_roi"]
        order = {f"{method}|{slot.w}|{slot.x}": index for index, (slot, method) in enumerate(run["kept"])}
        rank = np.argsort([order[key] for key in rows["slot_key"]], kind="stable") if len(rows) else []
        items = {f"{method}|{slot.w}|{slot.x}": (slot, method) for slot, method in run["kept"]}
        rois = []
        for number, position in enumerate(rank, start=1):
            row = rows.iloc[position]
            slot, method = items[row["slot_key"]]
            rois.append({"index": number, "key": row["slot_key"], "projection": method, "w": int(slot.w),
                         "x": int(slot.x), "y0": int(slot.y0), "y1": int(slot.y1),
                         "sigma": None if pd.isna(row["sigma"]) else float(row["sigma"]),
                         "period_count": int(row["period_count"]), "prediction": float(per_roi[position]),
                         "slot": slot})
        image_name = path.name if path else (name or "image")
        gt = parse_ground_truth(image_name)
        prediction = run["prediction"]
        return {"image": image_name, "source": str(path) if path else None, "case": self.case,
                "model": self.model_choice, "model_case": self.model_case,
                "method": self.runtime.manifest["method"], "model_type": self.runtime.manifest["model_type"],
                "N_pred": prediction, "gt": gt["count"] if gt else None,
                "hit": (prediction == gt["count"]) if gt else None, "rois": rois, "timings_ms": timings,
                "source_sha256_before": sha_before, "rotated": run["rotated"], "_gray": gray}

    def _signals(self, result):
        ratio = self.runtime.grid.gaussian_sigma_ratio
        signals = []
        for roi in result["rois"]:
            slot = roi["slot"]
            signal = project(result["rotated"], slot, roi["projection"], projection_sigma(roi["projection"], slot.w, ratio))
            _, high_idx, low_idx = count(signal, self.marks) if signal.size >= 2 else (0, [], [])
            signals.append({"rows": np.arange(slot.y0, slot.y1), "signal": signal, "high_idx": high_idx,
                            "low_idx": low_idx, "prediction": roi["prediction"],
                            "label": f"{roi['projection']} w{slot.w} x{slot.x}"})
        return signals

    def write_outputs(self, result, image_dir, options=OutputOptions()):
        """Write the enabled per-image outputs; verifies the source SHA after writing (before result.json)."""
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        written = []
        if options.corrected:
            ok, encoded = cv2.imencode(".png", result["rotated"])
            if not ok:
                raise IOError("could not encode corrected.png")
            (image_dir / OUTPUT_FILES["corrected"]).write_bytes(encoded.tobytes())
            written.append(OUTPUT_FILES["corrected"])
        signals = self._signals(result) if options.analysis_plot or options.signals_csv else []
        if options.analysis_plot:
            title = (f"{result['image']}  N_pred = {result['N_pred']}"
                     + (f"  GT = {result['gt']}" if result["gt"] is not None else "")
                     + f"\n{result['model']} model ({result['model_case']}, "
                     + f"{result['model_type'] if result['method'] == 'ml' else 'baseline'})")
            save_analysis(image_dir / OUTPUT_FILES["analysis_plot"], signals, title, options.dpi)
            written.append(OUTPUT_FILES["analysis_plot"])
        if options.signals_csv:
            height = result["rotated"].shape[0]
            table = {"row": np.arange(height)}
            for roi, panel in zip(result["rois"], signals):
                column = np.full(height, np.nan)
                column[panel["rows"]] = panel["signal"]
                table[f"N{roi['index']}|{roi['key']}"] = column
            frame = pd.DataFrame(table)
            if len(table) > 1:
                frame = frame[frame.iloc[:, 1:].notna().any(axis=1)]
            frame.to_csv(image_dir / OUTPUT_FILES["signals_csv"], index=False, encoding="utf-8")
            written.append(OUTPUT_FILES["signals_csv"])
        source = result["source"]
        sha_after = file_sha256(source) if source else _array_sha256(result["_gray"])
        result["source_sha256_after"] = sha_after
        result["source_unchanged"] = sha_after == result["source_sha256_before"]
        if options.result_json:
            written.append(OUTPUT_FILES["result_json"])
            write_json(image_dir / OUTPUT_FILES["result_json"], self.result_record(result, written, options))
        result["outputs"] = written
        if not result["source_unchanged"]:
            raise RuntimeError(f"source image changed during inference: {source or result['image']}")
        return written

    @staticmethod
    def result_record(result, outputs=(), options=OutputOptions()):
        """JSON-serialisable result (no arrays)."""
        record = {key: value for key, value in result.items() if key not in ("rotated", "_gray", "rois")}
        record["rois"] = [{key: value for key, value in roi.items() if key != "slot"} for roi in result["rois"]]
        record.update(outputs=list(outputs), dpi=options.dpi if options.analysis_plot else None)
        return record

    def run(self, source, out_dir, options=OutputOptions()):
        """Infer a file or folder; per-image outputs + <out_dir>/predictions.csv. Returns result dicts."""
        paths = collect_inputs(source)
        out_dir = Path(out_dir).resolve()
        input_root = Path(source).resolve() if Path(source).is_dir() else paths[0].resolve().parent
        if out_dir == input_root or input_root in out_dir.parents:
            raise ValueError(f"output folder {out_dir} must not be inside the input folder {input_root}")
        out_dir.mkdir(parents=True, exist_ok=True)
        results, records = [], []
        for path in paths:
            result = self.infer(path)
            self.write_outputs(result, out_dir / path.stem, options)
            results.append(result)
            records.append({"image": result["image"], "GT": result["gt"], "N_pred": result["N_pred"],
                            "hit": result["hit"]})
        pd.DataFrame(records, columns=["image", "GT", "N_pred", "hit"]).astype(object).to_csv(
            out_dir / PREDICTIONS_NAME, index=False, encoding="utf-8")
        return results
