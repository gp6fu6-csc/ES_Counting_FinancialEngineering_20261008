"""Single-thread CPU latency benchmark of the minimal-ROI pipeline (Task 7).

Stages per image (time.perf_counter, ms): read -> rotate -> extent (stack rows + clip selected ROIs)
-> features (width projection + indicators) -> model -> aggregate. Model / profile loading is done
once before timing and is excluded. Threads: OMP/MKL/OpenBLAS = 1, cv2.setNumThreads(1),
threadpoolctl limit 1, XGBoost n_jobs = 1, LightGBM num_threads = 1.
"""
import os
import time
from pathlib import Path

import numpy as np

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
LATENCY_LIMIT_MS = 1000.0


def single_thread():
    for name in THREAD_ENV:
        os.environ[name] = "1"
    import cv2
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    threadpool_limits(limits=1)


class ProjectRuntime:
    """Everything needed to predict one image of a project with its selected ROIs."""

    def __init__(self, profile, items, ext_slots, grid, model, manifest):
        self.profile, self.items, self.ext_slots, self.grid = profile, items, ext_slots, grid
        self.model, self.manifest = model, manifest
        x0, _, x1, _ = profile.valid_region["bbox"]
        self.valid_x0, self.valid_width = x0, x1 - x0

    @classmethod
    def load(cls, case, profiles_root="outputs/projects", single_thread=True):
        from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
        from escount.es_counting import read_gray
        from escount.features.extract import AugmentGrid, config_hash, extent_slots, feature_config
        from escount.models.store import load_model
        from escount.signals.roi_bank import build_roi_bank

        case_dir = Path(profiles_root) / case
        profile = ProjectProfile.load(case_dir / PROFILE_NAME)
        mask = read_gray(case_dir / ROTATED_MASK_NAME)
        model, manifest = load_model(case_dir / "model", single_thread=single_thread)
        if not manifest.get("selected_rois"):
            raise ValueError(f"{case}: no selected_rois in the model manifest; run `escount select-rois --project {case}`")
        grid = AugmentGrid.from_dict(manifest["feature_config"]["grid"])
        if config_hash(manifest["feature_config"]) != config_hash(feature_config(grid)):
            raise ValueError(f"{case}: model was trained with a different feature config; re-run prepare/train")
        items = []
        for roi in manifest["selected_rois"]:
            bank = build_roi_bank(mask, (roi["w"],), pitch=roi["pitch"], side_margin=grid.side_margin,
                                  x_offset=roi["x_offset"])
            items.append((next(slot for slot in bank if slot.x == roi["x"]), roi["projection"]))
        return cls(profile, items, extent_slots(mask, grid.side_margin), grid, model, manifest)

    def kept_items(self, extent):
        """Selected (slot, projection) items clipped to this image's stack extent."""
        from escount.features.extract import clip_items

        return clip_items(self.items, extent)

    def predict(self, gray):
        """Return (rounded prediction or None, per-ROI predictions, stage timings in ms)."""
        result = self.run(gray)
        return result["prediction"], result["per_roi"], result["timings"]

    def run(self, gray):
        """Timed pipeline keeping the intermediates (rotated image, kept items, feature rows)."""
        from escount.calibration.rotation_correction import rotate_image
        from escount.features.extract import roi_features
        from escount.models.train import aggregate, feature_frame
        from escount.signals.roi_bank import stack_rows

        timings, start = {}, time.perf_counter()

        def lap(name):
            nonlocal start
            now = time.perf_counter()
            timings[name] = 1000.0 * (now - start)
            start = now

        rotated = rotate_image(gray, self.profile.angle_deg)
        lap("rotate")
        extent, _ = stack_rows(rotated, self.ext_slots)
        kept = self.kept_items(extent)
        lap("extent")
        rows = roi_features(rotated, kept, self.valid_x0, self.valid_width, self.grid.gaussian_sigma_ratio)
        lap("features")
        if rows.empty:
            per_roi = np.array([])
        elif self.manifest["method"] == "ml":
            delta = self.model.predict(feature_frame(rows, self.manifest["scale_normalised"]))
            per_roi = rows["period_count"].to_numpy(dtype=np.float64) + delta
        else:
            per_roi = rows[self.manifest["baseline"]["column"]].to_numpy(dtype=np.float64)
        lap("model")
        prediction = int(np.rint(aggregate(per_roi, self.manifest["aggregation"]))) if per_roi.size else None
        lap("aggregate")
        return {"prediction": prediction, "per_roi": per_roi, "timings": timings, "rotated": rotated,
                "extent": extent, "kept": kept, "rows": rows}


def benchmark(case, data_root="Data", profiles_root="outputs/projects", write=True):
    """Time every labelled image of Data/<case>; returns a dict with median/max total ms."""
    single_thread()
    from escount.es_counting import read_gray, write_json
    from escount.features.cache import labelled_images

    runtime = ProjectRuntime.load(case, profiles_root, single_thread=True)   # excluded from timing
    per_image = []
    for path, gt in labelled_images(Path(data_root) / case):
        start = time.perf_counter()
        gray = read_gray(path)
        read_ms = 1000.0 * (time.perf_counter() - start)
        prediction, per_roi, timings = runtime.predict(gray)
        timings = {"read": read_ms, **timings}
        per_image.append({"image": path.name, "gt": gt["count"], "prediction": prediction,
                          "hit": prediction == gt["count"], "rois_used": int(per_roi.size),
                          "timings_ms": timings, "total_ms": sum(timings.values())})
    totals = [item["total_ms"] for item in per_image]
    report = {"case": case, "threads": 1, "method": runtime.manifest["method"],
              "model_type": runtime.manifest["model_type"], "roi_count": len(runtime.items),
              "median_ms": float(np.median(totals)), "max_ms": float(np.max(totals)),
              "limit_ms": LATENCY_LIMIT_MS, "within_limit": float(np.max(totals)) < LATENCY_LIMIT_MS,
              "hits": sum(item["hit"] for item in per_image), "total": len(per_image),
              "note": "in-sample predictions (model trained on these images); model load excluded",
              "per_image": per_image}
    if write:
        write_json(Path(profiles_root) / case / "benchmark.json", report)
    return report
