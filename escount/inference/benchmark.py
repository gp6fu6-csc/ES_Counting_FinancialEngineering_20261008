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
RETRAIN = "re-run `escount prepare / train / select-rois` for {case}"


def check_profile_fingerprint(manifest, case, profile, rotated_mask_path, required=True):
    """ValueError unless the model was trained on the current calibration of `case`.

    The model manifest stores the feature cache's profile fingerprint (angle, valid region, mask
    SHAs) per training project; a calibration saved later (CLI or web UI) changes it. A project the
    model was not trained on is skipped when `required` is False (General on another project).
    """
    from escount.features.cache import profile_fingerprint

    recorded = (manifest.get("profile_fingerprints") or {}).get(case)
    if recorded is None:
        if required:
            raise ValueError(f"{case}: the model manifest records no calibration fingerprint (trained before "
                             f"fingerprints were stored); {RETRAIN.format(case=case)}")
        return
    if recorded != profile_fingerprint(profile, rotated_mask_path):
        raise ValueError(f"{case}: the calibration (project.json / background_BW_rotated.png) changed after the "
                         f"model was trained; {RETRAIN.format(case=case)}")


def single_thread():
    for name in THREAD_ENV:
        os.environ[name] = "1"
    import cv2
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    threadpool_limits(limits=1)


class ProjectRuntime:
    """Everything needed to predict one image of a project with its selected ROIs."""

    def __init__(self, profile, items, ext_slots, grid, model, manifest, all_items=None, ensemble_meta=None):
        self.profile, self.items, self.ext_slots, self.grid = profile, items, ext_slots, grid
        self.model, self.manifest = model, manifest
        self.all_items = all_items or items
        self.ensemble_meta = ensemble_meta
        self.roi_mode = "minimal"
        x0, _, x1, _ = profile.valid_region["bbox"]
        self.valid_x0, self.valid_width = x0, x1 - x0

    @classmethod
    def load(cls, case, profiles_root="outputs/projects", single_thread=True):
        from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
        from escount.es_counting import read_gray
        from escount.features.extract import AugmentGrid, config_hash, extent_slots, feature_config
        from escount.models.store import load_model
        from escount.signals.roi_bank import build_roi_bank

        from escount.models.select_rois import check_selected_overlap

        case_dir = Path(profiles_root) / case
        profile = ProjectProfile.load(case_dir / PROFILE_NAME)
        mask = read_gray(case_dir / ROTATED_MASK_NAME)
        model, manifest = load_model(case_dir / "model", single_thread=single_thread)
        check_profile_fingerprint(manifest, case, profile, case_dir / ROTATED_MASK_NAME)
        if not manifest.get("selected_rois"):
            raise ValueError(f"{case}: no selected_rois in the model manifest; run `escount select-rois --project {case}`")
        check_selected_overlap(manifest["selected_rois"])
        grid = AugmentGrid.from_dict(manifest["feature_config"]["grid"])
        if config_hash(manifest["feature_config"]) != config_hash(feature_config(grid)):
            raise ValueError(f"{case}: model was trained with a different feature config; re-run prepare/train")
        items = []
        for roi in manifest["selected_rois"]:
            bank = build_roi_bank(mask, (roi["w"],), pitch=roi["pitch"], side_margin=grid.side_margin,
                                  x_offset=roi["x_offset"])
            slot = next((slot for slot in bank if slot.x == roi["x"]), None)
            if slot is None:
                raise ValueError(f"{case}: selected ROI {roi['key']} does not fit the current mask; {RETRAIN.format(case=case)}")
            items.append((slot, roi["projection"]))

        all_items = []
        all_spec = manifest.get("all_rois")
        if all_spec:
            for roi in all_spec:
                bank = build_roi_bank(mask, (roi["w"],), pitch=roi["pitch"], side_margin=grid.side_margin,
                                      x_offset=roi["x_offset"])
                slot = next((s for s in bank if s.x == roi["x"]), None)
                if slot is not None:
                    all_items.append((slot, roi["projection"]))

        ensemble_meta = None
        ens_file = case_dir / "model" / "model_all_rois_ensemble.joblib"
        if ens_file.is_file():
            try:
                import joblib
                ensemble_meta = joblib.load(ens_file)
            except Exception:
                ensemble_meta = None

        return cls(profile, items, extent_slots(mask, grid.side_margin), grid, model, manifest,
                   all_items=all_items, ensemble_meta=ensemble_meta)

    def kept_items(self, extent, dropped=None, roi_mode=None):
        """Selected (slot, projection) items clipped to this image's stack extent (`dropped`: General only)."""
        from escount.features.extract import clip_items

        mode = roi_mode or getattr(self, "roi_mode", "minimal")
        chosen = self.all_items if (mode == "all" and getattr(self, "all_items", None)) else self.items
        return clip_items(chosen, extent)

    def predict(self, gray, roi_mode=None):
        """Return (rounded prediction or None, per-ROI predictions, stage timings in ms)."""
        result = self.run(gray, roi_mode=roi_mode)
        return result["prediction"], result["per_roi"], result["timings"]

    def run(self, gray, roi_mode=None):
        """Timed pipeline keeping the intermediates (rotated image, kept items, feature rows)."""
        from escount.calibration.rotation_correction import rotate_image
        from escount.features.extract import roi_features
        from escount.signals.roi_bank import stack_rows

        mode = roi_mode or getattr(self, "roi_mode", "minimal")
        timings, start = {}, time.perf_counter()

        def lap(name):
            nonlocal start
            now = time.perf_counter()
            timings[name] = 1000.0 * (now - start)
            start = now

        rotated = rotate_image(gray, self.profile.angle_deg)
        lap("rotate")
        extent, _ = stack_rows(rotated, self.ext_slots)
        dropped = []
        kept = self.kept_items(extent, dropped, roi_mode=mode)
        lap("extent")
        rows = roi_features(rotated, kept, self.valid_x0, self.valid_width, self.grid.gaussian_sigma_ratio)
        lap("features")
        per_roi = self.roi_predictions(rows)
        lap("model")
        prediction = self.image_prediction(per_roi, roi_mode=mode)
        lap("aggregate")
        return {"prediction": prediction, "per_roi": per_roi, "timings": timings, "rotated": rotated,
                "extent": extent, "kept": kept, "dropped": dropped, "rows": rows, "roi_mode": mode}

    def roi_predictions(self, rows):
        """Per-ROI sheet counts of feature rows (ML: period_count + predicted delta; baseline: count column)."""
        from escount.models.train import feature_frame

        if rows.empty:
            return np.array([])
        # 支援多方案預測模式 (Q7: baseline / kd / macd / xgboost)
        method_choice = getattr(self, "forced_method", None) or self.manifest.get("method", "baseline")
        
        if method_choice == "xgboost" or (method_choice == "ml" and self.manifest["method"] == "ml"):
            delta = self.model.predict(feature_frame(rows, self.manifest["scale_normalised"]))
            return rows["period_count"].to_numpy(dtype=np.float64) + delta
        elif method_choice == "kd":
            kd_cols = [c for c in rows.columns if c.startswith("cnt_kd_")]
            chosen_col = kd_cols[0] if kd_cols else self.manifest.get("baseline", {}).get("column", "period_count")
            return rows[chosen_col].to_numpy(dtype=np.float64)
        elif method_choice == "macd":
            macd_cols = [c for c in rows.columns if c.startswith("cnt_macd_")]
            chosen_col = macd_cols[0] if macd_cols else self.manifest.get("baseline", {}).get("column", "period_count")
            return rows[chosen_col].to_numpy(dtype=np.float64)
        elif method_choice == "baseline":
            base_col = self.manifest.get("baseline", {}).get("column")
            if base_col and base_col in rows.columns:
                return rows[base_col].to_numpy(dtype=np.float64)
            return rows["period_count"].to_numpy(dtype=np.float64)
        
        # 預設行為
        if self.manifest["method"] == "ml":
            delta = self.model.predict(feature_frame(rows, self.manifest["scale_normalised"]))
            return rows["period_count"].to_numpy(dtype=np.float64) + delta
        return rows[self.manifest["baseline"]["column"]].to_numpy(dtype=np.float64)

    def image_prediction(self, per_roi, roi_mode=None):
        """Rounded aggregate (median / trimmed mean / ensemble) of the per-ROI predictions; None without ROIs."""
        if not per_roi.size:
            return None
        from escount.models.train import aggregate

        mode = roi_mode or getattr(self, "roi_mode", "minimal")
        method = getattr(self, "forced_method", None) or "auto"

        if mode == "all":
            rec = (self.manifest.get("all_rois_ensemble") or {}).get("recommended", "trimmed_mean")
            chosen = rec if method == "auto" else method
            if chosen in ("xgboost", "ridge") and getattr(self, "ensemble_meta", None) and chosen in self.ensemble_meta:
                from escount.models.all_rois import _ensemble_features
                feat = _ensemble_features(per_roi.reshape(1, -1))
                val = float(self.ensemble_meta[chosen].predict(feat)[0])
                return int(np.rint(val))
            elif chosen == "trimmed_mean":
                row = np.sort(per_roi)
                val = float(np.mean(row[1:-1])) if len(row) > 2 else float(np.mean(row))
                return int(np.rint(val))
            else:
                return int(np.rint(np.median(per_roi)))

        return int(np.rint(aggregate(per_roi, self.manifest["aggregation"])))


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
