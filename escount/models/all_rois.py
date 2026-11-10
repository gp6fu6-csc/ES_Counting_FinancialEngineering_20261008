"""All-ROIs multi-strip ensemble training and inference.

Evaluates all N counting strips defined in calibration (strip_params), and trains a second-stage
ensemble meta-learner (XGBoost / Ridge / Median) to aggregate multi-ROI predictions into a final N_pred.
"""
from dataclasses import asdict
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
from escount.calibration.rotation_correction import rotate_image
from escount.es_counting import parse_ground_truth, read_gray
from escount.features.extract import (
    AugmentGrid,
    clip_items,
    extent_slots,
    roi_features,
)
from escount.models.store import read_manifest, update_manifest
from escount.models.train import feature_frame
from escount.signals.roi_bank import RoiSlot, build_roi_bank, stack_rows

ENSEMBLE_MODEL_NAME = "model_all_rois_ensemble.joblib"
DEFAULT_STRIPS = {"n": 10, "w": 30, "pitch": 50, "side_margin": 50, "x_offset": 0}


def build_all_rois_layout(case, profiles_root="outputs/projects", data_root="Data"):
    """Compute the N counting strips defined in project.json strip_params.

    Returns (all_rois_list, roi_slots_list).
    """
    case_dir = Path(profiles_root) / case
    profile = ProjectProfile.load(case_dir / PROFILE_NAME)
    mask = read_gray(case_dir / ROTATED_MASK_NAME)
    strip_cfg = dict(profile.strip_params or DEFAULT_STRIPS)
    n = int(strip_cfg.get("n", 10))
    w = int(strip_cfg.get("w", 30))
    pitch = int(strip_cfg.get("pitch", 50))
    side_margin = int(strip_cfg.get("side_margin", 50))
    x_offset = int(strip_cfg.get("x_offset", 0))

    bank = build_roi_bank(mask, (w,), pitch=pitch, side_margin=side_margin, x_offset=x_offset)

    # Use reference image or first available image to compute strip extent
    ref_name = getattr(profile, "reference_image", None)
    img_path = None
    if ref_name and (Path(data_root) / case / ref_name).is_file():
        img_path = Path(data_root) / case / ref_name
    else:
        for ext in ("*.png", "*.bmp", "*.jpg"):
            candidates = [p for p in (Path(data_root) / case).glob(ext) if "background" not in p.name]
            if candidates:
                img_path = sorted(candidates)[0]
                break

    if img_path is not None:
        ref_img = rotate_image(read_gray(img_path), profile.angle_deg)
        extent, kept = stack_rows(ref_img, bank, full_span=True)
    else:
        kept = bank

    if len(kept) < n:
        n = len(kept)

    x0, _, x1, _ = profile.valid_region["bbox"]
    centre = (x0 + x1) / 2.0
    start = min(
        range(len(kept) - n + 1),
        key=lambda i: abs((kept[i].x + kept[i + n - 1].x + w) / 2.0 - centre),
    )
    chosen = kept[start : start + n]

    all_rois = []
    slots = []
    for i, slot in enumerate(chosen, 1):
        item = {
            "key": f"gaussian|{slot.w}|{slot.x}",
            "projection": "gaussian",
            "w": int(slot.w),
            "x": int(slot.x),
            "pitch": pitch,
            "x_offset": x_offset,
            "sigma": round(0.25 * slot.w, 3),
            "index": i,
        }
        all_rois.append(item)
        slots.append(slot)
    return all_rois, slots


def extract_all_rois_predictions(case, all_rois, profiles_root="outputs/projects", data_root="Data"):
    """Run the base trained model on all_rois for all labeled images in the case.

    Returns:
      images: list of image names
      gts: list of GT integer counts
      roi_preds_matrix: np.ndarray shape (n_images, n_rois)
    """
    case_dir = Path(profiles_root) / case
    profile = ProjectProfile.load(case_dir / PROFILE_NAME)
    mask = read_gray(case_dir / ROTATED_MASK_NAME)
    manifest = read_manifest(case_dir / "model")

    # Load base model
    model_file = next(iter(manifest["files"]))
    base_model = joblib.load(case_dir / "model" / model_file)

    grid = AugmentGrid.from_dict(manifest["feature_config"]["grid"])
    x0, _, x1, _ = profile.valid_region["bbox"]
    ext_slots = extent_slots(mask, grid.side_margin)

    slots = [
        RoiSlot(x=r["x"], y0=0, y1=mask.shape[0], w=r["w"], pitch=r["pitch"], x_offset=r["x_offset"])
        for r in all_rois
    ]
    items = [(s, r["projection"]) for s, r in zip(slots, all_rois)]

    data_dir = Path(data_root) / case
    labeled_files = []
    for ext in ("*.png", "*.bmp", "*.jpg"):
        for path in sorted(data_dir.glob(ext)):
            if "background" in path.name:
                continue
            gt = parse_ground_truth(path.name)
            if gt and gt.get("count") is not None:
                labeled_files.append((path, gt["count"]))

    if not labeled_files:
        return [], np.array([]), np.empty((0, len(all_rois)))

    preds_list, gts, names = [], [], []
    for path, gt_val in labeled_files:
        img = rotate_image(read_gray(path), profile.angle_deg)
        extent, _ = stack_rows(img, ext_slots)
        kept = clip_items(items, extent)
        rows = roi_features(img, kept, x0, x1 - x0, grid.gaussian_sigma_ratio)
        if len(rows) != len(all_rois):
            continue
        feats = feature_frame(rows, manifest["scale_normalised"])
        deltas = base_model.predict(feats)
        preds = rows["period_count"].to_numpy(dtype=np.float64) + deltas
        preds_list.append(preds)
        gts.append(gt_val)
        names.append(path.name)

    return names, np.array(gts, dtype=np.int64), np.array(preds_list, dtype=np.float64)


def _ensemble_features(preds):
    """Features for the meta-learner: (preds, median, mean, std, min, max, iqr)."""
    med = np.median(preds, axis=1, keepdims=True)
    mean = np.mean(preds, axis=1, keepdims=True)
    std = np.std(preds, axis=1, keepdims=True)
    p_min = np.min(preds, axis=1, keepdims=True)
    p_max = np.max(preds, axis=1, keepdims=True)
    diff = preds - med
    return np.hstack([preds, diff, med, mean, std, p_min, p_max])


def evaluate_ensembles_loio(preds_matrix, gts):
    """Evaluate LOIO exact hit rate and MAE for median, trimmed_mean, ridge, and xgboost.

    Returns dict of evaluations and the best recommended method.
    """
    n_images, n_rois = preds_matrix.shape
    if n_images == 0:
        return {"median": {"hits": 0, "total": 0, "hit_rate": 0.0, "mae": 0.0}}, "median"

    evals = {}
    methods = ["median", "trimmed_mean", "ridge", "xgboost"]

    for method in methods:
        loio_preds = []
        for i in range(n_images):
            val_preds = preds_matrix[i : i + 1]
            train_preds = np.delete(preds_matrix, i, axis=0)
            train_gts = np.delete(gts, i, axis=0)

            if method == "median":
                pred = float(np.median(val_preds))
            elif method == "trimmed_mean":
                row = np.sort(val_preds[0])
                pred = float(np.mean(row[1:-1])) if len(row) > 2 else float(np.mean(row))
            elif method == "ridge":
                if len(train_gts) >= 2:
                    X_train = _ensemble_features(train_preds)
                    X_val = _ensemble_features(val_preds)
                    reg = Ridge(alpha=1.0)
                    reg.fit(X_train, train_gts)
                    pred = float(reg.predict(X_val)[0])
                else:
                    pred = float(np.median(val_preds))
            elif method == "xgboost":
                if len(train_gts) >= 3:
                    try:
                        from xgboost import XGBRegressor
                        reg = XGBRegressor(n_estimators=15, max_depth=2, learning_rate=0.1, random_state=42)
                    except ImportError:
                        from sklearn.ensemble import GradientBoostingRegressor
                        reg = GradientBoostingRegressor(n_estimators=15, max_depth=2, learning_rate=0.1, random_state=42)
                    X_train = _ensemble_features(train_preds)
                    X_val = _ensemble_features(val_preds)
                    reg.fit(X_train, train_gts)
                    pred = float(reg.predict(X_val)[0])
                else:
                    pred = float(np.median(val_preds))
            loio_preds.append(pred)

        rounded = np.rint(loio_preds).astype(np.int64)
        hits = int(np.sum(rounded == gts))
        mae = float(np.mean(np.abs(np.array(loio_preds) - gts)))
        evals[method] = {
            "hits": hits,
            "total": n_images,
            "hit_rate": round(hits / n_images, 4) if n_images else 0.0,
            "mae": round(mae, 4),
        }

    # Best method: highest exact hits, then lowest MAE, prefer simple median in tie
    def sort_key(m):
        e = evals[m]
        # slight preference bonus for median if hits and mae are essentially identical
        bias = 0.001 if m == "median" else 0.0
        return (e["hits"], -e["mae"] + bias)

    recommended = max(methods, key=sort_key)
    return evals, recommended


def train_all_rois(case, profiles_root="outputs/projects", data_root="Data", device="cuda"):
    """Build all_rois geometry, evaluate multi-ROI ensemble on labeled images, and persist to manifest."""
    case_dir = Path(profiles_root) / case
    model_dir = case_dir / "model"
    if not (model_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{case}: manifest.json not found; run `escount train` first")

    all_rois, slots = build_all_rois_layout(case, profiles_root, data_root)
    names, gts, preds_matrix = extract_all_rois_predictions(case, all_rois, profiles_root, data_root)

    evals, recommended = evaluate_ensembles_loio(preds_matrix, gts)

    # Train final ensemble model on all available labeled data
    final_meta = {}
    if len(gts) >= 2:
        X_all = _ensemble_features(preds_matrix)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_all, gts)
        final_meta["ridge"] = ridge

        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=15, max_depth=2, learning_rate=0.1, random_state=42)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            xgb = GradientBoostingRegressor(n_estimators=15, max_depth=2, learning_rate=0.1, random_state=42)
        xgb.fit(X_all, gts)
        final_meta["xgboost"] = xgb

        joblib.dump(final_meta, model_dir / ENSEMBLE_MODEL_NAME)

    ensemble_info = {
        "recommended": recommended,
        "evaluation": evals,
        "model_file": ENSEMBLE_MODEL_NAME if final_meta else None,
        "roi_count": len(all_rois),
    }

    # Update manifest
    update_manifest(model_dir, all_rois=all_rois, all_rois_ensemble=ensemble_info)
    return ensemble_info
