"""Feature cache (Task 5, autoresearch "prepare").

prepare(case) computes feature rows for every image of Data/<case> with a numeric file name (new
images are picked up automatically) over the augmentation grid, in a process pool, and writes
  <cache_root>/<case>/features.parquet
  <cache_root>/<case>/manifest.json   (feature config + hash, project.json SHA, image SHA list)
The cache is rebuilt whenever the feature config, the project profile or the image set changes.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
from escount.calibration.rotation_correction import rotate_image
from escount.es_counting import file_sha256, list_images, parse_ground_truth, read_gray, scenario_dirs, write_json
from escount.features.extract import (
    AugmentGrid, bank_items, config_hash, extent_slots, feature_config, image_features,
)

FEATURES_NAME = "features.parquet"
MANIFEST_NAME = "manifest.json"


def labelled_images(case_dir):
    """[(path, gt dict)] for every analysable image whose file name is a sheet count."""
    return [(path, gt) for path in list_images(case_dir) if (gt := parse_ground_truth(path.name))]


def case_ids(data_root="Data"):
    return [path.name for path in scenario_dirs(data_root)]


def _image_worker(task):
    """Top-level (picklable) worker: one image -> feature rows (Windows spawn safe)."""
    import cv2

    cv2.setNumThreads(1)
    case, path, gt, angle, items, ext_slots, valid_x0, valid_width, grid = task
    rotated = rotate_image(read_gray(path), angle)
    table, _ = image_features(rotated, items, ext_slots, valid_x0, valid_width, grid)
    table.insert(0, "gt_group", gt["group"])
    table.insert(0, "gt", gt["count"])
    table.insert(0, "image_id", f"{case}/{Path(path).name}")
    table.insert(0, "image", Path(path).name)
    table.insert(0, "case", case)
    return table


def profile_fingerprint(profile, rotated_mask_path):
    """SHA-256 of the profile fields that change features (not e.g. selected_rois / created)."""
    fields = {"angle_deg": profile.angle_deg, "valid_region": profile.valid_region,
              "mask_sha256": profile.mask_sha256, "rotated_mask_sha256": file_sha256(rotated_mask_path)}
    return config_hash(fields)


def _manifest(case, grid, profile, rotated_mask_path, images):
    config = feature_config(grid)
    return {"case": case, "feature_config": config, "config_hash": config_hash(config),
            "profile_sha256": profile_fingerprint(profile, rotated_mask_path),
            "images": [{"name": path.name, "sha256": file_sha256(path), "gt": gt["count"]} for path, gt in images]}


def is_fresh(case_dir, manifest):
    keys = ("config_hash", "profile_sha256", "images")
    return case_dir.joinpath(FEATURES_NAME).is_file() and case_dir.joinpath(MANIFEST_NAME).is_file() and \
        {k: json.loads(case_dir.joinpath(MANIFEST_NAME).read_text(encoding="utf-8")).get(k) for k in keys} == \
        {k: manifest[k] for k in keys}


def prepare(case, data_root="Data", profiles_root="outputs/projects", cache_root="outputs/features",
            workers=None, grid=AugmentGrid(), force=False):
    """Build (or reuse a fresh) feature cache for one case. Returns (parquet path, rebuilt flag)."""
    case_dir = Path(data_root) / case
    profile_dir = Path(profiles_root) / case
    profile_path = profile_dir / PROFILE_NAME
    if not profile_path.is_file():
        raise FileNotFoundError(f"{profile_path} missing; run `escount calibrate --project {case}` first")
    profile = ProjectProfile.load(profile_path)
    if profile.quality_status != "ok":
        raise ValueError(f"{case}: project.json quality_status is {profile.quality_status!r}")
    images = labelled_images(case_dir)
    if not images:
        raise ValueError(f"{case_dir}: no image with a numeric file name")
    out_dir = Path(cache_root) / case
    manifest = _manifest(case, grid, profile, profile_dir / ROTATED_MASK_NAME, images)
    if not force and is_fresh(out_dir, manifest):
        return out_dir / FEATURES_NAME, False
    rotated_mask = read_gray(profile_dir / ROTATED_MASK_NAME)
    items = bank_items(rotated_mask, grid)
    ext = extent_slots(rotated_mask, grid.side_margin)
    x0, _, x1, _ = profile.valid_region["bbox"]
    tasks = [(case, str(path), gt, profile.angle_deg, items, ext, x0, x1 - x0, grid) for path, gt in images]
    workers = max(1, min(workers or os.cpu_count() or 1, len(tasks)))
    if workers == 1:
        tables = [_image_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tables = list(pool.map(_image_worker, tasks))
    table = pd.concat(tables, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / FEATURES_NAME, index=False)
    manifest.update(rows=len(table), workers=workers, created=datetime.now().isoformat(timespec="seconds"))
    write_json(out_dir / MANIFEST_NAME, manifest)
    return out_dir / FEATURES_NAME, True


def load_features(case, cache_root="outputs/features"):
    path = Path(cache_root) / case / FEATURES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `escount prepare --project {case}` first")
    return pd.read_parquet(path)


def load_manifest(case, cache_root="outputs/features"):
    return json.loads((Path(cache_root) / case / MANIFEST_NAME).read_text(encoding="utf-8"))
