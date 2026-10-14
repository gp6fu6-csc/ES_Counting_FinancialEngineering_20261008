"""Calibration v2 (Task 2): square-ROI FFT rotation calibration and the project profile.

Per project (one Data/<case> folder = one shooting situation):
  1. pick the reference image (largest file-name number, or --reference),
  2. place `roi_count` square ROIs (side >= 300 px, default 384) on the horizontal centreline of the
     background_BW valid region, evenly spaced vertically (or use manually adjusted ROIs),
  3. FFT angle per ROI (legacy `estimate_roi`), angle = median,
  4. rotate image + mask, re-estimate on the rotated image,
  5. gate: after_residual < before_residual AND after_residual <= tolerance (0.25 deg),
  6. write <profiles_root>/<case>/project.json + background_BW_rotated.png.
"""
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from escount.calibration.rotation_correction import (
    QUALITY_TOLERANCE_DEG, estimate_boxes, normalize_mask, roi_contained, rotate_image, rotate_mask, write_image,
)
from escount.es_counting import file_sha256, list_images, longest_run, pick_reference, read_gray, read_mask

SCHEMA_VERSION = 1
PROFILE_NAME = "project.json"
ROTATED_MASK_NAME = "background_BW_rotated.png"


class CalibrationQualityError(RuntimeError):
    """Raised when the rotation compensation fails the quality gate (profile is still written)."""


@dataclass(frozen=True)
class SquareRoi:
    x: int
    y: int
    side: int

    @property
    def box(self):
        """xywh tuple as used by rotation_correction.estimate_boxes / roi_contained."""
        return (self.x, self.y, self.side, self.side)


@dataclass(frozen=True)
class CalibrationConfig:
    roi_side: int = 384
    roi_count: int = 3
    min_side: int = 300
    tolerance_deg: float = QUALITY_TOLERANCE_DEG


@dataclass(kw_only=True)
class ProjectProfile:
    schema_version: int = SCHEMA_VERSION
    case: str
    angle_deg: float
    rois: list
    roi_angles_deg: list
    before_residual_deg: float
    after_residual_deg: float
    quality_status: str
    mask_sha256: str
    image_size: dict
    valid_region: dict
    reference_image: str
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    selected_rois: list = field(default_factory=list)   # written by `escount select-rois` (Task 7)
    strip_params: dict = field(default_factory=dict)    # counting strips N, w, D set in the calibration UI (Task 11)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported project.json schema_version {version!r} (expected {SCHEMA_VERSION})")
        data["rois"] = [SquareRoi(**item) for item in data["rois"]]
        return cls(**data)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- valid region / ROIs
def main_valid_region(mask):
    """Boolean map of the largest 8-connected valid component (drops stray valid specks at the border)."""
    valid = normalize_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=8)
    if count < 2:
        raise ValueError("mask has no valid (black) region")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def valid_region_info(mask):
    """{bbox: [x0, y0, x1, y1] exclusive end, centre_x: median per-row run midpoint} of the main region."""
    region = main_valid_region(mask)
    ys, xs = np.nonzero(region)
    midpoints = []
    for y in np.unique(ys):
        start, end = longest_run(region[y])
        midpoints.append((start + end) / 2.0)
    return {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centre_x": float(np.median(midpoints))}


def validate_rois(rois, mask, config=CalibrationConfig()):
    """Raise ValueError unless every ROI has side >= min_side and lies fully inside the valid region."""
    valid = normalize_mask(mask)
    for index, roi in enumerate(rois, 1):
        if roi.side < config.min_side:
            raise ValueError(f"ROI {index}: side {roi.side} px is smaller than the minimum {config.min_side} px")
        if not roi_contained(valid, roi.box):
            raise ValueError(f"ROI {index} {roi.box} is out of bounds or not fully inside the valid mask region")
    return list(rois)


def default_rois(mask, config=CalibrationConfig()):
    """`roi_count` squares centred on the valid region centreline, evenly spaced over its valid rows."""
    side = config.roi_side
    if side < config.min_side:
        raise ValueError(f"ROI side {side} px is smaller than the minimum {config.min_side} px")
    region = main_valid_region(mask)
    x = int(round(valid_region_info(mask)["centre_x"] - side / 2.0))
    if x < 0 or x + side > region.shape[1]:
        raise ValueError(f"square ROI of side {side} px does not fit horizontally on the valid-region centreline")
    start, end = longest_run(region[:, x:x + side].all(axis=1))
    if end - start < side:
        raise ValueError(f"valid region is shorter than one {side} px square ROI on its centreline")
    band = (end - start) / config.roi_count
    rois = []
    for index in range(config.roi_count):
        y = int(round(start + (index + 0.5) * band - side / 2.0))
        rois.append(SquareRoi(x, min(max(y, start), end - side), side))
    return validate_rois(rois, mask, config)


def adjust_roi(roi, dx=0, dy=0, side=None):
    """Shift by (dx, dy) px and optionally resize around the ROI centre (manual / UI adjustment)."""
    if side is None:
        return replace(roi, x=roi.x + dx, y=roi.y + dy)
    offset = (roi.side - side) // 2
    return SquareRoi(roi.x + dx + offset, roi.y + dy + offset, side)


# ---------------------------------------------------------------- angle estimate / quality gate
def estimate_angle(image, rois):
    """Return (median angle in deg, per-ROI angles) using the legacy FFT estimator on square boxes."""
    estimates, aggregate = estimate_boxes(image, [roi.box for roi in rois])
    return float(aggregate["theta_deg"]), [float(item["theta_deg"]) for item in estimates]


def quality_status(before_residual, after_residual, tolerance_deg=QUALITY_TOLERANCE_DEG):
    """'ok' iff after < before (strict) and after <= tolerance, else 'quality_failure'."""
    ok = after_residual < before_residual and after_residual <= tolerance_deg
    return "ok" if ok else "quality_failure"


def evaluate_angle(image, mask, rois, angle, config=CalibrationConfig()):
    """Rotate by `angle`, re-estimate and gate. Returns a dict with residuals, status and rotated arrays.

    Re-estimation reuses the same ROIs when they still fit the rotated mask, else default ROIs on it.
    """
    before, _ = estimate_angle(image, rois)
    rotated = rotate_image(image, angle)
    rotated_mask = rotate_mask(mask, angle)
    try:
        after_rois = validate_rois(rois, rotated_mask, config)
    except ValueError:
        after_rois = default_rois(rotated_mask, config)
    after, after_angles = estimate_angle(rotated, after_rois)
    before_residual, after_residual = abs(before), abs(after)
    return {"before_residual_deg": before_residual, "after_residual_deg": after_residual,
            "after_roi_angles_deg": after_angles, "after_rois": after_rois,
            "quality_status": quality_status(before_residual, after_residual, config.tolerance_deg),
            "rotated_image": rotated, "rotated_mask": rotated_mask}


def calibrate_project(case_dir, profiles_root="outputs/projects", reference=None, rois=None,
                      config=CalibrationConfig()):
    """Calibrate one project folder and write project.json + background_BW_rotated.png.

    Raises CalibrationQualityError (after writing the profile with quality_status 'quality_failure')
    when the compensation does not pass the gate.
    """
    case_dir = Path(case_dir).resolve()
    mask_path, mask = read_mask(case_dir)
    reference_path = case_dir / reference if reference else pick_reference(list_images(case_dir))
    image = read_gray(reference_path)
    if image.shape != mask.shape[:2]:
        raise ValueError(f"reference image {image.shape} and mask {mask.shape[:2]} sizes differ")
    rois = validate_rois(rois, mask, config) if rois is not None else default_rois(mask, config)
    angle, roi_angles = estimate_angle(image, rois)
    result = evaluate_angle(image, mask, rois, angle, config)
    profile = ProjectProfile(
        case=case_dir.name, angle_deg=angle, rois=list(rois), roi_angles_deg=roi_angles,
        before_residual_deg=result["before_residual_deg"], after_residual_deg=result["after_residual_deg"],
        quality_status=result["quality_status"], mask_sha256=file_sha256(mask_path),
        image_size={"width": int(image.shape[1]), "height": int(image.shape[0])},
        valid_region=valid_region_info(result["rotated_mask"]), reference_image=reference_path.name,
    )
    out_dir = Path(profiles_root) / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(out_dir / ROTATED_MASK_NAME, result["rotated_mask"])
    profile.save(out_dir / PROFILE_NAME)
    if profile.quality_status != "ok":
        raise CalibrationQualityError(
            f"{case_dir.name}: rotation quality failure (before {profile.before_residual_deg:.4f} deg, "
            f"after {profile.after_residual_deg:.4f} deg, tolerance {config.tolerance_deg} deg)")
    return profile
