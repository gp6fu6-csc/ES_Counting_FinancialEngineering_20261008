import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from escount.calibration.calibrate import (
    CalibrationConfig, CalibrationQualityError, ProjectProfile, SquareRoi, adjust_roi, calibrate_project,
    default_rois, estimate_angle, evaluate_angle, quality_status, validate_rois,
)
from escount.calibration.rotation_correction import normalize_mask, rotate_image
from escount.cli import main as cli_main

PERIOD = 20


def tilted_stripes(angle, height=1200, width=1400, period=PERIOD):
    """Horizontal sinusoidal stripes rotated so that rotate_image(result, angle) straightens them."""
    pad = 200
    rows = np.arange(height + 2 * pad, dtype=np.float64)[:, None]
    straight = 128 + 80 * np.sin(2 * np.pi * rows / period) * np.ones((1, width + 2 * pad))
    tilted = rotate_image(straight.astype(np.float32), -angle)
    return np.clip(tilted[pad:pad + height, pad:pad + width], 0, 255).astype(np.uint8)


def slanted_mask(height=1200, width=1400, left=300, right=1000, slope=0.15):
    """Black (valid) band whose left/right edges shift by `slope` px per row; white elsewhere."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    for y in range(height):
        shift = int(round(slope * y))
        mask[y, left + shift:right + shift] = 0
    mask[:40] = 255
    return mask


class SyntheticTiltTests(unittest.TestCase):
    def test_estimated_tilt_error_below_0_05_deg(self):
        mask = np.zeros((1200, 1400), dtype=np.uint8)
        rois = default_rois(mask)
        for truth in (-0.7, 0.4, 1.3):
            angle, per_roi = estimate_angle(tilted_stripes(truth), rois)
            self.assertLess(abs(angle - truth), 0.05, (truth, angle, per_roi))

    def test_quality_gate_accepts_true_angle_and_rejects_forced_bad_angle(self):
        image, mask = tilted_stripes(1.3), np.zeros((1200, 1400), dtype=np.uint8)
        rois = default_rois(mask)
        angle, _ = estimate_angle(image, rois)
        self.assertEqual(evaluate_angle(image, mask, rois, angle)["quality_status"], "ok")
        bad = evaluate_angle(image, mask, rois, angle - 2.0)  # over-rotated: residual about 2 deg
        self.assertEqual(bad["quality_status"], "quality_failure")
        self.assertGreater(bad["after_residual_deg"], 0.25)

    def test_quality_status_rule(self):
        self.assertEqual(quality_status(1.0, 0.1), "ok")
        self.assertEqual(quality_status(1.0, 0.3), "quality_failure")   # above 0.25 tolerance
        self.assertEqual(quality_status(0.1, 0.1), "quality_failure")   # not strictly smaller
        self.assertEqual(quality_status(0.1, 0.2), "quality_failure")


class RoiTests(unittest.TestCase):
    def test_default_rois_inside_slanted_mask(self):
        mask = slanted_mask()
        valid = normalize_mask(mask)
        rois = default_rois(mask)
        self.assertEqual(len(rois), 3)
        for roi in rois:
            self.assertGreaterEqual(roi.side, 300)
            self.assertTrue(valid[roi.y:roi.y + roi.side, roi.x:roi.x + roi.side].all(), roi)
        ys = [roi.y for roi in rois]
        self.assertEqual(ys, sorted(ys))

    def test_side_below_300_raises(self):
        mask = np.zeros((1200, 1400), dtype=np.uint8)
        with self.assertRaises(ValueError):
            default_rois(mask, CalibrationConfig(roi_side=299))
        with self.assertRaises(ValueError):
            validate_rois([SquareRoi(100, 100, 299)], mask)

    def test_out_of_bounds_and_outside_mask_raise(self):
        mask = slanted_mask()
        with self.assertRaises(ValueError):
            validate_rois([SquareRoi(1200, 100, 384)], mask)    # beyond the right image border
        with self.assertRaises(ValueError):
            validate_rois([SquareRoi(-1, 100, 384)], mask)      # negative x
        with self.assertRaises(ValueError):
            validate_rois([SquareRoi(250, 100, 384)], mask)     # partially left of the valid band

    def test_adjust_roi_shift_and_resize_keep_centre(self):
        roi = SquareRoi(100, 200, 384)
        self.assertEqual(adjust_roi(roi, dx=5, dy=-3), SquareRoi(105, 197, 384))
        self.assertEqual(adjust_roi(roi, side=300), SquareRoi(142, 242, 300))


class ProfileTests(unittest.TestCase):
    def test_json_round_trip(self):
        profile = ProjectProfile(
            case="demo", angle_deg=-1.25, rois=[SquareRoi(1, 2, 384), SquareRoi(3, 4, 400)],
            roi_angles_deg=[-1.2, -1.25], before_residual_deg=1.25, after_residual_deg=0.01,
            quality_status="ok", mask_sha256="ab" * 32, image_size={"width": 10, "height": 20},
            valid_region={"bbox": [0, 0, 10, 20], "centre_x": 5.0}, reference_image="100.png")
        restored = ProjectProfile.from_json(profile.to_json())
        self.assertEqual(restored, profile)
        self.assertEqual(json.loads(profile.to_json())["schema_version"], 1)
        data = json.loads(profile.to_json())
        data["schema_version"] = 99
        with self.assertRaises(ValueError):
            ProjectProfile.from_json(json.dumps(data))


class CalibrateProjectTests(unittest.TestCase):
    def test_calibrate_project_and_cli_write_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "Data" / "synthetic"
            case.mkdir(parents=True)
            cv2.imwrite(str(case / "050.png"), tilted_stripes(0.8))
            cv2.imwrite(str(case / "040.png"), tilted_stripes(0.0))
            cv2.imwrite(str(case / "background_BW.png"), np.zeros((1200, 1400), dtype=np.uint8))
            profiles = Path(tmp) / "projects"
            profile = calibrate_project(case, profiles)
            self.assertEqual(profile.reference_image, "050.png")
            self.assertLess(abs(profile.angle_deg - 0.8), 0.05)
            self.assertEqual(profile.quality_status, "ok")
            self.assertEqual(ProjectProfile.load(profiles / "synthetic" / "project.json"), profile)
            self.assertTrue((profiles / "synthetic" / "background_BW_rotated.png").is_file())
            # the 0-degree image cannot improve a ~0 residual strictly -> quality failure is written and raised
            with self.assertRaises(CalibrationQualityError):
                calibrate_project(case, profiles, reference="040.png")
            self.assertEqual(ProjectProfile.load(profiles / "synthetic" / "project.json").quality_status,
                             "quality_failure")
            code = cli_main(["calibrate", "--project", "synthetic", "--data-root", str(Path(tmp) / "Data"),
                             "--profiles-root", str(profiles)])
            self.assertEqual(code, 0)
            self.assertEqual(ProjectProfile.load(profiles / "synthetic" / "project.json").quality_status, "ok")


if __name__ == "__main__":
    unittest.main()
