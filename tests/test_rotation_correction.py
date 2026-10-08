import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from escount.calibration.rotation_correction import (
    BATCH_DATASETS,
    _baseline_signatures,
    _calibration_quality,
    _masked_horizontal_accumulation,
    estimate_image,
    normalize_angle,
    roi_boxes,
    roi_contained,
    rotate_image,
    rotate_mask,
    run,
    run_batch,
    run_with_parameters,
    select_mask_roi,
)


class RotationCorrectionTests(unittest.TestCase):
    def test_normalize_angle(self):
        self.assertAlmostEqual(normalize_angle(95), -85)
        self.assertAlmostEqual(normalize_angle(-95), 85)

    def test_roi_order_and_bounds(self):
        boxes = roi_boxes(2048, 2448)
        self.assertEqual(len(boxes), 3)
        self.assertLess(boxes[0][1], boxes[1][1])
        self.assertLess(boxes[1][1], boxes[2][1])
        for x, y, w, h in boxes:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 2448)
            self.assertLessEqual(y + h, 2048)

    def test_synthetic_horizontal_texture_is_finite(self):
        image = np.tile(np.arange(2448, dtype=np.uint8), (2048, 1))
        _, estimates, median = estimate_image(image)
        self.assertEqual(len(estimates), 3)
        self.assertTrue(np.isfinite(median))

    def test_physical_frequency_oracles(self):
        # Build two independent wave vectors from the analytic relation:
        # fx=-f*sin(theta), fy=f*cos(theta), in cycles/pixel.  The selected
        # lower-half FFT peak is the conjugate vector, so the estimator still
        # returns the same theta after normalize(angle - 90).
        height, width = 2048, 2448
        yy, xx = np.mgrid[:height, :width]
        for theta, frequency in ((-4.0, 1 / 32.0), (11.0, 1 / 47.0)):
            radians = np.deg2rad(theta)
            fx = -frequency * np.sin(radians)
            fy = frequency * np.cos(radians)
            image = (127 + 100 * np.sin(2 * np.pi * (fx * xx + fy * yy))).astype(np.uint8)
            before = estimate_image(image)[2]
            corrected = rotate_image(image, before)
            after = estimate_image(corrected)[2]
            self.assertAlmostEqual(before, theta, delta=0.5)
            self.assertLess(abs(after), abs(before))
            self.assertAlmostEqual(after, 0.0, delta=0.5)

    def test_run_with_parameters_preserves_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parameters = root / "parameters.json"
            output = root / "nested" / "corrected.png"
            metadata_path = root / "nested" / "metadata.json"
            image = np.arange(120, dtype=np.uint8).reshape(10, 12)
            self.assertTrue(cv2.imwrite(str(source), image))
            parameters.write_text(json.dumps({
                "compensation_angle_deg": -1.25,
                "rotation": {
                    "interpolation": "INTER_LINEAR",
                    "border": "BORDER_REFLECT_101",
                    "same_size": True,
                },
            }), encoding="utf-8")
            before = source.read_bytes()
            result = run_with_parameters(source, parameters, output, metadata_path)
            self.assertEqual(source.read_bytes(), before)
            corrected = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(corrected)
            self.assertEqual(corrected.shape, image.shape)
            self.assertEqual(corrected.dtype, image.dtype)
            self.assertEqual(result["compensation_angle_deg"], -1.25)
            self.assertTrue(result["verification"]["original_unchanged"])
            self.assertTrue(result["verification"]["shape_matches_input"])
            self.assertEqual(result["original_file_size_before"], result["original_file_size_after"])
            self.assertEqual(result["original_sha256_before"], result["original_sha256_after"])

        source = Path("Data/T-type/160.png")
        if not source.exists():
            self.skipTest("requested sample is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = run(source, Path(directory))
            self.assertTrue(np.isfinite(result["estimated_angle_deg"]))
            self.assertTrue(np.isfinite(result["compensation_angle_deg"]))
            output = Path(directory)
            for name in ("original.png", "corrected.png", "comparison.png", "fft_diagnostic.png", "edges_before.png", "edges_after.png"):
                image = cv2.imread(str(output / name))
                self.assertIsNotNone(image, name)
            self.assertTrue((output / "verification_summary.txt").is_file())
            self.assertLess(result["after_abs_median_angle_deg"], result["before_abs_median_angle_deg"])
            self.assertGreater(result["angle_improvement_deg"], 1.0)
            self.assertEqual(cv2.imread(str(output / "corrected.png"), 0).shape, cv2.imread(str(source), 0).shape)
            data = json.loads((output / "rotation_parameters.json").read_text(encoding="utf-8"))
            required = {"original.png", "corrected.png", "comparison.png", "fft_diagnostic.png", "edges_before.png", "edges_after.png", "rotation_parameters.json", "verification_summary.txt"}
            self.assertEqual({Path(path).name for path in data["output_files"]}, required)
            self.assertEqual(len(data["before_theta_i_deg"]), 3)
            self.assertLess(data["after_abs_median_angle_deg"], data["before_abs_median_angle_deg"])
            self.assertIn("dc_exclusion_radius_px", data["fft"])
    def test_mask_roi_containment_and_impossible_mask(self):
        # Black (< 128) is the steel/valid region; near-black noise values are still valid.
        mask = np.full((128, 256), 255, dtype=np.uint8)
        mask[:, 32:224] = 5
        roi = (32, 0, 192, 128)
        self.assertTrue(roi_contained(mask, roi))
        self.assertFalse(roi_contained(mask, (31, 0, 192, 128)))
        with self.assertRaises(ValueError):
            select_mask_roi(np.full((128, 256), 255, dtype=np.uint8))
        rotated = rotate_mask(mask, 10)
        self.assertEqual(int(rotated[0, 0]), 255)  # outside the source image is invalid

    def test_quality_gate_and_roi_limited_accumulation(self):
        gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
        mask = np.ones_like(gray, dtype=np.uint8)
        roi = {"xywh": [1, 1, 3, 2]}
        counts, totals, means, x_start = _masked_horizontal_accumulation(gray, mask, roi)
        self.assertEqual(x_start, 1)
        self.assertEqual(len(counts), 3)
        self.assertEqual(counts.tolist(), [2, 2, 2])
        self.assertEqual(totals.tolist(), [20.0, 22.0, 24.0])
        self.assertEqual(rotate_mask(mask, 0).shape, mask.shape)

        measured = {"theta_deg": 1.0, "peak_value": 2.0}
        after = {"theta_deg": 0.1, "peak_value": 3.0}
        quality = _calibration_quality(measured, after, np.zeros((2048, 2448), dtype=np.uint8))
        self.assertEqual(quality["status"], "ok")
        self.assertTrue(quality["checks"]["within_tolerance"])
        invalid = _calibration_quality(measured, after, np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(invalid["status"], "quality_failure")
        self.assertFalse(invalid["checks"]["input_shape_valid"])

    def test_three_roi_policy_and_normalized_peak_contract(self):
        mask = np.zeros((768, 1024), dtype=np.uint8)
        roi = select_mask_roi(mask)
        self.assertEqual(roi["roi_count"], 3)
        for box in roi["rois_xywh"]:
            self.assertTrue(roi_contained(mask, box))
        measured = {"theta_deg": 1.0, "peak_value": 2.0, "normalized_peak_magnitude": 1e-7}
        after = {"theta_deg": 0.1, "peak_value": 3.0, "normalized_peak_magnitude": 2e-6}
        quality = _calibration_quality(measured, after, np.zeros((2048, 2448), dtype=np.uint8))
        self.assertEqual(quality["normalized_peak_threshold"], 1e-6)
        self.assertEqual(quality["status"], "quality_failure")
        self.assertFalse(quality["checks"]["normalized_peak_above_threshold"])

    def test_color_input_is_rejected_before_grayscale_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "Data"
            for dataset in BATCH_DATASETS:
                dataset_dir = data_root / dataset
                dataset_dir.mkdir(parents=True)
                cv2.imwrite(str(dataset_dir / "background_BW.png"), np.zeros((128, 256), dtype=np.uint8))
                cv2.imwrite(str(dataset_dir / "color.png"), np.zeros((128, 256, 3), dtype=np.uint8))
                if dataset == BATCH_DATASETS[0]:
                    cv2.imwrite(str(dataset_dir / "096.png"), np.zeros((128, 256, 3), dtype=np.uint8))
            result = run_batch(data_root, root / "outputs")
            item = json.loads((root / "outputs" / BATCH_DATASETS[1] / "color" / "result.json").read_text())
            self.assertEqual(item["status"], "failure")
            self.assertTrue(item["failure_reasons"])
            self.assertTrue(item["source"]["before"]["sha256"])
            self.assertTrue(item["corrected_image_absent"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Data"
            for dataset in BATCH_DATASETS:
                (root / dataset).mkdir(parents=True)
                (root / dataset / "image.png").write_bytes(b"source")
            missing = _baseline_signatures(root, Path(directory) / "missing.json", BATCH_DATASETS)
            self.assertFalse(missing["valid"])
            self.assertIn("baseline_missing", missing["mismatches"])
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(json.dumps([{
                "relative_path": f"{BATCH_DATASETS[0]}\\\\image.png",
                "byte_size": 6,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }]), encoding="utf-8")
            exact = _baseline_signatures(root, baseline_path, BATCH_DATASETS)
            self.assertFalse(exact["valid"])
            self.assertIn(f"{BATCH_DATASETS[1]}/image.png", exact["extra"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Data"
            output_root = Path(directory) / "outputs"
            image = (127 + 80 * np.sin(2 * np.pi * np.arange(256, dtype=np.float32) / 32)).astype(np.uint8)
            image = np.tile(image, (128, 1))
            source_bytes = {}
            for dataset, names in ((BATCH_DATASETS[0], ("010.png", "096.png")), (BATCH_DATASETS[1], ("158.png", "160.png"))):
                dataset_dir = root / dataset
                dataset_dir.mkdir(parents=True)
                self.assertTrue(cv2.imwrite(str(dataset_dir / "background_BW.png"), np.zeros((128, 256), dtype=np.uint8)))
                self.assertTrue(cv2.imwrite(str(dataset_dir / "background.png"), image))
                for name in names:
                    path = dataset_dir / name
                    self.assertTrue(cv2.imwrite(str(path), image))
                    source_bytes[path] = path.read_bytes()

            result = run_batch(root, output_root)
            self.assertEqual(result["status"], "failed")
            for dataset in BATCH_DATASETS:
                dataset_output = output_root / dataset
                summary = json.loads((dataset_output / "dataset_summary.json").read_text(encoding="utf-8"))
                aggregate = json.loads((dataset_output / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["analysis"]["status"], "unavailable")
                self.assertEqual(summary["analysis"]["model_status"], "not_supplied")
                self.assertTrue(json.loads((dataset_output / "roi_diagnostics.json").read_text(encoding="utf-8"))["roi_contained"])
                self.assertIsNotNone(cv2.imread(str(dataset_output / "mask_roi_diagnostic.png"), cv2.IMREAD_UNCHANGED))
                self.assertEqual(aggregate["result_count"], len(aggregate["results"]))
                with (dataset_output / "results.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(aggregate["results"]))
                for item in aggregate["results"]:
                    self.assertEqual(item["roi"]["valid_ratio"], 1.0)
                    self.assertTrue(roi_contained(np.ones((128, 256), dtype=np.uint8), item["roi"]["xywh"]))
                    if item["status"] == "ok":
                        self.assertEqual(item["analysis"]["status"], "unavailable")
                        self.assertIn("No validated KD/MACD/count model", item["analysis"]["reason"])
                        self.assertEqual(item["analysis"]["roi_count"], 3)
                        self.assertEqual(len(item["analysis"]["signals"]), 3)
                        for signal in item["analysis"]["signals"]:
                            self.assertTrue(Path(signal["signal_csv"]).is_file())
                            self.assertIsNotNone(cv2.imread(signal["signal_png"], cv2.IMREAD_UNCHANGED))
                        self.assertEqual(cv2.imread(item["output_files"][0], cv2.IMREAD_UNCHANGED).shape, image.shape)
                    else:
                        self.assertEqual(item["analysis"]["status"], "unavailable")
                        self.assertEqual(item["analysis"]["model_status"], "not_supplied")
                        self.assertEqual(item["analysis"]["execution_status"], "not_run")
                        self.assertEqual(item["calibration_quality"]["status"], "quality_failure")
                        self.assertGreaterEqual(item["calibration_quality"]["after_residual_deg"], item["calibration_quality"]["before_residual_deg"])
                        self.assertFalse((Path(item["output_dir"]) / "corrected.png").exists())

            reference = json.loads((output_root / BATCH_DATASETS[0] / "reference_096.json").read_text(encoding="utf-8"))
            self.assertEqual(reference["reference_image"]["status"], "quality_failure")
            self.assertFalse(reference["reference_image"]["usable"])
            self.assertEqual(reference["reference_image"]["calibration_status"], "quality_failure")
            self.assertGreaterEqual(reference["reference_image"]["quality"]["after_residual_deg"], reference["reference_image"]["quality"]["before_residual_deg"])
            self.assertEqual(reference["reference_image"]["path"], str((root / BATCH_DATASETS[0] / "096.png").resolve()))
            first_results = json.loads((output_root / BATCH_DATASETS[0] / "results.json").read_text(encoding="utf-8"))["results"]
            self.assertEqual({item["calibration"]["angle_deg"] for item in first_results}, {first_results[0]["calibration"]["angle_deg"]})
            for path, before in source_bytes.items():
                self.assertEqual(path.read_bytes(), before)

