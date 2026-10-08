import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from escount.calibration.rotation_correction import normalize_mask
from escount.es_counting import (
    _rank, analyze, calibrate, count_strips, list_images, parse_ground_truth, pick_reference, predict,
    stack_extent, strip_layout, strip_signals,
)

PERIOD = 12


def synthetic_stack(count, height=640, width=400, base=40, period=PERIOD, seed=0):
    """Dark smooth background with `count` bright sheets (dark gaps) sitting on a base strip."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 12, dtype=np.float64)
    bottom = height - base
    top = bottom - count * period
    rows = np.arange(top, bottom)
    phase = (rows - top) % period
    image[top:bottom] = np.where(phase < period * 0.7, 170.0, 40.0)[:, None]
    image[bottom:] = 220
    image += rng.normal(0, 3, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def synthetic_mask(height=640, width=400, base=40):
    mask = np.zeros((height, width), dtype=np.uint8)  # black = steel region
    mask[height - base:] = 255
    return mask


class GroundTruthTests(unittest.TestCase):
    def test_file_name_formats(self):
        self.assertEqual(parse_ground_truth("096.png"), {"count": 96, "group": 1})
        self.assertEqual(parse_ground_truth("083_2.png"), {"count": 83, "group": 2})
        self.assertEqual(parse_ground_truth("075-1.png"), {"count": 75, "group": 1})
        self.assertEqual(parse_ground_truth("159_1.png"), {"count": 159, "group": 1})
        self.assertIsNone(parse_ground_truth("background.png"))
        self.assertIsNone(parse_ground_truth("sample_a.png"))

    def test_reference_is_largest_number(self):
        paths = [Path(name) for name in ("065.png", "122.png", "121.png", "note.png")]
        self.assertEqual(pick_reference(paths).name, "122.png")
        paths = [Path(name) for name in ("170_2.png", "170_1.png", "159_1.png")]
        self.assertEqual(pick_reference(paths).name, "170_1.png")

    def test_black_mask_is_valid_including_near_black_noise(self):
        mask = np.array([[0, 5, 127, 128, 255]], dtype=np.uint8)
        self.assertEqual(normalize_mask(mask).tolist(), [[True, True, True, False, False]])


class StripTests(unittest.TestCase):
    def test_strips_stay_inside_slanted_band_and_fill_it(self):
        height, width = 400, 300
        mask = np.full((height, width), 255, dtype=np.uint8)
        for y in range(height):
            left = 60 + y // 20
            mask[y, left:left + 150] = 0
        layout = strip_layout(mask, 15, side_margin=0)
        valid = normalize_mask(mask)
        self.assertGreater(layout["count"], 5)
        for strip in layout["strips"]:
            self.assertTrue(valid[strip["y0"]:strip["y1"], strip["x"]:strip["x"] + 15].all())
        # rectangular band 90..240: exactly band_width / w adjacent strips without side margin
        rect = np.full((200, 300), 255, dtype=np.uint8)
        rect[:, 90:240] = 0
        self.assertEqual(strip_layout(rect, 15, side_margin=0)["count"], 10)

    def test_side_margin_and_pitch(self):
        rect = np.full((200, 600), 255, dtype=np.uint8)
        rect[:, 100:500] = 0                       # steel band x = 100..499
        layout = strip_layout(rect, 10, side_margin=60)
        xs = [s["x"] for s in layout["strips"]]
        self.assertGreaterEqual(min(xs) - 100, 60)              # >= 60 px from the left edge
        self.assertGreaterEqual(500 - (max(xs) + 10), 60)       # >= 60 px from the right edge
        self.assertEqual(layout["count"], (400 - 2 * 60) // 10)  # N maximized inside the margins
        spaced = strip_layout(rect, 10, step=30, side_margin=60)
        self.assertEqual(spaced["gap_px"], 20)
        self.assertTrue(all(b["x"] - a["x"] == 30 for a, b in zip(spaced["strips"], spaced["strips"][1:])))
        with self.assertRaises(ValueError):
            strip_layout(rect, 20, step=10)                      # D < w is rejected
        with self.assertRaises(ValueError):
            strip_layout(rect, 10, side_margin=250)              # margins leave no room

    def test_stack_extent_and_signal_period(self):
        image, mask = synthetic_stack(30), synthetic_mask()
        layout = strip_layout(mask, 15)
        extent = stack_extent(image, layout)
        expected_top = 640 - 40 - 30 * PERIOD
        self.assertLess(abs(extent["y0"] - expected_top), 25)
        signals, lengths, starts = strip_signals(image, layout, extent)
        self.assertTrue((starts >= extent["y0"]).all())
        spectrum = np.abs(np.fft.rfft(signals[0, :lengths[0]] - signals[0, :lengths[0]].mean()))
        cycles = int(np.argmax(spectrum[1:]) + 1)
        self.assertAlmostEqual(lengths[0] / cycles, PERIOD, delta=1.0)


class CounterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        h = np.arange(50 * 14)
        self.clean = (100 + 60 * np.sin(2 * np.pi * h / 14))[None, :]
        self.noisy = self.clean + rng.normal(0, 4, self.clean.shape)
        self.lengths = np.array([self.clean.shape[1]])

    def test_kd_and_macd_count_fifty_periods(self):
        kd, _, _ = count_strips("kd", self.noisy, self.lengths, {"n": 9, "k": 3, "d": 3}, 20, "up")
        macd, _, _ = count_strips("macd", self.noisy, self.lengths, {"ema1": 3, "ema2": 8, "day": 3}, 0.0, "up")
        self.assertLessEqual(abs(kd[0] - 50), 1)
        self.assertLessEqual(abs(macd[0] - 50), 1)

    def test_direction_and_gate(self):
        line = {"n": 9, "k": 3, "d": 3}
        up, _, _ = count_strips("kd", self.clean, self.lengths, line, 0, "up")
        down, _, _ = count_strips("kd", self.clean, self.lengths, line, 0, "down")
        both, _, _ = count_strips("kd", self.clean, self.lengths, line, 0, "both")
        self.assertEqual(both[0], (up[0] + down[0]) / 2)
        rng = np.random.default_rng(2)
        jittery = self.clean + rng.normal(0, 15, self.clean.shape)
        ungated, _, _ = count_strips("macd", jittery, self.lengths, {"ema1": 2, "ema2": 5, "day": 2}, None, "up")
        gated, _, _ = count_strips("macd", jittery, self.lengths, {"ema1": 2, "ema2": 5, "day": 2}, 0.0, "up")
        self.assertLess(gated[0], ungated[0])

    def test_rank_ignores_held_out_image(self):
        predictions = np.array([[10.0, 20.0, 99.0], [11.0, 21.0, 30.0]])
        truth = np.array([10.0, 20.0, 30.0])
        self.assertEqual(_rank(predictions, truth, np.array([0, 1]))[0], 0)
        self.assertEqual(_rank(predictions, truth, np.array([0, 1, 2]))[0], 1)


class EndToEndTests(unittest.TestCase):
    def test_calibrate_and_analyze_synthetic_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = root / "Data" / "synthetic"
            scenario.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(scenario / "background_BW.png"), synthetic_mask()))
            counts = (20, 25, 30, 34)
            for seed, count in enumerate(counts):
                self.assertTrue(cv2.imwrite(str(scenario / f"{count:03d}.png"), synthetic_stack(count, seed=seed)))
            sources = {p: p.read_bytes() for p in scenario.iterdir()}
            output_root = root / "outputs"
            parameters = calibrate(scenario, output_root)
            results = analyze(scenario, output_root)
            out = output_root / "synthetic"
            for name in ("rotation_parameters.json", "background_BW_rotated.png", "strip_layout.json",
                         "counting_parameters.json", "calibration_report.json", "summary.csv"):
                self.assertTrue((out / name).is_file(), name)
            self.assertFalse(parameters["ml"]["eligible"])
            self.assertIsNone(parameters["ml"]["model_path"])
            self.assertIn(parameters["final_method"], ("kd", "macd"))
            self.assertLessEqual(parameters["loo_metrics"][parameters["final_method"]]["loo_mae"], 1.0)
            rotated = normalize_mask(cv2.imread(str(out / "background_BW_rotated.png"), cv2.IMREAD_UNCHANGED))
            self.assertEqual(len(results), len(counts))
            for item in results:
                saved = json.loads((out / Path(item["image"]).stem / "result.json").read_text(encoding="utf-8"))
                for key in ("rotation", "parameters", "predictions", "final", "ground_truth", "stack_extent", "source"):
                    self.assertIn(key, saved)
                self.assertTrue(saved["source"]["unchanged"])
                self.assertLessEqual(abs(saved["error"]), 1)
                for strip in saved["predictions"]["per_strip"]:
                    self.assertIn("kd_count", strip)
                    self.assertTrue(rotated[strip["y0"]:strip["y1"], strip["x"]:strip["x"] + saved["strips"]["width_px"]].all())
                self.assertIsNotNone(cv2.imread(str(out / Path(item["image"]).stem / "analysis.png")))
            with (out / "summary.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(counts))
            for path, data in sources.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertEqual(len(list_images(scenario)), len(counts))

            # predict: arbitrary/sequential names, no ground truth, separate output, no source changes
            batch = root / "new_batch"
            batch.mkdir()
            (batch / "0001.png").write_bytes((scenario / "025.png").read_bytes())
            (batch / "shot_b.png").write_bytes((scenario / "030.png").read_bytes())
            summary_before = (out / "summary.csv").read_bytes()
            predicted = predict(out, [batch], root / "pred")
            by_name = {item["image"]: item for item in predicted}
            reference = {item["image"]: item["final"]["count"] for item in results}
            self.assertEqual(by_name["0001.png"]["final"]["count"], reference["025.png"])
            self.assertEqual(by_name["shot_b.png"]["final"]["count"], reference["030.png"])
            self.assertIsNone(by_name["0001.png"]["ground_truth"])
            self.assertFalse(by_name["0001.png"]["calibration"]["used_in_calibration"])
            self.assertEqual((out / "summary.csv").read_bytes(), summary_before)
            with (root / "pred" / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            # a folder whose own background_BW differs from the calibrated scenario is rejected
            other = root / "other_scene"
            other.mkdir()
            (other / "x.png").write_bytes((scenario / "025.png").read_bytes())
            different = synthetic_mask()
            different[:, :20] = 255
            self.assertTrue(cv2.imwrite(str(other / "background_BW.png"), different))
            rejected = predict(out, [other / "x.png"], root / "pred_other")
            self.assertEqual(rejected[0]["status"], "failure")

    def test_xgboost_model_round_trip(self):
        import xgboost
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(60, 8)), rng.normal(size=60)
        model = xgboost.XGBRegressor(max_depth=3, n_estimators=20, random_state=0, n_jobs=1).fit(x, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save_model(str(path))
            loaded = xgboost.XGBRegressor()
            loaded.load_model(str(path))
            np.testing.assert_allclose(model.predict(x), loaded.predict(x))


if __name__ == "__main__":
    unittest.main()
