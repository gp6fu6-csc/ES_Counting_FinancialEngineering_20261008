import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from escount.calibration.calibrate import ProjectProfile
from escount.features.cache import load_features, prepare
from escount.features.extract import (
    FEATURE_INDICATORS, AugmentGrid, batch_counts, bank_items, count_columns, extent_slots, image_features,
)
from escount.indicators.counters import IndicatorConfig, count
from escount.research.evaluate import EvalResult, evaluate, exact_hit, rank_key, wilson_interval
from feature_tables import synthetic_table

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
LEGACY = ROOT / "outputs" / "es_counting"


class SpyProcedure:
    """Records every training call; predicts the training-set GT mean (never sees test GT)."""

    def __init__(self):
        self.calls = []

    def fit(self, train_rows):
        self.calls.append(set(train_rows["image_id"]))
        self.cases = set(train_rows["case"])
        self.mean = train_rows["gt"].mean()

    def predict(self, test_rows):
        assert "gt" not in test_rows.columns
        return pd.Series(self.mean, index=test_rows["image_id"].unique())


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.table = synthetic_table(cases=("A", "B", "C"), images_per_case=4, rows_per_image=6)

    def test_loio_never_trains_on_held_out_image_rows(self):
        spy = SpyProcedure()
        result = evaluate(spy, self.table, "LOIO")
        images = sorted(self.table["image_id"].unique())
        self.assertEqual(len(spy.calls), len(images))
        for held, trained in zip(images, spy.calls):
            self.assertNotIn(held, trained)                       # all rows / augmentations of it excluded
            self.assertEqual(trained, set(images) - {held})
        self.assertEqual(result.total, len(images))

    def test_lopo_never_trains_on_held_out_project(self):
        spy = SpyProcedure()
        seen = []
        original = spy.fit

        def fit(train_rows):
            original(train_rows)
            seen.append(set(train_rows["case"]))

        spy.fit = fit
        result = evaluate(spy, self.table, "LOPO")
        self.assertEqual(seen, [{"B", "C"}, {"A", "C"}, {"A", "B"}])
        self.assertEqual(set(result.per_image["fold"]), {"A", "B", "C"})

    def test_exact_hit_only(self):
        self.assertTrue(exact_hit([9.6], [10])[0])
        self.assertFalse(exact_hit([10.6], [10])[0])
        self.assertFalse(exact_hit([np.nan], [10])[0])
        self.assertFalse(exact_hit([11.0], [10])[0])               # no +-1 tolerance

    def test_rank_key_ties_roi_count_then_latency(self):
        def make(rate, rois, latency):
            return EvalResult(rate, 0, 0, rois, latency, pd.DataFrame())

        results = [make(0.5, 3, 1.0), make(0.75, 9, 9.0), make(0.75, 4, 5.0), make(0.75, 4, 2.0)]
        ordered = sorted(results, key=rank_key)
        self.assertEqual([(r.hit_rate, r.roi_count, r.latency_ms) for r in ordered],
                         [(0.75, 4, 2.0), (0.75, 4, 5.0), (0.75, 9, 9.0), (0.5, 3, 1.0)])

    def test_wilson_interval_known_values(self):
        for hits, total, expected in ((0, 10, (0.0, 0.2775)), (10, 10, (0.7225, 1.0)), (5, 10, (0.2366, 0.7634))):
            low, high = wilson_interval(hits, total)
            self.assertAlmostEqual(low, expected[0], places=4)
            self.assertAlmostEqual(high, expected[1], places=4)


def stack_image(sheets=40, period=12, top=300, height=900, width=600, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 15.0)
    rows = np.arange(top, top + sheets * period)
    image[rows] = np.where((rows - top) % period < period * 2 // 3, 170.0, 50.0)[:, None]
    image += rng.normal(0, 3, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class FeatureTests(unittest.TestCase):
    def test_batch_counts_equal_counters_per_row(self):
        rng = np.random.default_rng(1)
        lengths = np.array([300, 280, 260])
        base = np.sin(2 * np.pi * np.arange(300) / 11.0)[None, :] * 40 + 120
        matrix = base + rng.normal(0, 6, (3, 300))
        for row, length in enumerate(lengths):
            matrix[row, length:] = matrix[row, length - 1]
        for config in FEATURE_INDICATORS:
            up, down = batch_counts(matrix, lengths, config)
            for row, length in enumerate(lengths):
                for direction, expected in (("up", up[row]), ("down", down[row]), ("both", (up[row] + down[row]) / 2)):
                    single = IndicatorConfig(config.name, config.params, config.gate, direction)
                    self.assertEqual(count(matrix[row, :length], single)[0], expected, (config, direction))

    def test_features_and_parquet_round_trip(self):
        image = stack_image()
        mask = np.zeros(image.shape, dtype=np.uint8)
        grid = AugmentGrid(widths=(10, 20), pitches=(30,), x_offsets=(0, 15), projections=("mean", "gaussian"))
        table, extent = image_features(image, bank_items(mask, grid), extent_slots(mask), 0, image.shape[1], grid)
        self.assertLessEqual(abs(extent["y0"] - 300), 25)
        self.assertTrue(set(count_columns()) <= set(table.columns))
        self.assertTrue(((table["period"] - 12).abs() <= 1).all())
        self.assertTrue(((table["period_count"] - 40).abs() <= 2).all())
        self.assertEqual(set(table["x_offset"]), {0, 15})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.parquet"
            table.to_parquet(path, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(path), table)

    def test_prepare_multiprocess_cache_and_new_images(self):
        grid = AugmentGrid(widths=(10,), pitches=(30,), x_offsets=(0,), projections=("mean",))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            case_dir, profile_dir = tmp / "Data" / "Syn", tmp / "projects" / "Syn"
            case_dir.mkdir(parents=True)
            profile_dir.mkdir(parents=True)
            mask = np.zeros((900, 600), dtype=np.uint8)
            cv2.imwrite(str(case_dir / "background_BW.png"), mask)
            cv2.imwrite(str(profile_dir / "background_BW_rotated.png"), mask)
            for sheets in (30, 40):
                cv2.imwrite(str(case_dir / f"{sheets:03d}.png"), stack_image(sheets, seed=sheets))
            ProjectProfile(case="Syn", angle_deg=0.0, rois=[], roi_angles_deg=[], before_residual_deg=0.0,
                           after_residual_deg=0.0, quality_status="ok", mask_sha256="", image_size={},
                           valid_region={"bbox": [0, 0, 600, 900], "centre_x": 300.0},
                           reference_image="040.png").save(profile_dir / "project.json")
            args = dict(data_root=tmp / "Data", profiles_root=tmp / "projects", cache_root=tmp / "features",
                        workers=2, grid=grid)
            path, rebuilt = prepare("Syn", **args)
            self.assertTrue(rebuilt)
            table = load_features("Syn", tmp / "features")
            self.assertEqual(sorted(table["image"].unique()), ["030.png", "040.png"])
            self.assertEqual(sorted(table["gt"].unique()), [30, 40])
            manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([i["name"] for i in manifest["images"]], ["030.png", "040.png"])
            self.assertFalse(prepare("Syn", **args)[1])               # fresh cache reused
            profile = ProjectProfile.load(profile_dir / "project.json")
            profile.selected_rois = [{"key": "mean|10|50"}]
            profile.save(profile_dir / "project.json")
            self.assertFalse(prepare("Syn", **args)[1])               # selected_rois does not change features
            cv2.imwrite(str(case_dir / "050.png"), stack_image(50, seed=5))
            self.assertTrue(prepare("Syn", **args)[1])                # new image -> rebuilt
            self.assertIn("050.png", set(load_features("Syn", tmp / "features")["image"]))


class LegacyRegressionTests(unittest.TestCase):
    @unittest.skipUnless((LEGACY / "T-type" / "calibration_report.json").is_file(), "legacy outputs missing")
    def test_legacy_report_folds_exact_hits(self):
        from escount.research.baseline import legacy_report_hits

        expected = {"20260904-thinkness02_postition1": 8, "20260904-thinkness02_postition2": 10,
                    "Ring-type": 1, "T-type": 3}
        totals = {"20260904-thinkness02_postition1": 19, "20260904-thinkness02_postition2": 14,
                  "Ring-type": 6, "T-type": 4}
        for case, hits in expected.items():
            _, got, total = legacy_report_hits(LEGACY / case)
            self.assertEqual((got, total), (hits, totals[case]), case)

    @unittest.skipUnless((DATA / "T-type").is_dir(), "Data/T-type missing")
    def test_legacy_procedure_reproduces_t_type(self):
        from escount.research.baseline import legacy_baseline

        final, results = legacy_baseline(DATA / "T-type")
        self.assertEqual(final, "macd")
        self.assertEqual((results["macd"].hits, results["macd"].total), (3, 4))


if __name__ == "__main__":
    unittest.main()
