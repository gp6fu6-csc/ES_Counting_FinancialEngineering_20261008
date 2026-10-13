import time
import unittest

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from escount.calibration.calibrate import ProjectProfile
from escount.features.extract import AugmentGrid, bank_items, extent_slots, feature_config, image_features
from escount.inference.benchmark import ProjectRuntime
from escount.models.select_rois import check_selected_overlap, greedy_select, select_units, unit_geometry
from escount.models.train import feature_frame, fit_model
from escount.research.evaluate import evaluate
from feature_tables import synthetic_table


class ColumnProcedure:
    """Row prediction = the `signal` column (no fitting): isolates the selection logic."""

    def fit(self, train_rows):
        pass

    def predict_rows(self, test_rows):
        return test_rows["signal"].astype(float)

    def predict(self, test_rows):
        return self.predict_rows(test_rows).groupby(test_rows["image_id"]).median()


def selection_table():
    table = synthetic_table(images_per_case=10, rows_per_image=24, seed=3)
    rng = np.random.default_rng(7)
    good = table["x"] < 200                      # slots left of x=200 measure GT exactly
    table["signal"] = np.where(good, table["gt"], table["gt"] + rng.choice([-3, 2, 4], len(table)))
    return table


class SelectionTests(unittest.TestCase):
    def test_subset_hit_rate_at_least_full_set_and_deterministic(self):
        table = selection_table()
        full = evaluate(ColumnProcedure(), table, "LOIO")
        first = select_units(ColumnProcedure(), table, "slot_key")
        second = select_units(ColumnProcedure(), table, "slot_key")
        self.assertGreaterEqual(first["hit_rate"], full.hit_rate)
        self.assertEqual(first["hits"], first["total"])
        self.assertEqual(first["roi_count"], 1)
        self.assertLess(first["roi_count"], full.roi_count)
        self.assertEqual([r["key"] for r in first["selected_rois"]], [r["key"] for r in second["selected_rois"]])
        self.assertLess(first["selected_rois"][0]["x"], 200)
        self.assertEqual(first["full_set"]["hits"], full.hits)

    def test_general_position_units(self):
        table = selection_table()
        result = select_units(ColumnProcedure(), table, "pos_key")
        self.assertEqual(result["hits"], result["total"])
        self.assertIn("x_norm_bin", result["selected_rois"][0])

    def test_nested_estimate_is_reported(self):
        result = select_units(ColumnProcedure(), selection_table(), "slot_key")
        nested = result["nested_loio"]
        self.assertEqual(nested["total"], result["total"])
        self.assertLessEqual(nested["hits"], nested["total"])
        self.assertEqual(nested["hits"], nested["total"])        # the good x < 200 slots win in every fold


class OverlapTests(unittest.TestCase):
    def test_greedy_skips_units_overlapping_by_a_third(self):
        gt = np.array([10, 10, 10, 10])
        values = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10], [9, 11, 11]], dtype=float)
        cost = np.array([30.0, 30.0, 30.0])
        keys = ["mean|30|1490", "median|30|1490", "mean|30|1600"]   # unit 1 = same strip as unit 0
        self.assertEqual(greedy_select(values, gt, cost)[0], [0, 1])
        self.assertEqual(greedy_select(values, gt, cost, geometry=unit_geometry(keys, "slot_key"))[0], [0, 2])
        shifted = unit_geometry(["mean|30|1490", "mean|30|1505", "mean|30|1600"], "slot_key")   # 15 px >= 10 px
        self.assertEqual(greedy_select(values, gt, cost, geometry=shifted)[0], [0, 2])
        bins = unit_geometry(["mean|30|0.80", "median|20|0.80", "mean|30|0.30"], "pos_key")
        self.assertEqual(greedy_select(values, gt, cost, geometry=bins)[0], [0, 2])

    def test_selected_rois_overlap_check(self):
        check_selected_overlap([{"key": "a", "x": 100, "w": 30}, {"key": "b", "x": 121, "w": 30}])   # 9 < 10 px
        check_selected_overlap([{"key": "a", "x_norm_bin": 0.8}, {"key": "b", "x_norm_bin": 0.3}])
        for rois in ([{"key": "a", "x": 100, "w": 30}, {"key": "b", "x": 120, "w": 30}],          # 10 px = w/3
                     [{"key": "a", "x": 100, "w": 30}, {"key": "b", "x": 100, "w": 30}],
                     [{"key": "a", "x": 100, "w": 12}, {"key": "b", "x": 104, "w": 40}],
                     [{"key": "a", "x_norm_bin": 0.8}, {"key": "b", "x_norm_bin": 0.8}]):
            with self.assertRaises(ValueError):
                check_selected_overlap(rois)

    def test_selection_never_overlaps(self):
        table = selection_table()
        shifted = table.copy()
        shifted["x"] = shifted["x"] + 15                         # second grid 15 px to the right
        shifted["slot_key"] = shifted["projection"] + "|" + shifted["w"].astype(str) + "|" + shifted["x"].astype(str)
        result = select_units(ColumnProcedure(), pd.concat([table, shifted], ignore_index=True), "slot_key", 25)
        check_selected_overlap(result["selected_rois"])


def stack_image(height=2048, width=2448, sheets=150, period=11, top=250, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 15.0)
    rows = np.arange(top, top + sheets * period)
    image[rows] = np.where((rows - top) % period < 7, 170.0, 50.0)[:, None]
    image += rng.normal(0, 3, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class BenchmarkTests(unittest.TestCase):
    def test_selected_rois_single_thread_under_one_second(self):
        gray = stack_image()
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[:, :150] = 255
        mask[:, -150:] = 255
        grid = AugmentGrid(widths=(12, 24), pitches=(30,), x_offsets=(0,), projections=("mean", "gaussian"))
        items = bank_items(mask, grid)
        table, _ = image_features(gray, items[::40], extent_slots(mask), 150, gray.shape[1] - 300, grid)
        model, _ = fit_model("ridge", feature_frame(table), np.zeros(len(table)))
        profile = ProjectProfile(case="Syn", angle_deg=0.0, rois=[], roi_angles_deg=[], before_residual_deg=0.0,
                                 after_residual_deg=0.0, quality_status="ok", mask_sha256="",
                                 image_size={"width": 2448, "height": 2048},
                                 valid_region={"bbox": [150, 0, 2298, 2048], "centre_x": 1224.0}, reference_image="")
        selected = [items[i] for i in (0, 41, 82, 123, 164)]
        manifest = {"method": "ml", "scale_normalised": False, "aggregation": "median", "feature_config": feature_config(grid)}
        runtime = ProjectRuntime(profile, selected, extent_slots(mask), grid, model, manifest)
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with threadpool_limits(limits=1):
                runtime.predict(gray)                                  # warm-up (imports)
                start = time.perf_counter()
                prediction, per_roi, timings = runtime.predict(gray)
                elapsed = time.perf_counter() - start
        finally:
            cv2.setNumThreads(previous)
        self.assertEqual(per_roi.size, len(selected))
        self.assertLessEqual(abs(prediction - 150), 2)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(set(timings), {"rotate", "extent", "features", "model", "aggregate"})
        self.assertIsInstance(pd.Series(timings).sum(), float)


if __name__ == "__main__":
    unittest.main()
