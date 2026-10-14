import time
import unittest
import warnings

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from escount.calibration.calibrate import ProjectProfile
from escount.features.extract import AugmentGrid, bank_items, extent_slots, feature_config, image_features
from escount.inference.benchmark import ProjectRuntime
from escount.inference.engine import GeneralRuntime
from escount.signals.roi_bank import RoiSlot
from escount.models.select_rois import (
    check_selected_overlap, choose_size, greedy_select, medians_with, select_units, unit_geometry,
)
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


class BankProcedure(ColumnProcedure):
    """ColumnProcedure whose full ROI set is one deployable bank (the w=10 strips)."""

    def deployable_rows(self, rows):
        return rows[rows["w"] == 10]


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
        self.assertEqual(len(nested["curve"]), len(result["history_hits"]))
        deployed = result["deployed"]
        self.assertEqual((deployed["source"], deployed["roi_count"]), ("greedy_subset", result["roi_count"]))
        self.assertGreaterEqual(deployed["hits"], result["full_set"]["hits"])   # honest subset >= full set

    def test_size_by_nested_hits_fewest_on_ties(self):
        self.assertEqual(choose_size(np.array([1, 3, 3, 2]), 4), 2)
        self.assertEqual(choose_size(np.array([1, 3, 3, 2]), 1), 1)

    def test_full_bank_deployed_when_it_has_more_honest_hits(self):
        table = synthetic_table(images_per_case=6, rows_per_image=24, seed=5)
        # every strip is off by one (+1 / -1 alternating along x); only the median of a whole bank is exact
        table["signal"] = table["gt"] + np.where((table["x"] - 100) // 30 % 2 == 0, 1.0, -1.0)
        result = select_units(BankProcedure(), table, "slot_key", max_rois=1)
        self.assertEqual(result["deployed"]["source"], "full_set")
        self.assertEqual(result["deployed"]["hits"], result["total"])
        self.assertEqual([r["x"] for r in result["selected_rois"]], sorted(table.loc[table["w"] == 10, "x"].unique()))
        check_selected_overlap(result["selected_rois"])
        subset = select_units(ColumnProcedure(), table, "slot_key", max_rois=1)   # no deployable full set
        self.assertEqual((subset["deployed"]["source"], subset["deployed"]["hits"]), ("greedy_subset", 0))

    def test_fast_medians_equal_nanmedian(self):
        rng = np.random.default_rng(1)
        values = rng.integers(0, 6, (7, 9)).astype(float)
        values[rng.random(values.shape) < 0.2] = np.nan
        candidates = np.arange(9)
        for selected in ([], [3], [3, 5], [0, 2, 4, 6]):
            stacked = np.concatenate([np.repeat(values[:, selected][None], 9, axis=0), values.T[:, :, None]], axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = np.nanmedian(stacked, axis=2)
            np.testing.assert_array_equal(medians_with(values, selected, candidates), expected)


class OverlapTests(unittest.TestCase):
    def test_greedy_skips_units_overlapping_by_a_third(self):
        gt = np.array([10, 10, 10, 10])
        values = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10], [9, 11, 11]], dtype=float)
        cost = np.array([30.0, 30.0, 30.0])
        keys = ["mean|30|1490", "median|30|1490", "mean|30|1600"]   # unit 1 = same strip as unit 0
        self.assertEqual(greedy_select(values, gt, cost)[0], [0, 1, 2])          # full path up to max_rois
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

    def test_general_runtime_records_dropped_overlap(self):
        runtime = GeneralRuntime.__new__(GeneralRuntime)            # only the unit -> strip resolution
        runtime.valid_x0, runtime.valid_width = 0, 1000
        first, second, third = (RoiSlot(x, 30, 0, 1000, 30, 0) for x in (500, 510, 700))
        runtime.units = [({"key": "mean|30|0.50", "x_norm_bin": 0.50}, [(first, "mean")]),
                         ({"key": "mean|30|0.55", "x_norm_bin": 0.55}, [(second, "mean")]),   # 20 px >= w/3
                         ({"key": "mean|30|0.70", "x_norm_bin": 0.70}, [(third, "mean")])]
        dropped = []
        kept = runtime.kept_items({"y0": 0, "y1": 1000}, dropped)
        self.assertEqual([slot.x for slot, _ in kept], [500, 700])
        self.assertEqual([(d["unit"], d["strip"], d["overlaps"]) for d in dropped],
                         [("mean|30|0.55", "mean|30|510", "mean|30|500")])

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
