import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from escount.models import train
from escount.models.store import ModelIntegrityError, load_model, save_model
from escount.models.train import (
    BaselineProcedure, ModelProcedure, feature_frame, fit_model, general_table, general_variants, train_case,
)
from escount.research.evaluate import evaluate
from feature_tables import synthetic_table


class StoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(200, 6))
        self.y = self.x[:, 0] * 2 - self.x[:, 1] + rng.normal(0, 0.1, 200)

    def test_save_load_identical_predictions_all_model_types(self):
        for kind in train.MODEL_TYPES:
            model, _ = fit_model(kind, self.x, self.y, device="cpu", params={"n_estimators": 20} if kind in ("xgb", "lgbm") else None)
            with tempfile.TemporaryDirectory() as tmp:
                manifest = save_model(model, kind, tmp, features=["f"] * 6)
                loaded, loaded_manifest = load_model(tmp)
                self.assertEqual(loaded_manifest["files"], manifest["files"])
                np.testing.assert_array_equal(loaded.predict(self.x), model.predict(self.x), err_msg=kind)

    def test_manifest_sha_mismatch_raises(self):
        model, _ = fit_model("ridge", self.x, self.y)
        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, "ridge", tmp)
            path = Path(tmp) / "model_ridge.joblib"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(ModelIntegrityError):
                load_model(tmp)

    def test_forced_cuda_failure_falls_back_to_cpu(self):
        import xgboost

        original = train.make_model

        def failing(kind, device="cpu", params=None):
            if device == "cuda":
                raise xgboost.core.XGBoostError("forced CUDA failure")
            return original(kind, device, params)

        with mock.patch.object(train, "make_model", side_effect=failing):
            model, device = fit_model("xgb", self.x, self.y, device="cuda", params={"n_estimators": 10})
        self.assertEqual(device, "cpu")
        self.assertEqual(model.predict(self.x).shape, (200,))


class TrainingTests(unittest.TestCase):
    def test_learnable_delta_ml_beats_baseline_and_is_adopted(self):
        table = synthetic_table(images_per_case=8)
        baseline = evaluate(BaselineProcedure(), table, "LOIO")
        ml = evaluate(ModelProcedure("ridge", device="cpu"), table, "LOIO")
        self.assertGreaterEqual(ml.hit_rate, baseline.hit_rate)
        self.assertEqual(ml.hits, ml.total)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = train_case("A", table, tmp, device="cpu", model_types=("ridge",), log=lambda *_: None)
            self.assertTrue(manifest["adopted_ml"])
            self.assertEqual(manifest["method"], "ml")
            loaded, _ = load_model(tmp)
            self.assertEqual(loaded.predict(feature_frame(table)).shape, (len(table),))

    def test_not_adopted_when_baseline_is_equal(self):
        table = synthetic_table(images_per_case=6, learnable=False)
        table["cnt_kd_n4_k2_d2_g20_up"] = table["gt"].astype(float)     # a perfect single indicator
        with tempfile.TemporaryDirectory() as tmp:
            manifest = train_case("A", table, tmp, device="cpu", model_types=("ridge",), log=lambda *_: None)
        self.assertEqual(manifest["evaluation"]["LOIO"]["baseline"]["hits"], 6)
        self.assertFalse(manifest["adopted_ml"])
        self.assertEqual(manifest["method"], "baseline")
        self.assertEqual(manifest["baseline"]["column"], "cnt_kd_n4_k2_d2_g20_up")

    def test_exclude_from_general_removes_rows(self):
        tables = {case: synthetic_table(cases=(case,), images_per_case=2, rows_per_image=3, seed=i)
                  for i, case in enumerate(("P1", "Ring-type", "T-type"))}
        combined = general_table(tables, exclude=["T-type"])
        self.assertEqual(set(combined["case"]), {"P1", "Ring-type"})
        self.assertEqual(len(combined), len(tables["P1"]) + len(tables["Ring-type"]))
        variants = general_variants(tables, exclude=["T-type"])
        self.assertEqual(variants, {"general": ["P1", "Ring-type"], "general_no_ring": ["P1"]})
        self.assertEqual(general_variants(tables)["general"], ["P1", "Ring-type", "T-type"])

    def test_excluded_projects_drop_ring_and_general_no_ring(self):
        config = {"excluded_projects": ["Ring-type"], "exclude_from_general": ["Ring-type"]}
        cases = ["P1", "Ring-type", "T-type"]
        active = train.active_projects(cases, config)
        self.assertEqual(active, ["P1", "T-type"])
        self.assertEqual(general_variants(active, config["exclude_from_general"]), {"general": ["P1", "T-type"]})
        self.assertEqual(train.general_cases(config), ["general"])
        self.assertEqual(train.production_general(config), "general")
        with self.assertRaises(ValueError):
            train.check_active("Ring-type", config)
        enabled = {"excluded_projects": [], "exclude_from_general": ["Ring-type"]}   # re-enabled Ring-type
        self.assertEqual(train.general_cases(enabled, cases), ["general", "general_no_ring"])
        self.assertEqual(train.production_general(enabled), "general_no_ring")

    def test_committed_config_excludes_ring(self):
        config = train.load_general_config(Path(__file__).resolve().parents[1] / "configs" / "general.json")
        self.assertEqual(config["excluded_projects"], ["Ring-type"])
        self.assertIn("Ring-type", config["exclude_from_general"])

    def test_scale_normalised_features_have_no_absolute_px(self):
        table = synthetic_table(images_per_case=2, rows_per_image=3)
        x = feature_frame(table, scale_normalised=True)
        self.assertEqual(x.shape[1], len(train.feature_names(True)))
        self.assertNotIn("L", train.feature_names(True))
        self.assertNotIn("w", train.feature_names(True))


if __name__ == "__main__":
    unittest.main()
