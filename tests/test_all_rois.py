import json
from pathlib import Path
import unittest

from asgi_client import Client
from escount.inference.engine import InferenceEngine
from escount.models.all_rois import build_all_rois_layout, train_all_rois
from escount.server.app import ServerConfig, create_app


class TestAllRoisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = "20261020_TH01mm_03x_non-press"
        # Train all_rois ensemble
        cls.info = train_all_rois(cls.case)

    def test_build_all_rois_layout(self):
        rois, slots = build_all_rois_layout(self.case)
        self.assertEqual(len(rois), 10)
        self.assertEqual(len(slots), 10)
        self.assertEqual(rois[0]["index"], 1)
        self.assertEqual(rois[9]["index"], 10)

    def test_train_all_rois_evaluation(self):
        self.assertEqual(self.info["roi_count"], 10)
        self.assertIn("recommended", self.info)
        self.assertIn("evaluation", self.info)
        self.assertIn("median", self.info["evaluation"])
        self.assertIn("ridge", self.info["evaluation"])
        self.assertIn("xgboost", self.info["evaluation"])
        self.assertIn("trimmed_mean", self.info["evaluation"])

    def test_inference_dual_roi_modes(self):
        engine = InferenceEngine.load(self.case)

        # 1. Minimal ROI mode (2 ROIs)
        res_min = engine.infer(f"Data/{self.case}/170.png", roi_mode="minimal")
        self.assertEqual(len(res_min["rois"]), 2)
        self.assertEqual(res_min["N_pred"], 170)
        self.assertEqual(res_min["roi_mode"], "minimal")

        # 2. All ROI mode (10 ROIs)
        res_all = engine.infer(f"Data/{self.case}/170.png", roi_mode="all")
        self.assertEqual(len(res_all["rois"]), 10)
        self.assertEqual(res_all["N_pred"], 170)
        self.assertEqual(res_all["roi_mode"], "all")
        self.assertIsNotNone(res_all["all_rois_ensemble"])

    def test_preview_api_dual_roi_modes(self):
        app = create_app(ServerConfig())
        client = Client(app)

        # 1. Minimal preview
        resp_min = client.post("/preview", json={
            "case": self.case,
            "image": "170.png",
            "model": "project",
            "roi_mode": "minimal",
        })
        self.assertEqual(resp_min.status_code, 200)
        data_min = resp_min.json()
        self.assertEqual(len(data_min["rois"]), 2)
        self.assertEqual(data_min["N_pred"], 170)

        # 2. All preview
        resp_all = client.post("/preview", json={
            "case": self.case,
            "image": "170.png",
            "model": "project",
            "roi_mode": "all",
        })
        self.assertEqual(resp_all.status_code, 200)
        data_all = resp_all.json()
        self.assertEqual(len(data_all["rois"]), 10)
        self.assertEqual(data_all["N_pred"], 170)


if __name__ == "__main__":
    unittest.main()
