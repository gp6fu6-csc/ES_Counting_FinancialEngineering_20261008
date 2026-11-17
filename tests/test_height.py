"""Pixel-to-height model (PDF 高度量測): accumulated mm per row, left/right mean, extrapolation flag, outputs."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from asgi_client import Client
from escount.es_counting import list_images
from escount.height import MODEL_NAME, HeightModel, SideScale, copy_model
from escount.inference.engine import HEIGHTS_NAME, PREDICTIONS_NAME, InferenceEngine
from escount.server.app import ServerConfig, create_app
from synthetic_project import build


def uniform(step_px):
    """10 mm icons every `step_px` rows from row 0 to 400 (constant scale 10 / step_px mm per px)."""
    return {"10": [[0, y] for y in range(0, 401, step_px)], "5": [], "1": []}


class HeightModelTests(unittest.TestCase):
    def test_constant_scale_gives_proportional_height(self):
        side = SideScale(uniform(100))
        self.assertAlmostEqual(side.height(50, 150)[0], 10.0)
        self.assertFalse(side.height(50, 150)[1])

    def test_height_is_accumulated_not_a_single_ratio(self):
        # perspective: 10 mm span 100 px near the top and 200 px lower down
        side = SideScale({"10": [[0, 0], [0, 100], [0, 300]]})
        mm, _ = side.height(0, 300)
        self.assertAlmostEqual(mm, 21.25)            # sum of the interpolated scale, not 20 / 300 * 300
        self.assertNotAlmostEqual(mm, 20.0)

    def test_one_mm_list_is_five_mm_apart(self):
        # the 1 mm clicks are the first icon after each 5 mm icon, so consecutive clicks are 5 mm apart
        self.assertAlmostEqual(SideScale({"1": [[0, 10], [0, 60]]}).height(10, 60)[0], 5.0)

    def test_extrapolation_is_computed_and_flagged(self):
        mm, outside = SideScale(uniform(100)).height(-50, 100)
        self.assertAlmostEqual(mm, 15.0)
        self.assertTrue(outside)

    def test_left_right_mean(self):
        model = HeightModel({"left": uniform(100), "right": uniform(50)})
        result = model.measure(0, 100)
        self.assertAlmostEqual(result["height_left_mm"], 10.0)
        self.assertAlmostEqual(result["height_right_mm"], 20.0)
        self.assertAlmostEqual(result["height_mm"], 15.0)

    def test_side_without_two_points_is_rejected(self):
        with self.assertRaises(ValueError):
            HeightModel({"left": uniform(100), "right": {"10": [[0, 5]]}})


class HeightOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, _, cls.config = build(cls.root)
        case_dir = cls.data / "SynA"
        cls.image = list_images(case_dir)[0]
        shutil.copy2(cls.image, case_dir / "ratio.png")
        cls.client = Client(create_app(ServerConfig(profiles_root=str(cls.profiles), allowed_roots=(str(cls.data),),
                                                    general_config=str(cls.config))))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        (self.profiles / "SynA" / MODEL_NAME).unlink(missing_ok=True)

    def run_engine(self, out):
        engine = InferenceEngine.load("SynA", profiles_root=str(self.profiles), general_config=str(self.config))
        return engine.run(self.image, out)[0]

    def test_ratio_is_not_a_counting_image(self):
        self.assertNotIn("ratio.png", [p.name for p in list_images(self.data / "SynA")])
        self.assertEqual(self.client.get("/calibration/SynA/images").json()["images"][-1], "ratio.png")

    def test_without_model_only_edges_are_added(self):
        out = self.root / "no_model"
        result = self.run_engine(out)
        for roi in result["rois"]:
            self.assertEqual((roi["y_top"], roi["y_bottom"]), (roi["y0"], roi["y1"]))
            self.assertNotIn("height_mm", roi)
        self.assertNotIn("stack_height_mm", result)
        self.assertFalse((out / HEIGHTS_NAME).exists())
        self.assertEqual(list(pd.read_csv(out / PREDICTIONS_NAME).columns), ["image", "GT", "N_pred", "hit"])

    def test_saved_model_adds_every_roi_height(self):
        points = {"left": uniform(100), "right": uniform(100)}
        bad = self.client.post("/calibration/SynA/ratio", json={"angle_deg": 0, "points": {"left": uniform(100)}})
        self.assertEqual(bad.status_code, 422)
        saved = self.client.post("/calibration/SynA/ratio", json={"angle_deg": 0, "points": points})
        self.assertEqual(saved.status_code, 200, saved.text)
        out = self.root / "with_model"
        result = self.run_engine(out)
        for roi in result["rois"]:
            self.assertAlmostEqual(roi["height_mm"], (roi["y_bottom"] - roi["y_top"]) / 10.0)
        table = pd.read_csv(out / HEIGHTS_NAME)
        self.assertEqual(len(table), len(result["rois"]))
        record = json.loads((out / self.image.stem / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(len([r for r in record["rois"] if r["height_mm"] is not None]), len(result["rois"]))
        preview = self.client.post("/preview", json={"case": "SynA", "image": self.image.name}).json()
        self.assertTrue(preview["has_height_model"])
        self.assertTrue(all("height_mm" in roi for roi in preview["rois"]))

    def test_copy_model_to_another_project(self):
        (self.profiles / "SynA" / MODEL_NAME).write_text('{"points": {}}\n', encoding="utf-8")
        self.assertTrue(copy_model(self.profiles / "SynA", self.profiles / "Other"))
        self.assertTrue((self.profiles / "Other" / MODEL_NAME).is_file())
        self.assertFalse(copy_model(self.profiles / "Missing", self.profiles / "Other2"))


if __name__ == "__main__":
    unittest.main()
