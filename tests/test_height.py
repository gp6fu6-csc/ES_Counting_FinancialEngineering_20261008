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
from test_inference import PERIOD, TOP


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

    def test_one_mm_list_is_one_mm_apart(self):
        self.assertAlmostEqual(SideScale({"1": [[0, 10], [0, 60]]}).height(10, 60)[0], 1.0)

    def test_version_1_file_keeps_five_mm_for_one_mm_list(self):
        self.assertAlmostEqual(SideScale({"1": [[0, 10], [0, 60]]}, version=1).height(10, 60)[0], 5.0)

    def test_one_mm_points_set_the_scale_and_five_ten_mm_only_check(self):
        one = [[0, y] for y in range(0, 101, 10)]            # 1 mm every 10 px
        side = SideScale({"1": one, "5": [[0, 0], [0, 50]], "10": [[0, 0], [0, 100]]})
        self.assertAlmostEqual(side.height(0, 100)[0], 10.0)
        self.assertEqual(side.warnings, [])
        off = SideScale({"1": one, "10": [[0, 0], [0, 80]]})   # 80 px = 8 mm on the 1 mm scale, not 10 mm
        self.assertEqual(len(off.warnings), 1)

    def test_without_one_mm_points_five_ten_mm_are_used_with_a_warning(self):
        side = SideScale({"5": [[0, 0], [0, 50]]})
        self.assertAlmostEqual(side.height(0, 50)[0], 5.0)
        self.assertEqual(len(side.warnings), 1)

    def test_clicks_in_any_order_give_the_same_scale(self):
        ones = [[0, y] for y in range(0, 101, 10)]
        shuffled = [ones[i] for i in (5, 0, 10, 3, 7, 1, 9, 2, 8, 4, 6)]
        self.assertEqual(SideScale({"1": shuffled}).knots, SideScale({"1": ones}).knots)

    def test_skipped_one_mm_icon_is_inferred_with_a_warning(self):
        side = SideScale({"1": [[0, 0], [0, 10], [0, 30], [0, 40], [0, 50]]})   # the icon at 20 was not clicked
        self.assertAlmostEqual(side.height(0, 50)[0], 5.0)
        self.assertEqual(len(side.warnings), 1)

    def test_five_ten_mm_points_extend_beyond_the_one_mm_rows(self):
        # 1 mm rows 100..200 (10 px per mm); the 10 mm points 100 -> 400 reach 300 px below them (30 px per mm)
        side = SideScale({"1": [[0, y] for y in range(100, 201, 10)], "10": [[0, 0], [0, 100], [0, 400]]})
        self.assertAlmostEqual(side.height(250, 400)[0], 5.0)
        self.assertFalse(side.height(0, 400)[1])

    def test_knots_align_the_three_lists_by_height(self):
        side = SideScale({"1": [[0, y] for y in range(0, 101, 10)], "5": [[0, 1], [0, 51], [0, 101]],
                          "10": [[0, 100], [0, 0]]})
        self.assertEqual([round(y) for y, _ in side.knots], list(range(0, 101, 10)))
        self.assertAlmostEqual(side.knots[-1][1], 10.0)

    def test_double_click_on_one_icon_is_ignored(self):
        side = SideScale({"1": [[0, 0], [0, 1], [0, 10], [0, 20]]})
        self.assertAlmostEqual(side.height(0, 20)[0], 2.0)

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
        sheets = int(self.image.stem)
        for roi in result["rois"]:
            # top = first sheet from the top, bottom = first sheet from the bottom (not the strip ends at row ~0)
            self.assertLess(abs(roi["y_top"] - TOP), PERIOD + 8, roi)
            self.assertLess(abs(roi["y_bottom"] - (TOP + (sheets - 1) * PERIOD)), PERIOD + 8, roi)
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

    def test_results_page_measures_any_two_rows(self):
        self.assertEqual(self.client.post("/results/SynA/height", json={"y_a": 0, "y_b": 100}).status_code, 404)
        points = {"left": uniform(100), "right": uniform(50)}
        self.assertEqual(self.client.post("/calibration/SynA/ratio", json={"angle_deg": 0, "points": points}).status_code, 200)
        # the clicks may come in any order; the height is always measured from the upper to the lower row
        result = self.client.post("/results/SynA/height", json={"y_a": 300, "y_b": 100}).json()
        self.assertEqual((result["y_top"], result["y_bottom"]), (100, 300))
        self.assertAlmostEqual(result["height_mm"], (20.0 + 40.0) / 2)
        self.assertFalse(result["height_extrapolated"])

    def test_copy_model_to_another_project(self):
        (self.profiles / "SynA" / MODEL_NAME).write_text('{"points": {}}\n', encoding="utf-8")
        self.assertTrue(copy_model(self.profiles / "SynA", self.profiles / "Other"))
        self.assertTrue((self.profiles / "Other" / MODEL_NAME).is_file())
        self.assertFalse(copy_model(self.profiles / "Missing", self.profiles / "Other2"))


if __name__ == "__main__":
    unittest.main()
