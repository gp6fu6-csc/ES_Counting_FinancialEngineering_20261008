import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np

from asgi_client import Client
from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
from escount.es_counting import tree_signatures
from escount.server.app import UI_DIR, ServerConfig, create_app
from test_inference import ANGLE, HEIGHT, WIDTH, stack_image


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        self.refs += [value for name, value in attrs if name in ("src", "href") and value]


class CalibrationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.data, cls.profiles = root / "Data", root / "projects"
        case = cls.data / "SynA"
        case.mkdir(parents=True)
        cv2.imwrite(str(case / "background_BW.png"), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
        cv2.imwrite(str(case / "background.png"), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
        for seed, sheets in enumerate((76, 82)):
            cv2.imwrite(str(case / f"{sheets:03d}.png"), stack_image(sheets, seed))
        cls.before = tree_signatures(cls.data)
        cls.client = Client(cls.app())
        cls.info = cls.client.get("/calibration/SynA/images").json()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def app(cls, api_key=None):
        return create_app(ServerConfig(profiles_root=str(cls.profiles), allowed_roots=(str(cls.data),), api_key=api_key))

    def body(self, rois=None, **strips):
        return {"reference": "082.png", "rois": rois or self.info["default_rois"],
                "strips": {"n": 3, "w": 12, "pitch": 30, "side_margin": 50, "x_offset": 0, **strips}}

    def test_image_list_excludes_background_files(self):
        self.assertEqual(self.client.get("/calibration/cases").json()["cases"], ["SynA"])
        self.assertEqual(self.info["images"], ["076.png", "082.png"])
        self.assertEqual(self.info["reference"], "082.png")
        self.assertEqual(self.info["image_size"], {"width": WIDTH, "height": HEIGHT})
        self.assertEqual(len(self.info["default_rois"]), 3)
        self.assertTrue(all(roi["side"] == 384 for roi in self.info["default_rois"]))
        self.assertIsNone(self.info["profile"])
        for name in ("background_BW.png", "background.png", "..%2Fx.png"):
            self.assertEqual(self.client.get(f"/calibration/SynA/image/{name}").status_code, 404, name)
        response = self.client.get("/calibration/SynA/image/076.png", params={"max_width": 700, "angle": ANGLE})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape[1], 700)
        self.assertEqual(self.client.get("/calibration/NoCase/images").status_code, 404)

    def test_preview_returns_angle_residuals_and_strips(self):
        response = self.client.post("/calibration/preview", json={"case": "SynA", **self.body()})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertAlmostEqual(result["angle_deg"], ANGLE, delta=0.1)
        self.assertEqual(len(result["roi_angles_deg"]), 3)
        self.assertLess(result["after_residual_deg"], result["before_residual_deg"])
        self.assertEqual(result["quality_status"], "ok")
        strips = result["strips"]
        self.assertEqual(len(strips), 3)
        self.assertEqual({s["w"] for s in strips}, {12})
        self.assertEqual([b["x"] - a["x"] for a, b in zip(strips, strips[1:])], [30, 30])
        x0, _, x1, _ = result["rotated_valid_region"]["bbox"]
        for strip in strips:
            self.assertGreaterEqual(strip["x"] - 50, x0)
            self.assertLessEqual(strip["x"] + strip["w"] + 50, x1)
            self.assertLessEqual(strip["y0"], result["stack_extent"]["y0"] + 5)
        self.assertFalse(self.profiles.exists())                     # preview writes nothing

    def test_preview_rejects_invalid_geometry(self):
        small = [{**roi, "side": 299} for roi in self.info["default_rois"]]
        outside = [{"x": WIDTH - 100, "y": 10, "side": 384}]
        cases = {"side < 300": self.body(rois=small), "margin < 50": self.body(side_margin=49),
                 "D <= 2w/3": self.body(w=30, pitch=20), "ROI outside": self.body(rois=outside),
                 "too many strips": self.body(n=64, w=40, pitch=200)}
        for label, body in cases.items():
            response = self.client.post("/calibration/preview", json={"case": "SynA", **body})
            self.assertEqual(response.status_code, 422, label)
            self.assertIn("detail", response.json(), label)
        self.assertIn("overlap", self.client.post("/calibration/preview", json={"case": "SynA", **cases["D <= 2w/3"]})
                      .json()["detail"])
        self.assertEqual(self.client.post("/calibration/SynA/save", json=cases["side < 300"]).status_code, 422)
        self.assertFalse((self.profiles / "SynA" / PROFILE_NAME).exists())

    def test_save_writes_valid_project_profile(self):
        rois = [{**roi, "x": roi["x"] + 10} for roi in self.info["default_rois"]]
        response = self.client.post("/calibration/SynA/save", json=self.body(rois=rois, n=2, w=14, pitch=40))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["quality_status"], "ok")
        profile = ProjectProfile.load(self.profiles / "SynA" / PROFILE_NAME)
        self.assertEqual(profile.case, "SynA")
        self.assertEqual(profile.reference_image, "082.png")
        self.assertEqual([(r.x, r.y, r.side) for r in profile.rois], [(r["x"], r["y"], r["side"]) for r in rois])
        self.assertEqual(profile.strip_params, {"n": 2, "w": 14, "pitch": 40, "side_margin": 50, "x_offset": 0})
        self.assertAlmostEqual(profile.angle_deg, ANGLE, delta=0.1)
        self.assertTrue((self.profiles / "SynA" / ROTATED_MASK_NAME).is_file())
        reloaded = self.client.get("/calibration/SynA/images").json()
        self.assertEqual(reloaded["profile"]["rois"], rois)
        self.assertEqual(reloaded["strips"]["w"], 14)
        self.assertEqual(tree_signatures(self.data), self.before)

    def test_ui_page_is_served_offline_from_local_assets(self):
        response = self.client.get("/ui/calibration.html")
        self.assertEqual(response.status_code, 200)
        parser = AssetParser()
        parser.feed(response.text)
        self.assertIn("/ui/vendor/vue.global.prod.js", parser.refs)
        scripts_and_styles = [ref for ref in parser.refs if ref.endswith((".js", ".css"))]
        for ref in parser.refs:
            self.assertTrue(ref.startswith("/ui/"), ref)
        for ref in scripts_and_styles:
            self.assertTrue(ref.startswith("/ui/vendor/") or ref.count("/") == 2, ref)  # vendor or own /ui/ file
            self.assertEqual(self.client.get(ref).status_code, 200, ref)
        for own in ("calibration.html", "calibration.js", "common.js", "app.css"):
            text = (UI_DIR / own).read_text(encoding="utf-8")
            self.assertNotIn("http://", text, own)
            self.assertNotIn("https://", text, own)
        for name in ("LICENSE-vue", "LICENSE-echarts", "NOTICE-echarts", "LICENSE-zrender", "VERSIONS.md"):
            self.assertTrue((UI_DIR / "vendor" / name).is_file(), name)
        keyed = Client(self.app(api_key="k"))
        self.assertEqual(keyed.get("/ui/calibration.html").status_code, 200)
        self.assertEqual(keyed.get("/calibration/cases").status_code, 401)


if __name__ == "__main__":
    unittest.main()
