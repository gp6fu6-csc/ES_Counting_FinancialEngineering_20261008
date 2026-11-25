"""Variant / run endpoints of the server: calibration-page save, training-page batch job, results-page loading."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from asgi_client import Client
from escount.features.cache import prepare
from escount.models.all_rois import train_all_rois
from escount.models.select_rois import select_project
from escount.models.train import train_project
from escount.server.app import ServerConfig, create_app
from escount.variants import snapshot_run, sync_variant
from synthetic_project import GRID, build
from test_training_page import FakeProcess

STRIPS = {"n": 3, "w": 20, "pitch": 30, "side_margin": 50, "x_offset": 0}


class VariantApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, cls.features, cls.config = build(cls.root)
        cls.results = cls.root / "infer"
        cls.client = Client(create_app(ServerConfig(
            profiles_root=str(cls.profiles), allowed_roots=(str(cls.data),), general_config=str(cls.config),
            results_root=str(cls.results), features_root=str(cls.features))))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def wait_done(self):
        for _ in range(200):
            job = self.client.get("/training/SynA/status").json()["job"]
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def train_variant(self, name, strips=STRIPS):
        case = f"SynA@{name}"
        sync_variant(case, self.profiles, strips=strips)
        prepare(case, self.data, self.profiles, self.features, workers=1, grid=GRID)
        train_project(case, self.features, self.profiles, device="cpu", log=lambda *a, **k: None)
        select_project(case, self.features, self.profiles, device="cpu")
        train_all_rois(case, self.profiles, self.data, "cpu")
        return case

    def test_calibration_page_saves_only_the_strips_as_a_variant(self):
        info = self.client.get("/calibration/SynA/images").json()
        body = {"rois": info["default_rois"], "strips": STRIPS}
        saved = self.client.post("/calibration/SynA/variants/cal1", json=body)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["case"], "SynA@cal1")
        listing = self.client.get("/variants/SynA").json()
        item = next(v for v in listing["variants"] if v["name"] == "cal1")
        self.assertEqual(item["strip_params"], STRIPS)
        self.assertFalse(item["has_model"])
        self.assertTrue(listing["default"]["has_model"])
        overview = self.client.get("/variants").json()
        self.assertIn("cal1", overview["projects"]["SynA"])
        self.assertEqual(next(v for v in overview["variants"] if v["name"] == "cal1")["strip_params"], STRIPS)
        # variants are not projects of the results-page project list
        self.assertEqual([p["case"] for p in self.client.get("/projects").json()["projects"]], ["SynA"])
        for bad_name in ("a b", "x@y"):
            self.assertEqual(self.client.post(f"/calibration/SynA/variants/{bad_name}", json=body).status_code, 422)
        too_close = {"rois": info["default_rois"], "strips": {**STRIPS, "pitch": 5}}
        self.assertEqual(self.client.post("/calibration/SynA/variants/cal2", json=too_close).status_code, 422)
        self.assertEqual(self.client.get("/variants/NoSuchCase").status_code, 404)

    def test_results_page_loads_variant_latest_model_and_saved_run(self):
        case = self.train_variant("res1")
        run = "20261120-120000"
        snapshot_run(case, run, {"run": run, "status": "ok", "created": "2026-11-20T12:00:00", "strip_params": STRIPS,
                                 "model": {"minimal": {"hits": 4, "total": 4}, "all_rois": {"hits": 3, "total": 4}},
                                 "options": {"device": "cpu"}}, self.profiles)
        self.client.post("/calibration/SynA/variants/res1", json={
            "rois": self.client.get("/calibration/SynA/images").json()["default_rois"], "strips": STRIPS})
        item = next(v for v in self.client.get("/variants/SynA").json()["variants"] if v["name"] == "res1")
        self.assertTrue(item["has_model"])
        self.assertEqual([r["run"] for r in item["runs"]], [run])
        self.assertIn("最少 ROI 4/4", item["runs"][0]["label"])
        self.assertEqual(self.client.get(f"/runs/{case}~{run}").json()["options"], {"device": "cpu"})
        self.assertEqual(self.client.get(f"/runs/{case}").status_code, 422)
        self.assertEqual(self.client.get(f"/runs/{case}~nope").status_code, 404)
        for target in (case, f"{case}~{run}"):
            images = self.client.get(f"/results/{target}/images")
            self.assertEqual(images.status_code, 200, images.text)
            self.assertEqual(images.json()["models"], ["project"])
            preview = self.client.post("/preview", json={"case": target, "image": "070.png", "roi_mode": "all"})
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(len(preview.json()["rois"]), STRIPS["n"])
        missing = self.client.post("/preview", json={"case": f"{case}~19990101-000000", "image": "070.png"})
        self.assertEqual(missing.status_code, 404)             # a run that was never saved

    def test_batch_training_starts_one_run_variants_job(self):
        sync_variant("SynA@bat1", self.profiles, strips=STRIPS)
        fake = FakeProcess()
        with mock.patch("subprocess.Popen", fake):
            response = self.client.post("/training/batch", json={
                "cases": ["SynA"], "variants": ["bat1"], "device": "cpu", "max_rois": 9, "max_rows_per_image": 300,
                "research": True, "trials": 4, "track": "mae", "infer": False, "note": "from ui"})
            self.assertEqual(response.status_code, 200, response.text)
            job = self.wait_done()
        self.assertEqual((job["case"], job["step"], job["status"]), ("batch", "batch", "done"))
        command = fake.command[3:]
        self.assertEqual(command[:5], ["run-variants", "--projects", "SynA", "--variants", "bat1"])
        for flag, value in (("--device", "cpu"), ("--max-rois", "9"), ("--max-rows-per-image", "300"),
                            ("--trials", "4"), ("--track", "mae"), ("--note", "from ui"),
                            ("--profiles-root", str(self.profiles)), ("--cache-root", str(self.features)),
                            ("--results-root", str(self.results))):
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        self.assertIn("--no-infer", command)
        self.assertIn("--report", command)

    def test_research_checkbox_maps_to_trials_and_report(self):
        sync_variant("SynA@chk1", self.profiles, strips=STRIPS)
        base = {"cases": ["SynA"], "variants": ["chk1"], "device": "cpu", "trials": 7, "track": "exact"}
        for research in (True, False):
            fake = FakeProcess()
            with mock.patch("subprocess.Popen", fake):
                response = self.client.post("/training/batch", json={**base, "research": research})
                self.assertEqual(response.status_code, 200, response.text)
                self.wait_done()
            command = fake.command[3:]
            self.assertEqual(command[command.index("--trials") + 1], "7" if research else "0")
            self.assertEqual("--report" in command, research)
        # the default request (no checkbox field) stays a plain training
        fake = FakeProcess()
        with mock.patch("subprocess.Popen", fake):
            self.client.post("/training/batch", json={**base})
            self.wait_done()
        self.assertNotIn("--report", fake.command[3:])
        self.assertEqual(self.client.post("/training/batch", json={**base, "research": True, "trials": 0}).status_code, 422)

    def test_batch_training_validates_its_input(self):
        base = {"cases": ["SynA"], "variants": ["bat2"]}
        self.assertEqual(self.client.post("/training/batch", json=base).status_code, 409)           # variant unknown
        self.assertEqual(self.client.post("/training/batch", json={**base, "cases": []}).status_code, 422)
        self.assertEqual(self.client.post("/training/batch", json={**base, "variants": ["a b"]}).status_code, 422)
        self.assertEqual(self.client.post("/training/batch", json={**base, "trials": -1}).status_code, 422)
        self.assertEqual(self.client.post("/training/batch", json={**base, "cases": ["Nope"]}).status_code, 404)
        sync_variant("SynA@bat2", self.profiles, strips=STRIPS)
        self.assertEqual(self.client.post("/training/batch", json={**base, "cases": ["SynA@bat2"]}).status_code, 409)

    def test_training_status_and_single_steps_work_on_a_variant(self):
        sync_variant("SynA@step1", self.profiles, strips=STRIPS)
        steps = self.client.get("/training/SynA@step1/status").json()["steps"]
        self.assertTrue(steps["calibrate"])
        self.assertFalse(steps["train"])


if __name__ == "__main__":
    unittest.main()
