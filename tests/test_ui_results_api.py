import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from asgi_client import Client
from escount.cli import main as cli_main
from escount.es_counting import tree_signatures
from escount.inference.engine import InferenceEngine
from escount.models.store import read_manifest, update_manifest
from escount.server.app import ServerConfig, create_app
from synthetic_project import build

KD_COLUMN = "cnt_kd_n3_k5_d3_g30"      # a FEATURE_INDICATORS KD config used as the baseline column


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        self.refs += [value for name, value in attrs if name in ("src", "href") and value]


class ResultsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, _, cls.config = build(cls.root, general=True)
        # SynB: baseline column switched to a KD counter so KD edits must change N_pred
        model_dir = cls.profiles / "SynB" / "model"
        baseline = dict(read_manifest(model_dir)["baseline"], column=f"{KD_COLUMN}_both")
        update_manifest(model_dir, method="baseline", baseline=baseline)
        cls.results = cls.root / "infer"
        cls.cli = {}
        for case, image in (("SynA", "076.png"), ("SynB", "078.png")):
            out = cls.results / case
            status = cli_main(["infer", "--project", case, "--input", str(cls.data / case / image), "--output",
                               str(out), "--profiles-root", str(cls.profiles), "--config", str(cls.config)])
            assert status == 0
            cls.cli[case] = json.loads((out / Path(image).stem / "result.json").read_text(encoding="utf-8"))
        cls.before = tree_signatures(cls.data)
        cls.client = Client(create_app(ServerConfig(
            profiles_root=str(cls.profiles), allowed_roots=(str(cls.data),), general_config=str(cls.config),
            results_root=str(cls.results))))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def preview(self, **body):
        response = self.client.post("/preview", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_preview_with_profile_params_equals_cli_infer(self):
        for case, image in (("SynA", "076.png"), ("SynB", "078.png")):
            cli = self.cli[case]
            default = self.preview(case=case, image=image)
            trained = {family: {k: default["params"][family]["trained"][k] for k in keys}
                       for family, keys in (("macd", ("ema1", "ema2", "dea")), ("kd", ("n", "k", "d")))}
            explicit = self.preview(case=case, image=image, **trained)
            for result in (default, explicit):
                self.assertEqual(result["N_pred"], cli["N_pred"], case)
                self.assertEqual([r["prediction"] for r in result["rois"]], [r["prediction"] for r in cli["rois"]])
                self.assertEqual([r["key"] for r in result["rois"]], [r["key"] for r in cli["rois"]])
            roi = default["rois"][0]
            self.assertEqual(len(roi["signal"]), roi["y1"] - roi["y0"])
            for family, lines in (("macd", ("dif", "dea", "hist")), ("kd", ("rsv", "k", "d"))):
                for name in lines:
                    self.assertEqual(len(roi[family][name]), len(roi["signal"]), (family, name))
                self.assertGreater(len(roi[family]["golden"]) + len(roi[family]["death"]), 0)
        self.assertEqual(tree_signatures(self.data), self.before)

    def test_changed_params_change_n_pred_deterministically(self):
        base = self.preview(case="SynB", image="078.png")
        self.assertTrue(base["params"]["kd"]["affects_prediction"])
        self.assertEqual(base["params"]["kd"]["column"], KD_COLUMN)
        self.assertEqual({k: base["params"]["kd"][k] for k in "nkd"}, {"n": 3, "k": 5, "d": 3})
        edit = {"kd": {"n": 40, "k": 9, "d": 9}}
        first, second = self.preview(case="SynB", image="078.png", **edit), self.preview(case="SynB", image="078.png", **edit)
        self.assertNotEqual(first["N_pred"], base["N_pred"])
        self.assertEqual(first["N_pred"], second["N_pred"])
        self.assertEqual(first["rois"], second["rois"])
        self.assertEqual(first["params"]["kd"]["n"], 40)
        self.assertNotEqual(first["rois"][0]["kd"]["k"], base["rois"][0]["kd"]["k"])
        # the baseline (bollinger) model of SynA does not read MACD: N_pred unchanged, charts change
        synA = self.preview(case="SynA", image="076.png")
        self.assertFalse(synA["params"]["macd"]["affects_prediction"])
        edited = self.preview(case="SynA", image="076.png", macd={"ema1": 5, "ema2": 20, "dea": 9})
        self.assertEqual(edited["N_pred"], synA["N_pred"])
        self.assertNotEqual(edited["rois"][0]["macd"]["dif"], synA["rois"][0]["macd"]["dif"])
        bad = self.client.post("/preview", json={"case": "SynA", "image": "076.png", "kd": {"n": 0, "k": 1, "d": 1}})
        self.assertEqual(bad.status_code, 422)

    def test_project_and_general_model_selection(self):
        project = self.preview(case="SynA", image="082.png", model="project")
        general = self.preview(case="SynA", image="082.png", model="general")
        self.assertEqual((project["model"], project["model_case"]), ("project", "SynA"))
        self.assertEqual((general["model"], general["model_case"]), ("general", "general"))
        engine = InferenceEngine.load("SynA", "general", self.profiles, self.config)
        self.assertEqual(general["N_pred"], engine.infer(self.data / "SynA" / "082.png")["N_pred"])
        self.assertEqual(project["N_pred"], 82)
        edited = self.preview(case="SynA", image="082.png", model="general", kd={"n": 9, "k": 3, "d": 3})
        self.assertEqual(edited["model_case"], "general")

    def test_image_list_marks_complete_results(self):
        info = self.client.get("/results/SynA/images").json()
        self.assertEqual(info["models"], ["project", "general"])
        by_name = {item["name"]: item for item in info["images"]}
        self.assertEqual(sorted(by_name), ["070.png", "076.png", "082.png", "088.png"])
        self.assertTrue(by_name["076.png"]["complete"])
        self.assertEqual(by_name["076.png"]["stored_N_pred"], self.cli["SynA"]["N_pred"])
        self.assertFalse(by_name["070.png"]["complete"])
        self.assertEqual(by_name["070.png"]["gt"], 70)
        self.assertEqual(self.client.post("/preview", json={"case": "SynA", "image": "background_BW.png"}).status_code, 404)

    def test_results_page_served_offline(self):
        response = self.client.get("/ui/results.html")
        self.assertEqual(response.status_code, 200)
        parser = AssetParser()
        parser.feed(response.text)
        for vendor in ("/ui/vendor/vue.global.prod.js", "/ui/vendor/echarts.min.js"):
            self.assertIn(vendor, parser.refs)
        for ref in parser.refs:
            self.assertTrue(ref.startswith("/ui/"), ref)
            if ref.endswith((".js", ".css")):
                self.assertEqual(self.client.get(ref).status_code, 200, ref)
        self.assertNotIn("http", self.client.get("/ui/results.js").text)


if __name__ == "__main__":
    unittest.main()
