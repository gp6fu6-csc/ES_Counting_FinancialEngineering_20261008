"""ROI-strip variants: case ids, shared calibration, own strips / models / runs, CLI plumbing, run-variants."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from escount.calibration.calibrate import ProjectProfile
from escount.cli import _apply_variant, main as cli_main, resolve_case
from escount.es_counting import file_sha256
from escount.features.cache import prepare
from escount.inference.engine import InferenceEngine
from escount.models.all_rois import train_all_rois
from escount.models.select_rois import select_project
from escount.models.train import train_project
from escount.runs import run_variants
from escount.variants import (
    check_name, list_runs, normalize_strips, profile_case, read_run, run_model_dir, split_case, sync_variant,
    variant_names, variant_strips,
)
from synthetic_project import GRID, build

STRIPS_A = {"n": 3, "w": 20, "pitch": 30, "side_margin": 50, "x_offset": 0}
STRIPS_B = {"n": 4, "w": 16, "pitch": 30, "side_margin": 60, "x_offset": 0}


class CaseIdTests(unittest.TestCase):
    def test_split_and_profile_case(self):
        self.assertEqual(split_case("P"), ("P", None, None))
        self.assertEqual(split_case("P@a"), ("P", "a", None))
        self.assertEqual(split_case("P@a~20261120-101500"), ("P", "a", "20261120-101500"))
        self.assertEqual(profile_case("P@a~20261120-101500"), "P@a")

    def test_invalid_names_and_strips_are_refused(self):
        for bad in ("", "a b", "a@b", "a~b", "../x", "-x"):
            with self.assertRaises(ValueError, msg=bad):
                check_name(bad)
        with self.assertRaises(ValueError):
            normalize_strips({**STRIPS_A, "side_margin": 10})       # < 50 px from the valid-region edge
        with self.assertRaises(ValueError):
            normalize_strips({**STRIPS_A, "pitch": 5})              # D <= 2w/3
        with self.assertRaises(ValueError):
            normalize_strips({"n": 3})


class VariantPipelineTests(unittest.TestCase):
    """Base project trained once; variants share its calibration but train / infer on their own folders."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, cls.features, cls.config = build(cls.root)
        cls.base_model_sha = file_sha256(cls.profiles / "SynA" / "model" / "manifest.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_variant_shares_calibration_and_keeps_own_strips(self):
        sync_variant("SynA@shared", self.profiles, strips=STRIPS_A)
        base = ProjectProfile.load(self.profiles / "SynA" / "project.json")
        own = ProjectProfile.load(self.profiles / "SynA@shared" / "project.json")
        self.assertEqual(own.case, "SynA@shared")
        self.assertEqual((own.angle_deg, own.mask_sha256, own.valid_region), (base.angle_deg, base.mask_sha256, base.valid_region))
        self.assertEqual(own.strip_params, STRIPS_A)
        self.assertEqual(own.selected_rois, [])
        # a second sync (before every CLI step) keeps the variant's strips and its selected ROIs
        own.selected_rois = [{"key": "x"}]
        own.save(self.profiles / "SynA@shared" / "project.json")
        sync_variant("SynA@shared", self.profiles)
        again = ProjectProfile.load(self.profiles / "SynA@shared" / "project.json")
        self.assertEqual((again.strip_params, again.selected_rois), (STRIPS_A, [{"key": "x"}]))
        self.assertIn("shared", variant_names("SynA", self.profiles))
        self.assertEqual(variant_strips("SynA@shared", self.profiles), STRIPS_A)

    def test_variant_without_base_calibration_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            sync_variant("Nope@a", self.profiles)

    def test_two_variants_train_independently_and_leave_the_base_untouched(self):
        engines = {}
        for name, strips in (("pa", STRIPS_A), ("pb", STRIPS_B)):
            case = f"SynA@{name}"
            sync_variant(case, self.profiles, strips=strips)
            prepare(case, self.data, self.profiles, self.features, workers=1, grid=GRID)      # reads Data/SynA
            train_project(case, self.features, self.profiles, device="cpu", log=lambda *a, **k: None)
            select_project(case, self.features, self.profiles, device="cpu")
            info = train_all_rois(case, self.profiles, self.data, "cpu")
            self.assertEqual(info["roi_count"], strips["n"])
            engines[name] = InferenceEngine.load(case, profiles_root=self.profiles)
        image = self.data / "SynA" / "070.png"
        counts = {name: len(engine.infer(image, roi_mode="all")["rois"]) for name, engine in engines.items()}
        self.assertEqual(counts, {"pa": 3, "pb": 4})                # each variant counts with its own strips
        self.assertEqual(file_sha256(self.profiles / "SynA" / "model" / "manifest.json"), self.base_model_sha)
        self.assertEqual(ProjectProfile.load(self.profiles / "SynA" / "project.json").strip_params,
                         ProjectProfile.load(self.profiles / "SynA" / "project.json").strip_params)

    def test_resolve_case_and_apply_variant(self):
        self.assertEqual(resolve_case("SynA@pa", self.data), "SynA@pa")
        self.assertEqual(resolve_case(str(self.data / "SynA"), self.data), "SynA")
        with self.assertRaises(SystemExit):
            resolve_case("Missing@pa", self.data)
        args = argparse.Namespace(command="train", project="SynA", variant="viaflag", profiles_root=str(self.profiles))
        _apply_variant(args)
        self.assertEqual(args.project, "SynA@viaflag")
        self.assertTrue((self.profiles / "SynA@viaflag" / "project.json").is_file())
        args = argparse.Namespace(command="infer", project="SynA@viaflag~20261120-101500", variant=None,
                                  profiles_root=str(self.profiles), run=None)
        _apply_variant(args)
        self.assertEqual((args.project, args.run), ("SynA@viaflag", "20261120-101500"))
        with self.assertRaises(SystemExit):                           # --variant without --project
            _apply_variant(argparse.Namespace(command="train", project=None, variant="x", profiles_root="p"))

    def test_variant_cli_create_and_list(self):
        common = ["--project", "SynA", "--profiles-root", str(self.profiles)]
        self.assertEqual(cli_main(["variant", "create", "--name", "cli1", "--n", "5", "--w", "14", *common]), 0)
        strips = variant_strips("SynA@cli1", self.profiles)
        self.assertEqual((strips["n"], strips["w"]), (5, 14))
        self.assertEqual(cli_main(["variant", "list", *common]), 0)
        with self.assertRaises(SystemExit):
            cli_main(["variant", "create", "--name", "bad", "--side-margin", "10", *common])
        self.assertFalse((self.profiles / "SynA@bad").exists())


class RunVariantsTests(unittest.TestCase):
    """run-variants orchestration with the real library steps (small grid) behind an in-process runner."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, cls.features, cls.config = build(cls.root)
        sync_variant("SynA@one", cls.profiles, strips=STRIPS_A)
        cls.calls = []

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def runner(self, argv):
        """Stands in for escount.cli.main: the same library calls, with the test grid."""
        self.calls.append(argv)
        step = argv[0]
        flag = "--cases" if step == "research" else "--project"
        case = argv[argv.index(flag) + 1]
        if step == "prepare":
            prepare(case, self.data, self.profiles, self.features, workers=1, grid=GRID)
        elif step == "train":
            train_project(case, self.features, self.profiles, device="cpu", log=lambda *a, **k: None)
        elif step == "select-rois":
            select_project(case, self.features, self.profiles, device="cpu")
        elif step == "all-rois":
            train_all_rois(case, self.profiles, self.data, "cpu")
        elif step == "infer":
            engine = InferenceEngine.load(case, profiles_root=self.profiles)
            engine.run(argv[argv.index("--input") + 1], argv[argv.index("--output") + 1])
        elif step == "research":
            best = Path(argv[argv.index("--candidates-root") + 1]) / case / "best_params.json"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_text(json.dumps({"case": case, "trial": 2, "config_hash": "abc"}), encoding="utf-8")
        return 0

    def args(self, **overrides):
        values = dict(projects=["SynA"], variants=["one"], device="cpu", max_rois=7, max_rows_per_image=200, workers=1,
                      trials=0, track="default", no_infer=False, note="first try", data_root=str(self.data),
                      cache_root=str(self.features), results_root=str(self.root / "infer"),
                      profiles_root=str(self.profiles), config=str(self.config), storage="sqlite:///x.db",
                      research_root=str(self.root / "research"), candidates_root=str(self.root / "candidates"))
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_run_is_saved_with_options_model_and_results_and_can_be_loaded_later(self):
        self.calls.clear()
        sync_variant("SynA@two", self.profiles, strips=STRIPS_A)
        self.assertEqual(run_variants(self.args(trials=3, variants=["two"]), self.runner), 0)
        self.assertEqual([c[0] for c in self.calls], ["research", "prepare", "train", "select-rois", "all-rois", "infer"])
        (run,) = [item["run"] for item in list_runs("SynA@two", self.profiles)]
        record = read_run("SynA@two", run, self.profiles)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["options"]["max_rois"], 7)
        self.assertEqual(record["options"]["max_rows_per_image"], 200)
        self.assertEqual(record["note"], "first try")
        self.assertEqual(record["strip_params"], STRIPS_A)
        self.assertEqual((record["research"]["trials"], record["research"]["best_params"]["trial"]), (3, 2))
        self.assertEqual(record["model"]["all_rois"]["roi_count"], STRIPS_A["n"])
        self.assertEqual([s["step"] for s in record["steps"]], ["prepare", "train", "select-rois", "all-rois", "infer"])
        self.assertTrue((self.root / "infer" / f"SynA@two~{run}" / "predictions.csv").is_file())
        # the options really reached the CLI steps
        train_call = next(c for c in self.calls if c[0] == "train")
        self.assertEqual(train_call[train_call.index("--max-rows-per-image") + 1], "200")
        # the saved run loads on its own, even after the variant is retrained
        self.runner(["train", "--project", "SynA@two"])
        engine = InferenceEngine.load("SynA@two", profiles_root=self.profiles, run=run)
        self.assertEqual(len(engine.infer(self.data / "SynA" / "070.png", roi_mode="all")["rois"]), STRIPS_A["n"])
        self.assertTrue(run_model_dir("SynA@two", run, self.profiles).joinpath("manifest.json").is_file())

    def test_second_run_keeps_the_first(self):
        before = len(list_runs("SynA@one", self.profiles))
        run_variants(self.args(no_infer=True), self.runner)
        self.assertEqual(len(list_runs("SynA@one", self.profiles)), before + 1)

    def test_failed_step_is_recorded_and_stops_the_pair(self):
        def failing(argv):
            return 5 if argv[0] == "train" else self.runner(argv)

        self.assertEqual(run_variants(self.args(), failing), 3)
        newest = list_runs("SynA@one", self.profiles)[0]
        record = read_run("SynA@one", newest["run"], self.profiles)
        self.assertEqual((record["status"], record["steps"][-1]["step"]), ("failed", "train"))
        self.assertFalse(newest["has_model"])

    def test_missing_variant_is_an_error_and_other_projects_borrow_strips(self):
        with self.assertRaises(SystemExit):
            run_variants(self.args(variants=["nowhere"]), self.runner)
        with self.assertRaises(SystemExit):
            run_variants(self.args(projects=["NotAProject"]), self.runner)


if __name__ == "__main__":
    unittest.main()
