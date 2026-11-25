"""Purge: deletes only the named data set's generated data (never Data/, never a similarly named case)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from asgi_client import Client
from escount import cli
from escount.purge import plan_purge, purge_case
from escount.server.app import ServerConfig, create_app
from synthetic_project import build


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


class PurgeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        r = self.root = Path(self._tmp.name)
        self.opts = dict(profiles_root=r / "projects", cache_root=r / "features", results_root=r / "infer",
                         research_root=r / "research", candidates_root=r / "candidates")
        for case in ("A", "A@v", "AB"):
            touch(r / "projects" / case / "model" / "manifest.json")
            touch(r / "projects" / case / "project.json")
            touch(r / "projects" / case / "height_model.json")
            touch(r / "features" / case / "features.parquet")
        touch(r / "infer" / "A" / "predictions.csv")
        touch(r / "infer" / "A@v~run1" / "predictions.csv")
        touch(r / "infer" / "AB" / "predictions.csv")
        touch(r / "candidates" / "A" / "best_params.json")
        touch(r / "candidates" / "mae" / "A" / "best_params.json")
        touch(r / "candidates" / "AB" / "best_params.json")
        touch(r / "Data" / "A" / "1.png")

    def exists(self, *parts):
        return self.root.joinpath(*parts).exists()

    def test_dry_run_deletes_nothing(self):
        items = purge_case("A", dry_run=True, **self.opts)
        self.assertTrue(items)
        self.assertTrue(self.exists("projects", "A", "model"))

    def test_keeps_calibration_and_similar_names_and_data(self):
        purge_case("A", dry_run=False, **self.opts)
        self.assertFalse(self.exists("projects", "A", "model"))
        self.assertFalse(self.exists("features", "A"))
        self.assertFalse(self.exists("infer", "A"))
        self.assertTrue(self.exists("projects", "A", "project.json"))          # calibration kept
        self.assertTrue(self.exists("projects", "A", "height_model.json"))
        self.assertTrue(self.exists("projects", "A@v", "model"))               # variants only with with_variants
        self.assertTrue(self.exists("projects", "AB", "model"))                # "AB" is another case
        self.assertTrue(self.exists("infer", "AB"))
        self.assertTrue(self.exists("Data", "A", "1.png"))
        self.assertTrue(self.exists("candidates", "A"))                        # research needs the option

    def test_all_variants_research_and_calibration(self):
        purge_case("A", dry_run=False, with_variants=True, research=True, all_files=True, **self.opts)
        for parts in (("projects", "A"), ("projects", "A@v"), ("infer", "A@v~run1"), ("candidates", "A"),
                      ("candidates", "mae", "A")):
            self.assertFalse(self.exists(*parts), parts)
        for parts in (("projects", "AB"), ("candidates", "AB"), ("Data", "A", "1.png")):
            self.assertTrue(self.exists(*parts), parts)

    def test_variant_only(self):
        purge_case("A@v", dry_run=False, **self.opts)
        self.assertFalse(self.exists("projects", "A@v", "model"))
        self.assertFalse(self.exists("infer", "A@v~run1"))
        self.assertTrue(self.exists("projects", "A", "model"))
        self.assertTrue(self.exists("infer", "A"))

    def test_invalid_case_rejected(self):
        for bad in ("", "..", "../x", "a/b"):
            with self.assertRaises(ValueError):
                plan_purge(bad, **self.opts)

    def test_cli_needs_yes_and_does_not_create_variant(self):
        argv = ["purge", "--project", "A", "--profiles-root", str(self.opts["profiles_root"]),
                "--cache-root", str(self.opts["cache_root"]), "--results-root", str(self.opts["results_root"])]
        self.assertEqual(cli.main(argv), 0)
        self.assertTrue(self.exists("features", "A"))                          # no --yes: only listed
        self.assertEqual(cli.main([*argv, "--variant", "new", "--yes"]), 0)
        self.assertFalse(self.exists("projects", "A@new"))                     # --variant must not create a folder
        self.assertEqual(cli.main([*argv, "--yes"]), 0)
        self.assertFalse(self.exists("features", "A"))


class PurgeServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        data, profiles, _, general = build(root)
        self.features, self.results = root / "features", root / "infer"
        touch(self.features / "SynA" / "features.parquet")
        touch(self.results / "SynA" / "predictions.csv")
        self.data = data
        self.client = Client(create_app(ServerConfig(
            profiles_root=str(profiles), allowed_roots=(str(data),), general_config=str(general),
            results_root=str(self.results), features_root=str(self.features),
            research_root=str(root / "research"), candidates_root=str(root / "candidates"))))

    def test_dry_run_then_delete(self):
        url = "/training/SynA/purge"
        plan = self.client.post(url, json={"dry_run": True})
        self.assertEqual(plan.status_code, 200, plan.text)
        self.assertTrue(plan.json()["items"])
        self.assertTrue((self.features / "SynA").exists())
        done = self.client.post(url, json={"dry_run": False})
        self.assertEqual(done.status_code, 200, done.text)
        self.assertFalse((self.features / "SynA").exists())
        self.assertFalse((self.results / "SynA").exists())
        self.assertTrue((self.data / "SynA").exists())

    def test_refused_while_job_running_and_for_unknown_case(self):
        self.assertEqual(self.client.post("/training/NoSuch/purge", json={}).status_code, 404)
        with mock.patch("subprocess.Popen") as popen:
            import threading
            release = threading.Event()
            popen.return_value.stdout = iter(())
            popen.return_value.wait = lambda: release.wait(10) and 0
            self.client.post("/training/SynA/run", json={"step": "prepare"})
            response = self.client.post("/training/SynA/purge", json={"dry_run": False})
            release.set()
        self.assertEqual(response.status_code, 409, response.text)


if __name__ == "__main__":
    unittest.main()
