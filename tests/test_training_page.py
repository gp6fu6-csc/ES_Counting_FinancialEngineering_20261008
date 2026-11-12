"""Training page API: step status, step order, single running job, model unload, CLI arguments.

The CLI subprocess is replaced by a fake process, so no real prepare / train runs here.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from asgi_client import Client
from escount.server.app import ServerConfig, create_app
from synthetic_project import build


class FakeProcess:
    """Stands in for subprocess.Popen: yields log lines, optionally waits for `release` before ending."""

    def __init__(self, returncode=0, release=None):
        self.returncode, self.release = returncode, release

    def __call__(self, command, **kwargs):
        self.command = command
        return self

    @property
    def stdout(self):
        yield "step started\n"
        if self.release is not None:
            self.release.wait(10)
        yield "step finished\n"

    def wait(self):
        return self.returncode


class TrainingPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, _, cls.config = build(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.features = self.root / f"features_{self._testMethodName}"   # fresh: step 1 not done
        self.results = self.root / "infer"
        self.client = Client(create_app(ServerConfig(
            profiles_root=str(self.profiles), allowed_roots=(str(self.data),), general_config=str(self.config),
            results_root=str(self.results), features_root=str(self.features))))

    def status(self):
        response = self.client.get("/training/SynA/status")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def run_step(self, step, fake, **extra):
        with mock.patch("subprocess.Popen", fake):
            response = self.client.post("/training/SynA/run", json={"step": step, **extra})
            if response.status_code == 200:
                self.wait_done()
        return response

    def wait_done(self):
        for _ in range(200):
            job = self.status()["job"]
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_status_reports_each_step(self):
        steps = self.status()["steps"]
        self.assertEqual(set(steps), {"calibrate", "prepare", "train", "select", "infer"})
        self.assertTrue(steps["calibrate"])
        self.assertFalse(steps["prepare"])                     # empty features_root
        self.assertEqual(self.client.get("/training/NoSuchCase/status").status_code, 404)

    def test_steps_must_run_in_order(self):
        response = self.client.post("/training/SynA/run", json={"step": "train"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("prepare", response.json()["detail"])
        self.assertEqual(self.client.post("/training/SynA/run", json={"step": "bogus"}).status_code, 422)

    def test_successful_job_records_log_and_command(self):
        fake = FakeProcess()
        self.assertEqual(self.run_step("prepare", fake).status_code, 200)
        job = self.status()["job"]
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["returncode"], 0)
        self.assertEqual(job["log"], ["$ escount " + " ".join(["prepare", "--project", "SynA", *self.roots()]),
                                      "step started", "step finished"])
        self.assertEqual(fake.command[1:], ["-m", "escount.cli", "prepare", "--project", "SynA", *self.roots()])

    def roots(self):
        """The server's own folders, appended by the training page to every step command."""
        return ["--data-root", str(self.data.resolve()), "--profiles-root", str(self.profiles),
                "--cache-root", str(self.features)]

    def test_failed_job_is_reported(self):
        self.run_step("prepare", FakeProcess(returncode=1))
        self.assertEqual(self.status()["job"]["status"], "failed")

    def test_only_one_job_runs_at_a_time(self):
        release = threading.Event()
        with mock.patch("subprocess.Popen", FakeProcess(release=release)):
            self.assertEqual(self.client.post("/training/SynA/run", json={"step": "prepare"}).status_code, 200)
            second = self.client.post("/training/SynA/run", json={"step": "prepare"})
            self.assertEqual(second.status_code, 409)
            self.assertEqual(self.status()["job"]["status"], "running")
            release.set()
            self.assertEqual(self.wait_done()["status"], "done")

    def test_train_unloads_model_and_infer_writes_to_results_root(self):
        (self.features / "SynA").mkdir(parents=True, exist_ok=True)
        (self.features / "SynA" / "features.parquet").write_bytes(b"stub")   # step 1 done
        self.assertEqual(self.client.post("/projects/SynA/load").status_code, 200)
        self.assertEqual(self.loaded_models(), ["project"])
        fake = FakeProcess()
        self.assertEqual(self.run_step("train", fake, device="cpu").status_code, 200)
        self.assertEqual(fake.command[3:], ["train", "--project", "SynA", "--device", "cpu", *self.roots()])
        self.assertEqual(self.loaded_models(), [])            # next infer / preview loads the new model
        if self.status()["steps"]["select"]:
            self.assertEqual(self.run_step("infer", fake).status_code, 200)
            self.assertEqual(fake.command[3:], ["infer", "--project", "SynA", "--input", str(self.data / "SynA"),
                                                "--output", str(self.results / "SynA"),
                                                "--profiles-root", str(self.profiles)])

    def loaded_models(self):
        projects = self.client.get("/projects").json()["projects"]
        return next(p["loaded_models"] for p in projects if p["case"] == "SynA")


if __name__ == "__main__":
    unittest.main()
