import shutil
import tempfile
import unittest
from pathlib import Path

import cv2

from asgi_client import Client
from escount.cli import main as cli_main
from escount.es_counting import tree_signatures
from escount.inference.engine import InferenceEngine
from escount.server.app import ServerConfig, check_bind, create_app
from synthetic_project import build

KEY = "test-key-123"


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data, cls.profiles, _, cls.config = build(cls.root)
        cls.image = cls.data / "SynA" / "076.png"
        cls.outside = cls.root / "outside" / "076.png"
        cls.outside.parent.mkdir()
        shutil.copy(cls.image, cls.outside)
        cls.before = tree_signatures(cls.data)
        cls.expected = InferenceEngine.load("SynA", profiles_root=cls.profiles).infer(cls.image)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def app(self, api_key=None):
        return create_app(ServerConfig(profiles_root=str(self.profiles), allowed_roots=(str(self.data),),
                                       api_key=api_key, general_config=str(self.config)))

    def loaded_client(self):
        client = Client(self.app())
        response = client.post("/projects/SynA/load")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["case"], "SynA")
        return client

    def assert_same_result(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["N_pred"], self.expected["N_pred"])
        self.assertEqual(result["N_pred"], 76)
        self.assertEqual(result["gt"], 76)
        self.assertTrue(result["hit"])
        self.assertEqual([r["prediction"] for r in result["rois"]], [r["prediction"] for r in self.expected["rois"]])
        self.assertEqual(set(result["timings_ms"]) - {"read"}, set(self.expected["timings_ms"]) - {"read"})

    def test_load_then_path_and_image_inference_match_engine(self):
        client = self.loaded_client()
        projects = client.get("/projects").json()
        self.assertEqual([p["case"] for p in projects["projects"]], ["SynA"])
        self.assertEqual(projects["current"]["case"], "SynA")
        self.assert_same_result(client.post("/infer/path", json={"path": str(self.image)}))
        png = self.image.read_bytes()
        self.assert_same_result(client.post("/infer/image", content=png, content_type="image/png",
                                            params={"name": "076.png"}))
        self.assert_same_result(client.post("/infer/image", files={"file": ("076.png", png, "image/png")}))
        ok, bmp = cv2.imencode(".bmp", cv2.imread(str(self.image), cv2.IMREAD_UNCHANGED))
        self.assertTrue(ok)
        self.assert_same_result(client.post("/infer/image", content=bmp.tobytes(), params={"name": "076.bmp"}))
        self.assertEqual(client.post("/infer/image", content=b"GIF89a....").status_code, 415)
        self.assertEqual(tree_signatures(self.data), self.before)

    def test_infer_before_load_is_rejected(self):
        client = Client(self.app())
        self.assertEqual(client.post("/infer/path", json={"path": str(self.image)}).status_code, 409)
        self.assertEqual(client.post("/infer/image", content=self.image.read_bytes()).status_code, 409)
        self.assertEqual(client.post("/projects/NoSuchCase/load").status_code, 404)
        loaded = self.loaded_client()
        self.assertEqual(loaded.post("/infer/path", json={"path": str(self.image), "project": "Other"}).status_code, 409)

    def test_api_key_required_except_health(self):
        app = self.app(api_key=KEY)
        anonymous, wrong, good = Client(app), Client(app, {"X-API-Key": "wrong"}), Client(app, {"X-API-Key": KEY})
        self.assertEqual(anonymous.get("/health").status_code, 200)
        for client in (anonymous, wrong):
            self.assertEqual(client.get("/projects").status_code, 401)
            self.assertEqual(client.post("/projects/SynA/load").status_code, 401)
            self.assertEqual(client.post("/infer/path", json={"path": str(self.image)}).status_code, 401)
            self.assertEqual(client.post("/infer/image", content=self.image.read_bytes()).status_code, 401)
        self.assertEqual(good.get("/projects").status_code, 200)
        self.assertEqual(good.post("/projects/SynA/load").status_code, 200)
        self.assert_same_result(good.post("/infer/path", json={"path": str(self.image)}))

    def test_path_traversal_and_outside_root_are_forbidden(self):
        client = self.loaded_client()
        traversal = str(self.data / "SynA" / ".." / "SynA" / "076.png")
        self.assertEqual(client.post("/infer/path", json={"path": traversal}).status_code, 403)
        self.assertEqual(client.post("/infer/path", json={"path": str(self.outside)}).status_code, 403)
        self.assertEqual(client.post("/infer/path", json={"path": str(self.config)}).status_code, 403)
        self.assertEqual(client.post("/infer/path", json={"path": "076.png"}).status_code, 403)   # relative to cwd
        self.assertEqual(client.post("/infer/path", json={"path": str(self.data / "SynA" / "999.png")}).status_code, 404)

    def test_exposed_bind_requires_api_key(self):
        with self.assertRaises(ValueError):
            check_bind("0.0.0.0", None)
        check_bind("127.0.0.1", None)
        check_bind("0.0.0.0", KEY)
        with self.assertRaises(SystemExit):
            cli_main(["serve", "--host", "0.0.0.0"])

    def test_openapi_documents_both_infer_modes(self):
        client = Client(self.app())
        self.assertEqual(client.get("/docs").status_code, 200)
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/health", "/projects", "/projects/{case}/load", "/infer/path", "/infer/image"):
            self.assertIn(path, paths)
        self.assertIn("multipart/form-data", paths["/infer/image"]["post"]["requestBody"]["content"])


if __name__ == "__main__":
    unittest.main()
