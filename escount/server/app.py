"""Inference server (Task 10): FastAPI app for LabVIEW and other systems.

  GET  /health                     liveness (never needs the API key)
  GET  /projects                   calibrated projects under profiles_root (+ General models, loaded engines)
  POST /projects/{case}/load       load an InferenceEngine once ({"model": "project"|"general"}); becomes current
  POST /infer/path                 {"path": ..., "project"?, "model"?} -> image read by the server
  POST /infer/image                PNG/BMP bytes (multipart field "file" or raw body) -> cv2.imdecode
OpenAPI / Swagger UI at /docs. Inference is serial (one lock), single-thread CPU, nothing is written.

Security: `escount serve` binds 127.0.0.1 by default and refuses any other host without an API key
(check_bind). With an API key every request except /health, the OpenAPI docs and the static UI
files needs the header X-API-Key (401 otherwise). /infer/path only reads files whose resolved path
lies inside one of the configured allowed roots; '..' segments are rejected (403).
"""
import json
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from escount import __version__
from escount.calibration.calibrate import PROFILE_NAME, ProjectProfile
from escount.inference.engine import InferenceEngine

API_KEY_HEADER = "X-API-Key"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PUBLIC_PATHS = frozenset({"/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
IMAGE_SUFFIXES = (".png", ".bmp")
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"BM")
GENERAL_MODELS = ("general", "general_no_ring")
ModelChoice = Literal["project", "general"]


@dataclass(frozen=True)
class ServerConfig:
    profiles_root: str = "outputs/projects"
    allowed_roots: tuple = ("Data",)      # /infer/path may only read inside these folders
    api_key: Optional[str] = None
    general_config: str = "configs/general.json"


def check_bind(host, api_key):
    """Refuse to expose the server beyond the loopback interface without an API key."""
    if host not in LOOPBACK_HOSTS and not api_key:
        raise ValueError(f"refusing to bind {host!r} without an API key: use --api-key (or bind 127.0.0.1)")


def _is_public(path):
    return path in PUBLIC_PATHS or path == "/ui" or path.startswith("/ui/")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def jsonable(value):
    """Plain JSON types (numpy scalars/arrays converted, NaN kept out by the callers)."""
    return json.loads(json.dumps(value, default=_json_default))


def check_case(case):
    if not CASE_NAME.match(case or ""):
        raise HTTPException(422, f"invalid project id {case!r}")
    return case


def decode_image(data):
    """PNG/BMP bytes -> single-channel uint8 array (colour images are converted to gray)."""
    if not data:
        raise HTTPException(400, "empty image body")
    if not data.startswith(IMAGE_MAGIC):
        raise HTTPException(415, "only PNG or BMP images are accepted")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(400, "image bytes could not be decoded")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        raise HTTPException(422, f"expected an 8-bit image, got {image.dtype}")
    return image


class ServerState:
    """Loaded engines keyed by (case, model); the last loaded one is the default for inference."""

    def __init__(self, config):
        self.config = config
        self.profiles_root = Path(config.profiles_root)
        self.roots = [Path(root).resolve() for root in config.allowed_roots]
        self.engines, self.current = {}, None
        self.lock = threading.Lock()

    def load(self, case, model="project"):
        check_case(case)
        try:
            engine = InferenceEngine.load(self.profiles_root / case, model, self.profiles_root,
                                          self.config.general_config)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, KeyError, StopIteration) as error:
            raise HTTPException(422, f"{case}: cannot load {model} model: {error}") from error
        with self.lock:
            self.engines[(case, model)] = engine
            self.current = (case, model)
        return engine

    def engine(self, project=None, model=None):
        if project is None and self.current is None:
            raise HTTPException(409, "no project loaded: POST /projects/{id}/load first")
        key = (project or self.current[0], model or (self.current[1] if project is None else "project"))
        if key not in self.engines:
            raise HTTPException(409, f"project {key[0]!r} ({key[1]} model) is not loaded: "
                                     f"POST /projects/{key[0]}/load first")
        return self.engines[key]

    def allowed_path(self, raw):
        path = Path(raw)
        if ".." in path.parts:
            raise HTTPException(403, "path traversal ('..') is not allowed")
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise HTTPException(403, "path is outside the allowed root folder(s)")
        if not resolved.is_file():
            raise HTTPException(404, "image file not found")
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(415, "only .png or .bmp files are accepted")
        return resolved

    def infer(self, engine, source, name=None):
        from threadpoolctl import threadpool_limits

        try:
            with self.lock, threadpool_limits(limits=1):
                result = engine.infer(source, name=name)
        except (ValueError, IOError) as error:
            raise HTTPException(422, str(error)) from error
        record = InferenceEngine.result_record(result)
        for key in ("outputs", "dpi"):
            record.pop(key, None)
        return jsonable(record)


class LoadRequest(BaseModel):
    model: ModelChoice = "project"


class PathRequest(BaseModel):
    path: str
    project: Optional[str] = None
    model: Optional[ModelChoice] = None


def _engine_info(case, model, engine):
    manifest = engine.runtime.manifest
    return {"case": case, "model": model, "model_case": engine.model_case, "method": manifest["method"],
            "model_type": manifest["model_type"], "roi_count": len(manifest["selected_rois"]),
            "image_size": engine.runtime.profile.image_size}


IMAGE_BODY = {"requestBody": {"required": True, "content": {
    "image/png": {"schema": {"type": "string", "format": "binary"}},
    "image/bmp": {"schema": {"type": "string", "format": "binary"}},
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"],
                                       "properties": {"file": {"type": "string", "format": "binary"}}}},
}}}


def create_app(config=ServerConfig()):
    state = ServerState(config)
    app = FastAPI(title="ESCounting V4 inference server", version=__version__,
                  description="Sheet-count inference for LabVIEW / other systems. See docs/labview_api.md.")
    app.state.server = state

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if config.api_key and not _is_public(request.url.path):
            supplied = request.headers.get(API_KEY_HEADER, "")
            if not secrets.compare_digest(supplied.encode("utf-8"), config.api_key.encode("utf-8")):
                return JSONResponse({"detail": f"missing or invalid {API_KEY_HEADER} header"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/projects")
    def projects():
        items = []
        if state.profiles_root.is_dir():
            for folder in sorted(p for p in state.profiles_root.iterdir() if (p / PROFILE_NAME).is_file()):
                profile = ProjectProfile.load(folder / PROFILE_NAME)
                items.append({"case": folder.name, "quality_status": profile.quality_status,
                              "angle_deg": profile.angle_deg, "reference_image": profile.reference_image,
                              "has_model": (folder / "model" / "manifest.json").is_file(),
                              "selected_rois": len(profile.selected_rois),
                              "loaded_models": sorted(m for c, m in state.engines if c == folder.name)})
        general = [name for name in GENERAL_MODELS if (state.profiles_root / name / "model" / "manifest.json").is_file()]
        current = _engine_info(*state.current, state.engines[state.current]) if state.current else None
        return {"projects": items, "general_models": general, "current": current}

    @app.post("/projects/{case}/load")
    def load_project(case: str, body: Optional[LoadRequest] = None):
        model = (body or LoadRequest()).model
        return {"loaded": True, **_engine_info(case, model, state.load(case, model))}

    @app.post("/infer/path")
    def infer_path(body: PathRequest):
        engine = state.engine(body.project, body.model)
        return state.infer(engine, state.allowed_path(body.path))

    @app.post("/infer/image", openapi_extra=IMAGE_BODY)
    async def infer_image(request: Request, project: Optional[str] = None, model: Optional[ModelChoice] = None,
                          name: Optional[str] = None):
        engine = state.engine(project, model)
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(422, "multipart field 'file' is missing")
            data, name = await upload.read(), name or upload.filename
        else:
            data = await request.body()
        image = decode_image(data)
        return await run_in_threadpool(state.infer, engine, image, name or "image")

    return app
