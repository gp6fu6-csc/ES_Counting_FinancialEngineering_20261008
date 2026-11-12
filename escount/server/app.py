"""Inference server (Task 10): FastAPI app for LabVIEW and other systems.

  GET  /health                     liveness (never needs the API key)
  GET  /projects                   calibrated projects under profiles_root (+ General models, loaded engines)
  POST /projects/{case}/load       load an InferenceEngine once ({"model": "project"|"general"}); becomes current
  POST /infer/path                 {"path": ..., "project"?, "model"?} -> image read by the server
  POST /infer/image                PNG/BMP bytes (multipart field "file" or raw body) -> cv2.imdecode
OpenAPI / Swagger UI at /docs. Inference is serial (one lock), single-thread CPU, nothing is written.

Web UI (static, offline, vendored Vue 3 + ECharts) at /ui/calibration.html (PDF p12) with
  GET  /calibration/cases, GET /calibration/{case}/images, GET /calibration/{case}/image/{name}
  POST /calibration/preview        ROIs + strips -> angle, residuals, quality, strip overlay (nothing written)
  POST /calibration/{case}/save    same validation, then project.json + background_BW_rotated.png
and /ui/results.html (PDF p13-14) with
  GET  /results/{case}/images      images + whether `escount infer` outputs are complete
  POST /preview                    InferenceEngine with overridden MACD / KD params -> N_pred + chart series
Project image folders are read from the first allowed root.

Security: `escount serve` binds 127.0.0.1 by default and refuses any other host without an API key
(check_bind). With an API key every request except /health, the OpenAPI docs and the static UI
files needs the header X-API-Key (401 otherwise). Without an API key only loopback Host headers
(127.0.0.1, localhost, ::1) are answered (400 otherwise; DNS-rebinding guard). /infer/path only
reads files whose path lies inside one of the configured allowed roots, checked lexically before
any file-system access and again after resolve(); '..' segments and UNC paths are rejected (403).
"""
import json
import os
import re
import secrets
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from escount import __version__
from escount.calibration.calibrate import (
    PROFILE_NAME, CalibrationConfig, CalibrationQualityError, ProjectProfile, SquareRoi, calibrate_project,
    default_rois, estimate_angle, evaluate_angle, validate_rois, valid_region_info,
)
from escount.calibration.rotation_correction import normalize_mask, rotate_image
from escount.es_counting import list_images, pick_reference, read_gray, read_mask
from escount.inference.engine import InferenceEngine
from escount.signals.roi_bank import MIN_SIDE_MARGIN, build_roi_bank, check_geometry, stack_rows

API_KEY_HEADER = "X-API-Key"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PUBLIC_PATHS = frozenset({"/", "/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
IMAGE_SUFFIXES = (".png", ".bmp")
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"BM")
GENERAL_MODELS = ("general", "general_no_ring")
ModelChoice = Literal["project", "general"]
UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
DEFAULT_STRIPS = {"n": 4, "w": 16, "pitch": 30, "side_margin": MIN_SIDE_MARGIN, "x_offset": 0}


@dataclass(frozen=True)
class ServerConfig:
    profiles_root: str = "outputs/projects"
    allowed_roots: tuple = ("Data",)      # /infer/path may only read inside these folders
    api_key: Optional[str] = None
    general_config: str = "configs/general.json"
    results_root: str = "outputs/infer"   # `escount infer` outputs (<case>/<image stem>/...) for the results page
    features_root: str = "outputs/features"  # `escount prepare` cache (training page step status)


def check_bind(host, api_key):
    """Refuse to expose the server beyond the loopback interface without an API key."""
    if host not in LOOPBACK_HOSTS and not api_key:
        raise ValueError(f"refusing to bind {host!r} without an API key: use --api-key (or bind 127.0.0.1)")


def host_name(header):
    """Host header without the port ('[::1]:8000' -> '::1', 'localhost:8000' -> 'localhost')."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


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


def strip_layout(rotated_image, rotated_mask, strips):
    """N adjacent counting strips (w, D, side margin, x offset) closest to the valid-region centre.

    Uses the ROI bank rules (>= 50 px per-row side margin, D > 2w/3) and keeps only strips that span
    the detected stack; raises ValueError when fewer than N strips fit.
    """
    bank = build_roi_bank(rotated_mask, (strips.w,), pitch=strips.pitch, side_margin=strips.side_margin,
                          x_offset=strips.x_offset)
    extent, kept = stack_rows(rotated_image, bank, full_span=True)
    if len(kept) < strips.n:
        raise ValueError(f"only {len(kept)} strips of w={strips.w} px, D={strips.pitch} px span the stack "
                         f"(requested N={strips.n})")
    centre = valid_region_info(rotated_mask)["centre_x"]
    start = min(range(len(kept) - strips.n + 1),
                key=lambda i: abs((kept[i].x + kept[i + strips.n - 1].x + strips.w) / 2.0 - centre))
    chosen = kept[start:start + strips.n]
    return {"stack_extent": {"y0": int(extent["y0"]), "y1": int(extent["y1"])}, "available_strips": len(kept),
            "strips": [{"index": i, "x": s.x, "w": s.w, "y0": s.y0, "y1": s.y1} for i, s in enumerate(chosen, 1)]}


class ServerState:
    """Loaded engines keyed by (case, model); the last loaded one is the default for inference."""

    def __init__(self, config):
        self.config = config
        self.profiles_root = Path(config.profiles_root)
        self.roots = [Path(root).resolve() for root in config.allowed_roots]
        self.lexical_roots = [Path(os.path.abspath(root)) for root in config.allowed_roots]
        self.data_root = self.roots[0]            # project image folders (calibration / results pages)
        self.engines, self.current = {}, None
        self.lock = threading.Lock()

    # ------------------------------------------------------------ project image folders
    def cases(self):
        from escount.features.cache import case_ids

        return case_ids(self.data_root) if self.data_root.is_dir() else []

    def case_dir(self, case, need_mask=True):
        folder = self.data_root / check_case(case)
        if not folder.is_dir() or (need_mask and not (folder / "background_BW.png").is_file()):
            raise HTTPException(404, f"project folder {case!r} with background_BW.png not found")
        return folder

    def case_image(self, case, name):
        """Path of an analysable image of the case (background files are never served)."""
        images = {path.name: path for path in list_images(self.case_dir(case, need_mask=False))}
        if name not in images:
            raise HTTPException(404, f"image {name!r} not found in project {case!r}")
        return images[name]

    def calibration(self, case, body):
        """Validate ROIs and strips, estimate the angle and lay out the strips (no file is written)."""
        case_dir = self.case_dir(case)
        _, mask = read_mask(case_dir)
        try:
            reference = body.reference or pick_reference(list_images(case_dir)).name
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        image = read_gray(self.case_image(case, reference))
        rois = [SquareRoi(roi.x, roi.y, roi.side) for roi in body.rois]
        strips = body.strips
        try:
            check_geometry(strips.w, strips.pitch, strips.side_margin)
            validate_rois(rois, mask)
            config = CalibrationConfig(roi_side=max(roi.side for roi in rois), roi_count=len(rois))
            angle, roi_angles = estimate_angle(image, rois)
            result = evaluate_angle(image, mask, rois, angle, config)
            layout = strip_layout(result["rotated_image"], result["rotated_mask"], strips)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"case": case, "reference": reference, "angle_deg": angle, "roi_angles_deg": roi_angles,
                "before_residual_deg": result["before_residual_deg"],
                "after_residual_deg": result["after_residual_deg"],
                "after_roi_angles_deg": result["after_roi_angles_deg"], "quality_status": result["quality_status"],
                "tolerance_deg": config.tolerance_deg, "rois": [asdict(roi) for roi in rois],
                "rotated_valid_region": valid_region_info(result["rotated_mask"]), **layout,
                "strip_params": strips.model_dump(), "_config": config, "_rois": rois}

    def cached_engine(self, case, model="project"):
        """Engine for the results page: reuse a loaded one, else load it without changing `current`."""
        key = (check_case(case), model)
        if key not in self.engines:
            self.load(case, model, make_current=False)
        return self.engines[key]

    def load(self, case, model="project", make_current=True):
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
            if make_current:
                self.current = (case, model)
        return engine

    def unload(self, case):
        """Drop every loaded engine of `case` (after its calibration changed)."""
        with self.lock:
            for key in [key for key in self.engines if key[0] == case]:
                del self.engines[key]
            if self.current and self.current[0] == case:
                self.current = None

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
        if raw.startswith(("\\\\", "//")):           # UNC / device paths: never touch the network
            raise HTTPException(403, "UNC / network paths are not allowed")
        lexical = Path(os.path.abspath(raw))         # no file-system access before the root check
        if not any(lexical.is_relative_to(root) for root in self.lexical_roots + self.roots):
            raise HTTPException(403, "path is outside the allowed root folder(s)")
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise HTTPException(403, "path is outside the allowed root folder(s)")
        if not resolved.is_file():
            raise HTTPException(404, "image file not found")
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(415, "only .png or .bmp files are accepted")
        return resolved

    def run(self, engine, source, name=None, overrides=None, roi_mode="minimal"):
        """engine.infer serialised by the lock, single-thread CPU; bad images -> 422."""
        from threadpoolctl import threadpool_limits

        try:
            with self.lock, threadpool_limits(limits=1):
                return engine.infer(source, name=name, overrides=overrides, roi_mode=roi_mode)
        except (ValueError, IOError) as error:
            raise HTTPException(422, str(error)) from error

    def infer(self, engine, source, name=None):
        record = InferenceEngine.result_record(self.run(engine, source, name))
        for key in ("outputs", "dpi"):
            record.pop(key, None)
        return jsonable(record)


class LoadRequest(BaseModel):
    model: ModelChoice = "project"


class PathRequest(BaseModel):
    path: str
    project: Optional[str] = None
    model: Optional[ModelChoice] = None


class RoiModel(BaseModel):
    x: int
    y: int
    side: int


class StripModel(BaseModel):
    n: int = Field(DEFAULT_STRIPS["n"], ge=1, le=64)
    w: int = Field(DEFAULT_STRIPS["w"], ge=1, le=400)
    pitch: int = Field(DEFAULT_STRIPS["pitch"], ge=1, le=2000)
    side_margin: int = DEFAULT_STRIPS["side_margin"]   # < 50 is rejected by the ROI bank rules (422)
    x_offset: int = Field(DEFAULT_STRIPS["x_offset"], ge=0, le=2000)


class CalibrationBody(BaseModel):
    reference: Optional[str] = None
    rois: list[RoiModel] = Field(min_length=1, max_length=9)
    strips: StripModel = StripModel()


class PreviewCalibrationRequest(CalibrationBody):
    case: str


class MaskBody(BaseModel):
    """Valid-region polygon in full-resolution image pixels (x, y), at least 3 vertices."""
    points: list[tuple[float, float]] = Field(min_length=3, max_length=2000)


class MACDBody(BaseModel):
    ema1: int = Field(ge=1, le=500)
    ema2: int = Field(ge=1, le=500)
    dea: int = Field(ge=1, le=500)


class KDBody(BaseModel):
    n: int = Field(ge=1, le=500)      # RSV window (px)
    k: int = Field(ge=1, le=500)
    d: int = Field(ge=1, le=500)


class PreviewRequest(BaseModel):
    case: str
    image: str
    model: ModelChoice = "project"
    roi_mode: Literal["minimal", "all"] = "minimal"
    method: Optional[str] = "auto"    # Q7: auto / baseline / kd / macd / xgboost / ridge / trimmed_mean
    macd: Optional[MACDBody] = None   # None = the model's trained parameters
    kd: Optional[KDBody] = None


def _series(values):
    return np.round(np.asarray(values, dtype=np.float64), 4).tolist()


def preview_payload(engine, result, overrides):
    """N_pred + per-ROI signal / MACD / KD series and events for the results page charts."""
    from escount.features.extract import projection_sigma
    from escount.indicators.core import kd, macd
    from escount.indicators.counters import IndicatorConfig, count
    from escount.signals.projection import crop_horizontal_slice, project

    params, configs = {}, {}
    for family, source in engine.indicator_sources.items():
        config = source["config"]
        value = overrides.get(family, config.params)
        configs[family] = IndicatorConfig(family, value, config.gate, "both")
        params[family] = {**asdict(value), "gate": config.gate, "column": source["column"],
                          "trained": asdict(config.params), "affects_prediction": source["affects_prediction"]}
    rois = []
    rotated_img = result.get("rotated")
    ratio = engine.runtime.grid.gaussian_sigma_ratio if hasattr(engine, "runtime") and hasattr(engine.runtime, "grid") else 0.5

    # 建立全有效範圍（非 supporter）槽位對照表，確保結果頁圖表與切片均從頂部（0 訊號）畫到底部 (PDF P.8 & P.10)
    items_by_key = {}
    candidate_items = engine.runtime.all_items if (result.get("roi_mode") == "all" and getattr(engine.runtime, "all_items", None)) else getattr(engine.runtime, "items", [])
    if hasattr(engine, "runtime"):
        for s, proj in candidate_items:
            items_by_key[(s.x, s.w, proj)] = s

    for roi, panel in zip(result["rois"], engine._signals(result)):
        disp_slot = roi.get("slot")
        proj = roi.get("projection")
        if disp_slot is not None and (disp_slot.x, disp_slot.w, proj) in items_by_key:
            disp_slot = items_by_key[(disp_slot.x, disp_slot.w, proj)]
            if rotated_img is not None:
                signal = project(rotated_img, disp_slot, proj, projection_sigma(proj, disp_slot.w, ratio))
                _, high_idx, low_idx = count(signal, engine.marks) if signal.size >= 2 else (0, [], [])
            else:
                signal = panel["signal"]
                high_idx, low_idx = panel["high_idx"], panel["low_idx"]
        else:
            signal = panel["signal"]
            high_idx, low_idx = panel["high_idx"], panel["low_idx"]

        lines = {"macd": macd(signal, configs["macd"].params), "kd": kd(signal, configs["kd"].params)}
        charts = {}
        for family, line in lines.items():
            number, death, golden = count(signal, configs[family])
            charts[family] = {**{name: _series(getattr(line, name)) for name in line._fields},
                              "golden": [int(i) for i in golden], "death": [int(i) for i in death], "count": number}
        slice_b64 = None
        if rotated_img is not None and disp_slot is not None:
            try:
                slice_b64 = crop_horizontal_slice(rotated_img, disp_slot, as_png_base64=True)
            except Exception:
                slice_b64 = None
        roi_data = {key: value for key, value in roi.items() if key != "slot"}
        if disp_slot is not None:
            roi_data["y0"] = int(disp_slot.y0)
            roi_data["y1"] = int(disp_slot.y1)
        rois.append({**roi_data, "signal": _series(signal),
                     "highs": [int(i) for i in high_idx], "lows": [int(i) for i in low_idx],
                     "label": panel["label"], "slice_image": slice_b64, **charts})
    record = {key: result[key] for key in ("image", "case", "model", "model_case", "method", "model_type", "N_pred",
                                           "gt", "hit", "timings_ms", "roi_mode", "all_rois_ensemble") if key in result}
    marks = engine.marks
    return jsonable({**record, "params": params, "marks": {"name": marks.name, "params": asdict(marks.params),
                                                           "gate": marks.gate}, "rois": rois})


def _public_calibration(result):
    return jsonable({key: value for key, value in result.items() if not key.startswith("_")})


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
        if not config.api_key and host_name(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
            # keyless server = loopback only: refuse foreign Host headers (DNS rebinding)
            return JSONResponse({"detail": "invalid Host header: the keyless server only answers 127.0.0.1 / "
                                           "localhost"}, status_code=400)
        if config.api_key and not _is_public(request.url.path):
            supplied = request.headers.get(API_KEY_HEADER, "")
            if not secrets.compare_digest(supplied.encode("utf-8"), config.api_key.encode("utf-8")):
                return JSONResponse({"detail": f"missing or invalid {API_KEY_HEADER} header"}, status_code=401)
        response = await call_next(request)
        if request.url.path.startswith("/ui/"):
            response.headers["Cache-Control"] = "no-cache"   # revalidate (ETag) so UI updates are never stale
        return response

    @app.get("/", include_in_schema=False)
    def root():
        """Root URL opens the web UI (training page links to calibration and results)."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/ui/training.html")

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

    # ------------------------------------------------------------ calibration page (Task 11, PDF p12)
    @app.get("/calibration/cases")
    def calibration_cases():
        # folders without background_BW.png are listed too, so the mask can be drawn on the calibration page
        folders = sorted(p.name for p in state.data_root.iterdir()
                         if p.is_dir() and ((p / "background_BW.png").is_file() or list_images(p))) \
            if state.data_root.is_dir() else []
        return {"cases": folders}

    @app.get("/calibration/{case}/images")
    def calibration_images(case: str):
        case_dir = state.case_dir(case, need_mask=False)
        paths = list_images(case_dir)
        has_mask = (case_dir / "background_BW.png").is_file()
        try:
            reference = pick_reference(paths).name
        except ValueError:
            reference = paths[0].name if paths else None
        if has_mask:
            _, mask = read_mask(case_dir)
            image_size = {"width": int(mask.shape[1]), "height": int(mask.shape[0])}
            region = valid_region_info(mask)
            try:
                defaults = [asdict(roi) for roi in default_rois(mask)]
            except ValueError:
                defaults = []
        else:                                     # no mask yet: size from the reference image, ROIs wait for the mask
            if reference is None:
                raise HTTPException(404, f"project folder {case!r} has no analysable image")
            shape = read_gray(case_dir / reference).shape
            image_size, region, defaults = {"width": int(shape[1]), "height": int(shape[0])}, None, []
        profile_path = state.profiles_root / case / PROFILE_NAME
        profile = json.loads(ProjectProfile.load(profile_path).to_json()) if profile_path.is_file() else None
        config = CalibrationConfig()
        return jsonable({"case": case, "images": [path.name for path in paths], "reference": reference,
                         "has_mask": has_mask,
                         "image_size": image_size,
                         "valid_region": region, "default_rois": defaults,
                         "min_side": config.min_side, "default_side": config.roi_side,
                         "min_side_margin": MIN_SIDE_MARGIN,
                         "strips": (profile or {}).get("strip_params") or DEFAULT_STRIPS, "profile": profile})

    @app.get("/calibration/{case}/image/{name}", response_class=Response,
             responses={200: {"content": {"image/png": {}}}})
    def calibration_image(case: str, name: str, angle: float = 0.0, max_width: int = 1224):
        """Downscaled PNG of a project image (optionally rotated by `angle` degrees, as in inference)."""
        if not 64 <= max_width <= 8192 or abs(angle) > 45:
            raise HTTPException(422, "max_width must be in [64, 8192] and |angle| <= 45")
        image = read_gray(state.case_image(case, name))
        if angle:
            image = rotate_image(image, angle)
        scale = min(1.0, max_width / image.shape[1])
        if scale < 1.0:
            image = cv2.resize(image, (max_width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(500, "could not encode PNG")
        return Response(encoded.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/calibration/{case}/mask-overlay/{name}", response_class=Response,
             responses={200: {"content": {"image/png": {}}}})
    def calibration_mask_overlay(case: str, name: str, max_width: int = 1224):
        """Project image with the valid region (black in background_BW.png) tinted transparent red."""
        if not 64 <= max_width <= 8192:
            raise HTTPException(422, "max_width must be in [64, 8192]")
        case_dir = state.case_dir(case)
        image = read_gray(state.case_image(case, name))
        _, mask = read_mask(case_dir)
        valid = normalize_mask(mask)
        if valid.shape != image.shape:
            raise HTTPException(422, f"mask {valid.shape} and image {image.shape} sizes differ")
        colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        red = np.zeros_like(colour)
        red[:, :] = (0, 0, 255)
        colour[valid] = cv2.addWeighted(colour, 0.6, red, 0.4, 0)[valid]
        scale = min(1.0, max_width / colour.shape[1])
        if scale < 1.0:
            colour = cv2.resize(colour, (max_width, max(1, round(colour.shape[0] * scale))),
                                interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", colour)
        if not ok:
            raise HTTPException(500, "could not encode PNG")
        return Response(encoded.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/calibration/{case}/mask")
    def calibration_mask_save(case: str, body: MaskBody):
        """Rasterise the polygon (inside = valid = black 0, outside = white 255) into background_BW.png.

        An existing mask is first copied to <profiles_root>/<case>/background_BW.prev.png.
        """
        from escount.calibration.rotation_correction import write_image

        case_dir = state.case_dir(case, need_mask=False)
        paths = list_images(case_dir)
        if not paths:
            raise HTTPException(404, f"project folder {case!r} has no analysable image")
        height, width = read_gray(paths[0]).shape
        points = np.array(body.points, dtype=np.float64)
        if (points[:, 0] < -0.5).any() or (points[:, 1] < -0.5).any() \
                or (points[:, 0] > width - 0.5).any() or (points[:, 1] > height - 0.5).any():
            raise HTTPException(422, f"polygon vertices must lie inside the {width}x{height} image")
        mask = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], (0, 0, 0))
        if not (mask[:, :, 0] < 128).any():
            raise HTTPException(422, "polygon encloses no pixel")
        target = case_dir / "background_BW.png"
        if target.is_file():
            backup = state.profiles_root / case / "background_BW.prev.png"
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(target.read_bytes())
        write_image(target, mask)
        state.unload(case)                        # geometry changed
        return {"saved": True, "path": str(target.resolve()),
                "note": "background_BW.png updated: re-run rotation calibration, prepare, train and select-rois"}

    @app.post("/calibration/preview")
    def calibration_preview(body: PreviewCalibrationRequest):
        return _public_calibration(state.calibration(body.case, body))

    @app.post("/calibration/{case}/save")
    def calibration_save(case: str, body: CalibrationBody):
        """Validate like /calibration/preview, then write project.json + background_BW_rotated.png."""
        checked = state.calibration(case, body)
        try:
            calibrate_project(state.case_dir(case), state.profiles_root, checked["reference"], checked["_rois"],
                              checked["_config"])
        except CalibrationQualityError:
            pass                                  # the profile is written with quality_status 'quality_failure'
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        path = state.profiles_root / case / PROFILE_NAME
        profile = ProjectProfile.load(path)
        profile.strip_params = checked["strip_params"]
        profile.save(path)
        state.unload(case)                        # loaded engines hold the old geometry
        return jsonable({"saved": True, "path": str(path.resolve()), "quality_status": profile.quality_status,
                         "profile": json.loads(profile.to_json()),
                         "note": "geometry changed: re-run escount prepare / train / select-rois for this project "
                                 "(its trained models are refused until then)"})

    # ------------------------------------------------------------ results page (Task 12, PDF p13-14)
    @app.get("/results/{case}/images")
    def results_images(case: str):
        """Project images; `complete` = every per-image output of `escount infer` exists in results_root."""
        from escount.es_counting import parse_ground_truth
        from escount.inference.engine import OUTPUT_FILES

        paths = list_images(state.case_dir(case))
        out_root = Path(config.results_root) / case
        images = []
        for path in paths:
            folder = out_root / path.stem
            complete = all((folder / name).is_file() for name in OUTPUT_FILES.values())
            stored = None
            if complete:
                try:
                    stored = json.loads((folder / OUTPUT_FILES["result_json"]).read_text(encoding="utf-8")).get("N_pred")
                except (OSError, ValueError):
                    complete = False
            gt = parse_ground_truth(path.name)
            images.append({"name": path.name, "gt": gt["count"] if gt else None, "complete": complete,
                           "stored_N_pred": stored})
        models = [name for name, ok in (
            ("project", (state.profiles_root / case / "model" / "manifest.json").is_file()),
            ("general", any((state.profiles_root / g / "model" / "manifest.json").is_file() for g in GENERAL_MODELS)),
        ) if ok]
        return {"case": case, "images": images, "models": models, "results_root": str(out_root.resolve())}

    @app.post("/preview")
    def preview(body: PreviewRequest):
        """InferenceEngine on a project image with optional MACD / KD parameter overrides (live N_pred)."""
        from escount.indicators.core import KDParams, MACDParams

        engine = state.cached_engine(body.case, body.model)
        if body.method and body.method != "auto":
            engine.runtime.forced_method = body.method
        else:
            engine.runtime.forced_method = None
        engine.runtime.roi_mode = body.roi_mode
        path = state.case_image(body.case, body.image)
        overrides = {}
        if body.macd:
            overrides["macd"] = MACDParams(**body.macd.model_dump())
        if body.kd:
            overrides["kd"] = KDParams(**body.kd.model_dump())
        result = state.run(engine, path, overrides=overrides or None, roi_mode=body.roi_mode)
        if body.method and body.method != "auto":
            result["method"] = body.method
        result["roi_mode"] = body.roi_mode
        return preview_payload(engine, result, overrides)

    # ---- training page: run the CLI steps (prepare -> train -> select-rois -> infer) as one background job
    import subprocess
    import sys

    job = {"case": None, "step": None, "status": "idle", "log": [], "returncode": None}
    job_lock = threading.Lock()

    def step_status(case):
        model_manifest = state.profiles_root / case / "model" / "manifest.json"
        selected = False
        all_rois = False
        if model_manifest.is_file():
            try:
                m_info = json.loads(model_manifest.read_text(encoding="utf-8"))
                selected = bool(m_info.get("selected_rois"))
                all_rois = bool(m_info.get("all_rois"))
            except (OSError, ValueError):
                selected = False
        results = Path(config.results_root) / case
        return {"calibrate": (state.profiles_root / case / PROFILE_NAME).is_file(),
                "prepare": (Path(config.features_root) / case / "features.parquet").is_file(),
                "train": model_manifest.is_file(),
                "select": selected,
                "all_rois": all_rois,
                "infer": (results / "predictions.csv").is_file()}

    def run_job(case, step, args):
        command = [sys.executable, "-m", "escount.cli", *args]
        job["log"].append("$ escount " + " ".join(args))
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            for raw in process.stdout:
                job["log"].append(raw.rstrip())
                del job["log"][:-400]                     # keep the last 400 lines
            job["returncode"] = process.wait()
        except OSError as error:
            job["log"].append(f"cannot start: {error}")
            job["returncode"] = -1
        if step in ("train", "select", "all_rois"):
            state.unload(case)                            # next preview / infer loads the new model
        job["status"] = "done" if job["returncode"] == 0 else "failed"

    @app.get("/training/{case}/status")
    def training_status(case: str):
        state.case_dir(case)
        return {"case": case, "steps": step_status(case), "job": job}

    class TrainRequest(BaseModel):
        step: Literal["prepare", "train", "select", "all_rois", "infer"]
        device: Literal["cuda", "cpu"] = "cuda"

    @app.post("/training/{case}/run")
    def training_run(case: str, body: TrainRequest):
        folder = state.case_dir(case)
        steps = step_status(case)
        order = ["calibrate", "prepare", "train", "select", "infer"]
        if body.step == "all_rois":
            missing = [s for s in ["calibrate", "prepare", "train"] if not steps[s]]
        else:
            missing = [s for s in order[:order.index(body.step)] if not steps[s]]
        if missing:
            raise HTTPException(409, f"請先完成前面的步驟：{', '.join(missing)}")
        # the server's own folders, so the buttons act on the project the page shows (not the CLI defaults)
        roots = ["--data-root", str(state.data_root), "--profiles-root", str(state.profiles_root),
                 "--cache-root", str(config.features_root)]
        args = {"prepare": ["prepare", "--project", case, *roots],
                "train": ["train", "--project", case, "--device", body.device, *roots],
                "select": ["select-rois", "--project", case, *roots],
                "all_rois": ["all-rois", "--project", case, *roots],
                "infer": ["infer", "--project", case, "--input", str(folder),
                          "--output", str(Path(config.results_root) / case),
                          "--profiles-root", str(state.profiles_root)]}[body.step]
        with job_lock:
            if job["status"] == "running":
                raise HTTPException(409, f"已有工作執行中：{job['case']} / {job['step']}")
            job.update(case=case, step=body.step, status="running", log=[], returncode=None)
        threading.Thread(target=run_job, args=(case, body.step, args), daemon=True).start()
        return {"started": body.step, "case": case}

    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
