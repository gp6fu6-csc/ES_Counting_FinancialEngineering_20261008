"""Model store (Task 6): <profiles_root>/<case>/model/ with native/joblib model file + manifest.json.

XGBoost -> model_xgb.json (native JSON), LightGBM -> model_lgbm.txt (native text), Ridge / SVR
pipelines -> model_<type>.joblib. manifest.json records the SHA-256 of every model file, the feature
list, parameters and the training device; load_model verifies every SHA (mismatch -> error).
"""
from datetime import datetime
from pathlib import Path

from escount.es_counting import file_sha256, read_json, write_json

MANIFEST_NAME = "manifest.json"
MODEL_FILES = {"xgb": "model_xgb.json", "lgbm": "model_lgbm.txt",
               "ridge": "model_ridge.joblib", "svr": "model_svr.joblib"}
SCHEMA_VERSION = 1


class ModelIntegrityError(ValueError):
    """A model file is missing or its SHA-256 differs from manifest.json."""


class BoosterRegressor:
    """LightGBM Booster loaded from native text with the LGBMRegressor.predict interface."""

    def __init__(self, booster, num_threads=0):
        self.booster, self.num_threads = booster, num_threads

    def predict(self, x):
        return self.booster.predict(x, num_threads=self.num_threads)


def save_model(model, kind, model_dir, **fields):
    """Write the model file and manifest.json (SHA-256 per file). Returns the manifest dict."""
    if kind not in MODEL_FILES:
        raise ValueError(f"unknown model type {kind!r}; expected one of {sorted(MODEL_FILES)}")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILES[kind]
    if kind == "xgb":
        model.save_model(str(path))
    elif kind == "lgbm":
        model.booster_.save_model(str(path))
    else:
        import joblib

        joblib.dump(model, path)
    manifest = {"schema_version": SCHEMA_VERSION, "model_type": kind,
                "files": {path.name: file_sha256(path)}, **fields,
                "created": datetime.now().isoformat(timespec="seconds")}
    write_json(model_dir / MANIFEST_NAME, manifest)
    return manifest


def read_manifest(model_dir):
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `escount train` first")
    return read_json(path)


def update_manifest(model_dir, **fields):
    manifest = read_manifest(model_dir)
    manifest.update(fields)
    write_json(Path(model_dir) / MANIFEST_NAME, manifest)
    return manifest


def load_model(model_dir, single_thread=False):
    """Verify SHA-256 of every model file, then load. Returns (predictor with .predict(X), manifest)."""
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    for name, digest in manifest["files"].items():
        path = model_dir / name
        if not path.is_file():
            raise ModelIntegrityError(f"model file {path} is missing")
        if file_sha256(path) != digest:
            raise ModelIntegrityError(f"SHA-256 mismatch for {path}: manifest {digest}, file {file_sha256(path)}")
    kind = manifest["model_type"]
    path = model_dir / MODEL_FILES[kind]
    if kind == "xgb":
        import xgboost

        model = xgboost.XGBRegressor()
        try:
            model.load_model(str(path))
        except xgboost.core.XGBoostError:
            model.load_model(bytearray(path.read_bytes()))
        model.set_params(device="cpu", n_jobs=1 if single_thread else None)
    elif kind == "lgbm":
        import lightgbm

        model = BoosterRegressor(lightgbm.Booster(model_file=str(path)), num_threads=1 if single_thread else 0)
    else:
        import joblib

        model = joblib.load(path)
    return model, manifest
