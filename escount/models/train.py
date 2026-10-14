"""Project & General training (Task 6). Accuracy = exact hit rate only (escount.research.evaluate).

Target per ROI row: delta = GT - period_count, period_count = round(L / FFT period). Row prediction =
period_count + model(features); image prediction = median (or trimmed mean) of its row predictions,
rounded. Regressors: Ridge and SVR (StandardScaler pipelines), XGBoost (device='cuda', automatic
CPU fallback, device used is recorded) and LightGBM (CPU).

Baseline = best single indicator count (one count column on one ROI layout group w/D/x/projection)
chosen on the training fold by exact hits (ties: fewer ROIs, then lower cost). The ML model is
adopted only if its Grouped-LOO (LOIO) exact hit rate is strictly higher than the baseline's.

General: scale-normalised features (counts / period_count, w / period, no absolute px), trained on
all active projects; evaluated LOIO and LOPO. configs/general.json:
  excluded_projects     projects left out of prepare --all / train / research / report entirely
                        (the user excluded Ring-type: unreliable, LOIO 3/6); Data/ is never touched.
  exclude_from_general  projects left out of the General model only.
While Ring-type is active, `general` (incl. Ring-type) and `general_no_ring` are both trained for the
Ring inclusion comparison and `exclude_from_general` selects the production General model; with
Ring-type excluded only `general` exists.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from escount.features.extract import count_columns
from escount.research.evaluate import evaluate, hits_by_case, rank_key, wilson_interval

MODEL_TYPES = ("ridge", "svr", "xgb", "lgbm")
MODEL_PARAMS = {
    "ridge": {"alpha": 1.0},
    "svr": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"},
    "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": 0, "tree_method": "hist"},
    "lgbm": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.8, "subsample_freq": 1,
             "colsample_bytree": 0.8, "random_state": 0, "deterministic": True, "force_row_wise": True,
             "verbose": -1},
}
SVR_MAX_ROWS = 5000
MAX_ROWS_PER_IMAGE = 1000
GROUP = ["w", "pitch", "x_offset", "projection"]
PROJECT_BASE = ["w", "x_norm", "L", "proj_id", "period", "period_count", "snr", "std", "period_stability"]
GENERAL_BASE = ["x_norm", "proj_id", "period_count", "snr", "std", "period_stability", "w_over_period"]
RING_CASE = "Ring-type"
GENERAL_CASES = ("general", "general_no_ring")


# ---------------------------------------------------------------- features / helpers
def feature_names(scale_normalised=False):
    counts = count_columns()
    if scale_normalised:
        return GENERAL_BASE + [f"{c}_ratio" for c in counts]
    return PROJECT_BASE + counts


def feature_frame(rows, scale_normalised=False):
    """Model input matrix (float64 numpy) for feature rows."""
    if not scale_normalised:
        return rows[PROJECT_BASE + count_columns()].to_numpy(dtype=np.float64)
    period_count = rows["period_count"].to_numpy(dtype=np.float64)
    scale = np.where(period_count > 0, period_count, 1.0)[:, None]
    base = rows[GENERAL_BASE[:-1]].to_numpy(dtype=np.float64)
    w_over_period = (rows["w"] / rows["period"]).to_numpy(dtype=np.float64)[:, None]
    ratios = rows[count_columns()].to_numpy(dtype=np.float64) / scale
    return np.hstack([base, w_over_period, ratios])


def subsample(rows, max_rows_per_image, seed=0):
    """At most max_rows_per_image rows per image (deterministic)."""
    if not max_rows_per_image:
        return rows
    shuffled = rows.sample(frac=1.0, random_state=seed)
    return shuffled[shuffled.groupby("image_id").cumcount() < max_rows_per_image].sort_index()


def aggregate(values, how="median"):
    if how == "median":
        return float(np.median(values))
    if how == "trimmed":
        from scipy.stats import trim_mean

        return float(trim_mean(values, 0.1))
    raise ValueError(f"unknown aggregation {how!r}; expected 'median' or 'trimmed'")


def _xgb_device(model):
    config = json.loads(model.get_booster().save_config())
    return config["learner"]["generic_param"].get("device", "cpu")


def make_model(kind, device="cpu", params=None):
    params = {**MODEL_PARAMS[kind], **(params or {})}
    if kind in ("ridge", "svr"):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        return make_pipeline(StandardScaler(), Ridge(**params) if kind == "ridge" else SVR(**params))
    if kind == "xgb":
        import xgboost

        return xgboost.XGBRegressor(device=device, **params)
    if kind == "lgbm":
        import lightgbm

        return lightgbm.LGBMRegressor(**params)
    raise ValueError(f"unknown model type {kind!r}; expected one of {MODEL_TYPES}")


def fit_model(kind, x, y, device="cuda", params=None):
    """Fit one regressor. XGBoost tries CUDA first and falls back to CPU. Returns (model, device used)."""
    if kind != "xgb":
        return make_model(kind, params=params).fit(x, y), "cpu"
    import xgboost

    for attempt in (("cuda", "cpu") if device == "cuda" else ("cpu",)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = make_model("xgb", attempt, params).fit(x, y)
            used = _xgb_device(model)
            model.set_params(device="cpu")   # predict on host arrays without device-mismatch copies
            return model, "cuda" if used.startswith("cuda") else "cpu"
        except xgboost.core.XGBoostError:
            if attempt == "cpu":
                raise
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- procedures
class BaselineProcedure:
    """Best single indicator count column on one ROI layout group, chosen by training exact hits."""

    def __init__(self):
        self.columns = count_columns() + ["period_count"]
        self.choice, self.roi_count, self._memo = None, None, {}

    def _image_table(self, rows):
        parts = []
        for image_id, image_rows in rows.groupby("image_id", sort=False):
            if image_id not in self._memo:
                self._memo[image_id] = image_rows.groupby(GROUP)[self.columns].median().assign(
                    rois=image_rows.groupby(GROUP).size())
            parts.append(self._memo[image_id].assign(image_id=image_id))
        return pd.concat(parts)

    def fit(self, train_rows):
        table = self._image_table(train_rows)
        gt = table["image_id"].map(train_rows.groupby("image_id")["gt"].first()).to_numpy()
        hits = pd.DataFrame(np.rint(table[self.columns].to_numpy()) == gt[:, None],
                            index=table.index, columns=self.columns).groupby(level=list(range(len(GROUP)))).sum()
        rois = table["rois"].groupby(level=list(range(len(GROUP)))).median()
        long = hits.stack().rename("hits").reset_index().rename(columns={f"level_{len(GROUP)}": "column"})
        long["rois"] = rois.reindex(pd.MultiIndex.from_frame(long[GROUP])).to_numpy()
        long["cost"] = long["rois"] * long["w"]
        best = long.sort_values(["hits", "rois", "cost", *GROUP, "column"],
                                ascending=[False, True, True, True, True, True, True, True]).iloc[0]
        self.choice = {**{key: best[key] for key in GROUP}, "column": best["column"],
                       "train_hits": int(best["hits"])}
        self.choice = {k: (v.item() if hasattr(v, "item") else v) for k, v in self.choice.items()}
        self.roi_count = float(best["rois"])

    def _in_group(self, rows):
        mask = np.ones(len(rows), dtype=bool)
        for key in GROUP:
            mask &= (rows[key] == self.choice[key]).to_numpy()
        return rows[mask]

    def predict_rows(self, test_rows):
        return test_rows[self.choice["column"]].astype(np.float64)

    def deployable_rows(self, rows):
        """Rows of the chosen layout group: the full ROI set `predict` uses (one bank, overlap < w/3)."""
        return self._in_group(rows)

    def predict(self, test_rows):
        chosen = self._in_group(test_rows)
        result = chosen.groupby("image_id")[self.choice["column"]].median()
        return result.reindex(test_rows["image_id"].unique())


class ModelProcedure:
    """delta regression on all ROI rows of the training images; image prediction = median of rows."""

    roi_count = None

    def __init__(self, kind, scale_normalised=False, max_rows_per_image=MAX_ROWS_PER_IMAGE,
                 aggregation="median", device="cuda", params=None, seed=0):
        self.kind, self.scale_normalised, self.max_rows_per_image = kind, scale_normalised, max_rows_per_image
        self.aggregation, self.device, self.params, self.seed = aggregation, device, params, seed
        self.model, self.device_used, self.trained_rows = None, None, 0

    def fit(self, train_rows):
        rows = subsample(train_rows, self.max_rows_per_image, self.seed)
        if self.kind == "svr" and len(rows) > SVR_MAX_ROWS:
            rows = rows.sample(n=SVR_MAX_ROWS, random_state=self.seed).sort_index()
        target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
        self.model, self.device_used = fit_model(self.kind, feature_frame(rows, self.scale_normalised), target,
                                                 self.device, self.params)
        self.trained_rows = len(rows)
        return self

    def predict_rows(self, test_rows):
        delta = self.model.predict(feature_frame(test_rows, self.scale_normalised))
        return pd.Series(test_rows["period_count"].to_numpy(dtype=np.float64) + delta, index=test_rows.index)

    def predict(self, test_rows):
        rows = self.predict_rows(test_rows)
        return rows.groupby(test_rows["image_id"]).agg(lambda v: aggregate(v.to_numpy(), self.aggregation))


def make_procedure(spec, device="cuda"):
    """Procedure from a manifest-style spec {'method': 'baseline'|'ml', 'model_type', ...}."""
    if spec["method"] == "baseline":
        return BaselineProcedure()
    return ModelProcedure(spec["model_type"], spec["scale_normalised"], spec["max_rows_per_image"],
                          spec["aggregation"], device, spec.get("params"))


# ---------------------------------------------------------------- training a case
def _summary(result):
    low, high = wilson_interval(result.hits, result.total)
    return {"hits": result.hits, "total": result.total, "hit_rate": result.hit_rate, "wilson95": [low, high],
            "roi_count": result.roi_count, "latency_ms": result.latency_ms,
            "per_case": {case: list(value) for case, value in hits_by_case(result).items()}}


def train_case(case, table, out_dir, device="cuda", scale_normalised=False, schemes=("LOIO",),
               model_types=MODEL_TYPES, max_rows_per_image=MAX_ROWS_PER_IMAGE, aggregation="median",
               feature_config=None, profile_fingerprints=None, log=print):
    """Evaluate baseline + every model type (LOIO, optionally LOPO), fit the final model, write model/.

    profile_fingerprints = {project: profile_sha256 of its feature cache}; inference refuses to load
    the model once a project's calibration (project.json / rotated mask) no longer matches.
    """
    from escount.models.store import save_model

    results, devices, per_image = {}, {}, []
    results["baseline"] = evaluate(BaselineProcedure(), table, "LOIO")
    log(f"[{case}] LOIO baseline {results['baseline'].hits}/{results['baseline'].total}")
    for kind in model_types:
        procedure = ModelProcedure(kind, scale_normalised, max_rows_per_image, aggregation, device)
        results[kind] = evaluate(procedure, table, "LOIO")
        devices[kind] = procedure.device_used
        log(f"[{case}] LOIO {kind} {results[kind].hits}/{results[kind].total} (device {procedure.device_used})")
    best_kind = min(model_types, key=lambda k: (rank_key(results[k]), model_types.index(k)))
    adopted = results[best_kind].hit_rate > results["baseline"].hit_rate
    evaluation = {"LOIO": {name: _summary(result) for name, result in results.items()}}
    for name, result in results.items():
        per_image.append(result.per_image.assign(method=name))
    if "LOPO" in schemes:
        lopo = {"baseline": evaluate(BaselineProcedure(), table, "LOPO"),
                best_kind: evaluate(ModelProcedure(best_kind, scale_normalised, max_rows_per_image, aggregation,
                                                   device), table, "LOPO")}
        evaluation["LOPO"] = {name: _summary(result) for name, result in lopo.items()}
        for name, result in lopo.items():
            per_image.append(result.per_image.assign(method=name))
            log(f"[{case}] LOPO {name} {result.hits}/{result.total} {hits_by_case(result)}")
    final = ModelProcedure(best_kind, scale_normalised, max_rows_per_image, aggregation, device).fit(table)
    baseline = BaselineProcedure()
    baseline.fit(table)
    method = "ml" if adopted else "baseline"
    out_dir = Path(out_dir)
    manifest = save_model(
        final.model, best_kind, out_dir, case=case, method=method, adopted_ml=adopted,
        adoption_rule="ML adopted only if LOIO exact hit rate > baseline",
        features=feature_names(scale_normalised), scale_normalised=scale_normalised, target="delta = GT - period_count",
        aggregation=aggregation, max_rows_per_image=max_rows_per_image, params=MODEL_PARAMS[best_kind],
        device=final.device_used, devices_loio=devices, trained_rows=final.trained_rows,
        cases=sorted(table["case"].unique()), images=int(table["image_id"].nunique()),
        baseline=baseline.choice, baseline_roi_count=baseline.roi_count, evaluation=evaluation,
        feature_config=feature_config, profile_fingerprints=profile_fingerprints or {})
    pd.concat(per_image, ignore_index=True).to_csv(out_dir / "evaluation_per_image.csv", index=False,
                                                    encoding="utf-8")
    log(f"[{case}] best ML {best_kind} -> method={method} (device {final.device_used}); wrote {out_dir.resolve()}")
    return manifest


def train_project(case, cache_root="outputs/features", profiles_root="outputs/projects", device="cuda",
                  max_rows_per_image=MAX_ROWS_PER_IMAGE, log=print):
    from escount.features.cache import load_features, load_manifest

    table = load_features(case, cache_root)
    cache = load_manifest(case, cache_root)
    return train_case(case, table, Path(profiles_root) / case / "model", device=device,
                      max_rows_per_image=max_rows_per_image, feature_config=cache["feature_config"],
                      profile_fingerprints={case: cache["profile_sha256"]}, log=log)


# ---------------------------------------------------------------- General
def load_general_config(path="configs/general.json"):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config.setdefault("excluded_projects", [])
    config.setdefault("exclude_from_general", [])
    return config


def active_projects(cases, config):
    """`cases` without the configured excluded_projects (order kept)."""
    excluded = set(config["excluded_projects"])
    return [case for case in cases if case not in excluded]


def check_active(case, config):
    """Raise ValueError if `case` is an excluded project."""
    if case in config["excluded_projects"]:
        raise ValueError(f"{case} is listed in excluded_projects of the General config; remove it there to "
                         f"re-enable it")


def general_cases(config, projects=None):
    """General case ids in use: 'general', plus 'general_no_ring' while Ring-type is active."""
    ring = RING_CASE not in config["excluded_projects"] and (projects is None or RING_CASE in projects)
    return list(GENERAL_CASES) if ring else ["general"]


def general_table(tables, exclude=()):
    """Concatenate project feature tables, dropping every case in `exclude`."""
    kept = [table for case, table in sorted(tables.items()) if case not in set(exclude)]
    if not kept:
        raise ValueError("no project left for the General model")
    return pd.concat(kept, ignore_index=True)


def general_variants(cases, exclude=(), excluded=()):
    """{'general': cases minus `excluded` projects and non-Ring `exclude` entries} plus, while Ring-type is
    still among them, {'general_no_ring': the same without Ring-type} for the Ring inclusion comparison."""
    kept = sorted(set(cases) - set(excluded) - (set(exclude) - {RING_CASE}))
    variants = {"general": kept}
    if RING_CASE in kept:
        variants["general_no_ring"] = [case for case in kept if case != RING_CASE]
    return variants


def production_general(config):
    ring_active = RING_CASE not in config["excluded_projects"]
    return "general_no_ring" if ring_active and RING_CASE in config["exclude_from_general"] else "general"


def ring_comparison(manifests):
    """Hits of the non-Ring projects under `general` vs `general_no_ring` (adopted method)."""
    def other_hits(manifest, scheme):
        method = "baseline" if manifest["method"] == "baseline" else manifest["model_type"]
        per_case = manifest["evaluation"][scheme][method]["per_case"]
        return {case: value for case, value in per_case.items() if case != RING_CASE}

    rows = {}
    for scheme in ("LOIO", "LOPO"):
        with_ring, without = other_hits(manifests["general"], scheme), other_hits(manifests["general_no_ring"], scheme)
        rows[scheme] = {"with_ring": with_ring, "without_ring": without,
                        "with_ring_hits": sum(v[0] for v in with_ring.values()),
                        "without_ring_hits": sum(v[0] for v in without.values())}
    hurts = sum(rows[s]["with_ring_hits"] for s in rows) < sum(rows[s]["without_ring_hits"] for s in rows)
    return {"schemes": rows, "ring_hurts_others": hurts,
            "recommended_exclude_from_general": [RING_CASE] if hurts else [],
            "rule": "exclude Ring-type only if including it lowers the other projects' LOIO + LOPO exact hits"}


def train_general(cache_root="outputs/features", profiles_root="outputs/projects", config_path="configs/general.json",
                  device="cuda", max_rows_per_image=MAX_ROWS_PER_IMAGE, cases=None, log=print):
    from escount.es_counting import write_json
    from escount.features.cache import load_features, load_manifest

    config = load_general_config(config_path)
    cases = cases or sorted(p.name for p in Path(cache_root).iterdir() if (p / "features.parquet").is_file())
    cases = active_projects(cases, config)
    tables = {case: load_features(case, cache_root) for case in cases}
    manifests = {}
    for name, kept in general_variants(cases, config["exclude_from_general"]).items():
        table = general_table(tables, set(cases) - set(kept))
        manifests[name] = train_case(name, table, Path(profiles_root) / name / "model", device=device,
                                     scale_normalised=True, schemes=("LOIO", "LOPO"),
                                     max_rows_per_image=max_rows_per_image,
                                     feature_config=load_manifest(kept[0], cache_root)["feature_config"],
                                     profile_fingerprints={case: load_manifest(case, cache_root)["profile_sha256"]
                                                           for case in kept}, log=log)
    if "general_no_ring" in manifests:
        comparison = ring_comparison(manifests)
        log(f"[general] ring_hurts_others={comparison['ring_hurts_others']} "
            f"recommended exclude_from_general={comparison['recommended_exclude_from_general']}")
    else:
        comparison = {"ring_excluded": True, "excluded_projects": config["excluded_projects"],
                      "excluded_reason": config.get("excluded_reason", {}),
                      "note": "Ring-type is not part of the General data; no with/without Ring comparison"}
        log(f"[general] trained without {config['excluded_projects']} (excluded_projects); no Ring comparison")
    comparison.update(config_exclude_from_general=config["exclude_from_general"],
                      production_general=production_general(config), general_cases=list(manifests))
    write_json(Path(profiles_root) / "general" / "ring_comparison.json", comparison)
    return manifests, comparison
