"""Autoresearch daemon (Task 9). Rules and search space: escount/research/program.md.

One Optuna study per case (4 projects + `general` + `general_no_ring`) in one SQLite storage
(default sqlite:///research/study.db). Studies are opened with load_if_exists=True, so a restarted
run continues the trial numbering contiguously. Cases are visited round-robin, one trial at a time,
for `trials` new trials per case or until `hours` have elapsed.

Every trial: sample a configuration (ROI geometry w / D / x_offset, projection + sigma, indicator
families / directions, model type + hyper-parameters, ROI count cap) -> feature table for exactly
that geometry (escount.features.cache.prepare in research/features/<grid hash>/, multiprocess) ->
the FIXED evaluator escount.research.evaluate (LOIO; LOPO is recorded for the General cases) with
a fixed compute budget -> one row in research/<case>/results.tsv. status = keep iff rank_key
(-hit_rate, roi_count, latency_ms) beats the current candidate; on keep the complete, reusable
configuration is written to candidates/<case>/best_params.json (+ per_image.csv). Production
(outputs/projects/<case>/) is only touched by `escount adopt <case>`.
"""
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from escount.features.extract import (
    DIRECTIONS, FEATURE_INDICATORS, AugmentGrid, config_hash, feature_config, indicator_id,
)
from escount.models.train import (
    GENERAL_BASE, GENERAL_CASES, PROJECT_BASE, aggregate, fit_model, general_variants, load_general_config,
    subsample,
)
from escount.research.evaluate import evaluate, exact_hit, hits_by_case, rank_key, wilson_interval
from escount.signals.roi_bank import MIN_SIDE_MARGIN

DEFAULT_STORAGE = "sqlite:///research/study.db"
TSV_NAME = "results.tsv"
TSV_COLUMNS = ["trial", "case", "config_hash", "hit_rate", "hits/total", "roi_count", "latency_ms", "status",
               "timestamp"]
BEST_NAME = "best_params.json"
PER_IMAGE_NAME = "per_image.csv"
PER_IMAGE_COLUMNS = ["project", "image", "GT", "prediction", "hit", "scheme", "fold"]
ADOPTED_DIR = "research"
SEED = 0
MODEL_KINDS = ("baseline", "ridge", "svr", "xgb", "lgbm")
FAMILIES = ("kd", "macd", "peaks", "bollinger")
DIRECTION_SETS = {"both": ("both",), "up_down": ("up", "down"), "all": DIRECTIONS}
PROJECTIONS = ("mean", "gaussian", "median")
DEFAULT_SIGMA_RATIO = 0.25


@dataclass(frozen=True)
class TrialBudget:
    """Fixed compute budget of every trial (identical for all trials and cases)."""
    max_rows_per_image: int = 200     # training rows per image (deterministic subsample)
    n_estimators: int = 200           # boosting rounds of xgb / lgbm
    svr_max_rows: int = 3000          # SVR training rows


# ---------------------------------------------------------------- search space
def suggest(trial, budget=TrialBudget()):
    """Sample one configuration (static distributions so fANOVA can compare every trial)."""
    w = trial.suggest_int("w", 6, 40, step=2)
    d_sampled = trial.suggest_int("D", 6, 90, step=2)
    pitch = max(d_sampled, 2 * w // 3 + 1)                 # D > 2w/3  <=>  overlap w - D < w/3
    x_frac = trial.suggest_float("x_offset_frac", 0.0, 0.95, step=0.05)
    projection = trial.suggest_categorical("projection", list(PROJECTIONS))
    sigma_ratio = trial.suggest_float("sigma_ratio", 0.1, 0.6, step=0.05)
    families = [name for name in FAMILIES if trial.suggest_categorical(f"use_{name}", [True, False])]
    directions = trial.suggest_categorical("directions", list(DIRECTION_SETS))
    kind = trial.suggest_categorical("model", list(MODEL_KINDS))
    roi_cap = trial.suggest_int("roi_cap", 1, 8)
    aggregation = trial.suggest_categorical("aggregation", ["median", "trimmed"])
    params = {}
    if kind == "ridge":
        params["alpha"] = trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True)
    elif kind == "svr":
        params["C"] = trial.suggest_float("svr_C", 0.1, 100.0, log=True)
        params["epsilon"] = trial.suggest_float("svr_epsilon", 0.01, 1.0, log=True)
    elif kind == "xgb":
        params = {"max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
                  "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                  "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                  "n_estimators": budget.n_estimators}
    elif kind == "lgbm":
        params = {"num_leaves": trial.suggest_int("lgbm_num_leaves", 7, 63),
                  "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.3, log=True),
                  "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 50),
                  "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0),
                  "n_estimators": budget.n_estimators}
    return {"geometry": {"w": w, "D": pitch, "x_offset": min(int(x_frac * pitch), pitch - 1),
                         "projection": projection,
                         "sigma_ratio": sigma_ratio if projection == "gaussian" else None,
                         "side_margin": MIN_SIDE_MARGIN},
            "features": {"families": families, "directions": list(DIRECTION_SETS[directions])},
            "model": {"kind": kind, "params": params, "aggregation": aggregation, "roi_cap": roi_cap},
            "budget": asdict(budget)}


def grid_of(config):
    g = config["geometry"]
    return AugmentGrid(widths=(g["w"],), pitches=(g["D"],), x_offsets=(g["x_offset"],),
                       projections=(g["projection"],),
                       gaussian_sigma_ratio=g["sigma_ratio"] or DEFAULT_SIGMA_RATIO, side_margin=g["side_margin"])


def count_subset(features):
    """Count columns of the selected indicator families and directions."""
    return [f"cnt_{indicator_id(c)}_{d}" for c in FEATURE_INDICATORS if c.name in features["families"]
            for d in features["directions"]]


def design_matrix(rows, columns, scale_normalised):
    """Model inputs: project = absolute features, general = scale-normalised (no absolute px)."""
    if not scale_normalised:
        return rows[PROJECT_BASE + columns].to_numpy(dtype=np.float64)
    period_count = rows["period_count"].to_numpy(dtype=np.float64)
    scale = np.where(period_count > 0, period_count, 1.0)[:, None]
    base = rows[GENERAL_BASE[:-1]].to_numpy(dtype=np.float64)
    w_over_period = (rows["w"] / rows["period"]).to_numpy(dtype=np.float64)[:, None]
    return np.hstack([base, w_over_period, rows[columns].to_numpy(dtype=np.float64) / scale])


# ---------------------------------------------------------------- procedure
class ResearchProcedure:
    """Row predictor (one count column or a delta regressor) + ROI cap chosen on the training fold.

    fit(): learn the row predictor on the training rows (fixed budget), then rank ROI units
    (slot_key for projects, pos_key for General) by exact hits of round(median of the unit's rows)
    on the TRAINING images only (ties: higher mean SNR, then key) and keep the best `roi_cap`.
    predict(): image prediction = aggregate of the kept units' rows (all rows if none is present).
    """

    def __init__(self, config, scale_normalised=False, device="cuda", seed=SEED):
        self.config, self.scale_normalised, self.device, self.seed = config, scale_normalised, device, seed
        self.kind = config["model"]["kind"]
        self.columns = count_subset(config["features"])
        self.key_column = "pos_key" if scale_normalised else "slot_key"
        self.model, self.column, self.units, self.roi_count, self.device_used = None, None, [], None, None

    def fit(self, train_rows):
        budget = TrialBudget(**self.config["budget"])
        if self.kind == "baseline":
            candidates = self.columns + ["period_count"]
            medians = train_rows.groupby("image_id")[candidates].median()
            gt = train_rows.groupby("image_id")["gt"].first().reindex(medians.index).to_numpy()
            hits = (np.rint(medians.to_numpy()) == gt[:, None]).sum(axis=0)
            self.column = candidates[int(np.argmax(hits))]          # first column with the most hits
            self.device_used = "cpu"
        else:
            rows = subsample(train_rows, budget.max_rows_per_image, self.seed)
            if self.kind == "svr" and len(rows) > budget.svr_max_rows:
                rows = rows.sample(n=budget.svr_max_rows, random_state=self.seed).sort_index()
            target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
            self.model, self.device_used = fit_model(
                self.kind, design_matrix(rows, self.columns, self.scale_normalised), target, self.device,
                self.config["model"]["params"])
        self._select_units(train_rows)
        return self

    def _select_units(self, train_rows):
        frame = pd.DataFrame({"image_id": train_rows["image_id"].to_numpy(), "unit": train_rows[self.key_column].to_numpy(),
                              "gt": train_rows["gt"].to_numpy(), "snr": train_rows["snr"].to_numpy(),
                              "prediction": self.predict_rows(train_rows).to_numpy()})
        per_unit_image = frame.groupby(["unit", "image_id"]).agg(prediction=("prediction", "median"),
                                                                  gt=("gt", "first"), snr=("snr", "mean"))
        per_unit_image["hit"] = exact_hit(per_unit_image["prediction"], per_unit_image["gt"])
        score = per_unit_image.groupby(level="unit").agg(hits=("hit", "sum"), snr=("snr", "mean")).reset_index()
        score = score.sort_values(["hits", "snr", "unit"], ascending=[False, False, True])
        self.units = score["unit"].head(self.config["model"]["roi_cap"]).tolist()
        self.roi_count = float(len(self.units))

    def predict_rows(self, test_rows):
        if self.kind == "baseline":
            return test_rows[self.column].astype(np.float64)
        delta = self.model.predict(design_matrix(test_rows, self.columns, self.scale_normalised))
        return pd.Series(test_rows["period_count"].to_numpy(dtype=np.float64) + delta, index=test_rows.index)

    def predict(self, test_rows):
        rows = self.predict_rows(test_rows).to_numpy()
        chosen = test_rows[self.key_column].isin(self.units).to_numpy()
        how = self.config["model"]["aggregation"]
        out = {}
        for image_id, index in test_rows.groupby("image_id").indices.items():
            picked = index[chosen[index]]
            out[image_id] = aggregate(rows[picked if picked.size else index], how)
        return pd.Series(out).reindex(test_rows["image_id"].unique())

    def selected_rois(self, table):
        """Kept units with their geometry (after a final fit on all rows)."""
        rois = []
        for unit in self.units:
            row = table[table[self.key_column] == unit].iloc[0]
            item = {"key": unit, "projection": row["projection"], "w": int(row["w"]), "pitch": int(row["pitch"]),
                    "x_offset": int(row["x_offset"]), "sigma": None if pd.isna(row["sigma"]) else float(row["sigma"])}
            if self.key_column == "slot_key":
                item["x"] = int(row["x"])
            else:
                item["x_norm_bin"] = float(unit.rsplit("|", 1)[1])
            rois.append(item)
        return rois


# ---------------------------------------------------------------- evaluation helpers
def with_missing(result, expected):
    """Images without any ROI row count as misses (the denominator is every labelled image)."""
    have = set(result.per_image["image_id"])
    missing = [item for item in expected if item["image_id"] not in have]
    if not missing:
        return result
    extra = pd.DataFrame([{"image_id": m["image_id"], "case": m["case"], "image": m["image"], "gt": m["gt"],
                           "prediction": np.nan, "rounded": np.nan, "hit": False, "scheme": result.scheme,
                           "fold": "no-roi"} for m in missing])
    per_image = pd.concat([result.per_image, extra], ignore_index=True)
    hits, total = int(per_image["hit"].sum()), len(per_image)
    return replace(result, per_image=per_image, hits=hits, total=total, hit_rate=hits / total)


def summary(result):
    low, high = wilson_interval(result.hits, result.total)
    return {"scheme": result.scheme, "hits": result.hits, "total": result.total, "hit_rate": result.hit_rate,
            "wilson95": [low, high], "roi_count": result.roi_count, "latency_ms": result.latency_ms,
            "per_case": {case: list(v) for case, v in hits_by_case(result).items()}}


def keep_or_discard(result, best):
    """'keep' iff rank_key(result) is strictly better (smaller) than rank_key(best); best None -> keep."""
    if best is None:
        return "keep"
    return "keep" if rank_key(result) < rank_key(best) else "discard"


def per_image_table(results):
    """per_image.csv rows (project, image, GT, prediction, hit, scheme, fold) of LOIO (+ LOPO)."""
    parts = [r.per_image for r in results if r is not None]
    frame = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"project": frame["case"], "image": frame["image"], "GT": frame["gt"],
                         "prediction": frame["prediction"], "hit": frame["hit"].astype(bool),
                         "scheme": frame["scheme"], "fold": frame["fold"]})[PER_IMAGE_COLUMNS]


def read_tsv(path):
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=TSV_COLUMNS)
    return pd.read_csv(path, sep="\t", dtype={"hits/total": str, "config_hash": str})


def append_tsv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if new:
            writer.writerow(TSV_COLUMNS)
        writer.writerow([row[c] for c in TSV_COLUMNS])


def sqlite_path(storage):
    """File path of a sqlite:/// storage URL (None for other URLs)."""
    prefix = "sqlite:///"
    return Path(storage[len(prefix):]) if storage.startswith(prefix) else None


def open_study(case, storage=DEFAULT_STORAGE, seed=SEED):
    """Create or resume the study of `case`. The TPE seed depends on the trials already done, so a
    resumed study continues deterministically instead of replaying the first random trials."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = sqlite_path(storage)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(storage=storage, study_name=case, direction="maximize", load_if_exists=True)
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials), n_startup_trials=5)
    return study


def load_study(case, storage=DEFAULT_STORAGE):
    """Existing study or None (never creates a database)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = sqlite_path(storage)
    if path is not None and not path.is_file():
        return None
    try:
        return optuna.load_study(study_name=case, storage=storage)
    except KeyError:
        return None


# ---------------------------------------------------------------- daemon
class Researcher:
    """Round-robin autoresearch over cases. `table_fn(projects, grid) -> (table, expected images)`
    replaces the real feature computation (tests use synthetic tables)."""

    def __init__(self, projects, cases=None, storage=DEFAULT_STORAGE, research_root="research",
                 candidates_root="candidates", data_root="Data", profiles_root="outputs/projects",
                 config_path="configs/general.json", device="cuda", workers=None, budget=TrialBudget(), seed=SEED,
                 table_fn=None, log=print):
        self.projects = sorted(projects)
        variants = general_variants(self.projects, load_general_config(config_path)["exclude_from_general"])
        self.members = {**{case: [case] for case in self.projects}, **variants}
        self.cases = list(cases) if cases else self.projects + list(GENERAL_CASES)
        unknown = [case for case in self.cases if case not in self.members]
        if unknown:
            raise ValueError(f"unknown research cases {unknown}; expected {sorted(self.members)}")
        self.storage, self.research_root, self.candidates_root = storage, Path(research_root), Path(candidates_root)
        self.data_root, self.profiles_root, self.device, self.workers = data_root, profiles_root, device, workers
        self.budget, self.seed, self.log = budget, seed, log
        self.table_fn = table_fn or self.feature_table

    # -- features for exactly the trial geometry (cached per grid hash)
    def feature_table(self, projects, grid):
        from escount.features.cache import load_manifest, prepare

        cache_root = self.research_root / "features" / config_hash(grid.to_dict())[:16]
        tables, expected = [], []
        for case in projects:
            path, _ = prepare(case, self.data_root, self.profiles_root, cache_root, self.workers, grid)
            tables.append(pd.read_parquet(path))
            expected += [{"image_id": f"{case}/{item['name']}", "case": case, "image": item["name"], "gt": item["gt"]}
                         for item in load_manifest(case, cache_root)["images"]]
        return pd.concat(tables, ignore_index=True), expected

    def best(self, case):
        path = self.candidates_root / case / BEST_NAME
        if not path.is_file():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))["result"]
        return SimpleNamespace(hit_rate=result["hit_rate"], roi_count=result["roi_count"],
                               latency_ms=result["latency_ms"])

    def evaluate_config(self, case, config):
        general = case in GENERAL_CASES
        table, expected = self.table_fn(self.members[case], grid_of(config))
        if table.empty:
            raise ValueError(f"{case}: no ROI rows for geometry {config['geometry']}")
        procedure = ResearchProcedure(config, general, self.device, self.seed)
        loio = with_missing(evaluate(procedure, table, "LOIO"), expected)
        lopo = None
        if general and table["case"].nunique() >= 2:
            lopo = with_missing(evaluate(ResearchProcedure(config, general, self.device, self.seed), table, "LOPO"),
                                expected)
        return table, loio, lopo, procedure.device_used

    def objective(self, case, trial):
        config = suggest(trial, self.budget)
        digest = config_hash(config)[:12]
        row = {"trial": trial.number, "case": case, "config_hash": digest, "hit_rate": "nan", "hits/total": "",
               "roi_count": "", "latency_ms": "", "status": "discard", "timestamp": ""}
        trial.set_user_attr("config", config)
        trial.set_user_attr("config_hash", digest)
        try:
            table, loio, lopo, device = self.evaluate_config(case, config)
        except Exception as error:
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            append_tsv(self.research_root / case / TSV_NAME, row)
            self.log(f"[{case}] trial {trial.number} failed: {error}")
            raise
        status = keep_or_discard(loio, self.best(case))
        row.update({"hit_rate": f"{loio.hit_rate:.4f}", "hits/total": f"{loio.hits}/{loio.total}",
                    "roi_count": f"{loio.roi_count:g}", "latency_ms": f"{loio.latency_ms:.3f}", "status": status,
                    "timestamp": datetime.now().isoformat(timespec="seconds")})
        for key, value in {"hits": loio.hits, "total": loio.total, "roi_count": loio.roi_count,
                           "latency_ms": loio.latency_ms, "status": status, "device": device}.items():
            trial.set_user_attr(key, value)
        if lopo is not None:
            trial.set_user_attr("lopo_hits", lopo.hits)
            trial.set_user_attr("lopo_total", lopo.total)
        if status == "keep":
            self.write_candidate(case, trial.number, config, digest, table, loio, lopo, device)
        append_tsv(self.research_root / case / TSV_NAME, row)
        lopo_text = f" LOPO {lopo.hits}/{lopo.total}" if lopo is not None else ""
        self.log(f"[{case}] trial {trial.number}: LOIO {loio.hits}/{loio.total}{lopo_text} roi={loio.roi_count:g} "
                 f"latency={loio.latency_ms:.2f} ms model={config['model']['kind']} w={config['geometry']['w']} "
                 f"D={config['geometry']['D']} -> {status}")
        return loio.hit_rate

    def write_candidate(self, case, number, config, digest, table, loio, lopo, device):
        general = case in GENERAL_CASES
        final = ResearchProcedure(config, general, self.device, self.seed).fit(table)
        grid = grid_of(config)
        features = feature_config(grid)
        best = {"case": case, "trial": number, "study": case, "storage": self.storage, "config_hash": digest,
                "created": datetime.now().isoformat(timespec="seconds"),
                "metric": "exact hit rate round(prediction) == GT (LOIO); rank key (-hit_rate, roi_count, latency_ms)",
                "result": summary(loio), "lopo": summary(lopo) if lopo is not None else None,
                "projects": self.members[case],
                "extraction": {"grid": grid.to_dict(), "feature_config": features,
                               "feature_config_hash": config_hash(features), "vertical_smoothing": False,
                               "rule": "every strip row >= 50 px from the valid-region edge; D > 2w/3"},
                "geometry": config["geometry"],
                "features": {**config["features"], "columns": count_subset(config["features"]),
                             "base": GENERAL_BASE if general else PROJECT_BASE, "scale_normalised": general,
                             "target": "delta = GT - period_count" if config["model"]["kind"] != "baseline"
                             else "count column"},
                "model": {**config["model"], "device": device, "baseline_column": final.column,
                          "max_rows_per_image": config["budget"]["max_rows_per_image"]},
                "budget": config["budget"], "selected_rois": final.selected_rois(table),
                "rank_key": list(rank_key(loio))}
        out = self.candidates_root / case
        out.mkdir(parents=True, exist_ok=True)
        (out / BEST_NAME).write_text(json.dumps(best, ensure_ascii=False, indent=2, default=_json_default),
                                     encoding="utf-8")
        per_image_table([loio, lopo]).to_csv(out / PER_IMAGE_NAME, index=False, encoding="utf-8")

    def run_one(self, case):
        study = open_study(case, self.storage, self.seed)
        study.optimize(lambda trial: self.objective(case, trial), n_trials=1, catch=(Exception,))

    def run(self, trials=None, hours=None):
        """`trials` new trials per case, or round-robin until `hours` elapsed. Returns trials run."""
        if (trials is None) == (hours is None):
            raise ValueError("give exactly one of trials / hours")
        deadline = time.monotonic() + hours * 3600.0 if hours is not None else math.inf
        done, round_index = 0, 0
        while (trials is None or round_index < trials) and time.monotonic() < deadline:
            for case in self.cases:
                if time.monotonic() >= deadline:
                    break
                self.run_one(case)
                done += 1
            round_index += 1
        return done


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


# ---------------------------------------------------------------- adopt
def adopt(case, candidates_root="candidates", profiles_root="outputs/projects"):
    """Copy candidates/<case>/ (best_params.json, per_image.csv) into outputs/projects/<case>/research/."""
    source = Path(candidates_root) / case
    if not (source / BEST_NAME).is_file():
        raise FileNotFoundError(f"{source / BEST_NAME} missing; run `escount research` first")
    target = Path(profiles_root) / case / ADOPTED_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name in (BEST_NAME, PER_IMAGE_NAME):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    return target


def adopted_hash(case, profiles_root="outputs/projects"):
    """config_hash of the adopted research candidate of a case (None if never adopted)."""
    path = Path(profiles_root) / case / ADOPTED_DIR / BEST_NAME
    return json.loads(path.read_text(encoding="utf-8")).get("config_hash") if path.is_file() else None
