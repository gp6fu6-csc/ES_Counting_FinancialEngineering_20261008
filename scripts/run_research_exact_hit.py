"""Track 2: AutoResearch Engine for Industrial Absolute Exact Hit Maximization.

Directly optimizes for exact sheet count match (round(pred) == GT) using:
1. Soft Exact-Hit Loss (Sigmoid threshold gradient around |e|=0.5)
2. Discrete Residual Classifier (multi-class integer classification)
3. Standard GBDT and Baseline models with hierarchical (-exact_hits, mae) ranking.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from escount.features.extract import AugmentGrid, config_hash, feature_config
from escount.models.classifier import DiscreteResidualClassifier
from escount.models.loss import SoftExactHitObjective
from escount.models.train import (
    GENERAL_CASES, active_projects, aggregate, fit_model, general_variants, load_general_config, subsample,
)
from escount.research.daemon import (
    BEST_NAME, PER_IMAGE_COLUMNS, PER_IMAGE_NAME, TSV_COLUMNS, TSV_NAME, TrialBudget,
    append_tsv, count_subset, design_matrix, grid_of, open_study, per_image_table, read_tsv, suggest,
    summary, with_missing,
)
from escount.research.evaluate import evaluate, rank_key_exact

TRACK_NAME = "exact_hit"
DEFAULT_STORAGE = "sqlite:///research/study_exact_hit.db"
RESEARCH_ROOT = Path("research") / TRACK_NAME
CANDIDATES_ROOT = Path("candidates") / TRACK_NAME

EXACT_MODELS = ("baseline", "ridge", "svr", "xgb", "lgbm", "soft_loss_xgb", "soft_loss_lgbm", "discrete_xgb", "discrete_lgbm")


def suggest_exact(trial, budget=TrialBudget()):
    base_config = suggest(trial, budget)
    # Expand model search space with new exact-hit architectures
    kind = trial.suggest_categorical("exact_model_kind", list(EXACT_MODELS))
    base_config["model"]["kind"] = kind
    if "soft_loss" in kind:
        base_config["model"]["params"]["sharpness"] = trial.suggest_float("soft_sharpness", 4.0, 16.0, step=2.0)
        base_config["model"]["params"]["l2_reg"] = trial.suggest_float("soft_l2_reg", 0.01, 0.2, log=True)
    elif "discrete" in kind:
        base_config["model"]["params"]["max_offset"] = trial.suggest_int("discrete_max_offset", 2, 4)
    return base_config


class ResearchProcedureExact:
    """Procedure dedicated to maximizing absolute exact hits."""

    def __init__(self, config, scale_normalised=False, device="cuda", seed=0):
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
            self.column = candidates[int(np.argmax(hits))]
            self.device_used = "cpu"
        elif self.kind in ("discrete_xgb", "discrete_lgbm"):
            rows = subsample(train_rows, budget.max_rows_per_image, self.seed)
            target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
            X = design_matrix(rows, self.columns, self.scale_normalised)
            backend = "xgb" if "xgb" in self.kind else "lgbm"
            max_off = self.config["model"]["params"].get("max_offset", 3)
            self.model = DiscreteResidualClassifier(kind=backend, max_offset=max_off, device=self.device)
            self.model.fit(X, target)
            self.device_used = self.device
        elif self.kind in ("soft_loss_xgb", "soft_loss_lgbm"):
            rows = subsample(train_rows, budget.max_rows_per_image, self.seed)
            target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
            X = design_matrix(rows, self.columns, self.scale_normalised)
            sharpness = self.config["model"]["params"].get("sharpness", 8.0)
            l2_reg = self.config["model"]["params"].get("l2_reg", 0.05)
            obj = SoftExactHitObjective(sharpness=sharpness, l2_reg=l2_reg)

            if "xgb" in self.kind:
                import xgboost
                p = {**self.config["model"]["params"]}
                p.pop("sharpness", None)
                p.pop("l2_reg", None)
                p.setdefault("n_estimators", 200)
                p.setdefault("max_depth", 5)
                p.setdefault("learning_rate", 0.05)
                p.setdefault("random_state", 0)
                p.setdefault("tree_method", "hist")
                p.setdefault("device", self.device)
                xgb_reg = xgboost.XGBRegressor(objective=obj.xgb_objective, **p)
                self.model = xgb_reg.fit(X, target)
                self.device_used = self.device
            else:
                import lightgbm
                p = {**self.config["model"]["params"]}
                p.pop("sharpness", None)
                p.pop("l2_reg", None)
                p.setdefault("n_estimators", 200)
                p.setdefault("num_leaves", 31)
                p.setdefault("learning_rate", 0.05)
                p.setdefault("random_state", 0)
                p.setdefault("verbose", -1)
                lgb_reg = lightgbm.LGBMRegressor(objective=obj.lgb_objective, **p)
                self.model = lgb_reg.fit(X, target)
                self.device_used = "cpu"
        else:
            rows = subsample(train_rows, budget.max_rows_per_image, self.seed)
            target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
            from escount.models.train import MODEL_PARAMS
            valid_keys = set(MODEL_PARAMS.get(self.kind, {}).keys())
            filtered_params = {k: v for k, v in self.config["model"].get("params", {}).items() if k in valid_keys}
            self.model, self.device_used = fit_model(
                self.kind, design_matrix(rows, self.columns, self.scale_normalised), target, self.device,
                filtered_params)
        self._select_units(train_rows)
        return self

    def _select_units(self, train_rows):
        frame = pd.DataFrame({"image_id": train_rows["image_id"].to_numpy(), "unit": train_rows[self.key_column].to_numpy(),
                              "gt": train_rows["gt"].to_numpy(), "snr": train_rows["snr"].to_numpy(),
                              "prediction": self.predict_rows(train_rows).to_numpy()})
        per_unit_image = frame.groupby(["unit", "image_id"]).agg(prediction=("prediction", "median"),
                                                                  gt=("gt", "first"), snr=("snr", "mean"))
        per_unit_image["hit"] = np.rint(per_unit_image["prediction"]) == per_unit_image["gt"]
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


def run_exact_research(cases=None, trials=10, hours=None, storage=DEFAULT_STORAGE, device="cuda",
                       data_root="Data", profiles_root="outputs/projects", output_root=None):
    config = load_general_config()
    if output_root:         # isolated workspace (escount batch): shadows the module-level roots below
        RESEARCH_ROOT = Path(output_root) / "research" / TRACK_NAME
        CANDIDATES_ROOT = Path(output_root) / "candidates" / TRACK_NAME
    else:
        RESEARCH_ROOT, CANDIDATES_ROOT = globals()["RESEARCH_ROOT"], globals()["CANDIDATES_ROOT"]
    all_projects = [d.name for d in Path(data_root).iterdir() if d.is_dir()]
    projects = sorted(active_projects(all_projects, config))
    target_cases = cases or projects

    print(f"=== Starting AutoResearch Track 2 (Industrial Absolute Exact Hit) ===")
    print(f"Target Cases: {target_cases} | Storage: {storage} | Device: {device}")

    from escount.features.cache import prepare, load_manifest

    def get_table(case_name, grid):
        cache_root = RESEARCH_ROOT / "features" / config_hash(grid.to_dict())[:16]
        path, _ = prepare(case_name, data_root, profiles_root, cache_root, None, grid)
        expected = [{"image_id": f"{case_name}/{item['name']}", "case": case_name, "image": item["name"], "gt": item["gt"]}
                    for item in load_manifest(case_name, cache_root)["images"]]
        return pd.read_parquet(path), expected

    for case in target_cases:
        study = open_study(f"{case}_exact_hit", storage)
        print(f"\n--- Case: {case} (Existing Trials: {len(study.trials)}) ---")
        best_rank = (0, float("inf"), float("inf"))
        c_path = CANDIDATES_ROOT / case / BEST_NAME
        if c_path.is_file():
            try:
                res = json.loads(c_path.read_text(encoding="utf-8"))["result"]
                best_rank = (-res["hits"], res.get("mae", float("inf")), res["latency_ms"])
            except Exception:
                pass

        for i in range(trials):
            trial = study.ask()
            config_trial = suggest_exact(trial)
            digest = config_hash(config_trial)[:12]
            try:
                table, expected = get_table(case, grid_of(config_trial))
                procedure = ResearchProcedureExact(config_trial, False, device)
                loio = with_missing(evaluate(procedure, table, "LOIO"), expected)
                study.tell(trial, loio.hit_rate)

                curr_rank = rank_key_exact(loio)
                status = "keep" if curr_rank < best_rank else "discard"
                if status == "keep":
                    best_rank = curr_rank
                    out_cand = CANDIDATES_ROOT / case
                    out_cand.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "case": case, "trial": trial.number, "track": TRACK_NAME,
                        "config_hash": digest, "result": {**summary(loio), "mae": loio.mae},
                        "geometry": config_trial["geometry"], "model": config_trial["model"]
                    }
                    (out_cand / BEST_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    per_image_table([loio]).to_csv(out_cand / PER_IMAGE_NAME, index=False, encoding="utf-8")

                tsv_row = {
                    "trial": trial.number, "case": case, "config_hash": digest,
                    "hit_rate": f"{loio.hit_rate:.4f}", "hits/total": f"{loio.hits}/{loio.total}",
                    "roi_count": f"{loio.roi_count:g}", "latency_ms": f"{loio.latency_ms:.3f}",
                    "status": status, "timestamp": datetime.now().isoformat(timespec="seconds")
                }
                append_tsv(RESEARCH_ROOT / case / TSV_NAME, tsv_row)
                print(f"[{case}] Trial {trial.number}: Model={config_trial['model']['kind']} LOIO={loio.hits}/{loio.total} "
                      f"MAE={loio.mae:.3f} -> {status}")
            except Exception as e:
                print(f"[{case}] Trial {trial.number} failed: {e}")
                study.tell(trial, float("nan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Track 2 (Industrial Absolute Exact Hit)")
    parser.add_argument("--cases", nargs="*", default=None, help="Specific cases to research")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per case")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"], help="Compute device")
    args = parser.parse_args()
    run_exact_research(cases=args.cases, trials=args.trials, device=args.device)
