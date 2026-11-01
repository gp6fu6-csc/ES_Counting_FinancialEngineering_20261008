"""Track 3: AutoResearch Engine for Low-Resource Fast ML Collaborative Ensemble.

Explores joint collaborative prediction combining XGBoost, LightGBM, and Ridge
via Median Voting, Mean Blending, and Stacking with minimal compute overhead (< 50ms).
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
from escount.models.ensemble import FastCollaborativeEnsemble
from escount.models.train import (
    GENERAL_CASES, active_projects, aggregate, fit_model, general_variants, load_general_config, subsample,
)
from escount.research.daemon import (
    BEST_NAME, PER_IMAGE_COLUMNS, PER_IMAGE_NAME, TSV_COLUMNS, TSV_NAME, TrialBudget,
    append_tsv, count_subset, design_matrix, grid_of, open_study, per_image_table, read_tsv, suggest,
    summary, with_missing,
)
from escount.research.evaluate import evaluate, rank_key_exact

TRACK_NAME = "ensemble"
DEFAULT_STORAGE = "sqlite:///research/study_ensemble.db"
RESEARCH_ROOT = Path("research") / TRACK_NAME
CANDIDATES_ROOT = Path("candidates") / TRACK_NAME

STRATEGIES = ("median_vote", "mean_vote", "stacking")


def suggest_ensemble(trial, budget=TrialBudget()):
    base_config = suggest(trial, budget)
    strategy = trial.suggest_categorical("ensemble_strategy", list(STRATEGIES))
    use_xgb = trial.suggest_categorical("use_xgb", [True])
    use_lgbm = trial.suggest_categorical("use_lgbm", [True, False])
    use_ridge = trial.suggest_categorical("use_ridge", [True, False])
    # Ensure at least two models for ensemble
    if not use_lgbm and not use_ridge:
        use_lgbm = True

    base_config["ensemble"] = {
        "strategy": strategy,
        "models": {
            "xgb": use_xgb,
            "lgbm": use_lgbm,
            "ridge": use_ridge,
        }
    }
    return base_config


class ResearchProcedureEnsemble:
    """Procedure dedicated to fast collaborative ML ensemble."""

    def __init__(self, config, scale_normalised=False, device="cuda", seed=0):
        self.config, self.scale_normalised, self.device, self.seed = config, scale_normalised, device, seed
        self.ens_spec = config.get("ensemble", {"strategy": "median_vote", "models": {"xgb": True, "lgbm": True}})
        self.columns = count_subset(config["features"])
        self.key_column = "pos_key" if scale_normalised else "slot_key"
        self.ensemble = None
        self.units, self.roi_count, self.device_used = [], None, None

    def fit(self, train_rows):
        budget = TrialBudget(**self.config["budget"])
        rows = subsample(train_rows, budget.max_rows_per_image, self.seed)
        target = (rows["gt"] - rows["period_count"]).to_numpy(dtype=np.float64)
        X = design_matrix(rows, self.columns, self.scale_normalised)

        active_models = {}
        from escount.models.train import MODEL_PARAMS
        raw_params = self.config["model"].get("params", {})
        for m_name, is_active in self.ens_spec.get("models", {}).items():
            if is_active:
                valid_keys = set(MODEL_PARAMS.get(m_name, {}).keys())
                filtered_params = {k: v for k, v in raw_params.items() if k in valid_keys}
                m_fit, dev = fit_model(m_name, X, target, self.device, filtered_params)
                active_models[m_name] = m_fit
                self.device_used = dev

        self.ensemble = FastCollaborativeEnsemble(
            models=active_models,
            strategy=self.ens_spec.get("strategy", "median_vote")
        )

        if self.ens_spec.get("strategy") == "stacking" and len(active_models) > 1:
            oof = {name: model.predict(X) for name, model in active_models.items()}
            self.ensemble.fit_stacking(oof, target)

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
        delta = self.ensemble.predict(design_matrix(test_rows, self.columns, self.scale_normalised))
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


def run_ensemble_research(cases=None, trials=10, hours=None, storage=DEFAULT_STORAGE, device="cuda"):
    config = load_general_config()
    all_projects = [d.name for d in Path("Data").iterdir() if d.is_dir()]
    projects = sorted(active_projects(all_projects, config))
    target_cases = cases or projects

    print(f"=== Starting AutoResearch Track 3 (Collaborative Fast ML Ensemble) ===")
    print(f"Target Cases: {target_cases} | Storage: {storage} | Device: {device}")

    from escount.features.cache import prepare, load_manifest

    def get_table(case_name, grid):
        cache_root = RESEARCH_ROOT / "features" / config_hash(grid.to_dict())[:16]
        path, _ = prepare(case_name, "Data", "outputs/projects", cache_root, None, grid)
        expected = [{"image_id": f"{case_name}/{item['name']}", "case": case_name, "image": item["name"], "gt": item["gt"]}
                    for item in load_manifest(case_name, cache_root)["images"]]
        return pd.read_parquet(path), expected

    for case in target_cases:
        study = open_study(f"{case}_ensemble", storage)
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
            config_trial = suggest_ensemble(trial)
            digest = config_hash(config_trial)[:12]
            try:
                table, expected = get_table(case, grid_of(config_trial))
                procedure = ResearchProcedureEnsemble(config_trial, False, device)
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
                        "geometry": config_trial["geometry"], "ensemble": config_trial["ensemble"]
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
                m_str = "+".join([k for k, v in config_trial["ensemble"]["models"].items() if v])
                print(f"[{case}] Trial {trial.number}: Strat={config_trial['ensemble']['strategy']} "
                      f"Models={m_str} LOIO={loio.hits}/{loio.total} MAE={loio.mae:.3f} Latency={loio.latency_ms:.1f}ms -> {status}")
            except Exception as e:
                print(f"[{case}] Trial {trial.number} failed: {e}")
                study.tell(trial, float("nan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Track 3 (Collaborative Fast ML Ensemble)")
    parser.add_argument("--cases", nargs="*", default=None, help="Specific cases to research")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per case")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"], help="Compute device")
    args = parser.parse_args()
    run_ensemble_research(cases=args.cases, trials=args.trials, device=args.device)
