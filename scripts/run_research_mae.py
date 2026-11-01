"""Track 1: AutoResearch Engine for MAE Error Minimization.

Focuses on reducing macro-level error, eliminating outlier sheet count deviations,
and stabilizing baseline regressions.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from escount.features.extract import AugmentGrid, config_hash, feature_config
from escount.models.train import (
    GENERAL_CASES, active_projects, aggregate, fit_model, general_variants, load_general_config, subsample,
)
from escount.research.daemon import (
    BEST_NAME, PER_IMAGE_COLUMNS, PER_IMAGE_NAME, TSV_COLUMNS, TSV_NAME, TrialBudget,
    append_tsv, count_subset, design_matrix, grid_of, open_study, per_image_table, read_tsv, suggest,
    summary, with_missing,
)
from escount.research.evaluate import evaluate, rank_key_mae

TRACK_NAME = "mae"
DEFAULT_STORAGE = "sqlite:///research/study_mae.db"
RESEARCH_ROOT = Path("research") / TRACK_NAME
CANDIDATES_ROOT = Path("candidates") / TRACK_NAME


class ResearchProcedureMAE:
    """Procedure optimized for MAE minimization."""

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
            errors = np.abs(medians.to_numpy() - gt[:, None]).mean(axis=0)
            self.column = candidates[int(np.argmin(errors))]
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
        per_unit_image["error"] = (per_unit_image["prediction"] - per_unit_image["gt"]).abs()
        score = per_unit_image.groupby(level="unit").agg(mae=("error", "mean"), snr=("snr", "mean")).reset_index()
        score = score.sort_values(["mae", "snr", "unit"], ascending=[True, False, True])
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


def run_mae_research(cases=None, trials=10, hours=None, storage=DEFAULT_STORAGE, device="cuda"):
    config = load_general_config()
    all_projects = [d.name for d in Path("Data").iterdir() if d.is_dir()]
    projects = sorted(active_projects(all_projects, config))
    target_cases = cases or projects

    print(f"=== Starting AutoResearch Track 1 (MAE Minimization) ===")
    print(f"Target Cases: {target_cases} | Storage: {storage} | Device: {device}")

    from escount.features.cache import prepare, load_manifest

    def get_table(case_name, grid):
        cache_root = RESEARCH_ROOT / "features" / config_hash(grid.to_dict())[:16]
        path, _ = prepare(case_name, "Data", "outputs/projects", cache_root, None, grid)
        expected = [{"image_id": f"{case_name}/{item['name']}", "case": case_name, "image": item["name"], "gt": item["gt"]}
                    for item in load_manifest(case_name, cache_root)["images"]]
        return pd.read_parquet(path), expected

    for case in target_cases:
        study = open_study(f"{case}_mae", storage)
        print(f"\n--- Case: {case} (Existing Trials: {len(study.trials)}) ---")
        best_mae = float("inf")
        c_path = CANDIDATES_ROOT / case / BEST_NAME
        if c_path.is_file():
            try:
                best_mae = json.loads(c_path.read_text(encoding="utf-8"))["result"].get("mae", float("inf"))
            except Exception:
                pass

        for i in range(trials):
            trial = study.ask()
            config_trial = suggest(trial)
            digest = config_hash(config_trial)[:12]
            try:
                table, expected = get_table(case, grid_of(config_trial))
                procedure = ResearchProcedureMAE(config_trial, False, device)
                loio = with_missing(evaluate(procedure, table, "LOIO"), expected)
                mae_val = loio.mae
                study.tell(trial, mae_val)

                status = "keep" if mae_val < best_mae else "discard"
                if status == "keep":
                    best_mae = mae_val
                    out_cand = CANDIDATES_ROOT / case
                    out_cand.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "case": case, "trial": trial.number, "track": TRACK_NAME,
                        "config_hash": digest, "result": {**summary(loio), "mae": mae_val},
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
                print(f"[{case}] Trial {trial.number}: MAE={mae_val:.3f} sheets (LOIO {loio.hits}/{loio.total}) -> {status}")
            except Exception as e:
                print(f"[{case}] Trial {trial.number} failed: {e}")
                study.tell(trial, float("nan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Track 1 (MAE Minimization)")
    parser.add_argument("--cases", nargs="*", default=None, help="Specific cases to research")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per case")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"], help="Compute device")
    args = parser.parse_args()
    run_mae_research(cases=args.cases, trials=args.trials, device=args.device)
