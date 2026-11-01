"""End-to-end integration test for the three AutoResearch tracks using synthetic data."""
import numpy as np
import pandas as pd
import pytest

from escount.features.extract import count_columns
from escount.research.daemon import suggest
from escount.research.evaluate import evaluate
from scripts.run_research_ensemble import ResearchProcedureEnsemble, suggest_ensemble
from scripts.run_research_exact_hit import ResearchProcedureExact, suggest_exact
from scripts.run_research_mae import ResearchProcedureMAE


def make_synthetic_table(n_images=4, rows_per_image=10):
    rows = []
    cases = ["caseA"]
    columns = count_columns()
    for case in cases:
        for img_idx in range(n_images):
            image_name = f"img_{img_idx}.png"
            image_id = f"{case}/{image_name}"
            gt = 100.0 + img_idx * 10
            for r in range(rows_per_image):
                row = {
                    "case": case,
                    "image": image_name,
                    "image_id": image_id,
                    "gt": gt,
                    "gt_group": "group1",
                    "w": 16,
                    "pitch": 30,
                    "x_offset": 0,
                    "x": r * 10,
                    "y0": 0,
                    "y1": 500,
                    "projection": "mean",
                    "proj_id": 0,
                    "sigma": np.nan,
                    "slot_key": f"mean|16|{r * 10}",
                    "pos_key": f"mean|16|0.10",
                    "L": 500,
                    "x_norm": 0.1,
                    "period": 5.0,
                    "period_count": 100.0 + img_idx * 10,
                    "snr": 10.0,
                    "std": 20.0,
                    "period_stability": 0.01,
                }
                for col in columns:
                    row[col] = gt + np.random.choice([-1.0, 0.0, 1.0])
                rows.append(row)
    return pd.DataFrame(rows)


def test_track1_mae_procedure():
    table = make_synthetic_table(n_images=4)
    # Mock config
    config = {
        "geometry": {"w": 16, "D": 30, "x_offset": 0, "projection": "mean", "sigma_ratio": None, "side_margin": 50},
        "features": {"families": ["kd"], "directions": ["both"]},
        "model": {"kind": "ridge", "params": {"alpha": 1.0}, "aggregation": "median", "roi_cap": 2},
        "budget": {"max_rows_per_image": 50, "n_estimators": 10, "svr_max_rows": 50}
    }
    proc = ResearchProcedureMAE(config, device="cpu")
    result = evaluate(proc, table, "LOIO")
    assert result.total == 4
    assert result.mae >= 0.0
    assert result.hit_rate >= 0.0


def test_track2_exact_hit_procedure():
    table = make_synthetic_table(n_images=4)
    # Test soft-hit loss XGBoost
    config_soft = {
        "geometry": {"w": 16, "D": 30, "x_offset": 0, "projection": "mean", "sigma_ratio": None, "side_margin": 50},
        "features": {"families": ["kd"], "directions": ["both"]},
        "model": {"kind": "soft_loss_xgb", "params": {"sharpness": 6.0, "l2_reg": 0.05, "n_estimators": 5}, "aggregation": "median", "roi_cap": 2},
        "budget": {"max_rows_per_image": 50, "n_estimators": 5, "svr_max_rows": 50}
    }
    proc_soft = ResearchProcedureExact(config_soft, device="cpu")
    res_soft = evaluate(proc_soft, table, "LOIO")
    assert res_soft.total == 4

    # Test discrete classifier
    config_disc = {
        "geometry": {"w": 16, "D": 30, "x_offset": 0, "projection": "mean", "sigma_ratio": None, "side_margin": 50},
        "features": {"families": ["kd"], "directions": ["both"]},
        "model": {"kind": "discrete_xgb", "params": {"max_offset": 2, "n_estimators": 5}, "aggregation": "median", "roi_cap": 2},
        "budget": {"max_rows_per_image": 50, "n_estimators": 5, "svr_max_rows": 50}
    }
    proc_disc = ResearchProcedureExact(config_disc, device="cpu")
    res_disc = evaluate(proc_disc, table, "LOIO")
    assert res_disc.total == 4


def test_track3_ensemble_procedure():
    table = make_synthetic_table(n_images=4)
    config_ens = {
        "geometry": {"w": 16, "D": 30, "x_offset": 0, "projection": "mean", "sigma_ratio": None, "side_margin": 50},
        "features": {"families": ["kd"], "directions": ["both"]},
        "model": {"kind": "ensemble", "params": {"n_estimators": 5}, "aggregation": "median", "roi_cap": 2},
        "ensemble": {"strategy": "median_vote", "models": {"xgb": True, "ridge": True}},
        "budget": {"max_rows_per_image": 50, "n_estimators": 5, "svr_max_rows": 50}
    }
    proc_ens = ResearchProcedureEnsemble(config_ens, device="cpu")
    res_ens = evaluate(proc_ens, table, "LOIO")
    assert res_ens.total == 4
    assert res_ens.latency_ms >= 0.0
