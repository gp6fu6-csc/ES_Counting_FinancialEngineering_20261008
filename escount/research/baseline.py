"""Legacy V3 baseline, re-created only to regression-check the fixed evaluator (Task 5).

LegacyProcedure re-runs the V3 selection (strip width 15 px, pitch D = 30 px, side margin 100 px,
legacy rotation, es_counting KD/MACD coarse -> fine grids, legacy MAE-ranked choice inside the
training fold via es_counting._procedure). The MAE ranking and the kd/macd final-method choice by LOO
MAE below are the LEGACY rules, kept only to reproduce the V3 numbers; V4 ranks by exact hit rate.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from escount import es_counting as es
from escount.research.evaluate import EvalResult, evaluate, exact_hit

LEGACY_WIDTHS = (15,)
LEGACY_STEP = 30
LEGACY_SIDE_MARGIN = 100


class LegacyCase:
    """Legacy per-strip count caches of one case (built exactly like es_counting.calibrate)."""

    def __init__(self, case_dir):
        case_dir = Path(case_dir).resolve()
        with tempfile.TemporaryDirectory() as scratch:   # calibrate_rotation writes diagnostics
            rotation, rotated_mask = es.calibrate_rotation(case_dir, Path(scratch))
        self.angle = rotation["compensation_angle_deg"]
        images = [p for p in es.list_images(case_dir) if es.parse_ground_truth(p.name)]
        self.names = [p.name for p in images]
        self.truth = np.array([es.parse_ground_truth(p.name)["count"] for p in images], dtype=np.float64)
        layouts = {w: es.strip_layout(rotated_mask, w, step=LEGACY_STEP, side_margin=LEGACY_SIDE_MARGIN)
                   for w in LEGACY_WIDTHS}
        data = es._load_scenario_signals(case_dir, self.angle, layouts, images)
        self.caches = {}
        for w, layout in layouts.items():
            stacked, lengths = np.vstack(data[w]["signals"]), np.concatenate(data[w]["lengths"])
            self.caches[w] = {}
            for method, grid in (("kd", es.KD_COARSE), ("macd", es.MACD_COARSE)):
                cache = es.CountCache(method, stacked, lengths, len(images), layout["count"])
                for line in es._lines(method, grid):
                    cache.add(line)
                variants, predictions = zip(*cache.variants())
                index, _, _ = es._rank(np.array(predictions), self.truth, np.arange(len(images)))
                for line in es._neighbors(method, dict(variants[index][0])):
                    cache.add(line)
                variants, predictions = zip(*cache.variants())
                self.caches[w][method] = {"variants": list(variants), "predictions": np.array(predictions)}
        self.table = pd.DataFrame({"case": case_dir.name, "image": self.names,
                                   "image_id": [f"{case_dir.name}/{n}" for n in self.names], "gt": self.truth})


class LegacyProcedure:
    """Legacy KD or MACD selection on the training fold; one row per image in the table."""

    roi_count = None

    def __init__(self, legacy, method):
        self.legacy, self.method, self.choice = legacy, method, None

    def fit(self, train_rows):
        train = np.array([self.legacy.names.index(name) for name in train_rows["image"]])
        w, kd_index, macd_index, _ = es._procedure(self.legacy.caches, self.legacy.truth, train)
        self.choice = (w, kd_index if self.method == "kd" else macd_index)

    def predict(self, test_rows):
        w, index = self.choice
        predictions = self.legacy.caches[w][self.method]["predictions"][index]
        return pd.Series({image_id: float(predictions[self.legacy.names.index(name)])
                          for image_id, name in zip(test_rows["image_id"], test_rows["image"])})


def legacy_baseline(case_dir):
    """Re-run the legacy LOO for kd and macd; final method = legacy rule (lowest LOO MAE, ties higher
    within-1 rate, then kd). Returns (final_method, {method: EvalResult})."""
    legacy = LegacyCase(case_dir)
    results = {method: evaluate(LegacyProcedure(legacy, method), legacy.table, "LOIO") for method in ("kd", "macd")}

    def legacy_score(method):  # legacy-only rule (regression check), never used for V4 ranking
        errors = (results[method].per_image["prediction"] - results[method].per_image["gt"]).abs()
        return errors.mean(), -(errors <= 1).mean(), ("kd", "macd").index(method)

    final = min(results, key=legacy_score)
    return final, results


def legacy_report_hits(case_output_dir):
    """Exact hits of the legacy final_method from outputs/es_counting/<case>/calibration_report.json."""
    case_output_dir = Path(case_output_dir)
    report = es.read_json(case_output_dir / "calibration_report.json")
    final = report["final_method"]
    folds = report["loo_folds"]
    key = f"{final}_pred"
    hits = exact_hit([f[key] for f in folds], [f["truth"] for f in folds])
    return final, int(hits.sum()), len(folds)


def as_hits(result):
    return f"{result.hits}/{result.total}"


__all__ = ["EvalResult", "LegacyCase", "LegacyProcedure", "legacy_baseline", "legacy_report_hits", "as_hits"]
