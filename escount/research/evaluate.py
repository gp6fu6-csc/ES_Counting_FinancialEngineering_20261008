"""FIXED evaluator (Task 5). Research code and autoresearch agents must NOT edit this module.

The only accuracy metric is the exact hit rate: an image is a hit iff round(prediction) == GT
(np.rint). There is no MAE and no +-1 tolerance anywhere. Candidates are ranked with
rank_key(result) = (-hit_rate, roi_count, latency_ms): higher hit rate first, ties broken by fewer
ROIs, then by lower latency.

Validation is grouped: LOIO holds out ALL rows (every ROI / augmentation) of one image per fold,
LOPO holds out every row of one project (case). A procedure implements
  fit(train_rows)          -> learns only from the training rows
  predict(test_rows)       -> pandas Series of per-image predictions indexed by image_id
  predict_rows(test_rows)  -> optional, per-row predictions (used by minimal-ROI selection)
  roi_count                -> optional attribute, ROIs used per image (default: median rows per image)
Test rows are passed without the gt columns.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMES = {"LOIO": "image_id", "LOPO": "case"}
LABEL_COLUMNS = ("gt", "gt_group")


@dataclass
class EvalResult:
    hit_rate: float
    hits: int
    total: int
    roi_count: float
    latency_ms: float
    per_image: pd.DataFrame
    scheme: str = "LOIO"
    row_predictions: pd.Series = field(default=None, repr=False)
    mae: float = 0.0
    within_1_rate: float = 0.0


def exact_hit(prediction, truth):
    """round(prediction) == truth (np.rint); NaN predictions are misses."""
    prediction = np.asarray(prediction, dtype=np.float64)
    return np.isfinite(prediction) & (np.rint(np.nan_to_num(prediction, nan=-1.0)) == np.asarray(truth))


def rank_key(result):
    return (-result.hit_rate, result.roi_count, result.latency_ms)


def rank_key_mae(result):
    return (result.mae, result.roi_count, result.latency_ms)


def rank_key_exact(result):
    return (-result.hits, result.mae, result.roi_count, result.latency_ms)


def wilson_interval(hits, total, z=1.96):
    """Wilson score interval of a hit rate (95 % for z = 1.96)."""
    if total <= 0:
        return 0.0, 1.0
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(procedure, table, scheme="LOIO", keep_row_predictions=False):
    """Grouped cross-validation of `procedure` on a feature table; exact hit rate only."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(SCHEMES)}")
    groups = table[SCHEMES[scheme]]
    fold_ids = sorted(groups.unique())
    if len(fold_ids) < 2:
        raise ValueError(f"{scheme} needs at least 2 groups, got {len(fold_ids)}")
    images = table.groupby("image_id", sort=True).agg(case=("case", "first"), image=("image", "first"),
                                                       gt=("gt", "first"))
    predictions, folds, roi_counts, seconds, row_parts = {}, {}, [], 0.0, []
    for fold in fold_ids:
        held = (groups == fold).to_numpy()
        procedure.fit(table[~held])
        test = table[held].drop(columns=[c for c in LABEL_COLUMNS if c in table.columns])
        start = time.perf_counter()
        predicted = procedure.predict(test)
        seconds += time.perf_counter() - start
        for image_id in test["image_id"].unique():
            predictions[image_id] = float(predicted.get(image_id, np.nan))
            folds[image_id] = fold
        count = getattr(procedure, "roi_count", None)
        roi_counts.append(float(test.groupby("image_id").size().median()) if count is None else float(count))
        if keep_row_predictions:
            row_parts.append(procedure.predict_rows(test))
    per_image = images.reset_index()
    per_image["prediction"] = per_image["image_id"].map(predictions)
    per_image["rounded"] = np.rint(per_image["prediction"])
    per_image["hit"] = exact_hit(per_image["prediction"], per_image["gt"])
    per_image["scheme"] = scheme
    per_image["fold"] = per_image["image_id"].map(folds)
    hits, total = int(per_image["hit"].sum()), len(per_image)
    errors = (per_image["prediction"] - per_image["gt"]).abs()
    mae = float(np.nanmean(errors)) if len(errors) else 0.0
    within_1_rate = float(np.nanmean(errors <= 1.0)) if len(errors) else 0.0
    return EvalResult(hit_rate=hits / total, hits=hits, total=total, roi_count=float(np.median(roi_counts)),
                      latency_ms=1000.0 * seconds / total, per_image=per_image, scheme=scheme,
                      row_predictions=pd.concat(row_parts) if keep_row_predictions else None,
                      mae=mae, within_1_rate=within_1_rate)


def hits_by_case(result):
    """{case: (hits, total)} of an EvalResult."""
    grouped = result.per_image.groupby("case")["hit"]
    return {case: (int(hit.sum()), int(hit.size)) for case, hit in grouped}
