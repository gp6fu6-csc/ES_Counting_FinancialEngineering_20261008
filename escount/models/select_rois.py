"""Minimal ROI selection (Task 7): greedy forward selection over ROI units.

A unit is one (projection, w, position): for a project the exact strip x (`slot_key`), for General
the x_norm bin shared by all projects (`pos_key`, nearest strip per image). Per-row predictions are
out-of-fold Grouped-LOO (LOIO) predictions of the case's adopted procedure, so every image is
predicted by a model that never saw it. Step: add the unit that most improves the exact hit rate of
round(median of the selected units' predictions) (ties: lower cost w, then key order); stop when the
hit rate no longer improves. The selection itself uses all images, so its hit rate is optimistic.
"""
import warnings

import numpy as np
import pandas as pd

from escount.research.evaluate import evaluate, exact_hit

MAX_ROIS = 25


def unit_matrix(table, row_predictions, key_column):
    """(values[n_images, n_units], unit keys, gt per image, cost per unit, image ids)."""
    frame = table[["image_id", "gt", "w", "x_norm", key_column]].copy()
    frame["prediction"] = row_predictions.reindex(table.index).to_numpy()
    if key_column == "pos_key":
        centre = frame[key_column].str.rsplit("|", n=1).str[1].astype(float)
        frame["distance"] = (frame["x_norm"] - centre).abs()
        frame = frame.sort_values(["image_id", key_column, "distance"])
    frame = frame.drop_duplicates(["image_id", key_column], keep="first")
    values = frame.pivot(index="image_id", columns=key_column, values="prediction")
    gt = frame.groupby("image_id")["gt"].first().reindex(values.index).to_numpy()
    cost = frame.groupby(key_column)["w"].first().reindex(values.columns).to_numpy(dtype=np.float64)
    return values.to_numpy(dtype=np.float64), list(values.columns), gt, cost, list(values.index)


def subset_hits(values, gt, units):
    """Exact hits of round(median over `units`) per image."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN image -> NaN -> miss
        prediction = np.nanmedian(values[:, units], axis=1)
    return int(exact_hit(prediction, gt).sum())


def greedy_select(values, gt, cost, max_rois=MAX_ROIS):
    """Indices of the greedily selected units and the hits after each step."""
    selected, history, best_hits = [], [], -1
    remaining = np.arange(values.shape[1])
    while len(selected) < max_rois and remaining.size:
        stacked = np.concatenate([np.repeat(values[:, selected][None], remaining.size, axis=0),
                                  values.T[remaining][:, :, None]], axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions = np.nanmedian(stacked, axis=2)
        hits = exact_hit(predictions, gt[None, :]).sum(axis=1)
        order = np.lexsort((remaining, cost[remaining], -hits))
        choice = order[0]
        if selected and hits[choice] <= best_hits:
            break
        best_hits = int(hits[choice])
        selected.append(int(remaining[choice]))
        history.append(best_hits)
        remaining = np.delete(remaining, choice)
    return selected, history


def select_units(procedure, table, key_column="slot_key", max_rois=MAX_ROIS):
    """Run LOIO with row predictions, then greedy selection. Returns a result dict."""
    full = evaluate(procedure, table, "LOIO", keep_row_predictions=True)
    values, keys, gt, cost, image_ids = unit_matrix(table, full.row_predictions, key_column)
    selected, history = greedy_select(values, gt, cost, max_rois)
    hits = history[-1]
    chosen_keys = [keys[i] for i in selected]
    representative = table[table[key_column].isin(chosen_keys)].drop_duplicates(key_column).set_index(key_column)
    rois = []
    for key in chosen_keys:
        row = representative.loc[key]
        item = {"key": key, "projection": row["projection"], "w": int(row["w"]),
                "sigma": None if pd.isna(row["sigma"]) else float(row["sigma"])}
        if key_column == "slot_key":
            item.update(x=int(row["x"]), pitch=int(row["pitch"]), x_offset=int(row["x_offset"]))
        else:
            item.update(x_norm_bin=float(key.rsplit("|", 1)[1]))
        rois.append(item)
    return {"key_column": key_column, "selected_rois": rois, "roi_count": len(rois),
            "hits": hits, "total": len(gt), "hit_rate": hits / len(gt), "history_hits": history,
            "full_set": {"hits": full.hits, "total": full.total, "hit_rate": full.hit_rate,
                         "roi_count": full.roi_count},
            "candidate_units": len(keys),
            "note": "out-of-fold LOIO row predictions; the greedy subset is chosen on all images (optimistic)"}


def _persist(model_dir, result):
    from escount.models.store import update_manifest

    return update_manifest(model_dir, selected_rois=result["selected_rois"],
                           selection={k: v for k, v in result.items() if k != "selected_rois"})


def select_project(case, cache_root="outputs/features", profiles_root="outputs/projects", device="cuda",
                   max_rois=MAX_ROIS):
    """Select the minimal ROIs of a project; write them into project.json and the model manifest."""
    from pathlib import Path

    from escount.calibration.calibrate import PROFILE_NAME, ProjectProfile
    from escount.features.cache import load_features
    from escount.models.store import read_manifest
    from escount.models.train import make_procedure

    model_dir = Path(profiles_root) / case / "model"
    result = select_units(make_procedure(read_manifest(model_dir), device), load_features(case, cache_root),
                          "slot_key", max_rois)
    _persist(model_dir, result)
    profile_path = Path(profiles_root) / case / PROFILE_NAME
    profile = ProjectProfile.load(profile_path)
    profile.selected_rois = result["selected_rois"]
    profile.save(profile_path)
    return result


def select_general(cache_root="outputs/features", profiles_root="outputs/projects", device="cuda",
                   max_rois=MAX_ROIS, config_path="configs/general.json"):
    """Select x_norm-bin ROI units for the General cases in use (`general`, plus `general_no_ring`
    while Ring-type is active), stored in their manifests."""
    from pathlib import Path

    from escount.features.cache import load_features
    from escount.models.store import read_manifest
    from escount.models.train import general_cases, general_table, load_general_config, make_procedure

    results = {}
    for name in general_cases(load_general_config(config_path)):
        if name != "general" and not (Path(profiles_root) / name / "model" / "manifest.json").is_file():
            continue
        model_dir = Path(profiles_root) / name / "model"
        manifest = read_manifest(model_dir)
        table = general_table({case: load_features(case, cache_root) for case in manifest["cases"]})
        results[name] = select_units(make_procedure(manifest, device), table, "pos_key", max_rois)
        _persist(model_dir, results[name])
    return results
