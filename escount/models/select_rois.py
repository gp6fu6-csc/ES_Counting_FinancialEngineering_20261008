"""Minimal ROI selection (Task 7): greedy forward selection over ROI units.

A unit is one (projection, w, position): for a project the exact strip x (`slot_key`), for General
the x_norm bin shared by all projects (`pos_key`, nearest strip per image). Per-row predictions are
out-of-fold Grouped-LOO (LOIO) predictions of the case's adopted procedure, so every image is
predicted by a model that never saw it. Step: add the unit that most improves the exact hit rate of
round(median of the selected units' predictions) (ties: lower cost w, then key order), up to
`max_rois` units. Units that overlap an already selected ROI by >= w/3 (narrower width; same strip
under another projection included) or share its x_norm bin (General) are never added.

Subset size: in-sample hits (selection on all images) are optimistic, so the size k is chosen on
the `nested_loio` curve (for each image the path is re-selected without it, its first k units score
that image), most hits first, fewest ROIs on ties. `deployed` is the honest estimate of what
inference runs: an outer LOIO of the whole procedure (path + size chosen on the inner curve). A
baseline procedure's full ROI set (one bank, overlap < w/3) is deployed instead when its LOIO
(group chosen inside each fold) has more exact hits. Latency is checked by `escount benchmark`.
"""
import warnings

import numpy as np
import pandas as pd

from escount.research.evaluate import evaluate, exact_hit, wilson_interval

MAX_ROIS = 25


def unit_geometry(keys, key_column):
    """Column geometry of the units: {'x', 'w'} (project slots) or {'bin'} (General x_norm bins)."""
    parts = [str(key).rsplit("|", 2) for key in keys]
    if key_column == "pos_key":
        return {"bin": np.array([round(float(p[2]), 4) for p in parts])}
    return {"x": np.array([int(p[2]) for p in parts]), "w": np.array([int(p[1]) for p in parts])}


def conflicts_with(geometry, chosen, candidates):
    """Mask over `candidates`: units overlapping unit `chosen` by >= w/3 of the narrower strip, or
    (General) in the same x_norm bin."""
    if "bin" in geometry:
        return geometry["bin"][candidates] == geometry["bin"][chosen]
    x, w = geometry["x"], geometry["w"]
    overlap = np.minimum(x[candidates] + w[candidates], x[chosen] + w[chosen]) - np.maximum(x[candidates], x[chosen])
    return 3 * overlap >= np.minimum(w[candidates], w[chosen])


def check_selected_overlap(rois):
    """ValueError if two selected ROIs overlap by >= w/3 (narrower width) or two General units share
    an x_norm bin."""
    if not rois:
        return
    general = "x" not in rois[0]
    if general:
        geometry = {"bin": np.array([round(float(r["x_norm_bin"]), 4) for r in rois])}
    else:
        geometry = {"x": np.array([int(r["x"]) for r in rois]), "w": np.array([int(r["w"]) for r in rois])}
    for index in range(1, len(rois)):
        clash = np.flatnonzero(conflicts_with(geometry, index, np.arange(index)))
        if clash.size:
            other = rois[int(clash[0])]
            rule = "share the x_norm bin" if general else "overlap by >= w/3"
            raise ValueError(f"selected ROIs {other.get('key')} and {rois[index].get('key')} {rule}; "
                             f"re-run `escount select-rois`")


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


def medians_with(values, selected, candidates):
    """nanmedian over the `selected` units plus each candidate unit, per image: (n_candidates, n_images).

    Equal to np.nanmedian of the stacked columns, computed by inserting each candidate into the
    sorted selected values (O(images x candidates x k) comparisons instead of a sort per candidate).
    """
    cand = values[:, candidates]
    if not selected:
        return cand.T.copy()
    block = np.sort(values[:, selected], axis=1)                  # NaN sorted last
    valid = np.sum(~np.isnan(block), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(block, axis=1)                        # candidate NaN -> selected only
    position = (block[:, None, :] < cand[:, :, None]).sum(axis=2)  # insertion index among valid values
    size = np.broadcast_to((valid + 1)[:, None], position.shape)
    last = block.shape[1] - 1

    def merged(index):
        gathered = np.take_along_axis(block, np.clip(index - (index > position), 0, last), axis=1)
        return np.where(index == position, cand, gathered)

    result = (merged((size - 1) // 2) + merged(size // 2)) / 2.0
    return np.where(np.isnan(cand), base[:, None], result).T


def greedy_select(values, gt, cost, max_rois=MAX_ROIS, geometry=None):
    """Greedy path: indices of up to `max_rois` units in selection order and the in-sample hits after
    each step. With `geometry` (unit_geometry) a unit conflicting with a selected one is never added."""
    selected, history = [], []
    remaining = np.arange(values.shape[1])
    while len(selected) < max_rois and remaining.size:
        hits = exact_hit(medians_with(values, selected, remaining), gt[None, :]).sum(axis=1)
        choice = np.lexsort((remaining, cost[remaining], -hits))[0]
        selected.append(int(remaining[choice]))
        history.append(int(hits[choice]))
        remaining = np.delete(remaining, choice)
        if geometry is not None:
            remaining = remaining[~conflicts_with(geometry, selected[-1], remaining)]
    return selected, history


def nested_curve(values, gt, cost, max_rois=MAX_ROIS, geometry=None):
    """curve[k-1] = exact hits when every image is scored with the first k units of the greedy path
    selected without it (selection-inside-fold estimate per subset size)."""
    curve = np.zeros(max_rois, dtype=int)
    for index in range(len(gt)):
        others = np.arange(len(gt)) != index
        path, _ = greedy_select(values[others], gt[others], cost, max_rois, geometry)
        for size in range(1, max_rois + 1):
            curve[size - 1] += subset_hits(values[index:index + 1], gt[index:index + 1], path[:size])
    return curve


def choose_size(curve, path_length):
    """Subset size with the most nested hits; ties -> fewest ROIs."""
    return int(np.argmax(curve[:max(path_length, 1)])) + 1


def honest_hits(values, gt, cost, max_rois=MAX_ROIS, geometry=None):
    """Outer LOIO of the whole selection procedure (greedy path + size chosen on the nested curve of
    the other images): exact hits of the held-out images."""
    hits = 0
    for index in range(len(gt)):
        others = np.arange(len(gt)) != index
        path, _ = greedy_select(values[others], gt[others], cost, max_rois, geometry)
        size = choose_size(nested_curve(values[others], gt[others], cost, max_rois, geometry), len(path))
        hits += subset_hits(values[index:index + 1], gt[index:index + 1], path[:size])
    return hits


def _roi_items(table, key_column, keys):
    representative = table[table[key_column].isin(keys)].drop_duplicates(key_column).set_index(key_column)
    rois = []
    for key in keys:
        row = representative.loc[key]
        item = {"key": key, "projection": row["projection"], "w": int(row["w"]),
                "sigma": None if pd.isna(row["sigma"]) else float(row["sigma"])}
        if key_column == "slot_key":
            item.update(x=int(row["x"]), pitch=int(row["pitch"]), x_offset=int(row["x_offset"]))
        else:
            item.update(x_norm_bin=float(key.rsplit("|", 1)[1]))
        rois.append(item)
    return rois


def select_units(procedure, table, key_column="slot_key", max_rois=MAX_ROIS):
    """Run LOIO with row predictions, then greedy selection with the size chosen on the nested curve.
    Returns a result dict; `deployed` is the honest exact-hit estimate of the ROIs inference uses."""
    full = evaluate(procedure, table, "LOIO", keep_row_predictions=True)
    values, keys, gt, cost, image_ids = unit_matrix(table, full.row_predictions, key_column)
    geometry = unit_geometry(keys, key_column)
    path, history = greedy_select(values, gt, cost, max_rois, geometry)
    curve = nested_curve(values, gt, cost, max_rois, geometry)
    size = choose_size(curve, len(path))
    honest = honest_hits(values, gt, cost, max_rois, geometry)
    chosen_keys, source = [keys[i] for i in path[:size]], "greedy_subset"
    deployed_hits = honest
    estimate = ("outer LOIO of the whole selection (greedy path + size chosen on the nested curve, both "
                "without the held-out image)")
    if key_column == "slot_key" and hasattr(procedure, "deployable_rows"):
        procedure.fit(table)
        bank = procedure.deployable_rows(table).sort_values("x")["slot_key"].unique().tolist()
        if (-full.hits, len(bank)) < (-honest, size):
            chosen_keys, source, deployed_hits = bank, "full_set", full.hits
            estimate = "LOIO of the full ROI set (layout group and indicator chosen inside each fold)"
    rois = _roi_items(table, key_column, chosen_keys)
    total = len(gt)
    low, high = wilson_interval(deployed_hits, total)
    in_sample = history[size - 1] if source == "greedy_subset" else None
    return {"key_column": key_column, "selected_rois": rois, "roi_count": len(rois),
            "hits": in_sample, "total": total, "hit_rate": None if in_sample is None else in_sample / total,
            "history_hits": history,
            "full_set": {"hits": full.hits, "total": full.total, "hit_rate": full.hit_rate,
                         "roi_count": full.roi_count},
            "nested_loio": {"hits": int(curve[size - 1]), "total": total, "hit_rate": int(curve[size - 1]) / total,
                            "size": size, "curve": curve[:max(len(path), 1)].tolist(),
                            "note": "curve[k-1] = hits when each image uses the first k units of the path "
                                    "re-selected without it; the subset size is chosen on this curve"},
            "deployed": {"source": source, "roi_count": len(rois), "hits": deployed_hits, "total": total,
                         "hit_rate": deployed_hits / total, "wilson95": [low, high], "estimate": estimate,
                         "note": "honest exact-hit estimate of the ROIs inference uses; quote this"},
            "candidate_units": len(keys),
            "note": "out-of-fold LOIO row predictions; `hits` = in-sample hits of the greedy subset (optimistic)"}


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
    try:
        from escount.models.all_rois import train_all_rois
        train_all_rois(case, profiles_root=profiles_root, device=device)
    except Exception as err:
        warnings.warn(f"all_rois ensemble training skipped: {err}")
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
