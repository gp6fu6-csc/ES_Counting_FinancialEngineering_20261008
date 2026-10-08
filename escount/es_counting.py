"""Electrical-steel lamination counting (演算法流程規劃_20261006.pdf, p5-6).

Pipeline per scenario folder (each Data/<scenario> is one shooting situation):
  calibrate: rotation calibration on a reference image -> rotated background_BW mask ->
             strip layout inside the black (steel) mask region -> per-row strip signals ->
             KD / MACD crossover counting over a parameter grid -> leave-one-image-out (LOO)
             parameter selection -> optional XGBoost stage -> counting_parameters.json
  analyze:   apply the scenario parameters to every image and write one project folder per image.

All periods (KD n/k/d, MACD EMA1/EMA2/Day) are measured in pixels along the stack height.
Ground truth comes from file names: "096" = 96 sheets, "083_2" / "075-1" = second/first group.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

from escount.calibration.rotation_correction import (
    EXCLUDED_IMAGE_NAMES, MASK_NAME, MASK_VALID_THRESHOLD, estimate_boxes, normalize_mask,
    rotate_image, rotate_mask, save_comparison, select_mask_roi, write_image,
)

# ---- Engineering assumptions (not specified by the PDF; recorded in JSON outputs) ----
STRIP_WIDTHS = (10, 15, 20)          # w candidates in px (user: 10-20 px)
STRIP_MARGIN = 5                     # px trimmed at both ends of each strip's black-row run
STRIP_SIDE_MARGIN = 75               # px kept between every strip and the left/right edge of the steel region
STRIP_STEP = None                    # strip pitch D in px (left edge to left edge); None = D equals w (no gap)
STRIP_MIN_LENGTH_RATIO = 0.95        # drop strips shorter than 95% of the longest strip
KD_COARSE = {"n": (3, 5, 7, 9, 12, 15, 20), "k": (2, 3, 5), "d": (2, 3, 5)}
MACD_COARSE = {"ema1": (2, 3, 4, 6, 8), "ema2": (5, 8, 12, 16, 24), "day": (2, 3, 5, 8)}
KD_GATES = (0, 10, 20, 30, 40)       # up-cross counted if K <= 100-g, down-cross if K >= g
MACD_GATES = (None, 1.0, 0.5, 0.25, 0.0)  # up-cross counted if DIF <= tau*std, down if DIF >= -tau*std
DIRECTIONS = ("up", "down", "both")
FINE_RADIUS = 2                      # integer refinement radius around the best coarse line
FFT_PERIOD_RANGE_PX = (6, 60)        # plausible sheet period for the FFT feature
ML_MIN_IMAGES = 8
XGB_PARAMS = {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1, "random_state": 0, "n_jobs": 1}
ML_FEATURES = ("kd_count", "macd_count", "kd_raw_crossings", "macd_raw_crossings",
               "strip_length_px", "fft_count", "signal_std", "strip_width_px")
STACK_ACTIVITY_WINDOW_PX = 41        # moving-mean window of row-to-row texture for stack-extent detection
STACK_ACTIVITY_RATIO = 0.35          # stack rows: texture activity > ratio x its 90th percentile
STACK_GAP_BRIDGE_PX = 150            # low-texture gaps up to this length inside the stack are bridged
PLOT_STRIPS = 6
GT_PATTERN = re.compile(r"^(\d+)(?:[_-](\d+))?$")


# ---------------------------------------------------------------- helpers
def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_signatures(root):
    root = Path(root)
    return {str(p.relative_to(root)).replace("\\", "/"): (p.stat().st_size, file_sha256(p))
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- ground truth / images
def parse_ground_truth(name):
    """Return {"count", "group"} from a file name like 096, 083_2, 075-1; None if not numeric."""
    match = GT_PATTERN.match(Path(name).stem)
    if not match:
        return None
    return {"count": int(match.group(1)), "group": int(match.group(2)) if match.group(2) else 1}


def list_images(scenario_dir):
    """All analyzable PNGs of a scenario (background files excluded), sorted by name."""
    return [p for p in sorted(Path(scenario_dir).glob("*.png")) if p.name not in EXCLUDED_IMAGE_NAMES]


def pick_reference(paths):
    """Default rotation reference: the image whose file-name number is largest (smallest group on ties)."""
    labeled = [(parse_ground_truth(p.name), p) for p in paths]
    labeled = [(gt, p) for gt, p in labeled if gt]
    if not labeled:
        raise ValueError("no image with a numeric file name to use as rotation reference")
    return max(labeled, key=lambda item: (item[0]["count"], -item[0]["group"]))[1]


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"unreadable image: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"expected single-channel uint8 image: {path} {image.shape} {image.dtype}")
    return image


def read_mask(scenario_dir):
    path = Path(scenario_dir) / MASK_NAME
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"missing or unreadable mask: {path}")
    return path, mask


# ---------------------------------------------------------------- rotation calibration
def calibrate_rotation(scenario_dir, out_dir, reference=None):
    """Estimate the scenario rotation on the reference image inside the black mask region."""
    scenario_dir, out_dir = Path(scenario_dir), Path(out_dir)
    mask_path, mask = read_mask(scenario_dir)
    images = list_images(scenario_dir)
    reference_path = scenario_dir / reference if reference else pick_reference(images)
    gray = read_gray(reference_path)
    if gray.shape != mask.shape[:2]:
        raise ValueError("reference image and mask sizes differ")
    roi = select_mask_roi(mask)
    valid = normalize_mask(mask)
    boxes = roi["rois_xywh"]
    if not all(valid[y:y + h, x:x + w].all() for x, y, w, h in boxes):
        raise ValueError("rotation ROI is not fully inside the black mask region")
    estimates, aggregate = estimate_boxes(gray, boxes)
    angle = float(aggregate["theta_deg"])
    corrected = rotate_image(gray, angle)
    rotated_mask = rotate_mask(mask, angle)
    after_roi = select_mask_roi(rotated_mask)
    after_estimates, after_aggregate = estimate_boxes(corrected, after_roi["rois_xywh"])
    before_residual, after_residual = abs(angle), abs(float(after_aggregate["theta_deg"]))
    status = "ok" if after_residual <= before_residual else "quality_failure"
    out_dir.mkdir(parents=True, exist_ok=True)
    rotated_mask_path = out_dir / "background_BW_rotated.png"
    write_image(rotated_mask_path, rotated_mask)
    comparison_path = out_dir / "rotation_reference_comparison.png"
    save_comparison(gray, corrected, comparison_path, angle)
    parameters = {
        "status": status,
        "scenario": scenario_dir.name,
        "reference_image": str(reference_path.resolve()),
        "reference_selection": "user --reference" if reference else "largest file-name number",
        "compensation_angle_deg": angle,
        "rotation": {"interpolation": "INTER_LINEAR", "border": "BORDER_REFLECT_101", "same_size": True,
                     "mask_interpolation": "INTER_NEAREST", "mask_outside_fill": 255},
        "angle_convention": "theta=normalize(atan2(fy, fx)-90); OpenCV positive angle equals theta",
        "mask": {"source": str(mask_path.resolve()), "rotated": str(rotated_mask_path.resolve()),
                 "validity": f"first channel < {MASK_VALID_THRESHOLD} is the steel region (black)"},
        "fft": {"rois_xywh": boxes, "roi_angles_deg": [e["theta_deg"] for e in estimates],
                "after_rois_xywh": after_roi["rois_xywh"],
                "after_roi_angles_deg": [e["theta_deg"] for e in after_estimates]},
        "quality": {"before_residual_deg": before_residual, "after_residual_deg": after_residual,
                    "rule": "post-compensation residual must not exceed the pre-compensation residual"},
        "outputs": [str(rotated_mask_path.resolve()), str(comparison_path.resolve())],
    }
    write_json(out_dir / "rotation_parameters.json", parameters)
    if status != "ok":
        raise RuntimeError(f"{scenario_dir.name}: rotation compensation increased the FFT residual "
                           f"({before_residual:.3f} -> {after_residual:.3f} deg)")
    return parameters, rotated_mask


# ---------------------------------------------------------------- strips and signals
def longest_run(flags):
    """Return (start, end_exclusive) of the longest True run."""
    padded = np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0]))
    change = np.diff(padded)
    starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    if not starts.size:
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def strip_layout(valid, width, step=None, margin=STRIP_MARGIN, min_ratio=STRIP_MIN_LENGTH_RATIO,
                 side_margin=STRIP_SIDE_MARGIN):
    """Place vertical strips (width w, pitch D) inside the valid (black) region.

    N is not set directly: it is every pitch-D position whose strip, widened by `side_margin` px on
    both sides, still lies entirely in the steel region (so each strip keeps >= side_margin px from
    the left/right edge of the effective analysis area and from the image border).
    """
    valid = normalize_mask(valid)
    step = step or width
    if step < width:
        raise ValueError(f"strip pitch D={step} must be >= strip width w={width}")
    candidates = []
    for x in range(side_margin, valid.shape[1] - width - side_margin + 1, step):
        start, end = longest_run(valid[:, x - side_margin:x + width + side_margin].all(axis=1))
        start, end = start + margin, end - margin
        if end - start > 0:
            candidates.append((x, start, end))
    if not candidates:
        raise ValueError(f"no strip of width {width} with {side_margin}px side margin fits inside the steel region")
    longest = max(end - start for _, start, end in candidates)
    strips = [{"x": x, "y0": start, "y1": end, "width": width}
              for x, start, end in candidates if end - start >= min_ratio * longest]
    for strip in strips:
        if not valid[strip["y0"]:strip["y1"], strip["x"] - side_margin:strip["x"] + width + side_margin].all():
            raise AssertionError("strip or its side margin leaves the steel region")
    return {"width": width, "step": step, "gap_px": step - width, "side_margin_px": side_margin,
            "margin_px": margin, "min_length_ratio": min_ratio, "count": len(strips), "strips": strips,
            "rule": "per pitch-D position: longest run of rows where the strip plus side_margin on both "
                    "sides is all steel, trimmed vertically by margin_px"}


def stack_extent(image, layout):
    """PDF step 3 (電磁鋼計算區間): rows actually occupied by the stack inside the strips.

    Row profile p(y) = median over strips of the strip row mean. Sheets alternate bright/dark gaps,
    while rows above a short stack (empty dark area, white fixture, rods) are smooth, so the
    texture activity a(y) = centered moving mean of |p(y) - p(y-1)| separates them. The stack is the
    longest run where a(y) exceeds STACK_ACTIVITY_RATIO x its 90th percentile.
    """
    strips = layout["strips"]
    top, bottom = min(s["y0"] for s in strips), max(s["y1"] for s in strips)
    table = np.full((bottom - top, len(strips)), np.nan)
    for index, strip in enumerate(strips):
        table[strip["y0"] - top:strip["y1"] - top, index] = \
            image[strip["y0"]:strip["y1"], strip["x"]:strip["x"] + strip["width"]].mean(axis=1)
    profile = np.nanmedian(table, axis=1)
    texture = np.abs(np.diff(profile, prepend=profile[0]))
    activity = uniform_filter1d(texture, STACK_ACTIVITY_WINDOW_PX, mode="nearest")
    threshold = STACK_ACTIVITY_RATIO * float(np.percentile(activity, 90))
    active = activity > threshold
    # bridge short low-texture gaps inside the stack (e.g. locally blurred sheets)
    padded = np.concatenate(([0], active.astype(np.int8), [0]))
    change = np.diff(padded)
    starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    for gap_start, gap_end in zip(ends[:-1], starts[1:]):
        if gap_end - gap_start <= STACK_GAP_BRIDGE_PX:
            active[gap_start:gap_end] = True
    start, end = longest_run(active)
    return {"y0": top + start, "y1": top + end, "threshold": threshold,
            "window_px": STACK_ACTIVITY_WINDOW_PX, "ratio": STACK_ACTIVITY_RATIO,
            "gap_bridge_px": STACK_GAP_BRIDGE_PX}


def strip_signals(image, layout, extent=None):
    """Per-row mean intensity of each strip (horizontal accumulation) limited to the stack extent.

    Returns (signals end-padded to the layout's longest strip, valid lengths, absolute start rows).
    """
    strips = layout["strips"]
    width = max(s["y1"] - s["y0"] for s in strips)
    signals = np.empty((len(strips), width), dtype=np.float64)
    lengths = np.zeros(len(strips), dtype=np.int64)
    starts = np.zeros(len(strips), dtype=np.int64)
    for row, strip in enumerate(strips):
        y0, y1 = strip["y0"], strip["y1"]
        if extent is not None:
            y0, y1 = max(y0, extent["y0"]), min(y1, extent["y1"])
        if y1 - y0 < 2:
            y0, y1 = strip["y0"], strip["y0"] + 2
        values = image[y0:y1, strip["x"]:strip["x"] + strip["width"]].mean(axis=1)
        signals[row, :values.size] = values
        signals[row, values.size:] = values[-1]  # causal filters: padding never affects valid samples
        lengths[row], starts[row] = values.size, y0
    return signals, lengths, starts


# ---------------------------------------------------------------- KD / MACD
def kd_lines(signals, n, k, d):
    """Stochastic oscillator along h: RSV over a trailing n-px window, K/D smoothed with 1/k, 1/d."""
    length = signals.shape[1]
    padded = np.pad(signals, ((0, 0), (n - 1, 0)), mode="edge")
    low = minimum_filter1d(padded, n, axis=1)[:, n // 2:n // 2 + length]
    high = maximum_filter1d(padded, n, axis=1)[:, n // 2:n // 2 + length]
    span = high - low
    rsv = np.divide((signals - low) * 100.0, span, out=np.full_like(signals, 50.0), where=span > 1e-9)
    rows = signals.shape[0]
    k_line = lfilter([1.0 / k], [1.0, -(1.0 - 1.0 / k)], rsv, axis=1, zi=np.full((rows, 1), 50.0 * (1.0 - 1.0 / k)))[0]
    d_line = lfilter([1.0 / d], [1.0, -(1.0 - 1.0 / d)], k_line, axis=1, zi=np.full((rows, 1), 50.0 * (1.0 - 1.0 / d)))[0]
    return k_line, d_line


def _ema(values, span, initial):
    alpha = 2.0 / (span + 1.0)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], values, axis=1, zi=(initial * (1.0 - alpha))[:, None])[0]


def macd_lines(signals, ema1, ema2, day):
    """DIF = EMA(ema1) - EMA(ema2); DEA = EMA(DIF, day); spans in px."""
    dif = _ema(signals, ema1, signals[:, 0]) - _ema(signals, ema2, signals[:, 0])
    dea = _ema(dif, day, np.zeros(signals.shape[0]))
    return dif, dea


def crossings(fast, slow, lengths):
    """Boolean (rows, L-1) masks of up/down crossings of fast over slow at sample index 1..L-1."""
    diff = fast - slow
    previous, current = diff[:, :-1], diff[:, 1:]
    inside = np.arange(1, diff.shape[1])[None, :] < lengths[:, None]
    up = (previous <= 0) & (current > 0) & inside
    down = (previous >= 0) & (current < 0) & inside
    return up, down


def gate_masks(method, gate, fast, lengths):
    """Return (up_ok, down_ok) masks for a crossing-zone gate."""
    value = fast[:, 1:]
    if method == "kd":
        return value <= 100.0 - gate, value >= float(gate)
    if gate is None:
        ones = np.ones_like(value, dtype=bool)
        return ones, ones
    inside = np.arange(fast.shape[1])[None, :] < lengths[:, None]
    masked = np.where(inside, fast, 0.0)
    variance = np.sum(masked ** 2, axis=1) / lengths - (np.sum(masked, axis=1) / lengths) ** 2
    threshold = gate * np.sqrt(np.maximum(variance, 0.0))[:, None]
    return value <= threshold, value >= -threshold


def method_lines(method, signals, line):
    if method == "kd":
        return kd_lines(signals, line["n"], line["k"], line["d"])
    return macd_lines(signals, line["ema1"], line["ema2"], line["day"])


def count_strips(method, signals, lengths, line, gate, direction):
    """Per-strip crossing count plus the counted crossing masks (for plotting)."""
    fast, slow = method_lines(method, signals, line)
    up, down = crossings(fast, slow, lengths)
    up_ok, down_ok = gate_masks(method, gate, fast, lengths)
    up, down = up & up_ok, down & down_ok
    up_count, down_count = up.sum(axis=1), down.sum(axis=1)
    counts = {"up": up_count, "down": down_count, "both": (up_count + down_count) / 2.0}[direction]
    return counts.astype(np.float64), up, down


def image_prediction(strip_counts):
    return int(np.rint(np.median(strip_counts)))


# ---------------------------------------------------------------- calibration cache
def _lines(method, grid):
    if method == "kd":
        return [{"n": n, "k": k, "d": d} for n, k, d in product(grid["n"], grid["k"], grid["d"])]
    return [{"ema1": a, "ema2": b, "day": c} for a, b, c in product(grid["ema1"], grid["ema2"], grid["day"]) if a < b]


def _neighbors(method, line):
    keys = ("n", "k", "d") if method == "kd" else ("ema1", "ema2", "day")
    ranges = [range(max(2, line[key] - FINE_RADIUS), line[key] + FINE_RADIUS + 1) for key in keys]
    lines = [dict(zip(keys, values)) for values in product(*ranges)]
    return [item for item in lines if method == "kd" or item["ema1"] < item["ema2"]]


def _line_key(line):
    return tuple(sorted(line.items()))


class CountCache:
    """Per-strip up/down crossing counts for every evaluated (line, gate) on every image of one w."""

    def __init__(self, method, signals_all, lengths_all, n_images, n_strips):
        self.method, self.signals, self.lengths = method, signals_all, lengths_all
        self.n_images, self.n_strips = n_images, n_strips
        self.gates = KD_GATES if method == "kd" else MACD_GATES
        self.lines = {}

    def add(self, line):
        key = _line_key(line)
        if key in self.lines:
            return
        fast, slow = method_lines(self.method, self.signals, line)
        up, down = crossings(fast, slow, self.lengths)
        entry = {}
        for gate in self.gates:
            up_ok, down_ok = gate_masks(self.method, gate, fast, self.lengths)
            shape = (self.n_images, self.n_strips)
            entry[gate] = ((up & up_ok).sum(axis=1).reshape(shape).astype(np.int32),
                           (down & down_ok).sum(axis=1).reshape(shape).astype(np.int32))
        self.lines[key] = entry

    def strip_counts(self, line_key, gate, direction):
        up, down = self.lines[line_key][gate]
        return {"up": up, "down": down, "both": (up + down) / 2.0}[direction].astype(np.float64)

    def variants(self):
        """Yield (variant, image_predictions[n_images])."""
        for key in self.lines:
            for gate in self.gates:
                for direction in DIRECTIONS:
                    counts = self.strip_counts(key, gate, direction)
                    yield (key, gate, direction), np.rint(np.median(counts, axis=1))


def _rank(predictions, truth, train):
    errors = np.abs(predictions[:, train] - truth[train])
    mae = errors.mean(axis=1)
    within = (errors <= 1).mean(axis=1)
    order = np.lexsort((np.arange(len(mae)), -within, mae))
    return int(order[0]), float(mae[order[0]]), float(within[order[0]])


def _variant_dict(method, variant):
    key, gate, direction = variant
    params = dict(key)
    params["gate"] = gate
    params["direction"] = direction
    return params


# ---------------------------------------------------------------- features / ML
def fft_counts(signals, lengths):
    counts = np.zeros(len(lengths))
    for row, length in enumerate(lengths):
        values = signals[row, :length] - signals[row, :length].mean()
        spectrum = np.abs(np.fft.rfft(values))
        cycles = np.arange(spectrum.size)
        allowed = (cycles > 0) & (length / np.maximum(cycles, 1) >= FFT_PERIOD_RANGE_PX[0]) & \
                  (length / np.maximum(cycles, 1) <= FFT_PERIOD_RANGE_PX[1])
        counts[row] = cycles[allowed][np.argmax(spectrum[allowed])] if allowed.any() else 0
    return counts


def signal_stds(signals, lengths):
    return np.array([signals[row, :length].std() for row, length in enumerate(lengths)])


def feature_matrix(kd_counts, macd_counts, kd_raw, macd_raw, lengths, fft, stds, width):
    return np.column_stack([kd_counts, macd_counts, kd_raw, macd_raw, lengths, fft, stds,
                            np.full(len(lengths), width)]).astype(np.float64)


def _xgb():
    import xgboost  # imported lazily so the classical pipeline works without it
    return xgboost


# ---------------------------------------------------------------- calibrate (counting parameters)
def _load_scenario_signals(scenario_dir, angle, layouts, images):
    data = {w: {"signals": [], "lengths": [], "extents": []} for w in layouts}
    for path in images:
        corrected = rotate_image(read_gray(path), angle)
        for w, layout in layouts.items():
            extent = stack_extent(corrected, layout)
            signals, lengths, _ = strip_signals(corrected, layout, extent)
            data[w]["signals"].append(signals)
            data[w]["lengths"].append(lengths)
            data[w]["extents"].append(extent)
    return data


def _procedure(caches, truth, train):
    """Parameter selection procedure on a training subset: returns (w, kd_index, macd_index, scores)."""
    best = None
    scores = {}
    for w, per_method in caches.items():
        picks = {}
        for method in ("kd", "macd"):
            variants, predictions = per_method[method]["variants"], per_method[method]["predictions"]
            index, mae, within = _rank(predictions, truth, train)
            picks[method] = (index, mae, within)
        scores[w] = picks
        score = min(picks["kd"][1], picks["macd"][1])
        if best is None or score < best[0]:
            best = (score, w)
    w = best[1]
    return w, scores[w]["kd"][0], scores[w]["macd"][0], scores


def _features_for(caches, data, w, kd_index, macd_index, image_index):
    kd_variant = caches[w]["kd"]["variants"][kd_index]
    macd_variant = caches[w]["macd"]["variants"][macd_index]
    kd_cache, macd_cache = caches[w]["kd"]["cache"], caches[w]["macd"]["cache"]
    kd = kd_cache.strip_counts(*kd_variant)[image_index]
    macd = macd_cache.strip_counts(*macd_variant)[image_index]
    kd_raw = kd_cache.strip_counts(kd_variant[0], KD_GATES[0], "both")[image_index]
    macd_raw = macd_cache.strip_counts(macd_variant[0], MACD_GATES[0], "both")[image_index]
    extra = data[w]["extra"][image_index]
    return feature_matrix(kd, macd, kd_raw, macd_raw, data[w]["lengths"][image_index], extra["fft"], extra["std"], w)


def calibrate(scenario_dir, output_root, reference=None, widths=STRIP_WIDTHS, step=STRIP_STEP,
              side_margin=STRIP_SIDE_MARGIN):
    scenario_dir = Path(scenario_dir).resolve()
    out_dir = Path(output_root).resolve() / scenario_dir.name
    before = tree_signatures(scenario_dir)
    rotation, rotated_mask = calibrate_rotation(scenario_dir, out_dir, reference)
    angle = rotation["compensation_angle_deg"]
    images = [p for p in list_images(scenario_dir) if parse_ground_truth(p.name)]
    if len(images) < 2:
        raise ValueError(f"{scenario_dir.name}: at least two images with numeric names are required")
    truth = np.array([parse_ground_truth(p.name)["count"] for p in images], dtype=np.float64)
    layouts = {w: strip_layout(rotated_mask, w, step=step, side_margin=side_margin) for w in widths}
    data = _load_scenario_signals(scenario_dir, angle, layouts, images)
    caches = {}
    for w, layout in layouts.items():
        stacked = np.vstack(data[w]["signals"])
        lengths_all = np.concatenate(data[w]["lengths"])
        data[w]["extra"] = [{"fft": fft_counts(s, n), "std": signal_stds(s, n)}
                            for s, n in zip(data[w]["signals"], data[w]["lengths"])]
        caches[w] = {}
        for method, grid in (("kd", KD_COARSE), ("macd", MACD_COARSE)):
            cache = CountCache(method, stacked, lengths_all, len(images), layout["count"])
            for line in _lines(method, grid):
                cache.add(line)
            # coarse -> fine: refine integer periods around the best coarse line (all images)
            variants, predictions = zip(*cache.variants())
            index, _, _ = _rank(np.array(predictions), truth, np.arange(len(images)))
            for line in _neighbors(method, dict(variants[index][0])):
                cache.add(line)
            variants, predictions = zip(*cache.variants())
            caches[w][method] = {"cache": cache, "variants": list(variants), "predictions": np.array(predictions)}
        data[w]["signals"] = None  # release memory; per-strip counts are cached

    everyone = np.arange(len(images))
    w_star, kd_index, macd_index, scores = _procedure(caches, truth, everyone)

    # ---- leave-one-image-out evaluation of the whole selection procedure
    folds = []
    loo = {"kd": np.zeros(len(images)), "macd": np.zeros(len(images)), "ml": np.full(len(images), np.nan)}
    ml_enabled = len(images) >= ML_MIN_IMAGES
    for held in range(len(images)):
        train = np.delete(everyone, held)
        w_i, kd_i, macd_i, _ = _procedure(caches, truth, train)
        loo["kd"][held] = caches[w_i]["kd"]["predictions"][kd_i, held]
        loo["macd"][held] = caches[w_i]["macd"]["predictions"][macd_i, held]
        fold = {"held_out": images[held].name, "truth": truth[held], "w": w_i,
                "kd": _variant_dict("kd", caches[w_i]["kd"]["variants"][kd_i]),
                "macd": _variant_dict("macd", caches[w_i]["macd"]["variants"][macd_i]),
                "kd_pred": loo["kd"][held], "macd_pred": loo["macd"][held]}
        if ml_enabled:
            xgboost = _xgb()
            features = [_features_for(caches, data, w_i, kd_i, macd_i, j) for j in everyone]
            x_train = np.vstack([features[j] for j in train])
            y_train = np.concatenate([np.full(len(features[j]), truth[j]) for j in train])
            model = xgboost.XGBRegressor(**XGB_PARAMS).fit(x_train, y_train)
            loo["ml"][held] = image_prediction(model.predict(features[held]))
            fold["ml_pred"] = loo["ml"][held]
        folds.append(fold)

    metrics = {}
    for method, predictions in loo.items():
        if np.isnan(predictions).any():
            continue
        errors = np.abs(predictions - truth)
        metrics[method] = {"loo_mae": float(errors.mean()), "loo_within_1": float((errors <= 1).mean()),
                           "loo_max_abs_error": float(errors.max())}

    # ---- production model on all images
    ml_info = {"eligible": ml_enabled, "adopted": False, "min_images": ML_MIN_IMAGES, "features": list(ML_FEATURES),
               "xgb_params": XGB_PARAMS, "model_path": None, "loo_mae": metrics.get("ml", {}).get("loo_mae")}
    if ml_enabled:
        xgboost = _xgb()
        features = [_features_for(caches, data, w_star, kd_index, macd_index, j) for j in everyone]
        model = xgboost.XGBRegressor(**XGB_PARAMS).fit(
            np.vstack(features), np.concatenate([np.full(len(f), truth[j]) for j, f in enumerate(features)]))
        model_path = out_dir / "model_xgb.json"
        model.save_model(str(model_path))
        ml_info["model_path"] = str(model_path)
        best_classical = min(metrics["kd"]["loo_mae"], metrics["macd"]["loo_mae"])
        ml_info["adopted"] = metrics["ml"]["loo_mae"] < best_classical
    candidates = ["kd", "macd"] + (["ml"] if ml_info["adopted"] else [])
    final_method = min(candidates, key=lambda m: (metrics[m]["loo_mae"], -metrics[m]["loo_within_1"], candidates.index(m)))

    layout = layouts[w_star]
    write_json(out_dir / "strip_layout.json", {"scenario": scenario_dir.name, "rotated_mask": rotation["mask"]["rotated"], **layout})
    save_layout_diagnostic(rotated_mask, layout, out_dir / "strip_layout.png")
    kd_params = _variant_dict("kd", caches[w_star]["kd"]["variants"][kd_index])
    macd_params = _variant_dict("macd", caches[w_star]["macd"]["variants"][macd_index])
    train_mae = {m: scores[w_star][m][1] for m in ("kd", "macd")}
    parameters = {
        "scenario": scenario_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "rotation_parameters": str((out_dir / "rotation_parameters.json").resolve()),
        "strip_layout": str((out_dir / "strip_layout.json").resolve()),
        "strip_width_px": w_star, "strip_count": layout["count"], "strip_step_px": layout["step"],
        "strip_side_margin_px": side_margin,
        "kd": {**kd_params, "units": "px", "train_mae": train_mae["kd"]},
        "macd": {**macd_params, "units": "px", "train_mae": train_mae["macd"]},
        "ml": ml_info,
        "final_method": final_method,
        "selection_basis": "lowest leave-one-image-out MAE (ties: higher within-1 rate, then kd, macd, ml)",
        "loo_metrics": metrics,
        "image_aggregation": "median of per-strip counts, rounded to nearest integer",
        "ground_truth_rule": "file name number = sheet count; _N or -N suffix = N-th group of that count",
        "calibration_images": [p.name for p in images],
        "assumptions": {
            "strip_widths_px": list(widths), "strip_margin_px": STRIP_MARGIN,
            "strip_side_margin_px": side_margin,
            "strip_min_length_ratio": STRIP_MIN_LENGTH_RATIO,
            "strip_step": f"D = {step} px" if step else "D = w (adjacent, no gap)",
            "strip_count_rule": "N = all pitch-D positions whose strip plus side margin fits in the steel region",
            "signal": "per-row mean intensity over the strip width, steel rows only",
            "kd": "RSV=(s-min_n)/(max_n-min_n)*100 trailing window; K=K+(RSV-K)/k; D=D+(K-D)/d; K0=D0=50",
            "macd": "DIF=EMA(ema1)-EMA(ema2); DEA=EMA(DIF, day); EMA alpha=2/(span+1)",
            "kd_gate": "up-cross counted if K<=100-gate, down-cross if K>=gate",
            "macd_gate": "up-cross counted if DIF<=gate*std(DIF), down-cross if DIF>=-gate*std(DIF); null = no gate",
            "direction": "up = K/DIF crosses above D/DEA, down = crosses below, both = (up+down)/2",
            "kd_coarse_grid": KD_COARSE, "macd_coarse_grid": MACD_COARSE, "kd_gates": KD_GATES,
            "macd_gates": MACD_GATES, "fine_radius": FINE_RADIUS,
        },
    }
    write_json(out_dir / "counting_parameters.json", parameters)
    per_w = {str(w): {m: {"best": _variant_dict(m, caches[w][m]["variants"][scores[w][m][0]]),
                          "train_mae": scores[w][m][1], "train_within_1": scores[w][m][2],
                          "evaluated_variants": len(caches[w][m]["variants"])} for m in ("kd", "macd")}
             for w in caches}
    write_json(out_dir / "calibration_report.json", {
        "scenario": scenario_dir.name, "images": [p.name for p in images], "truth": truth,
        "per_strip_width": per_w, "selected_w": w_star, "loo_folds": folds, "loo_metrics": metrics,
        "final_method": final_method,
        "leakage_note": "LOO selection excludes the held-out image; the fine-grid candidate pool was built "
                        "around the all-image coarse optimum, so the pool itself is not fold-specific.",
    })
    if tree_signatures(scenario_dir) != before:
        raise RuntimeError(f"{scenario_dir.name}: source data changed during calibration")
    return parameters


# ---------------------------------------------------------------- analyze
def save_layout_diagnostic(rotated_mask, layout, path):
    canvas = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
    for strip in layout["strips"]:
        cv2.rectangle(canvas, (strip["x"], strip["y0"]), (strip["x"] + strip["width"] - 1, strip["y1"] - 1), (0, 0, 255), 1)
    cv2.putText(canvas, f"N={layout['count']} w={layout['width']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 160, 255), 4)
    write_image(path, canvas)


def save_analysis_figure(corrected, layout, signals, lengths, starts, extent, marks, per_strip, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    count = layout["count"]
    picks = sorted(set(np.linspace(0, count - 1, min(PLOT_STRIPS, count)).round().astype(int).tolist()))
    figure = plt.figure(figsize=(4 + 2.2 * len(picks), 8), constrained_layout=True)
    grid = figure.add_gridspec(1, len(picks) + 2)
    axis = figure.add_subplot(grid[0, :2])
    axis.imshow(corrected, cmap="gray", vmin=0, vmax=255)
    for index, strip in enumerate(layout["strips"]):
        chosen = index in picks
        axis.add_patch(Rectangle((strip["x"], starts[index]), strip["width"], lengths[index],
                                 fill=False, edgecolor="red" if chosen else "yellow", linewidth=1.2 if chosen else 0.3))
        if chosen:
            axis.text(strip["x"], starts[index] - 8, f"N{index + 1}", color="red", fontsize=7)
    axis.axhline(extent["y0"], color="cyan", linewidth=0.6)
    axis.axhline(extent["y1"], color="cyan", linewidth=0.6)
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    up, down = marks
    for column, index in enumerate(picks, start=2):
        sub = figure.add_subplot(grid[0, column])
        length = lengths[index]
        values = signals[index, :length]
        heights = np.arange(length) + starts[index]
        sub.plot(values, heights, color="black", linewidth=0.6)
        positions_down = np.flatnonzero(down[index]) + 1
        positions_up = np.flatnonzero(up[index]) + 1
        sub.scatter(values[positions_down], heights[positions_down], s=10, facecolors="none", edgecolors="red", linewidths=0.6)
        sub.scatter(values[positions_up], heights[positions_up], s=10, facecolors="none", edgecolors="blue", linewidths=0.6)
        sub.invert_yaxis()
        sub.set_title(f"N{index + 1}_pred={per_strip[index]:g}", fontsize=8)
        sub.set_xlabel("sum (row mean)", fontsize=7)
        sub.tick_params(labelsize=6)
        if column == 2:
            sub.set_ylabel("h (px)", fontsize=7)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def write_signals_csv(path, layout, signals, lengths, starts, height):
    table = np.full((height, layout["count"]), np.nan)
    for index in range(layout["count"]):
        table[starts[index]:starts[index] + lengths[index], index] = signals[index, :lengths[index]]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y"] + [f"strip_{i + 1}_x{s['x']}" for i, s in enumerate(layout["strips"])])
        for y in range(height):
            row = table[y]
            if np.isnan(row).all():
                continue
            writer.writerow([y] + ["" if np.isnan(v) else f"{v:.3f}" for v in row])


def analyze_image(path, out_dir, rotation, layout, parameters, model, loo_lookup, truth_from_name=True):
    image_dir = out_dir / path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    sha_before = file_sha256(path)
    gray = read_gray(path)
    corrected = rotate_image(gray, rotation["compensation_angle_deg"])
    write_image(image_dir / "corrected.png", corrected)
    extent = stack_extent(corrected, layout)
    signals, lengths, starts = strip_signals(corrected, layout, extent)
    results, marks = {}, {}
    for method in ("kd", "macd"):
        params = parameters[method]
        line = {k: params[k] for k in (("n", "k", "d") if method == "kd" else ("ema1", "ema2", "day"))}
        counts, up, down = count_strips(method, signals, lengths, line, params["gate"], params["direction"])
        raw, _, _ = count_strips(method, signals, lengths, line, KD_GATES[0] if method == "kd" else MACD_GATES[0], "both")
        results[method] = {"per_strip": counts, "raw": raw, "median": image_prediction(counts)}
        marks[method] = (up, down)
    ml = None
    if model is not None:
        features = feature_matrix(results["kd"]["per_strip"], results["macd"]["per_strip"], results["kd"]["raw"],
                                  results["macd"]["raw"], lengths, fft_counts(signals, lengths),
                                  signal_stds(signals, lengths), layout["width"])
        per_strip_ml = model.predict(features).astype(np.float64)
        ml = {"per_strip": per_strip_ml, "median": image_prediction(per_strip_ml)}
    final_method = parameters["final_method"]
    final_count = (ml if final_method == "ml" else results[final_method])["median"]
    plot_method = final_method if final_method != "ml" else min(
        ("kd", "macd"), key=lambda m: parameters["loo_metrics"][m]["loo_mae"])
    truth = parse_ground_truth(path.name) if truth_from_name else None
    title = (f"{path.name}  final={final_count} ({final_method})"
             + (f"  GT={truth['count']}" if truth else "") + f"  marks: {plot_method}")
    save_analysis_figure(corrected, layout, signals, lengths, starts, extent, marks[plot_method],
                         results[plot_method]["per_strip"], title, image_dir / "analysis.png")
    write_signals_csv(image_dir / "signals.csv", layout, signals, lengths, starts, corrected.shape[0])
    per_strip = []
    for index, strip in enumerate(layout["strips"]):
        item = {"strip": index + 1, "x": strip["x"], "y0": int(starts[index]), "y1": int(starts[index] + lengths[index]),
                "kd_count": float(results["kd"]["per_strip"][index]), "macd_count": float(results["macd"]["per_strip"][index])}
        if ml:
            item["ml_pred"] = float(ml["per_strip"][index])
        per_strip.append(item)
    sha_after = file_sha256(path)
    result = {
        "status": "ok",
        "scenario": parameters["scenario"],
        "image": path.name,
        "input_path": str(path.resolve()),
        "source": {"sha256_before": sha_before, "sha256_after": sha_after, "unchanged": sha_before == sha_after},
        "ground_truth": truth,
        "rotation": {"compensation_angle_deg": rotation["compensation_angle_deg"],
                     "reference_image": rotation["reference_image"], "parameters_file": parameters["rotation_parameters"],
                     "rotation": rotation["rotation"]},
        "mask": {"rotated_mask": rotation["mask"]["rotated"], "validity": rotation["mask"]["validity"]},
        "strips": {"width_px": layout["width"], "step_px": layout["step"], "count": layout["count"],
                   "layout_file": parameters["strip_layout"]},
        "stack_extent": extent,
        "parameters": {"kd": parameters["kd"], "macd": parameters["macd"],
                       "ml": {k: parameters["ml"][k] for k in ("eligible", "adopted", "model_path", "features", "xgb_params")}},
        "predictions": {
            "per_strip": per_strip,
            "kd_median": results["kd"]["median"], "macd_median": results["macd"]["median"],
            "ml_median": ml["median"] if ml else None,
        },
        "final": {"count": final_count, "method": final_method, "selection_basis": parameters["selection_basis"],
                  "method_loo_mae": parameters["loo_metrics"].get(final_method, {}).get("loo_mae")},
        "error": final_count - truth["count"] if truth else None,
        "calibration": {"used_in_calibration": path.name in loo_lookup,
                        "note": ("in-sample prediction; see loo_prediction for the held-out estimate"
                                 if path.name in loo_lookup else "image was not part of the calibration set"),
                        "loo_prediction": loo_lookup.get(path.name)},
        "outputs": [str((image_dir / name).resolve()) for name in ("corrected.png", "analysis.png", "signals.csv", "result.json")],
    }
    write_json(image_dir / "result.json", result)
    return result


def load_calibration(params_dir):
    """Load the scenario's calibrated parameters (written by `calibrate`)."""
    params_dir = Path(params_dir).resolve()
    missing = [name for name in ("rotation_parameters.json", "strip_layout.json", "counting_parameters.json",
                                 "calibration_report.json") if not (params_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{params_dir} is not a calibrated scenario folder; missing {missing}. "
                                f"Run: escount legacy-count calibrate --scenario Data/<scenario>")
    rotation = read_json(params_dir / "rotation_parameters.json")
    layout = read_json(params_dir / "strip_layout.json")
    parameters = read_json(params_dir / "counting_parameters.json")
    report = read_json(params_dir / "calibration_report.json")
    loo_lookup = {fold["held_out"]: {k: fold.get(k) for k in ("kd_pred", "macd_pred", "ml_pred")} for fold in report["loo_folds"]}
    model = None
    if parameters["ml"]["model_path"]:
        model = _xgb().XGBRegressor()
        model.load_model(parameters["ml"]["model_path"])
    return rotation, layout, parameters, model, loo_lookup


def analyze(scenario_dir, output_root, names=None):
    scenario_dir = Path(scenario_dir).resolve()
    out_dir = Path(output_root).resolve() / scenario_dir.name
    rotation, layout, parameters, model, loo_lookup = load_calibration(out_dir)
    paths = list_images(scenario_dir)
    if names:
        paths = [p for p in paths if p.name in set(names)]
    before = tree_signatures(scenario_dir)
    results = [analyze_image(p, out_dir, rotation, layout, parameters, model, loo_lookup) for p in paths]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "ground_truth", "kd", "macd", "ml", "final", "method", "error",
                         "loo_kd", "loo_macd", "loo_ml"])
        for item in results:
            loo = item["calibration"]["loo_prediction"] or {}
            writer.writerow([item["image"], item["ground_truth"]["count"] if item["ground_truth"] else "",
                             item["predictions"]["kd_median"], item["predictions"]["macd_median"],
                             "" if item["predictions"]["ml_median"] is None else item["predictions"]["ml_median"],
                             item["final"]["count"], item["final"]["method"],
                             "" if item["error"] is None else item["error"],
                             loo.get("kd_pred", ""), loo.get("macd_pred", ""),
                             "" if loo.get("ml_pred") is None else loo.get("ml_pred")])
    if tree_signatures(scenario_dir) != before:
        raise RuntimeError(f"{scenario_dir.name}: source data changed during analysis")
    return results


def _collect_inputs(inputs):
    paths = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            paths.extend(list_images(item))
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(f"input not found: {item}")
    stems = [p.stem for p in paths]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        raise ValueError(f"inputs share file names (output folders would collide): {duplicates}")
    return [p.resolve() for p in paths]


def _check_compatible(path, rotation):
    """Reject images that cannot belong to the calibrated scenario."""
    calibrated_mask = Path(rotation["mask"]["source"])
    shape = cv2.imread(str(calibrated_mask), cv2.IMREAD_UNCHANGED).shape[:2]
    gray = read_gray(path)
    if gray.shape != shape:
        raise ValueError(f"image size {gray.shape} differs from the calibrated scenario {shape}")
    local_mask = path.parent / MASK_NAME
    if local_mask.is_file() and file_sha256(local_mask) != file_sha256(calibrated_mask):
        raise ValueError(f"{local_mask} differs from the calibrated scenario mask {calibrated_mask}; "
                         f"calibrate that scenario separately")


def predict(params_dir, inputs, output_dir, truth_from_name=False):
    """Predict sheet counts for any image file names using an already calibrated scenario."""
    rotation, layout, parameters, model, _ = load_calibration(params_dir)
    output_dir = Path(output_dir).resolve()
    paths = _collect_inputs(inputs)
    if not paths:
        raise ValueError("no input images found")
    results = []
    for path in paths:
        try:
            _check_compatible(path, rotation)
            item = analyze_image(path, output_dir, rotation, layout, parameters, model, {}, truth_from_name)
            item["params_dir"] = str(Path(params_dir).resolve())
            write_json(output_dir / path.stem / "result.json", item)
        except (IOError, OSError, ValueError) as error:
            item = {"status": "failure", "image": path.name, "input_path": str(path), "reason": str(error)}
            write_json(output_dir / path.stem / "result.json", item)
        results.append(item)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "status", "final", "method", "kd", "macd", "ml", "ground_truth", "error",
                         "input_path", "reason"])
        for item in results:
            ok = item["status"] == "ok"
            writer.writerow([item["image"], item["status"],
                             item["final"]["count"] if ok else "", item["final"]["method"] if ok else "",
                             item["predictions"]["kd_median"] if ok else "", item["predictions"]["macd_median"] if ok else "",
                             "" if not ok or item["predictions"]["ml_median"] is None else item["predictions"]["ml_median"],
                             item["ground_truth"]["count"] if ok and item["ground_truth"] else "",
                             "" if not ok or item["error"] is None else item["error"],
                             item["input_path"], item.get("reason", "")])
    return results


def scenario_dirs(data_root):
    return [p for p in sorted(Path(data_root).iterdir()) if p.is_dir() and (p / MASK_NAME).is_file()]


def run_all(data_root, output_root, **layout_options):
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    before = tree_signatures(data_root)
    summary = {"data_root": str(data_root), "output_root": str(output_root), "scenarios": []}
    for scenario in scenario_dirs(data_root):
        entry = {"scenario": scenario.name}
        try:
            parameters = calibrate(scenario, output_root, **layout_options)
            results = analyze(scenario, output_root)
            errors = [abs(r["error"]) for r in results if r["error"] is not None]
            entry.update({"status": "ok", "final_method": parameters["final_method"], "strip_width_px": parameters["strip_width_px"],
                          "strip_count": parameters["strip_count"], "loo_metrics": parameters["loo_metrics"],
                          "images": len(results), "in_sample_mae": float(np.mean(errors)) if errors else None})
        except (IOError, OSError, ValueError, RuntimeError) as error:
            entry.update({"status": "failed", "reason": str(error)})
        summary["scenarios"].append(entry)
        print(json.dumps(entry, ensure_ascii=False, default=_json_default))
    summary["data_unchanged"] = tree_signatures(data_root) == before
    write_json(output_root / "run_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    def add_layout_options(command):
        command.add_argument("--widths", nargs="+", type=int, default=list(STRIP_WIDTHS),
                             help=f"strip width w candidates in px; calibration picks the best (default {list(STRIP_WIDTHS)})")
        command.add_argument("--step", type=int, default=STRIP_STEP,
                             help="strip pitch D in px, left edge to left edge (default: D = w, no gap)")
        command.add_argument("--side-margin", type=int, default=STRIP_SIDE_MARGIN,
                             help=f"px kept between strips and the left/right edge of the steel region (default {STRIP_SIDE_MARGIN})")

    for name in ("calibrate", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--scenario", required=True, type=Path)
        command.add_argument("--output-root", default=Path("outputs/es_counting"), type=Path)
        if name == "calibrate":
            command.add_argument("--reference", help="rotation reference file name (default: largest number)")
            add_layout_options(command)
        else:
            command.add_argument("--images", nargs="*", help="subset of file names to analyze")
    command = sub.add_parser("run-all")
    command.add_argument("--data-root", default=Path("Data"), type=Path)
    command.add_argument("--output-root", default=Path("outputs/es_counting"), type=Path)
    add_layout_options(command)
    command = sub.add_parser("predict", help="predict counts for any file names with a calibrated scenario")
    command.add_argument("--params", required=True, type=Path, help="calibrated folder, e.g. outputs/es_counting/T-type")
    command.add_argument("--input", required=True, nargs="+", type=Path, help="image files and/or folders")
    command.add_argument("--output", required=True, type=Path, help="output folder for this prediction batch")
    command.add_argument("--ground-truth-from-name", action="store_true",
                         help="parse the sheet count from file names (e.g. 096.png) to report the error")
    args = parser.parse_args()
    if args.command == "predict":
        for item in predict(args.params, args.input, args.output, args.ground_truth_from_name):
            if item["status"] == "ok":
                print(item["image"], "final =", item["final"]["count"], f"({item['final']['method']})")
            else:
                print(item["image"], "FAILED:", item["reason"])
        print("summary:", (args.output / "predictions.csv").resolve())
        return
    if args.command == "calibrate":
        result = calibrate(args.scenario, args.output_root, args.reference,
                           widths=tuple(args.widths), step=args.step, side_margin=args.side_margin)
        print(json.dumps({k: result[k] for k in ("scenario", "strip_width_px", "strip_count", "kd", "macd", "final_method", "loo_metrics")},
                         ensure_ascii=False, indent=2, default=_json_default))
    elif args.command == "analyze":
        for item in analyze(args.scenario, args.output_root, args.images):
            print(item["image"], item["final"]["count"], item["final"]["method"], item["error"])
    else:
        run_all(args.data_root, args.output_root,
                widths=tuple(args.widths), step=args.step, side_margin=args.side_margin)


if __name__ == "__main__":
    main()
