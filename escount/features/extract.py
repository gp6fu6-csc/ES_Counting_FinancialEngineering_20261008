"""Per-ROI features (Task 5).

One feature row = one (image, ROI slot, projection). Columns:
  identity    case, image, image_id, gt, gt_group, w, pitch, x_offset, x, y0, y1, projection, proj_id,
              sigma, slot_key (projection|w|x, project-level slot), pos_key (projection|w|x_norm bin)
  geometry    L (stack rows of the strip), x_norm = (x - valid_left) / valid_width
  signal      period (FFT dominant period, px), period_count = round(L / period), snr (FFT peak /
              median in-band magnitude), std, period_stability (std of the 4 quarter periods / period)
  counts      cnt_<indicator id>_<up|down|both> for every FEATURE_INDICATORS config
The projected signal is the raw width-direction projection (no vertical smoothing); the stack rows
come from es_counting.stack_extent over a fixed extent layout, so training and inference agree.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from escount.es_counting import count_strips
from escount.indicators.core import BollingerParams, KDParams, MACDParams, PeakParams, dominant_period
from escount.indicators.counters import IndicatorConfig, count
from escount.signals.projection import METHODS, project
from escount.signals.roi_bank import DEFAULT_WIDTHS, MIN_SIDE_MARGIN, build_roi_bank, check_geometry, clip_slots, stack_rows

FEATURE_VERSION = 1
PERIOD_RANGE_PX = (6, 60)
MIN_SIGNAL_PX = 4 * PERIOD_RANGE_PX[0]   # shortest strip signal kept (a 10-sheet stack is ~100 px)
EXTENT_WIDTH, EXTENT_PITCH = 16, 30      # fixed layout used only to detect the stack rows
POSITION_BIN = 0.05                      # x_norm bin of pos_key (General: position shared across projects)
DIRECTIONS = ("up", "down", "both")
PROJECTION_IDS = {name: index for index, name in enumerate(METHODS)}
ROW_COLUMNS = ["w", "pitch", "x_offset", "x", "y0", "y1", "projection", "proj_id", "sigma", "slot_key", "pos_key",
               "L", "x_norm", "period", "period_count", "snr", "std", "period_stability"]

FEATURE_INDICATORS = (
    IndicatorConfig("kd", KDParams(4, 2, 2), 20),
    IndicatorConfig("kd", KDParams(3, 5, 3), 30),
    IndicatorConfig("kd", KDParams(4, 3, 7), 40),
    IndicatorConfig("kd", KDParams(6, 3, 3), 20),
    IndicatorConfig("kd", KDParams(9, 3, 3), 0),
    IndicatorConfig("kd", KDParams(13, 5, 5), 40),
    IndicatorConfig("macd", MACDParams(2, 4, 2), None),
    IndicatorConfig("macd", MACDParams(2, 5, 3), 1.0),
    IndicatorConfig("macd", MACDParams(3, 8, 3), 0.5),
    IndicatorConfig("macd", MACDParams(4, 12, 4), 0.0),
    IndicatorConfig("macd", MACDParams(6, 16, 5), None),
    IndicatorConfig("macd", MACDParams(7, 22, 6), 0.0),
    IndicatorConfig("peaks", PeakParams(), 0.25),
    IndicatorConfig("peaks", PeakParams(), 0.5),
    IndicatorConfig("peaks", PeakParams(distance=4), 0.25),
    IndicatorConfig("peaks", PeakParams(), 1.0),
    IndicatorConfig("bollinger", BollingerParams(10), 0.5),
    IndicatorConfig("bollinger", BollingerParams(20), 0.3),
)


@dataclass(frozen=True)
class AugmentGrid:
    """Augmentation grid on the same image: strip width w, pitch D, grid shift x and projection."""
    widths: tuple = DEFAULT_WIDTHS
    pitches: tuple = (30,)
    x_offsets: tuple = (0, 15)
    projections: tuple = METHODS
    gaussian_sigma_ratio: float = 0.25   # gaussian projection sigma = ratio * w
    side_margin: int = MIN_SIDE_MARGIN

    def __post_init__(self):
        for w in self.widths:
            for pitch in self.pitches:
                check_geometry(w, pitch, self.side_margin)
        unknown = set(self.projections) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown projections {sorted(unknown)}; expected {METHODS}")

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})


def indicator_id(config):
    p, g = config.params, config.gate
    gate = "none" if g is None else f"{g:g}"
    if config.name == "kd":
        return f"kd_n{p.n}_k{p.k}_d{p.d}_g{gate}"
    if config.name == "macd":
        return f"macd_{p.ema1}_{p.ema2}_{p.dea}_g{gate}"
    if config.name == "peaks":
        return f"peaks_p{p.prominence}_d{p.distance}_g{gate}"
    return f"boll_n{p.n}_k{p.k:g}_g{gate}"


def count_columns(indicators=FEATURE_INDICATORS):
    return [f"cnt_{indicator_id(c)}_{d}" for c in indicators for d in DIRECTIONS]


def feature_config(grid, indicators=FEATURE_INDICATORS):
    """Everything that determines a feature row; its hash marks stale caches."""
    return {"feature_version": FEATURE_VERSION, "grid": grid.to_dict(),
            "indicators": [indicator_id(c) for c in indicators], "period_range_px": list(PERIOD_RANGE_PX),
            "min_signal_px": MIN_SIGNAL_PX, "extent_layout": {"width": EXTENT_WIDTH, "pitch": EXTENT_PITCH},
            "position_bin": POSITION_BIN}


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def projection_sigma(method, w, ratio):
    return max(0.5, ratio * w) if method == "gaussian" else None


# ---------------------------------------------------------------- ROI items
def bank_items(rotated_mask, grid):
    """All (RoiSlot, projection) pairs of the augmentation grid (slots not yet clipped to the stack)."""
    items = []
    for pitch in grid.pitches:
        for x_offset in grid.x_offsets:
            slots = build_roi_bank(rotated_mask, grid.widths, pitch=pitch, side_margin=grid.side_margin,
                                   x_offset=x_offset)
            items.extend((slot, method) for slot in slots for method in grid.projections)
    return items


def extent_slots(rotated_mask, side_margin=MIN_SIDE_MARGIN):
    return build_roi_bank(rotated_mask, (EXTENT_WIDTH,), pitch=EXTENT_PITCH, side_margin=side_margin)


def clip_items(items, extent):
    """Clip each item's slot to the stack extent; drop slots that do not span it or are too short."""
    kept = []
    for slot, method in items:
        clipped = clip_slots([slot], extent)
        if clipped and clipped[0].y1 - clipped[0].y0 >= MIN_SIGNAL_PX:
            kept.append((clipped[0], method))
    return kept


# ---------------------------------------------------------------- counting / stats
def batch_counts(signals, lengths, config):
    """(up, down) event counts per row of an end-padded signal matrix; equals counters.count per row."""
    if config.name in ("kd", "macd"):
        if config.name == "kd":
            line = {"n": config.params.n, "k": config.params.k, "d": config.params.d}
            gate = 0 if config.gate is None else config.gate
        else:
            line = {"ema1": config.params.ema1, "ema2": config.params.ema2, "day": config.params.dea}
            gate = config.gate
        _, up, down = count_strips(config.name, signals, lengths, line, gate, "both")
        return up.sum(axis=1).astype(np.float64), down.sum(axis=1).astype(np.float64)
    up, down = np.zeros(len(lengths)), np.zeros(len(lengths))
    for row, length in enumerate(lengths):
        _, high_idx, low_idx = count(signals[row, :length], config)
        up[row], down[row] = len(low_idx), len(high_idx)
    return up, down


def signal_stats(signal):
    """period, period_count, snr, std, period_stability of one raw strip signal."""
    low, high = PERIOD_RANGE_PX
    length = signal.size
    period = dominant_period(signal, "fft", low, high)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    cycles = np.arange(spectrum.size)
    periods = length / np.maximum(cycles, 1)
    band = spectrum[(cycles > 0) & (periods >= low) & (periods <= high)]
    snr = float(band.max() / max(float(np.median(band)), 1e-12))
    quarters = []
    for part in np.array_split(signal, 4):
        try:
            quarters.append(dominant_period(part, "fft", low, high))
        except ValueError:
            pass
    stability = float(np.std(quarters) / period) if quarters else 0.0
    return {"period": period, "period_count": int(round(length / period)), "snr": snr,
            "std": float(signal.std()), "period_stability": stability}


def roi_features(image, items, valid_x0, valid_width, sigma_ratio, indicators=FEATURE_INDICATORS):
    """Feature rows (DataFrame) for (slot, projection) items already clipped to the stack extent."""
    rows, blocks = [], []
    groups = {}
    for slot, method in items:
        groups.setdefault((slot.w, method), []).append(slot)
    for (w, method), slots in groups.items():
        sigma = projection_sigma(method, w, sigma_ratio)
        signals = [project(image, slot, method, sigma) for slot in slots]
        lengths = np.array([s.size for s in signals], dtype=np.int64)
        matrix = np.empty((len(signals), int(lengths.max())), dtype=np.float64)
        for row, values in enumerate(signals):
            matrix[row, :values.size] = values
            matrix[row, values.size:] = values[-1]   # causal filters: padding never affects valid samples
        counts = {}
        for config in indicators:
            up, down = batch_counts(matrix, lengths, config)
            name = indicator_id(config)
            counts[f"cnt_{name}_up"], counts[f"cnt_{name}_down"], counts[f"cnt_{name}_both"] = up, down, (up + down) / 2.0
        for row, (slot, values) in enumerate(zip(slots, signals)):
            x_norm = (slot.x - valid_x0) / valid_width
            pos_bin = round(x_norm / POSITION_BIN) * POSITION_BIN
            rows.append({"w": slot.w, "pitch": slot.pitch, "x_offset": slot.x_offset, "x": slot.x,
                         "y0": slot.y0, "y1": slot.y1, "projection": method, "proj_id": PROJECTION_IDS[method],
                         "sigma": np.nan if sigma is None else sigma, "slot_key": f"{method}|{slot.w}|{slot.x}",
                         "pos_key": f"{method}|{slot.w}|{pos_bin:.2f}", "L": int(values.size),
                         "x_norm": x_norm, **signal_stats(values)})
        blocks.append(pd.DataFrame(counts))
    if not rows:
        return pd.DataFrame(columns=ROW_COLUMNS + count_columns(indicators))
    return pd.concat([pd.DataFrame(rows), pd.concat(blocks, ignore_index=True)], axis=1)


def image_features(rotated_image, items, ext_slots, valid_x0, valid_width, grid, indicators=FEATURE_INDICATORS):
    """Stack extent -> clip items -> feature rows for one rotated image. Returns (DataFrame, extent)."""
    extent, _ = stack_rows(rotated_image, ext_slots)
    kept = clip_items(items, extent)
    return roi_features(rotated_image, kept, valid_x0, valid_width, grid.gaussian_sigma_ratio, indicators), extent


def load_indicator_configs(config_path="configs/indicators.json"):
    """Load indicator configurations from JSON if available, falling back to FEATURE_INDICATORS."""
    path = Path(config_path)
    if not path.is_file():
        return FEATURE_INDICATORS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = []
        for item in data.get("indicators", []):
            name = item["name"]
            params_dict = item.get("params", {})
            gate = item.get("gate")
            if name == "kd":
                params = KDParams(**params_dict)
            elif name == "macd":
                params = MACDParams(**params_dict)
            elif name == "peaks":
                params = PeakParams(**params_dict)
            elif name == "bollinger":
                params = BollingerParams(**params_dict)
            else:
                continue
            items.append(IndicatorConfig(name, params, gate))
        return tuple(items) if items else FEATURE_INDICATORS
    except Exception:
        return FEATURE_INDICATORS
