"""Counters (Task 4): turn one 1-D strip signal into a sheet count with an indicator config.

Events: an "up" event is where the signal turns upward (golden cross / trough), a "down" event is
where it turns downward (death cross / peak). count = #up, #down or (#up + #down) / 2 for direction
up / down / both. Returned (count, high_idx, low_idx): high_idx = down-event sample indices,
low_idx = up-event sample indices (for marking highs/lows in plots).

Gates ("focus & threshold"):
  kd        gate g in [0, 50]: up-cross counted if K <= 100 - g, down-cross if K >= g (legacy KD zone gate)
  macd      gate tau or None: up-cross counted if DIF <= tau*std(DIF), down if DIF >= -tau*std (legacy)
  peaks     gate tau or None: minimum prominence = max(params.prominence, tau * std(signal))
  bollinger gate g in (0, 0.5]: hysteresis on %b, up event when %b rises above 1 - g after being
            below g, down event when it falls below g after being above 1 - g (g = 0.5: mid-band crossings)
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from escount.es_counting import count_strips
from escount.indicators.core import (
    BollingerParams, KDParams, MACDParams, PeakParams, bollinger, dominant_period, peaks,
)

NAMES = ("kd", "macd", "peaks", "bollinger")
DIRECTIONS = ("up", "down", "both")
PARAM_TYPES = {"kd": KDParams, "macd": MACDParams, "peaks": PeakParams, "bollinger": BollingerParams}


@dataclass(frozen=True)
class IndicatorConfig:
    name: str
    params: Any
    gate: Optional[float] = None
    direction: str = "both"

    def __post_init__(self):
        if self.name not in NAMES:
            raise ValueError(f"unknown indicator {self.name!r}; expected one of {NAMES}")
        if not isinstance(self.params, PARAM_TYPES[self.name]):
            raise ValueError(f"{self.name} needs {PARAM_TYPES[self.name].__name__}, got {type(self.params).__name__}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}; expected one of {DIRECTIONS}")
        if self.name == "bollinger" and not (self.gate is not None and 0 < self.gate <= 0.5):
            raise ValueError(f"bollinger gate must be in (0, 0.5], got {self.gate}")


def _combine(up_idx, down_idx, direction):
    up, down = len(up_idx), len(down_idx)
    return float({"up": up, "down": down, "both": (up + down) / 2.0}[direction])


def _crossing_events(signal, config):
    """KD / MACD crossings through the legacy count_strips (identical gating and masks)."""
    if config.name == "kd":
        line = {"n": config.params.n, "k": config.params.k, "d": config.params.d}
        gate = 0 if config.gate is None else config.gate
    else:
        line = {"ema1": config.params.ema1, "ema2": config.params.ema2, "day": config.params.dea}
        gate = config.gate
    _, up, down = count_strips(config.name, signal[None, :], np.array([signal.size]), line, gate, config.direction)
    return np.flatnonzero(up[0]) + 1, np.flatnonzero(down[0]) + 1


def _peak_events(signal, config):
    prominence = config.params.prominence
    if config.gate is not None:
        prominence = max(prominence or 0.0, config.gate * float(signal.std()))
    highs, lows = peaks(signal, PeakParams(prominence=prominence, distance=config.params.distance))
    return lows, highs


def _bollinger_events(signal, config):
    percent_b = bollinger(signal, config.params).percent_b
    low, high = config.gate, 1.0 - config.gate
    # hysteresis: +1 in the high zone, -1 in the low zone; an event is a switch between the zones
    zone = np.where(percent_b > high, 1, np.where(percent_b < low, -1, 0))
    index = np.flatnonzero(zone)
    visited = zone[index]
    switch = np.flatnonzero(visited[1:] != visited[:-1]) + 1
    events, new_zone = index[switch].astype(np.int64), visited[switch]
    return events[new_zone == 1], events[new_zone == -1]


def count(signal, config):
    """Return (count, high_idx, low_idx) for one 1-D signal."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError(f"expected a 1-D signal with >= 2 samples, got shape {signal.shape}")
    events = {"kd": _crossing_events, "macd": _crossing_events,
              "peaks": _peak_events, "bollinger": _bollinger_events}[config.name]
    up_idx, down_idx = events(signal, config)
    return _combine(up_idx, down_idx, config.direction), down_idx, up_idx


def period_count(signal, method="fft", min_px=6, max_px=60):
    """round(L / dominant period)."""
    signal = np.asarray(signal, dtype=np.float64)
    return int(round(signal.size / dominant_period(signal, method, min_px, max_px)))
