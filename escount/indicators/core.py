"""Indicator library (Task 4): pure functions over 1-D signals, periods in px along the stack height.

KD and MACD reuse the legacy es_counting.kd_lines / macd_lines (same lfilter recursions and initial
conditions), so results are numerically identical to the V3 pipeline.
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks, lfilter

from escount.es_counting import _ema, kd_lines, macd_lines


@dataclass(frozen=True)
class MAParams:
    n: int


@dataclass(frozen=True)
class EMAParams:
    span: float


@dataclass(frozen=True)
class MACDParams:
    ema1: int
    ema2: int
    dea: int


@dataclass(frozen=True)
class KDParams:
    n: int
    k: int
    d: int


@dataclass(frozen=True)
class BollingerParams:
    n: int
    k: float = 2.0


@dataclass(frozen=True)
class PeakParams:
    prominence: Optional[float] = None
    distance: Optional[int] = None


class MACDResult(NamedTuple):
    dif: np.ndarray
    dea: np.ndarray
    hist: np.ndarray   # dif - dea


class KDResult(NamedTuple):
    rsv: np.ndarray
    k: np.ndarray
    d: np.ndarray


class BollingerResult(NamedTuple):
    percent_b: np.ndarray   # (x - lower) / (upper - lower)
    bandwidth: np.ndarray   # (upper - lower) / middle
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


def _signal(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"expected a 1-D signal with >= 2 samples, got shape {x.shape}")
    return x


def ma(x, params):
    """Trailing simple moving average over n px (start edge-padded)."""
    x, n = _signal(x), params.n
    padded = np.pad(x, (n - 1, 0), mode="edge")
    return lfilter(np.full(n, 1.0 / n), [1.0], padded)[n - 1:]


def ema(x, params):
    """Exponential moving average, alpha = 2/(span+1), initialised at x[0] (legacy _ema)."""
    x = _signal(x)
    return _ema(x[None, :], params.span, x[:1])[0]


def macd(x, params):
    """DIF = EMA(ema1) - EMA(ema2); DEA = EMA(DIF, dea); hist = DIF - DEA (legacy macd_lines)."""
    x = _signal(x)
    dif, dea = macd_lines(x[None, :], params.ema1, params.ema2, params.dea)
    return MACDResult(dif[0], dea[0], dif[0] - dea[0])


def kd(x, params):
    """RSV over a trailing n-px window, K/D smoothed with 1/k, 1/d (legacy kd_lines)."""
    x = _signal(x)
    n, length = params.n, x.size
    padded = np.pad(x, (n - 1, 0), mode="edge")
    low = minimum_filter1d(padded, n)[n // 2:n // 2 + length]
    high = maximum_filter1d(padded, n)[n // 2:n // 2 + length]
    span = high - low
    rsv = np.divide((x - low) * 100.0, span, out=np.full_like(x, 50.0), where=span > 1e-9)
    k_line, d_line = kd_lines(x[None, :], params.n, params.k, params.d)
    return KDResult(rsv, k_line[0], d_line[0])


def bollinger(x, params):
    """Trailing n-px mean +- k * population std (start edge-padded)."""
    x, n = _signal(x), params.n
    middle = ma(x, MAParams(n))
    padded = np.pad(x, (n - 1, 0), mode="edge")
    mean_sq = lfilter(np.full(n, 1.0 / n), [1.0], padded ** 2)[n - 1:]
    std = np.sqrt(np.maximum(mean_sq - middle ** 2, 0.0))
    upper, lower = middle + params.k * std, middle - params.k * std
    width = upper - lower
    percent_b = np.divide(x - lower, width, out=np.full_like(x, 0.5), where=width > 1e-9)
    bandwidth = np.divide(width, np.abs(middle), out=np.zeros_like(x), where=np.abs(middle) > 1e-9)
    return BollingerResult(percent_b, bandwidth, middle, upper, lower)


def peaks(x, params):
    """(high_idx, low_idx) from scipy.signal.find_peaks on x and -x."""
    x = _signal(x)
    highs, _ = find_peaks(x, prominence=params.prominence, distance=params.distance)
    lows, _ = find_peaks(-x, prominence=params.prominence, distance=params.distance)
    return highs, lows


def dominant_period(x, method="fft", min_px=6, max_px=60):
    """Dominant period in px within [min_px, max_px] by FFT peak or autocorrelation peak."""
    x = _signal(x)
    values = x - x.mean()
    length = values.size
    if method == "fft":
        spectrum = np.abs(np.fft.rfft(values))
        cycles = np.arange(spectrum.size)
        periods = length / np.maximum(cycles, 1)
        allowed = (cycles > 0) & (periods >= min_px) & (periods <= max_px)
        if not allowed.any():
            raise ValueError(f"no FFT period in [{min_px}, {max_px}] px for length {length}")
        return float(periods[allowed][np.argmax(spectrum[allowed])])
    if method == "acf":
        acf = np.correlate(values, values, mode="full")[length - 1:] / length  # biased estimator
        lags = np.arange(int(np.ceil(min_px)), min(int(max_px), length - 2) + 1)
        if not lags.size:
            raise ValueError(f"no ACF lag in [{min_px}, {max_px}] px for length {length}")
        lag = int(lags[np.argmax(acf[lags])])
        left, centre, right = acf[lag - 1], acf[lag], acf[lag + 1]
        denominator = left - 2 * centre + right
        offset = 0.5 * (left - right) / denominator if denominator < 0 else 0.0
        return float(lag + offset)
    raise ValueError(f"unknown dominant_period method {method!r}; expected 'acf' or 'fft'")
