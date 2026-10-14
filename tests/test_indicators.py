import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from escount.es_counting import kd_lines, macd_lines
from escount.indicators.core import (
    BollingerParams, EMAParams, KDParams, MACDParams, MAParams, PeakParams, bollinger, dominant_period, ema, kd,
    ma, macd, peaks,
)
from escount.indicators.counters import IndicatorConfig, count, period_count

PERIOD, LENGTH = 20, 1000  # 50 full periods


def sine(period=PERIOD, length=LENGTH):
    return 100.0 + 50.0 * np.sin(2 * np.pi * np.arange(length) / period)


def noisy_signal(seed=0, length=600):
    rng = np.random.default_rng(seed)
    return 100 + 40 * np.sin(2 * np.pi * np.arange(length) / 13.0) + rng.normal(0, 8, length)


class CounterTests(unittest.TestCase):
    def test_each_counter_counts_50_period_sine_exactly(self):
        x = sine()
        configs = [
            IndicatorConfig("kd", KDParams(9, 3, 3), gate=20, direction="down"),
            IndicatorConfig("kd", KDParams(5, 2, 3), gate=None, direction="down"),
            IndicatorConfig("macd", MACDParams(3, 8, 3), gate=None, direction="down"),
            IndicatorConfig("macd", MACDParams(4, 12, 3), gate=0.5, direction="down"),
            IndicatorConfig("peaks", PeakParams(), gate=None, direction="both"),
            IndicatorConfig("peaks", PeakParams(prominence=10, distance=5), gate=0.5, direction="up"),
            IndicatorConfig("bollinger", BollingerParams(20, 2.0), gate=0.3, direction="down"),
            IndicatorConfig("bollinger", BollingerParams(10, 2.0), gate=0.5, direction="both"),
        ]
        for config in configs:
            value, high_idx, low_idx = count(x, config)
            self.assertEqual(value, 50.0, config)
            self.assertTrue(len(high_idx) and len(low_idx), config)
        self.assertEqual(period_count(x), 50)
        self.assertEqual(period_count(x, method="acf"), 50)

    def test_peak_indices_are_highs_and_lows(self):
        x = sine()
        _, high_idx, low_idx = count(x, IndicatorConfig("peaks", PeakParams(), direction="both"))
        self.assertTrue(np.all(x[high_idx] > 149.0))
        self.assertTrue(np.all(x[low_idx] < 51.0))

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            IndicatorConfig("rsi", KDParams(9, 3, 3))
        with self.assertRaises(ValueError):
            IndicatorConfig("kd", MACDParams(3, 8, 3))
        with self.assertRaises(ValueError):
            IndicatorConfig("kd", KDParams(9, 3, 3), direction="sideways")
        with self.assertRaises(ValueError):
            IndicatorConfig("bollinger", BollingerParams(20), gate=None)


class LegacyEquivalenceTests(unittest.TestCase):
    def test_kd_matches_legacy_pointwise(self):
        for seed, (n, k, d) in enumerate(((9, 3, 3), (5, 2, 5), (15, 5, 2), (3, 3, 3))):
            x = noisy_signal(seed)
            result = kd(x, KDParams(n, k, d))
            legacy_k, legacy_d = kd_lines(x[None, :], n, k, d)
            np.testing.assert_array_equal(result.k, legacy_k[0])
            np.testing.assert_array_equal(result.d, legacy_d[0])
            self.assertTrue(np.all((result.rsv >= -1e-9) & (result.rsv <= 100 + 1e-9)))

    def test_macd_matches_legacy_pointwise(self):
        for seed, (a, b, c) in enumerate(((3, 8, 3), (2, 5, 2), (6, 24, 8), (4, 12, 5))):
            x = noisy_signal(seed)
            result = macd(x, MACDParams(a, b, c))
            legacy_dif, legacy_dea = macd_lines(x[None, :], a, b, c)
            np.testing.assert_array_equal(result.dif, legacy_dif[0])
            np.testing.assert_array_equal(result.dea, legacy_dea[0])
            np.testing.assert_array_equal(result.hist, legacy_dif[0] - legacy_dea[0])


class CoreTests(unittest.TestCase):
    def test_ma_ema_bollinger_basics(self):
        x = noisy_signal()
        self.assertAlmostEqual(ma(x, MAParams(5))[100], x[96:101].mean())
        np.testing.assert_allclose(ma(np.full(50, 7.0), MAParams(9)), 7.0)
        np.testing.assert_allclose(ema(np.full(50, 7.0), EMAParams(4)), 7.0)
        band = bollinger(x, BollingerParams(20, 2.0))
        window = x[81:101]
        self.assertAlmostEqual(band.middle[100], window.mean())
        self.assertAlmostEqual(band.upper[100], window.mean() + 2 * window.std())
        self.assertEqual(band.percent_b.shape, x.shape)
        self.assertTrue(np.all(band.bandwidth >= 0))

    def test_peaks_and_dominant_period(self):
        x = sine(period=17, length=850)
        highs, lows = peaks(x, PeakParams(prominence=10))
        self.assertEqual((len(highs), len(lows)), (50, 50))
        self.assertAlmostEqual(dominant_period(x, "fft"), 17.0)
        self.assertLess(abs(dominant_period(x, "acf") - 17.0), 0.1)
        with self.assertRaises(ValueError):
            dominant_period(x, "wavelet")

    def test_params_frozen_and_hashable(self):
        params = (MAParams(5), EMAParams(3), MACDParams(3, 8, 3), KDParams(9, 3, 3), BollingerParams(20, 2.0),
                  PeakParams(5.0, 3))
        self.assertEqual(len({*params, *params}), len(params))
        config = IndicatorConfig("kd", KDParams(9, 3, 3), gate=20, direction="up")
        self.assertEqual(hash(config), hash(IndicatorConfig("kd", KDParams(9, 3, 3), gate=20, direction="up")))
        with self.assertRaises(FrozenInstanceError):
            params[3].n = 7
        with self.assertRaises(FrozenInstanceError):
            config.gate = 10


if __name__ == "__main__":
    unittest.main()
