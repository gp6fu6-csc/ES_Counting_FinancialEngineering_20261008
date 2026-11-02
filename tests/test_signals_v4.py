import inspect
import unittest

import numpy as np

from escount.calibration.rotation_correction import normalize_mask
from escount.signals import projection
from escount.signals.projection import project
from escount.signals.roi_bank import RoiSlot, build_roi_bank, stack_rows


def slanted_mask(height=600, width=700, left=150, right=520, slope=0.12):
    mask = np.full((height, width), 255, dtype=np.uint8)  # white = invalid
    for y in range(height):
        shift = int(round(slope * y))
        mask[y, left + shift:right + shift] = 0          # black = valid
    mask[:30] = 255
    mask[-20:] = 255
    return mask


def row_margins(valid, slot):
    """Per-row distance from the strip to the nearest invalid pixel (or image border) on each side."""
    margins = []
    for y in range(slot.y0, slot.y1):
        row = valid[y]
        left_invalid = np.flatnonzero(~row[:slot.x])
        right_invalid = np.flatnonzero(~row[slot.x + slot.w:])
        left = slot.x - (left_invalid[-1] + 1) if left_invalid.size else slot.x
        right = right_invalid[0] if right_invalid.size else row.size - slot.x - slot.w
        margins.append(min(left, right))
    return margins


def sine_image(period, height=600, width=700, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.arange(height, dtype=np.float64)[:, None]
    image = 128 + 60 * np.sin(2 * np.pi * rows / period) + rng.normal(0, 4, (height, width))
    return np.clip(image, 0, 255).astype(np.uint8)


def dominant_period_fft(signal, min_px=6, max_px=60):
    values = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(values))
    cycles = np.arange(spectrum.size)
    allowed = (cycles > 0) & (len(values) / np.maximum(cycles, 1) >= min_px) & \
              (len(values) / np.maximum(cycles, 1) <= max_px)
    return len(values) / cycles[allowed][np.argmax(spectrum[allowed])]


class RoiBankGeometryTests(unittest.TestCase):
    def test_overlap_of_a_third_or_more_raises(self):
        mask = np.zeros((300, 400), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_roi_bank(mask, widths=[30], pitch=20)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_roi_bank(mask, widths=[30], pitch=19)
        self.assertTrue(build_roi_bank(mask, widths=[30], pitch=21))  # overlap 9 px < 10 px

    def test_side_margin_below_50_raises(self):
        mask = np.zeros((300, 400), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_roi_bank(mask, widths=[10], pitch=10, side_margin=49)

    def test_every_row_keeps_50_px_on_slanted_mask(self):
        mask = slanted_mask()
        valid = normalize_mask(mask)
        for x_offset in (0, 7):
            slots = build_roi_bank(mask, widths=range(6, 41, 2), pitch=30, x_offset=x_offset)
            self.assertGreater(len(slots), 0)
            self.assertEqual({s.w for s in slots}, set(range(6, 41, 2)))
            for slot in slots:
                self.assertGreaterEqual(min(row_margins(valid, slot)), 50, slot)
                self.assertEqual(slot.x_offset, x_offset)

    def test_x_offset_shifts_the_grid(self):
        mask = np.zeros((200, 500), dtype=np.uint8)
        base = [s.x for s in build_roi_bank(mask, widths=[12], pitch=20)]
        shifted = [s.x for s in build_roi_bank(mask, widths=[12], pitch=20, x_offset=5)]
        self.assertEqual(base[0], 50)
        self.assertEqual(shifted[0], 55)
        self.assertEqual(np.diff(base).tolist(), [20] * (len(base) - 1))

    def test_stack_rows_slots_span_whole_stack(self):
        mask = np.zeros((600, 400), dtype=np.uint8)
        image = np.full((600, 400), 12, dtype=np.uint8)
        rows = np.arange(150, 500)
        image[150:500] = np.where((rows - 150) % 12 < 8, 170, 40)[:, None]
        slots = build_roi_bank(mask, widths=[10, 20], pitch=30)
        extent, covering = stack_rows(image, slots)
        self.assertLessEqual(abs(extent["y0"] - 150), 25)
        self.assertLessEqual(abs(extent["y1"] - 500), 25)
        self.assertEqual(len(covering), len(slots))
        self.assertTrue(all((s.y0, s.y1) == (extent["y0"], extent["y1"]) for s in covering))
        # a fixture interrupting the valid rows of the right-hand strips inside the stack drops them
        mask[300:310, 250:] = 255
        slots = build_roi_bank(mask, widths=[10], pitch=30)
        extent, covering = stack_rows(image, slots)
        self.assertTrue(all(s.x + s.w + 50 <= 250 for s in covering))
        self.assertGreater(len(covering), 0)

    def test_stack_rows_full_span_preserves_valid_boundaries(self):
        mask = np.zeros((600, 400), dtype=np.uint8)
        image = np.full((600, 400), 12, dtype=np.uint8)
        rows = np.arange(150, 500)
        image[150:500] = np.where((rows - 150) % 12 < 8, 170, 40)[:, None]
        slots = build_roi_bank(mask, widths=[10], pitch=30)
        extent, covering = stack_rows(image, slots, full_span=True)
        self.assertEqual(len(covering), len(slots))
        self.assertTrue(all(s.y0 == 0 and s.y1 == 600 for s in covering))
        self.assertGreater(extent["y0"], 100)
        self.assertLess(extent["y1"], 550)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.image = sine_image(17)
        self.slot = RoiSlot(x=200, w=24, y0=40, y1=560, pitch=30, x_offset=0)

    def test_mean_projection_equals_raw_row_mean(self):
        s = self.slot
        expected = self.image[s.y0:s.y1, s.x:s.x + s.w].mean(axis=1)
        self.assertTrue(np.array_equal(project(self.image, s, "mean"), expected))

    def test_median_and_gaussian(self):
        s = self.slot
        block = self.image[s.y0:s.y1, s.x:s.x + s.w]
        self.assertTrue(np.array_equal(project(self.image, s, "median"), np.median(block, axis=1)))
        gaussian = project(self.image, s, "gaussian", sigma=4.0)
        self.assertEqual(gaussian.shape, (s.y1 - s.y0,))
        self.assertEqual(gaussian.dtype, np.float64)
        flat = np.full_like(self.image, 77)
        self.assertTrue(np.allclose(project(flat, s, "gaussian", sigma=4.0), 77.0))
        with self.assertRaises(ValueError):
            project(self.image, s, "gaussian")

    def test_dominant_period_of_projected_sinusoid(self):
        for method, sigma in (("mean", None), ("gaussian", 5.0), ("median", None)):
            period = dominant_period_fft(project(self.image, self.slot, method, sigma))
            self.assertLessEqual(abs(period - 17), 1.0, (method, period))

    def test_no_vertical_filtering_row_impulse_preserved(self):
        image = np.zeros((100, 60), dtype=np.uint8)
        image[50] = 200
        slot = RoiSlot(x=10, w=20, y0=0, y1=100, pitch=20, x_offset=0)
        for method, sigma in (("mean", None), ("gaussian", 3.0), ("median", None)):
            signal = project(image, slot, method, sigma)
            self.assertEqual(np.flatnonzero(signal).tolist(), [50], method)
        source = inspect.getsource(projection)
        for name in ("filter1d", "convolve", "savgol", "lfilter", "uniform_filter", "GaussianBlur"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
