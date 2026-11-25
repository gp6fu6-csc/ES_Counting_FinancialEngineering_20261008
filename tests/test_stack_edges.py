"""Top / bottom sheet edges (PDF 20261124 p24-25, 頂-底標註樣本): edges of the first / last sheet,
not the brightest row inside the stack and not the reflective support base below it."""
import numpy as np

from escount.signals.edges import consensus_edges, stack_edges

PITCH = 10


def _sheets(first, count, height=150.0, trough=40.0):
    """Rows of a cosine sheet train: peak at first + k*PITCH, sheet edges half a pitch around it."""
    rows = np.arange(first - PITCH // 2, first + count * PITCH - PITCH // 2)
    phase = 2 * np.pi * (rows - first) / PITCH
    return rows, trough + (height - trough) * (0.5 + 0.5 * np.cos(phase))


def _signal(n=600, dark=3.0):
    return np.full(n, dark)


def test_top_is_first_sheet_upper_edge_not_its_peak():
    s = _signal()
    rows, values = _sheets(first=100, count=40)
    s[rows] = values
    top, bottom, _ = stack_edges(s)
    assert 95 <= top < 100          # upper edge of the first sheet, before its peak at row 100
    assert 490 < bottom <= 496      # lower edge of the last sheet (peak at 490)


def test_bottom_ignores_reflective_base_after_dark_gap():
    s = _signal()
    rows, values = _sheets(first=100, count=30)       # last sheet peak at 390
    s[rows] = values
    base = np.arange(420, 520)                         # dark gap 396..420, then a weakly textured base
    s[base] = 50 + 20 * np.sin(2 * np.pi * base / 13)
    s[base[::25]] += 60                                # sparse base reflections
    top, bottom, _ = stack_edges(s)
    assert 95 <= top < 100
    assert 390 < bottom <= 397


def test_top_ignores_weak_reflection_above_stack():
    s = _signal()
    s[40:60] = 3 + 25 * np.hanning(20)                 # faint, non-black glint far above the stack
    rows, values = _sheets(first=150, count=30)
    s[rows] = values
    top, _, _ = stack_edges(s)
    assert 145 <= top < 150


def test_no_sheets_returns_none():
    assert stack_edges(_signal()) is None


def test_consensus_snaps_outlier_rois():
    edges = [(100, 500), (101, 499), (99, 501), (100, 500), (300, 380)]
    assert consensus_edges(edges)[-1] == (100, 500)
    assert consensus_edges(edges)[:4] == edges[:4]
