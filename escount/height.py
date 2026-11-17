"""Pixel-to-height model (optional, one per project): <profiles_root>/<case>/height_model.json.

The user clicks the reference-scale icons (10 / 5 / 1 mm lists) on the rotated `ratio.png`, left and right
side separately. Consecutive clicks of one list are a known distance apart (10 mm list: 10 mm, 5 mm list: 5 mm,
1 mm list: the first 1 mm icon after each 5 mm icon, so 5 mm). Each pair gives the local scale (mm per px) at its
mid row; the scale is interpolated between mid rows (nearest value beyond the first / last one) and a height is
the sum of the scale over the rows between two y positions, never a single ratio. The left and right scales
only describe the pixel size on each side; a stack is measured with both and the mean is reported.
"""
import json
import shutil
from pathlib import Path

import numpy as np

MODEL_NAME = "height_model.json"
RATIO_NAME = "ratio.png"                       # the scale photo in the project image folder
SIDES = ("left", "right")
PITCH_MM = {"10": 10.0, "5": 5.0, "1": 5.0}      # distance between consecutive clicks of each list


class SideScale:
    """mm-per-px profile of one reference scale built from its clicked points ([x, y] per point)."""

    def __init__(self, lists):
        segments = []
        for kind, pitch in PITCH_MM.items():
            ys = sorted(float(point[1]) for point in lists.get(kind, []))
            segments += [((a + b) / 2.0, pitch / (b - a)) for a, b in zip(ys, ys[1:]) if b > a]
        if not segments:
            raise ValueError("a side needs at least two different y positions in one of its lists")
        segments.sort()
        self.mid = np.array([s[0] for s in segments])
        self.scale = np.array([s[1] for s in segments])
        every = [float(point[1]) for kind in PITCH_MM for point in lists.get(kind, [])]
        self.low, self.high = min(every), max(every)      # rows covered by the clicked points

    def height(self, y_top, y_bottom):
        """(mm between the two rows, True if a row lies outside the clicked range = extrapolated)."""
        rows = np.linspace(y_top, y_bottom, max(2, int(np.ceil(y_bottom - y_top)) + 1))
        scale = np.interp(rows, self.mid, self.scale)
        mm = float(np.sum((scale[1:] + scale[:-1]) / 2.0 * np.diff(rows)))
        return mm, bool(y_top < self.low or y_bottom > self.high)


class HeightModel:
    def __init__(self, points):
        self.sides = {side: SideScale(points.get(side, {})) for side in SIDES}

    def measure(self, y_top, y_bottom):
        """Per-ROI fields: height_left_mm, height_right_mm, height_mm (their mean), height_extrapolated."""
        if y_bottom <= y_top:
            return {"height_left_mm": None, "height_right_mm": None, "height_mm": None, "height_extrapolated": False}
        (left, left_out), (right, right_out) = (self.sides[side].height(y_top, y_bottom) for side in SIDES)
        return {"height_left_mm": left, "height_right_mm": right, "height_mm": (left + right) / 2.0,
                "height_extrapolated": left_out or right_out}


def read_points(profile_dir):
    """The saved record ({image, angle_deg, points}) or None when the project has no height model."""
    path = Path(profile_dir) / MODEL_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_model(profile_dir):
    record = read_points(profile_dir)
    return HeightModel(record["points"]) if record else None


def save_points(profile_dir, image, angle_deg, points):
    """Validate (raises ValueError) and write the clicked points; returns the saved record."""
    points = {side: {kind: [[float(x), float(y)] for x, y in points.get(side, {}).get(kind, [])]
                     for kind in PITCH_MM} for side in SIDES}
    HeightModel(points)
    record = {"image": image, "angle_deg": float(angle_deg), "points": points}
    path = Path(profile_dir) / MODEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def copy_model(source_dir, target_dir):
    """Copy a project's height model to another project (copy-params / batch); True if there was one."""
    source = Path(source_dir) / MODEL_NAME
    if not source.is_file():
        return False
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, Path(target_dir) / MODEL_NAME)
    return True
