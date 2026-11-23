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
VERSION = 2                                    # 2: every 10 / 5 / 1 mm icon is clicked (1 mm list is 1 mm apart)
PITCH_MM = {"10": 10.0, "5": 5.0, "1": 1.0}     # distance between consecutive clicks of each list
PITCH_MM_V1 = {"10": 10.0, "5": 5.0, "1": 5.0}  # files without "version": the 1 mm list holds the first icon after each 5 mm
DUPLICATE_PX = 3.0                              # clicks of one list closer than this are one icon clicked twice
SKIP_TOLERANCE = 0.15                           # a gap this close to n typical spacings is n spacings (skipped icons are inferred)
CHECK_TOLERANCE = 0.03                          # 5 / 10 mm spans may differ this much from the 1 mm scale before a warning


def _segments(lists, pitches, dedupe, skipped=None):
    """[(mid_row, mm_per_px, kind, y_a, y_b)] of the consecutive clicks of each list (clicks may come in any order).

    With a `skipped` list, a gap of n times the typical spacing of a list means n - 1 icons were not clicked: the
    gap counts as n pitches and a warning is appended.
    """
    found = []
    for kind, pitch in pitches.items():
        ys = sorted(float(point[1]) for point in lists.get(kind, []))
        if dedupe:
            ys = [y for i, y in enumerate(ys) if i == 0 or y - ys[i - 1] >= DUPLICATE_PX]
        typical = float(np.percentile(np.diff(ys), 25)) if skipped is not None and len(ys) > 2 else 0.0
        for a, b in zip(ys, ys[1:]):
            n = max(1, round((b - a) / typical)) if typical > 0 else 1
            if typical > 0 and (n > 1 or abs((b - a) / typical - n) > SKIP_TOLERANCE * n):
                skipped.append(f"{kind} mm points at y={a:.0f}..{b:.0f} look like {n} icon spacings")
            if b > a:
                found.append(((a + b) / 2.0, n * pitch / (b - a), kind, a, b))
    return found


def _accumulate(mid, scale, y_top, y_bottom):
    rows = np.linspace(y_top, y_bottom, max(2, int(np.ceil(y_bottom - y_top)) + 1))
    values = np.interp(rows, mid, scale)
    return float(np.sum((values[1:] + values[:-1]) / 2.0 * np.diff(rows)))


class SideScale:
    """mm-per-px profile of one reference scale built from its clicked points ([x, y] per point).

    Version 2: the 1 mm clicks give the scale; the 5 / 10 mm clicks only check it (warnings). Without 1 mm clicks
    the 5 / 10 mm clicks give the scale. Version 1 (old files): all lists give the scale, the 1 mm list is 5 mm apart.
    """

    def __init__(self, lists, version=VERSION):
        self.warnings = []
        if version < 2:
            segments = _segments(lists, PITCH_MM_V1, False)
        else:
            fine = _segments(lists, {"1": PITCH_MM["1"]}, True, self.warnings)
            coarse = _segments(lists, {k: v for k, v in PITCH_MM.items() if k != "1"}, True)
            segments = fine or coarse
            if fine:
                self.warnings += self._check(fine, coarse)
                low, high = min(s[3] for s in fine), max(s[4] for s in fine)
                segments = fine + [s for s in coarse if s[0] < low or s[0] > high]      # 5 / 10 mm extend beyond the 1 mm rows
            elif coarse:
                self.warnings = ["no 1 mm points: the scale comes from the 5 / 10 mm points only"]
        if not segments:
            raise ValueError("a side needs at least two different y positions in one of its lists")
        segments.sort()
        self.mid = np.array([s[0] for s in segments])
        self.scale = np.array([s[1] for s in segments])
        every = [float(point[1]) for kind in PITCH_MM for point in lists.get(kind, [])]
        self.low, self.high = min(every), max(every)      # rows covered by the clicked points
        self.knots = [] if version < 2 else self._knots(every)

    def _knots(self, rows):
        """[[y, mm]] top to bottom: icons of different lists at one height are one reference point (closer than half
        a typical 1 mm spacing); mm is the accumulated length from the top-most point."""
        tolerance = 0.5 / float(np.median(self.scale))
        groups = []
        for y in sorted(rows):
            if groups and y - groups[-1][-1] < tolerance:
                groups[-1].append(y)
            else:
                groups.append([y])
        ys = [float(np.mean(group)) for group in groups]
        return [[y, _accumulate(self.mid, self.scale, ys[0], y) if y > ys[0] else 0.0] for y in ys]

    def _check(self, fine, coarse):
        """Warnings for 5 / 10 mm spans (inside the 1 mm rows) that the 1 mm scale measures off by > tolerance."""
        fine.sort()
        mid, scale = np.array([s[0] for s in fine]), np.array([s[1] for s in fine])
        low, high = min(s[3] for s in fine), max(s[4] for s in fine)
        warnings = []
        for _, _, kind, a, b in sorted(coarse, key=lambda s: s[3]):
            if a < low or b > high:
                continue
            nominal = PITCH_MM[kind]
            measured = _accumulate(mid, scale, a, b)
            if abs(measured - nominal) > CHECK_TOLERANCE * nominal:
                warnings.append(f"{kind} mm points at y={a:.0f}..{b:.0f} measure {measured:.2f} mm on the 1 mm scale")
        return warnings
    def height(self, y_top, y_bottom):
        """(mm between the two rows, True if a row lies outside the clicked range = extrapolated)."""
        mm = _accumulate(self.mid, self.scale, y_top, y_bottom)
        return mm, bool(y_top < self.low or y_bottom > self.high)


class HeightModel:
    def __init__(self, points, version=VERSION):
        self.sides = {side: SideScale(points.get(side, {}), version) for side in SIDES}
        self.warnings = [f"{side}: {text}" for side in SIDES for text in self.sides[side].warnings]

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
    return HeightModel(record["points"], record.get("version", 1)) if record else None


def save_points(profile_dir, image, angle_deg, points):
    """Validate (raises ValueError) and write the clicked points; returns the saved record plus its warnings."""
    points = {side: {kind: [[float(x), float(y)] for x, y in points.get(side, {}).get(kind, [])]
                     for kind in PITCH_MM} for side in SIDES}
    model = HeightModel(points)
    record = {"version": VERSION, "image": image, "angle_deg": float(angle_deg), "points": points,
              "knots": {side: model.sides[side].knots for side in SIDES}}
    path = Path(profile_dir) / MODEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**record, "warnings": model.warnings}


def copy_model(source_dir, target_dir):
    """Copy a project's height model to another project (copy-params / batch); True if there was one."""
    source = Path(source_dir) / MODEL_NAME
    if not source.is_file():
        return False
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, Path(target_dir) / MODEL_NAME)
    return True
