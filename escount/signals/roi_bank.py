"""ROI bank (Task 3): vertical counting strips inside the rotated background_BW valid region.

Geometry rules (agreed requirements, enforced with ValueError):
  * every row of every strip keeps >= 50 px to the left/right edge of the valid region
    (`side_margin` >= 50; per row y in [y0, y1): valid[y, x - side_margin : x + w + side_margin] all True),
  * adjacent strips of one width overlap by less than w/3, i.e. pitch D > 2w/3,
  * strips span the whole stack vertically (`stack_rows` clips slots to the stack extent and drops
    slots whose valid rows stop inside the stack).
Augmentation varies w, D and x_offset (grid shift) on the same image.
"""
from dataclasses import dataclass, replace

from escount.calibration.rotation_correction import normalize_mask
from escount.es_counting import longest_run, stack_extent

MIN_SIDE_MARGIN = 50
DEFAULT_WIDTHS = tuple(range(6, 41, 2))
STACK_EDGE_TOLERANCE_PX = 5   # max rows a kept slot may miss at either end of the stack extent


@dataclass(frozen=True)
class RoiSlot:
    x: int          # left column of the strip
    w: int          # strip width (projection is across these columns)
    y0: int         # first row (inclusive)
    y1: int         # last row (exclusive)
    pitch: int      # D: left-edge to left-edge distance between adjacent strips of this width
    x_offset: int   # grid shift used to build this bank


def check_geometry(w, pitch, side_margin):
    """Raise ValueError if the strip geometry violates the agreed ROI rules."""
    if side_margin < MIN_SIDE_MARGIN:
        raise ValueError(f"side_margin {side_margin} px < {MIN_SIDE_MARGIN} px from the valid-region edge")
    if w < 1:
        raise ValueError(f"strip width must be >= 1 px, got {w}")
    if pitch * 3 <= 2 * w:
        raise ValueError(f"pitch D={pitch} <= 2w/3 for w={w}: adjacent overlap {w - pitch} px >= w/3")


def build_roi_bank(rotated_mask, widths=DEFAULT_WIDTHS, *, pitch, side_margin=MIN_SIDE_MARGIN, x_offset=0):
    """All strips of every width in `widths` on a pitch-`pitch` grid starting at side_margin + x_offset.

    Each strip's rows are the longest run of rows where the strip widened by `side_margin` on both
    sides is entirely valid, so the >= side_margin px rule holds on every row (also on slanted masks).
    """
    if x_offset < 0:
        raise ValueError(f"x_offset must be >= 0, got {x_offset}")
    valid = normalize_mask(rotated_mask)
    width_px = valid.shape[1]
    slots = []
    for w in widths:
        check_geometry(w, pitch, side_margin)
        for x in range(side_margin + x_offset, width_px - w - side_margin + 1, pitch):
            y0, y1 = longest_run(valid[:, x - side_margin:x + w + side_margin].all(axis=1))
            if y1 - y0 > 1:
                slots.append(RoiSlot(x, w, y0, y1, pitch, x_offset))
    if not slots:
        raise ValueError(f"no strip with {side_margin} px side margin fits inside the valid region")
    for slot in slots:  # explicit per-row guarantee
        if not valid[slot.y0:slot.y1, slot.x - side_margin:slot.x + slot.w + side_margin].all():
            raise AssertionError(f"slot {slot} or its side margin leaves the valid region")
    return slots


def stack_rows(image, slots, edge_tolerance=STACK_EDGE_TOLERANCE_PX):
    """Detect the stack rows with es_counting.stack_extent and clip slots to them.

    Returns (extent dict with y0/y1, kept slots with y0/y1 = their intersection with the extent).
    A slot is kept only if it spans the whole stack: its valid rows may miss at most `edge_tolerance`
    px at the top/bottom end of the extent (the rotated mask edge is slanted by a few px); slots
    whose valid run stops inside the stack (e.g. interrupted by a fixture) are dropped.
    """
    layout = {"strips": [{"x": s.x, "y0": s.y0, "y1": s.y1, "width": s.w} for s in slots]}
    extent = stack_extent(image, layout)
    if extent["y1"] - extent["y0"] < 2:
        raise ValueError("stack extent is shorter than 2 rows")
    return extent, clip_slots(slots, extent, edge_tolerance)


def clip_slots(slots, extent, edge_tolerance=STACK_EDGE_TOLERANCE_PX):
    """Clip slots to a stack extent; drop slots that do not span it (same rule as stack_rows)."""
    y0, y1 = extent["y0"], extent["y1"]
    return [replace(s, y0=max(s.y0, y0), y1=min(s.y1, y1)) for s in slots
            if s.y0 <= y0 + edge_tolerance and s.y1 >= y1 - edge_tolerance]
