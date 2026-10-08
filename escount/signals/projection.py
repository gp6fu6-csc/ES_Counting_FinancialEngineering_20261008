"""Width-direction projection (Task 3): one raw value per stack row for a strip.

Rows are combined ONLY across the strip width (mean / Gaussian-weighted / median). There is NO
filtering along the vertical (row) direction: the returned signal keeps every row-to-row change;
smoothing, if any, happens inside the indicators.
"""
import numpy as np

METHODS = ("mean", "gaussian", "median")


def gaussian_weights(w, sigma):
    """Normalised Gaussian column weights centred on the strip, sigma in px."""
    if sigma is None or sigma <= 0:
        raise ValueError(f"gaussian projection needs sigma > 0, got {sigma}")
    offsets = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def project(image, slot, method="mean", sigma=None):
    """Return the raw per-row float64 signal of image[slot.y0:slot.y1, slot.x:slot.x + slot.w]."""
    height, width = image.shape[:2]
    if slot.x < 0 or slot.y0 < 0 or slot.x + slot.w > width or slot.y1 > height or slot.y1 <= slot.y0:
        raise ValueError(f"slot {slot} is outside the image {width}x{height}")
    block = image[slot.y0:slot.y1, slot.x:slot.x + slot.w]
    if method == "mean":
        return np.asarray(block.mean(axis=1), dtype=np.float64)
    if method == "median":
        return np.asarray(np.median(block, axis=1), dtype=np.float64)
    if method == "gaussian":
        return block.astype(np.float64) @ gaussian_weights(slot.w, sigma)
    raise ValueError(f"unknown projection method {method!r}; expected one of {METHODS}")
