"""Top / bottom edge of the sheet stack inside one ROI strip signal (PDF 20261124 p24-25, 頂-底標註樣本).

Top = upper edge of the first sheet, bottom = lower edge of the last sheet (not the support base below it).

The signal is the per-row projection of the strip. Sheets show up as a chain of regular, similarly tall peaks
(the same periodic structure MACD / KD count). Everything else is rejected by that structure:
  * chain split   : a gap between neighbouring peaks > GAP_RATIO x the median pitch whose signal falls to the
                    dark level ends the chain (dark gap above the stack / between the last sheet and a reflective
                    base); the chain with the most peaks is the stack.
  * amplitude trim: peaks at either end whose swing (MACD-like height above the troughs one pitch away) is
                    < TRIM_RATIO x the chain median are faint reflections or the base, not sheets, and are dropped.
                    Weak peaks between strong ones (blurred / saturated sheets) stay.
  * edge search   : from the first / last sheet peak walk outward to where the signal falls to
                    dark + EDGE_LEVEL x (median peak - dark); if it never gets that low (non-black reflection) the
                    valley within EDGE_SEARCH_PITCH pitches is used.
"""
import numpy as np
from scipy.signal import find_peaks

GAP_RATIO = 1.3            # a gap wider than this x the median pitch that also falls to dark breaks the chain
TRIM_RATIO = 0.45          # end peaks lower than this x the chain median swing are not sheets
EDGE_LEVEL = 0.30          # edge = where the signal crosses dark + EDGE_LEVEL x (peak - dark)
EDGE_SEARCH_PITCH = 2.2    # how far (in pitches) beyond the end peak the edge may lie
DARK_RATIO = 0.08         # a gap counts as dark when the signal drops below dark + DARK_RATIO x span
PROMINENCE_RATIO = 0.2     # peak prominence >= this x (p95 - dark)
MIN_PEAKS = 4
EDGE_OUTLIER_PX = 20       # ROI edges further than this from the image median snap to it (consensus_edges)


def _swing(signal, peaks, pitch):
    """Peak height above the higher of the lowest points within one pitch left / right (MACD-like swing).
    Local, so flat (saturated) tops and taller neighbours do not distort it."""
    reach = max(int(round(pitch)), 1)
    return np.asarray([signal[p] - max(signal[max(p - reach, 0):p + 1].min(), signal[p:p + reach + 1].min())
                       for p in peaks])


def _chain(signal, peaks, dark_level):
    """Longest run of peaks. A run breaks only at a dark gap: spacing > GAP_RATIO x the median pitch AND the
    signal between the two peaks falls to the dark level (a merely blurred / missed sheet does not break it)."""
    spacing = np.diff(peaks)
    pitch = float(np.median(spacing[spacing <= 60])) if (spacing <= 60).any() else float(np.median(spacing))
    breaks = [i for i in np.flatnonzero(spacing > GAP_RATIO * pitch)
              if signal[peaks[i]:peaks[i + 1] + 1].min() <= dark_level]
    breaks = np.asarray(breaks, dtype=int)
    runs = np.split(np.arange(peaks.size), breaks + 1)
    return max(runs, key=len), pitch


def stack_edges(signal):
    """(top, bottom, info) as sample indices of `signal`, or None when no regular sheet chain exists."""
    s = np.asarray(signal, dtype=np.float64)
    if s.size < 4 * MIN_PEAKS:
        return None
    dark = float(np.percentile(s, 2))
    span = float(np.percentile(s, 95)) - dark
    if span <= 0:
        return None
    peaks, _ = find_peaks(s, prominence=PROMINENCE_RATIO * span)
    if peaks.size < MIN_PEAKS:
        return None
    run, pitch = _chain(s, peaks, dark + DARK_RATIO * span)
    chain = peaks[run]
    swing = _swing(s, chain, pitch)
    strong = np.flatnonzero(swing >= TRIM_RATIO * float(np.median(swing)))
    chain = chain[strong[0]:strong[-1] + 1] if strong.size else chain[:0]
    if chain.size < MIN_PEAKS:
        return None
    height = float(np.median(s[chain])) - dark
    level = dark + EDGE_LEVEL * height
    reach = int(round(EDGE_SEARCH_PITCH * pitch))

    def walk(start, step):
        stop = min(max(start + step * reach, 0), s.size - 1)
        inside = s[start:stop + 1] if step > 0 else s[stop:start + 1][::-1]
        below = np.flatnonzero(inside <= level)
        offset = int(below[0]) if below.size else int(np.argmin(inside))
        return start + step * offset

    top, bottom = walk(int(chain[0]), -1), walk(int(chain[-1]), +1)
    return top, bottom, {"pitch": pitch, "peaks": int(chain.size), "dark": dark, "level": level}


def consensus_edges(edges, tolerance=EDGE_OUTLIER_PX):
    """[(top, bottom)] per ROI -> same list where an ROI far from the image-wide median (noisy / low-contrast strip)
    takes the median instead. All ROIs see the same physical first / last sheet, so they must agree."""
    if len(edges) < 3:
        return list(edges)
    top, bottom = (int(np.median([e[k] for e in edges])) for k in (0, 1))
    return [(t if abs(t - top) <= tolerance else top, b if abs(b - bottom) <= tolerance else bottom) for t, b in edges]
