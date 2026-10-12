"""Per-image analysis figure (Task 8, PDF p9-p10).

One subplot per selected ROI, at most MAX_COLUMNS per row, tight layout, dpi from OutputOptions
(>= 64). Each subplot shows the raw width-projected strip signal over the stack rows with the
counter's highs (down events) and lows (up events) marked; titles are N1_pred .. Nk_pred.
Uses matplotlib.figure.Figure directly (no pyplot / GUI backend).
"""
import math

import numpy as np
from matplotlib.figure import Figure

MAX_COLUMNS = 4


def grid_shape(count, max_columns=MAX_COLUMNS):
    """(rows, columns) of the subplot grid for `count` ROIs."""
    columns = max(1, min(max_columns, count))
    return max(1, math.ceil(count / columns)), columns


def analysis_figure(panels, title="", dpi=100):
    """panels: [{'rows', 'signal', 'high_idx', 'low_idx', 'prediction', 'label'}]; returns a Figure."""
    nrows, ncols = grid_shape(len(panels))
    figure = Figure(figsize=(max(5.0, 4.0 * ncols), 3.0 * nrows + 0.5), dpi=dpi)
    axes = figure.subplots(nrows, ncols, squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.set_visible(False)
            continue
        panel = panels[index]
        rows, signal = np.asarray(panel["rows"]), np.asarray(panel["signal"])
        ax.plot(rows, signal, color="0.25", linewidth=0.8)
        high, low = np.asarray(panel["high_idx"], dtype=int), np.asarray(panel["low_idx"], dtype=int)
        ax.plot(rows[high], signal[high], "v", color="tab:red", markersize=3, label=f"high ({high.size})")
        ax.plot(rows[low], signal[low], "^", color="tab:blue", markersize=3, label=f"low ({low.size})")
        prediction = panel["prediction"]
        ax.set_title(f"N{index + 1}_pred = {prediction:.2f}" if prediction is not None else f"N{index + 1}_pred = n/a",
                     fontsize=10)
        ax.set_xlabel(f"row (px)  {panel['label']}", fontsize=8)
        ax.set_ylabel("intensity", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
    if not panels:
        axes.flat[0].set_visible(True)
        axes.flat[0].text(0.5, 0.5, "no ROI inside the stack", ha="center", va="center")
        axes.flat[0].set_axis_off()
    if title:
        figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def save_analysis(path, panels, title="", dpi=100):
    analysis_figure(panels, title, dpi).savefig(str(path), dpi=dpi)
