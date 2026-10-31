"""Presentation figure style (Task 9): 16:9, 200 dpi (3200 x 1800 px), Chinese font Microsoft JhengHei.

Figures are built with matplotlib.figure.Figure (no pyplot, no GUI backend). Never save with
bbox_inches="tight": it would change the 3200 x 1800 px slide size.
"""
import warnings
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure

FONT_PATH = Path(r"C:\Windows\Fonts\msjh.ttc")
FONT_FAMILY = "Microsoft JhengHei"
FALLBACK_FAMILY = "DejaVu Sans"
FIGSIZE = (16, 9)
DPI = 200
_STATE = {}


def setup_style(font_path=FONT_PATH):
    """Register Microsoft JhengHei and set the rcParams. Warns and falls back if the font is missing."""
    if "family" in _STATE:
        return _STATE["family"]
    family = FONT_FAMILY
    path = Path(font_path)
    if path.is_file():
        font_manager.fontManager.addfont(str(path))
        if FONT_FAMILY not in {font.name for font in font_manager.fontManager.ttflist}:
            warnings.warn(f"{path} does not provide {FONT_FAMILY!r}; falling back to {FALLBACK_FAMILY}")
            family = FALLBACK_FAMILY
    else:
        warnings.warn(f"font {path} not found; Chinese labels fall back to {FALLBACK_FAMILY}")
        family = FALLBACK_FAMILY
    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 15
    matplotlib.rcParams["axes.titlesize"] = 18
    matplotlib.rcParams["axes.labelsize"] = 16
    _STATE["family"] = family
    return family


def new_figure(nrows=1, ncols=1, **subplot_kw):
    """(Figure 16:9 at 200 dpi, single axes / 1-D / 2-D axes array as matplotlib's squeeze)."""
    setup_style()
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = figure.subplots(nrows, ncols, **subplot_kw)
    return figure, axes


def save(figure, path):
    """Save at 200 dpi (fixed 3200 x 1800 px) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, format="png")
    return path


def placeholder(path, title, message):
    """Explanatory figure used when there is not enough data (the file is never skipped)."""
    figure, ax = new_figure()
    ax.set_axis_off()
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=30, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=20, color="#555555", wrap=True,
            transform=ax.transAxes)
    return save(figure, path)


def plot_param_sensitivity_matrix(path, title, x_param_name, x_values, y_param_name, y_values, hit_matrix):
    """Plot a 2D parameter sensitivity heatmap for quantitative comparison (Q6).

    hit_matrix shape: (len(y_values), len(x_values)) with exact hit rates [0.0 .. 1.0].
    """
    setup_style()
    fig, ax = new_figure(nrows=1, ncols=1)
    im = ax.imshow(hit_matrix, cmap="viridis", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.set_xlabel(f"{x_param_name}")
    ax.set_ylabel(f"{y_param_name}")
    ax.set_title(title)
    
    # Text annotate cells
    for r in range(len(y_values)):
        for c in range(len(x_values)):
            val = hit_matrix[r][c]
            text_color = "white" if val < 0.6 else "black"
            ax.text(c, r, f"{val:.1%}", ha="center", va="center", color=text_color, fontsize=11)
            
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("完全命中率 (Exact Hit Rate)")
    fig.tight_layout()
    save(fig, path)
    return path
