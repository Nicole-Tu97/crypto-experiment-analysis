"""
Shared plotting style + palette.

Colors are taken from a CVD-validated categorical palette (blue / orange / aqua),
so treatment vs control and adjusted vs unadjusted stay distinguishable for
colorblind readers. Figures are light-surface PNGs meant for a GitHub README.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# CVD-validated categorical hues
BLUE = "#2a78d6"     # treatment / primary emphasis
ORANGE = "#eb6834"   # adjusted / secondary
AQUA = "#1baf7a"     # positive / effect
GRAY = "#8a8a86"     # control / neutral
RED = "#e34948"      # guardrail breach / caution
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10.5,
        "legend.frameon": False,
        "figure.dpi": 130,
    })


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
