"""Shared matplotlib style so every figure in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#0f766e"      # teal - primary series
ACCENT2 = "#f59e0b"     # amber - comparison series
ACCENT3 = "#6366f1"     # indigo
STATUS_COLORS = {"LOW": "#16a34a", "MODERATE": "#eab308", "HIGH": "#f97316", "FULL": "#dc2626"}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })
