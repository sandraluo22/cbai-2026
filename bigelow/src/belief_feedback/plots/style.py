"""Shared figure style, watermarking, and save helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import Config

COLORS = {
    "baseline": "#4d4d4d",
    "positive": "#c23b22",
    "negative": "#2a6f97",
    "model": "#7b2d8b",
    "oracle": "#1b7837",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def new_fig(nrows: int, ncols: int, width: float = 11.0, height: float | None = None):
    height = height or 2.9 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    return fig, axes


def finish(cfg: Config, fig, name: str, data: pd.DataFrame | None = None, caption: str = "") -> None:
    """Watermark, save PDF + 300-dpi PNG, and store the plotting dataframe."""
    if cfg.label:
        fig.text(
            0.5, 0.5, cfg.label, fontsize=42, color="crimson", alpha=0.18,
            ha="center", va="center", rotation=22, zorder=100,
        )
        fig.text(0.01, 0.01, cfg.label, fontsize=8, color="crimson", alpha=0.85)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(cfg.paths.figures / f"{name}.pdf")
    fig.savefig(cfg.paths.figures / f"{name}.png", dpi=300)
    if data is not None:
        data.to_parquet(cfg.paths.figure_data / f"{name}.parquet", index=False)
        data.to_csv(cfg.paths.figure_data / f"{name}.csv", index=False)
    plt.close(fig)


def annotate_n(ax, n: int, extra: str = "") -> None:
    ax.text(
        0.98, 0.02, f"n={n}{(' ' + extra) if extra else ''}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555",
    )
