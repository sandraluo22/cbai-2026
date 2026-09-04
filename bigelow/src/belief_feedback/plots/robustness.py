"""Figure 14: robustness."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig

PANELS = [
    ("prompt", "prompt variant"),
    ("memory", "memory policy"),
    ("topology", "topology"),
    ("steering_scope", "steering scope"),
]


def make(cfg: Config) -> None:
    rob = pd.read_parquet(cfg.paths.runs / "robustness_results.parquet")
    fig, axes = new_fig(1, 5, width=15, height=3.2)
    for k, (dim, title) in enumerate(PANELS):
        ax = axes[0][k]
        sub = rob[rob["dimension"] == dim]
        g = sub.groupby("variant")["mean_effect"].agg(["mean", "std", "count"])
        ax.bar(
            range(len(g)), g["mean"],
            yerr=g["std"] / g["count"].pow(0.5),
            color=COLORS["model"], capsize=4,
        )
        ax.set_xticks(range(len(g)), list(g.index), rotation=25, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("mean impulse effect (Delta ell)")
        ax.set_title(title)
        annotate_n(ax, sub["world_id"].nunique(), "worlds")
    ax = axes[0][4]
    g = rob.groupby("dimension")[["malformed_rate", "hallucinated_citation_rate"]].mean()
    g.plot.bar(ax=ax, rot=25, color=[COLORS["negative"], COLORS["positive"]])
    ax.set_ylabel("rate per memo")
    ax.set_title("malformed and hallucinated outputs")
    finish(cfg, fig, "fig14_robustness", rob)
