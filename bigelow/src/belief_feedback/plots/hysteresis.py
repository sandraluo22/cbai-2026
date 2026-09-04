"""Figure 9: hysteresis."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    res = pd.read_parquet(cfg.paths.runs / "hysteresis_results.parquet")
    traj = pd.read_parquet(cfg.paths.runs / "hysteresis_trajectories.parquet")
    fig, axes = new_fig(1, 4, width=13, height=3.0)
    pos = traj[traj["sign"] == "positive"]

    for k, comm in enumerate(("live", "replay")):
        ax = axes[0][k]
        for sched, color in (("early", COLORS["positive"]), ("late", COLORS["negative"])):
            sub = pos[(pos["comm"] == comm) & (pos["schedule"] == sched)]
            m = sub.groupby("round")["mean_ell"].mean()
            ax.plot(m.index, m.to_numpy(), "o-", color=color, label=f"{sched} schedule")
        ax.set_xlabel("round")
        ax.set_ylabel("mean network ell")
        ax.legend()
        ax.set_title(f"{comm} communication (positive dose)")
        annotate_n(ax, sub["world_id"].nunique(), "worlds")

    ax = axes[0][2]
    g = res.groupby(["sign", "comm"])["hysteresis_gap"].mean().unstack()
    g.plot.bar(ax=ax, rot=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("final early - late gap (ell)")
    ax.set_title("final hysteresis gaps")

    ax = axes[0][3]
    piv = res.pivot_table(index=["world_id", "replicate_seed", "sign"], columns="comm", values="hysteresis_gap").reset_index()
    if {"live", "replay"} <= set(piv.columns):
        inter = (piv["live"] - piv["replay"]) * piv["sign"].map({"positive": 1, "negative": -1})
        ax.bar(["live - replay"], [inter.mean()], yerr=[inter.std() / max(len(inter), 1) ** 0.5],
               color=COLORS["model"], capsize=4)
        annotate_n(ax, piv["world_id"].nunique(), "worlds")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("sign-adjusted interaction (ell)")
    ax.set_title("live minus fixed-replay hysteresis")
    finish(cfg, fig, "fig09_hysteresis", res)
