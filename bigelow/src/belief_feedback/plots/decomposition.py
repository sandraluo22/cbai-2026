"""Figures 7 and 13: causal path decomposition and text mediation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig

DECOMP_ORDER = [
    "one_hop_effect",
    "forward_cascade_effect",
    "reciprocal_feedback_effect",
    "total_closed_loop_effect",
    "text_mediated_effect",
]


def make_fig07(cfg: Config) -> None:
    eff = pd.read_parquet(cfg.paths.runs / "branch_effects.parquet")
    eff = eff.assign(adj=lambda d: d["effect"] * d["sign"].map({"positive": 1, "negative": -1}))
    fig, axes = new_fig(1, 5, width=15, height=3.0)
    final_round = eff["round"].max()
    for k, cond in enumerate(DECOMP_ORDER[:4]):
        ax = axes[0][k]
        sub = eff[(eff["condition"] == cond) & (eff["round"] == final_round)]
        g = sub.groupby("world_id")["adj"].mean()
        ax.bar([0], [g.mean()], yerr=[g.std() / max(np.sqrt(len(g)), 1)], color=COLORS["model"], capsize=4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks([])
        ax.set_ylabel("sign-adjusted Delta ell")
        ax.set_title(cond.replace("_", " "))
        annotate_n(ax, len(g), "worlds")
    ax = axes[0][4]
    tot = eff[eff["condition"] == "total_closed_loop_effect"]
    piv = tot.pivot_table(index="round", columns="graph_distance", values="adj", aggfunc="mean")
    im = ax.imshow(piv.to_numpy().T, aspect="auto", origin="lower", cmap="RdBu_r")
    ax.set_xticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_yticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_xlabel("round")
    ax.set_ylabel("graph distance")
    ax.set_title("total effect by round x distance")
    fig.colorbar(im, ax=ax, shrink=0.8)
    finish(cfg, fig, "fig07_causal_path_decomposition", eff)


def make_fig13(cfg: Config) -> None:
    eff = pd.read_parquet(cfg.paths.runs / "branch_effects.parquet")
    eff = eff.assign(adj=lambda d: d["effect"] * d["sign"].map({"positive": 1, "negative": -1}))
    patch = pd.read_parquet(cfg.paths.runs / "mechanistic_patching.parquet")
    fig, axes = new_fig(1, 4, width=13, height=3.0)
    final_round = eff["round"].max()
    down = eff[eff["agent_id"] != 0]

    ax = axes[0][0]
    tot = down[(down["condition"] == "total_closed_loop_effect") & (down["round"] == final_round)]
    tme = down[(down["condition"] == "text_mediated_effect") & (down["round"] == final_round)]
    vals = [tot.groupby("world_id")["adj"].mean().mean(), (tot.groupby("world_id")["adj"].mean() - tme.groupby("world_id")["adj"].mean()).mean()]
    ax.bar(["live memo", "full-text clamp"], vals, color=[COLORS["positive"], COLORS["baseline"]])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("downstream sign-adjusted Delta ell")
    ax.set_title("live vs text-clamped intervention")

    ax = axes[0][1]
    live = patch[~patch["text_clamped"]]
    ax.bar(
        ["steering", "projection patch"],
        [live["steer_live_downstream"].mean(), live["delta_downstream_mean"].mean()],
        color=[COLORS["positive"], COLORS["model"]],
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("downstream mean Delta ell")
    ax.set_title("activation steering vs projection patch")
    annotate_n(ax, len(live), "worlds")

    ax = axes[0][2]
    for cond, color in (("total_closed_loop_effect", COLORS["positive"]), ("text_mediated_effect", COLORS["model"])):
        m = down[down["condition"] == cond].groupby("round")["adj"].mean()
        ax.plot(m.index, m.to_numpy(), "o-", color=color, label=cond.replace("_", " "))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("round")
    ax.set_ylabel("downstream effect")
    ax.legend()
    ax.set_title("downstream effect by round")

    ax = axes[0][3]
    per_world_tot = tot.groupby("world_id")["adj"].mean()
    per_world_tme = tme.groupby("world_id")["adj"].mean()
    frac = (per_world_tme / per_world_tot.replace(0, np.nan)).dropna()
    ax.hist(frac.clip(-1, 2), bins=10, color=COLORS["model"], alpha=0.8)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="fully text-mediated")
    ax.set_xlabel("fraction of total effect mediated by text")
    ax.set_ylabel("worlds")
    ax.legend()
    ax.set_title(f"median = {frac.median():.2f}" if len(frac) else "no data")
    finish(cfg, fig, "fig13_text_mediation", patch)
