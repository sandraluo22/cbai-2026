"""Figure 12: mechanistic alignment."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    probe = pd.read_parquet(cfg.paths.runs / "probe_results.parquet")
    beliefs = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    patch = pd.read_parquet(cfg.paths.runs / "mechanistic_patching.parquet")
    base = beliefs[beliefs["branch"] == "baseline"].dropna(subset=["caa_projection"])
    fig, axes = new_fig(1, 5, width=15, height=3.0)

    ax = axes[0][0]
    ax.plot(probe["layer"], probe["accuracy"], "o-", label="accuracy")
    ax.plot(probe["layer"], probe["auroc"], "s-", label="AUROC")
    ax.set_xlabel("layer")
    ax.set_ylabel("endogenous-world performance")
    ax.legend()
    ax.set_title("probe performance by layer")

    ax = axes[0][1]
    ax.plot(base["caa_projection"], base["semantic_log_odds"], ".", alpha=0.35, color=COLORS["model"])
    ax.set_xlabel("CAA projection (selected layer)")
    ax.set_ylabel("behavioral semantic log odds")
    corr = base[["caa_projection", "semantic_log_odds"]].corr().iloc[0, 1]
    ax.set_title(f"projection vs behavior (r={corr:.2f})")
    annotate_n(ax, len(base))

    ax = axes[0][2]
    for _wid, grp in list(base.groupby("world_id"))[:6]:
        m = grp.groupby("round")["caa_projection"].mean()
        ax.plot(m.index, m.to_numpy(), "-", alpha=0.7)
    ax.set_xlabel("round")
    ax.set_ylabel("mean CAA projection")
    ax.set_title("projection trajectories")

    ax = axes[0][3]
    ax.plot(probe["layer"], probe["cosine_with_caa"], "o-", color=COLORS["oracle"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(probe weight, CAA direction)")
    ax.set_title("probe / CAA alignment")

    ax = axes[0][4]
    live = patch[~patch["text_clamped"]]
    ax.bar(
        ["source belief\n(t=1)", "downstream\nmean"],
        [live["delta_source_belief_t1"].mean(), live["delta_downstream_mean"].mean()],
        color=[COLORS["positive"], COLORS["negative"]],
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Delta ell from projection patch")
    ax.set_title("belief-component patching")
    annotate_n(ax, len(live), "worlds")
    finish(cfg, fig, "fig12_mechanistic_alignment", probe)
