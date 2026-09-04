"""Figure 8: evidence recycling."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    rec = pd.read_parquet(cfg.paths.runs / "recycling_results.parquet")
    net = pd.read_parquet(cfg.paths.runs / "recycling_network.parquet")
    neutral = rec[~rec["provenance_aware_prompt"]]
    fig, axes = new_fig(1, 5, width=15, height=3.0)

    ax = axes[0][0]
    ind = neutral[neutral["condition"] == "independent"]
    ryc = neutral[neutral["condition"] == "recycled"]
    bars = {
        "1 report": neutral["gain_one_report"].abs().mean(),
        "3 independent": ind["gain_three_reports"].abs().mean(),
        "3 recycled": ryc["gain_three_reports"].abs().mean(),
        "oracle (aware,\n3 recycled)": ryc["oracle_aware_gain_three"].abs().mean(),
    }
    ax.bar(bars.keys(), bars.values(), color=[COLORS["baseline"], COLORS["oracle"], COLORS["positive"], "#999"])
    ax.set_ylabel("|belief gain| (log odds)")
    ax.set_title("belief gain by provenance")
    annotate_n(ax, neutral["world_id"].nunique(), "worlds")

    ax = axes[0][1]
    for cond, color in (("independent", COLORS["oracle"]), ("recycled", COLORS["positive"])):
        vals = neutral[neutral["condition"] == cond]["multiplier"]
        ax.bar(cond, vals.mean(), yerr=vals.std() / max(len(vals), 1) ** 0.5, color=color, capsize=4)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="provenance-aware oracle")
    ax.axhline(3.0, color="gray", ls=":", lw=0.8, label="provenance-blind oracle")
    ax.set_ylabel("three-report multiplier")
    ax.legend()
    ax.set_title("recycling multiplier")

    ax = axes[0][2]
    g = rec.groupby(["condition", "provenance_aware_prompt"])["double_counting_gap"].mean().unstack()
    g.plot.bar(ax=ax, color=[COLORS["baseline"], COLORS["oracle"]], rot=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("|LLM gain| - |aware-oracle gain|")
    ax.legend(title="provenance-aware prompt", fontsize=7)
    ax.set_title("double-counting gap")

    ax = axes[0][3]
    m = rec.groupby(["condition", "provenance_aware_prompt"])["multiplier"].mean().unstack()
    m.plot.bar(ax=ax, color=[COLORS["baseline"], COLORS["oracle"]], rot=0)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("multiplier")
    ax.legend(title="provenance-aware prompt", fontsize=7)
    ax.set_title("instruction control")

    ax = axes[0][4]
    piv = net.pivot_table(index=["provenance_role"], columns="condition", values="final_mean_ell", aggfunc="mean")
    piv.plot.bar(ax=ax, rot=0, color=[COLORS["positive"], COLORS["baseline"]])
    ax.set_ylabel("final mean network ell")
    ax.set_title("network outcome x steering impulse")
    finish(cfg, fig, "fig08_evidence_recycling", rec)
