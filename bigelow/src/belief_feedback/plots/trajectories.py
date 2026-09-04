"""Figures 5-6: closed-loop impulse trajectories and composition test."""

from __future__ import annotations

import pandas as pd

from ..analysis.bootstrap import cluster_bootstrap
from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig


def _paired_diff(beliefs: pd.DataFrame, branch: str) -> pd.DataFrame:
    piv = beliefs.pivot_table(
        index=["world_id", "replicate_seed", "agent_id", "round"],
        columns="branch",
        values="semantic_log_odds",
    ).reset_index()
    if branch not in piv.columns or "baseline" not in piv.columns:
        return pd.DataFrame()
    piv["diff"] = piv[branch] - piv["baseline"]
    return piv.dropna(subset=["diff"])


def make_fig05(cfg: Config) -> None:
    beliefs = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    effects = pd.read_parquet(cfg.paths.runs / "branch_effects.parquet")
    fig, axes = new_fig(1, 5, width=15, height=3.0)
    n_boot = min(cfg.analysis.bootstrap_samples, 500)

    for k, (branch, color) in enumerate(
        (("positive_impulse", COLORS["positive"]), ("negative_impulse", COLORS["negative"]))
    ):
        ax = axes[0][k]
        d = _paired_diff(beliefs, branch)
        if len(d):
            for r in sorted(d["round"].unique()):
                sub = d[d["round"] == r]
                res = cluster_bootstrap(
                    sub, lambda x: float(x["diff"].mean()), n_resamples=n_boot,
                    seed_parts=("fig5", branch, r),
                )
                ax.errorbar(
                    r, res.estimate,
                    yerr=[[res.estimate - res.ci_low], [res.ci_high - res.estimate]],
                    fmt="o", color=color, capsize=3,
                )
            annotate_n(ax, d["world_id"].nunique(), "worlds")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("round")
        ax.set_ylabel("mean paired Delta ell")
        ax.set_title(f"{branch} (95% cluster bootstrap)")

    ax = axes[0][2]
    tot = effects[effects["condition"] == "total_closed_loop_effect"].copy()
    tot["adj"] = tot["effect"] * tot["sign"].map({"positive": 1, "negative": -1})
    for dist, grp in tot.groupby("graph_distance"):
        m = grp.groupby("round")["adj"].mean()
        ax.plot(m.index, m.to_numpy(), "o-", label=f"distance {dist}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("round")
    ax.set_ylabel("sign-adjusted effect")
    ax.legend()
    ax.set_title("total effect by graph distance")

    ax = axes[0][3]
    d = _paired_diff(beliefs, "positive_impulse")
    if len(d):
        src = d[d["agent_id"] == 0].groupby("round")["diff"].mean()
        net = d.groupby("round")["diff"].mean()
        ax.plot(src.index, src.to_numpy(), "o-", color=COLORS["positive"], label="source agent")
        ax.plot(net.index, net.to_numpy(), "s-", color=COLORS["baseline"], label="network mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("round")
    ax.set_ylabel("Delta ell (positive impulse)")
    ax.legend()
    ax.set_title("source vs network mean")

    ax = axes[0][4]
    base = beliefs[beliefs["branch"] == "baseline"]
    m = base.groupby("round")["semantic_log_odds"].agg(["mean", "std"])
    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.2, color=COLORS["baseline"])
    ax.plot(m.index, m["mean"], "o-", color=COLORS["baseline"])
    ax.set_xlabel("round")
    ax.set_ylabel("semantic log odds")
    ax.set_title("baseline network trajectory (+/- sd)")
    finish(cfg, fig, "fig05_closed_loop_impulse_trajectories", tot)


def make_fig06(cfg: Config) -> None:
    preds = pd.read_parquet(cfg.paths.runs / "composition_predictions.parquet")
    comp = pd.read_parquet(cfg.paths.runs / "composition_metrics.parquet")
    fig, axes = new_fig(1, 4, width=13, height=3.0)

    ax = axes[0][0]
    tf = preds[(preds["kind"] == "teacher_forced") & (preds["model"] == "F4")]
    ax.plot(tf["observed"], tf["predicted"], ".", alpha=0.4, color=COLORS["model"])
    lim = max(1.0, tf[["observed", "predicted"]].abs().max().max()) if len(tf) else 1.0
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax.set_xlabel("observed next belief (ell)")
    ax.set_ylabel("F4 predicted")
    ax.set_title("teacher-forced one-step (F4)")
    annotate_n(ax, len(tf))

    ax = axes[0][1]
    fr = preds[(preds["kind"] == "free_rollout") & (preds["model"] == "F4") & (preds["condition"] == "baseline")]
    if len(fr):
        obs = fr.groupby("round")["observed"].mean()
        pre = fr.groupby("round")["predicted"].mean()
        lo = fr.groupby("round")["q10"].mean()
        hi = fr.groupby("round")["q90"].mean()
        ax.fill_between(lo.index, lo.to_numpy(), hi.to_numpy(), alpha=0.2, color=COLORS["model"], label="80% PI")
        ax.plot(pre.index, pre.to_numpy(), "o-", color=COLORS["model"], label="rollout mean")
        ax.plot(obs.index, obs.to_numpy(), "s-", color="k", label="observed")
    ax.set_xlabel("round")
    ax.set_ylabel("mean network ell")
    ax.legend()
    ax.set_title("free rollout vs observed (F4)")

    ax = axes[0][2]
    fr_all = preds[(preds["kind"] == "free_rollout") & (preds["model"] == "F4")]
    if len(fr_all):
        cons = fr_all.groupby(["world_id", "condition"]).first().reset_index()
        ax.plot(cons["pred_consensus_prob"], cons["obs_upstream_majority"], "o", alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        annotate_n(ax, len(cons), "episodes")
    ax.set_xlabel("predicted P(upstream majority)")
    ax.set_ylabel("observed outcome")
    ax.set_title("final consensus probability")

    ax = axes[0][3]
    ax.bar(comp["model"], comp["generalization_gap"], color=COLORS["model"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("endogenous - exogenous RMSE")
    ax.set_title("generalization gap by model")
    finish(cfg, fig, "fig06_composition_generalization", comp)
