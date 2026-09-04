"""Figure 11: empirical network Jacobian."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.protocol import graph_distance
from ..config import Config
from ..world.generator import load_worlds
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    jac = pd.read_parquet(cfg.paths.runs / "jacobian_results.parquet")
    summ = pd.read_parquet(cfg.paths.runs / "jacobian_summary.parquet")
    worlds = load_worlds(cfg)
    fig, axes = new_fig(1, 4, width=13, height=3.0)

    j_only = jac[jac["condition"] == "jacobian"]
    n = int(j_only["agent_id"].max()) + 1
    mat = np.zeros((n, n))
    counts = np.zeros((n, n))
    for _, r in j_only.iterrows():
        mat[int(r["agent_id"]), int(r["source_agent"])] += r["jacobian_value"]
        counts[int(r["agent_id"]), int(r["source_agent"])] += 1
    mat = np.divide(mat, np.maximum(counts, 1))
    ax = axes[0][0]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.abs(mat).max() or -1, vmax=np.abs(mat).max() or 1)
    ax.set_xlabel("source agent j")
    ax.set_ylabel("target agent i")
    ax.set_title("mean intervention Jacobian J[i,j]")
    fig.colorbar(im, ax=ax, shrink=0.8)
    annotate_n(ax, j_only["world_id"].nunique(), "worlds")

    ax = axes[0][1]
    imp = jac[jac["condition"] == "impulse_response"].copy()
    topo = cfg.network.topology
    imp["distance"] = [
        graph_distance(topo, worlds[w].n_agents, int(j), int(i))
        for w, j, i in zip(imp["world_id"], imp["source_agent"], imp["agent_id"])
    ]
    g = imp.groupby("distance")["jacobian_value"].apply(lambda s: float(np.mean(np.abs(s))))
    ax.plot(g.index, g.to_numpy(), "o-", color=COLORS["model"])
    ax.set_xlabel("graph distance from source")
    ax.set_ylabel("|impulse response| (ell per unit steer)")
    ax.set_title("impulse magnitude by distance")

    ax = axes[0][2]
    g2 = summ.groupby("round")["spectral_radius"].mean()
    ax.plot(g2.index, g2.to_numpy(), "o-", color=COLORS["positive"])
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("spectral radius (local diagnostic)")
    ax.set_title("spectral radius by round")

    ax = axes[0][3]
    obs, pred = [], []
    for wid, grp in imp.groupby("world_id"):
        world = worlds[wid]
        n_w = world.n_agents
        jmats = {}
        for t, g3 in j_only[j_only["world_id"] == wid].groupby("round"):
            m = np.zeros((n_w, n_w))
            for _, r in g3.iterrows():
                m[int(r["agent_id"]), int(r["source_agent"])] = r["jacobian_value"]
            jmats[int(t)] = m
        rounds = sorted(jmats)
        for j in grp["source_agent"].unique():
            e = np.zeros(n_w)
            e[int(j)] = 1.0
            prod = e.copy()
            for t in rounds:
                prod = jmats[t] @ prod
                sub = grp[(grp["source_agent"] == j) & (grp["round"] == t)]
                for _, r in sub.iterrows():
                    obs.append(r["jacobian_value"])
                    pred.append(prod[int(r["agent_id"])])
    if obs:
        ax.plot(obs, pred, ".", alpha=0.4, color=COLORS["model"])
        lim = max(1e-3, float(np.abs(obs).max()), float(np.abs(pred).max()))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        annotate_n(ax, len(obs))
    ax.set_xlabel("observed multi-round response")
    ax.set_ylabel("product-of-Jacobians prediction")
    ax.set_title("observed vs Jacobian-predicted")
    finish(cfg, fig, "fig11_empirical_jacobian", summ)
