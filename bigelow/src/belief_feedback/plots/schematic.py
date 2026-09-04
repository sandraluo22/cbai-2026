"""Figure 1: world, provenance, assignment, topology, and design schematic."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..config import Config
from ..world.generator import load_worlds, worlds_in_split
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    worlds = load_worlds(cfg)
    world = worlds_in_split(worlds, "endogenous_test")[0]
    fig, axes = new_fig(1, 5, width=15, height=3.1)

    # (a) hypothesis + events
    ax = axes[0][0]
    llrs = [e.llr for e in world.events]
    ax.barh(
        range(len(llrs)), llrs,
        color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in llrs],
    )
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("event LLR (semantic)")
    ax.set_ylabel("latent event")
    ax.set_title("(a) latent evidence events")
    annotate_n(ax, len(llrs))

    # (b) provenance
    ax = axes[0][1]
    g = nx.DiGraph()
    for rep in world.reports:
        g.add_edge(rep.event_id.split("-")[-1], rep.report_id.split("-")[-1])
    pos = nx.spring_layout(g, seed=3)
    events = {e.event_id.split("-")[-1] for e in world.events}
    colors = [COLORS["oracle"] if n in events else COLORS["model"] for n in g.nodes]
    nx.draw(g, pos, ax=ax, node_size=90, node_color=colors, with_labels=False, arrowsize=6)
    ax.set_title("(b) events (green) -> reports (purple)")

    # (c) assignment matrix
    ax = axes[0][2]
    mat = np.zeros((world.n_agents, len(world.reports)))
    ridx = {rep.report_id: k for k, rep in enumerate(world.reports)}
    for a, rids in world.assignments.items():
        for rid in rids:
            mat[a, ridx[rid]] = 1
    ax.imshow(mat, aspect="auto", cmap="Greys")
    ax.set_xlabel("report")
    ax.set_ylabel("agent")
    ax.set_title("(c) private report assignment")

    # (d) topology
    ax = axes[0][3]
    ring = nx.cycle_graph(world.n_agents)
    nx.draw_circular(
        ring, ax=ax, node_size=260, node_color="#dddddd", edgecolors="k",
        with_labels=True, font_size=7,
    )
    ax.set_title("(d) bidirectional ring")

    # (e) design logic
    ax = axes[0][4]
    ax.axis("off")
    ax.text(
        0.02, 0.5,
        "exogenous worlds\n  -> fit G (emission)\n  -> fit F (receiver)\n\n"
        "endogenous worlds\n  -> live closed loop\n  -> predict with\n     F(adj x G(ell))",
        fontsize=9, family="monospace", va="center",
    )
    ax.set_title("(e) identification vs composition")

    data = pd.DataFrame(
        {
            "event_id": [e.event_id for e in world.events],
            "llr": llrs,
            "family": [e.family for e in world.events],
        }
    )
    finish(cfg, fig, "fig01_world_and_network_schematic", data)
