"""Paper-reproduction sanity check (run BEFORE trusting alignment).

Park, Lee, Lubana et al.: at high in-context length, the residual-stream
representation of the concept words reorganizes to match the GRAPH geometry. We
reproduce the core effect by:

  1. taking per-node MEAN activations over high-context occurrences (here only,
     means are fine -- they are the paper's object; alignment uses per-occurrence
     vectors, never these means),
  2. PCA to 2D,
  3. checking the 2D layout recovers the grid: correlation between PCA pairwise
     distances and grid (Manhattan) distances, plus a saved scatter plot.

A clearly positive distance correlation that grows with context length means
activations were captured correctly.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from config import Config
from graph import Graph
from models import CaptureResult
from align import _pca


def per_node_means(cap: CaptureResult, layer: int, graph: Graph,
                   min_context: int) -> np.ndarray:
    """[n_nodes, d] mean activation per node over occurrences with
    context_length >= min_context. Nodes with no such occurrence -> NaN row."""
    acts = cap.acts[layer].astype(np.float64)
    node = cap.meta["node"]
    ctx = cap.meta["context_length"]
    keep = ctx >= min_context
    out = np.full((graph.n_nodes, acts.shape[1]), np.nan)
    for n in range(graph.n_nodes):
        m = keep & (node == n)
        if m.any():
            out[n] = acts[m].mean(0)
    return out


def grid_recovery_score(node_means: np.ndarray, graph: Graph) -> Dict:
    """PCA node means to 2D; correlate PCA pairwise distances with grid distances.
    Returns the correlation, the 2D coords, and the explained variance."""
    valid = ~np.isnan(node_means).any(1)
    X = node_means[valid]
    mean, comps = _pca(X, 2)
    coords2d = (X - mean) @ comps.T                      # [n_valid, 2]

    full = np.full((graph.n_nodes, 2), np.nan)
    full[valid] = coords2d

    gd = graph.grid_distance_matrix()
    iu = np.triu_indices(graph.n_nodes, k=1)
    # restrict to pairs where both nodes are valid
    vmask = valid[iu[0]] & valid[iu[1]]
    pca_d = np.linalg.norm(full[iu[0]] - full[iu[1]], axis=1)
    pairs = vmask & ~np.isnan(pca_d)
    if pairs.sum() < 3:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pca_d[pairs], gd[iu][pairs])[0, 1])

    return {"distance_corr": corr, "coords2d": full, "n_valid": int(valid.sum())}


def reproduce_for_model(cap: CaptureResult, layer: int, graph: Graph, cfg: Config,
                        tag: str, plot_path: str | None = None) -> Dict:
    """Distance-correlation as a function of context-length checkpoint, plus an
    optional scatter plot of the highest-context 2D layout."""
    results = []
    for C in cfg.context_checkpoints:
        nm = per_node_means(cap, layer, graph, min_context=C)
        score = grid_recovery_score(nm, graph)
        results.append({"min_context": C, "distance_corr": score["distance_corr"],
                        "n_valid": score["n_valid"]})

    top = grid_recovery_score(
        per_node_means(cap, layer, graph, cfg.context_checkpoints[-1]), graph)

    if plot_path is not None:
        _plot_layout(top["coords2d"], graph, plot_path,
                     title=f"{tag}: grid recovery (layer {layer}, "
                           f"ctx>={cfg.context_checkpoints[-1]}, "
                           f"r={top['distance_corr']:.2f})")

    return {"tag": tag, "layer": layer, "by_context": results,
            "top_distance_corr": top["distance_corr"]}


def _plot_layout(coords2d: np.ndarray, graph: Graph, path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # plotting is optional
        print(f"[reproduce] skipping plot ({e})")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    # draw graph edges in PCA space
    for n in range(graph.n_nodes):
        for m in graph.neighbors(n):
            if m > n and not (np.isnan(coords2d[n]).any() or np.isnan(coords2d[m]).any()):
                ax.plot([coords2d[n, 0], coords2d[m, 0]],
                        [coords2d[n, 1], coords2d[m, 1]], color="0.8", zorder=1)
    for n in range(graph.n_nodes):
        if np.isnan(coords2d[n]).any():
            continue
        ax.scatter(*coords2d[n], zorder=2)
        ax.annotate(graph.words[n], coords2d[n], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[reproduce] wrote {path}")
