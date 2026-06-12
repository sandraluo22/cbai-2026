"""Similarity matrices over the activation store + transition detection.

S[i,j] = ruler(activations at t_i, activations at t_j), N-axis = rollouts of one
source (5a). The SAME matrix is read two ways: raw cell values (linear-
transformability) and block structure along the diagonal (phase transition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from activations import pooled_matrix, rollout_matrix, source_matrix
from ruler import RulerConfig, ruler


def _mats_for_axis(arr, meta, layer, axis, source_id):
    pos = meta["positions_t"]
    if axis == "pooled":
        return pos, [pooled_matrix(arr, meta, t, layer) for t in pos]
    if axis == "source":
        return pos, [source_matrix(arr, meta, t, layer) for t in pos]
    return pos, [rollout_matrix(arr, meta, source_id, t, layer) for t in pos]


def build_similarity_matrix(arr, meta, layer, ruler_cfg: RulerConfig,
                            rng: np.random.Generator, axis: str = "pooled",
                            source_id: int = 0) -> np.ndarray:
    """Position x position similarity for a layer.

    axis="pooled" (default): N = all (source,rollout) points, not averaged.
    axis="source":           N = sources (each row a source's mean state).
    axis="rollout":           N = rollouts of ``source_id``.
    """
    pos, mats = _mats_for_axis(arr, meta, layer, axis, source_id)
    P = len(pos)
    S = np.empty((P, P))
    for i in range(P):
        for j in range(P):
            S[i, j] = ruler(mats[i], mats[j], ruler_cfg, rng)
    return S


def save_heatmap(S, positions_t, out_dir: Path, prefix: str, metric_label: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}.npy", S)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(positions_t))); ax.set_xticklabels(positions_t, rotation=45)
    ax.set_yticks(range(len(positions_t))); ax.set_yticklabels(positions_t)
    ax.set_title(f"Pairwise state similarity\n({metric_label})")
    ax.set_xlabel("context length t_j"); ax.set_ylabel("context length t_i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140); plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# Transition detection (block structure along the diagonal) — used in step 4   #
# --------------------------------------------------------------------------- #
def detect_transition(curve: np.ndarray, positions_t: list[int], frac: float = 0.5) -> dict:
    """Transition location + sharpness of a 1-D (monotone-ish) convergence curve.

    location_t (primary): threshold crossing — the first t where the curve reaches
      ``frac`` of the way through its own [min,max] range (robust 'halfway-rise'
      point; monotone curves give a stable, comparable location across runs).
    changepoint_t: best piecewise-constant split (kept for reference; noisy on
      smooth high-floor curves, which is why it is not primary).
    sharpness: max single-step normalized increase (how abruptly the curve rises).
    """
    x = np.asarray(curve, dtype=float)
    n = x.shape[0]
    if n < 3:
        return {"location_t": None, "sharpness": 0.0}
    lo, hi = float(x.min()), float(x.max())
    rng = (hi - lo) + 1e-9
    norm = (x - lo) / rng
    cross = next((i for i, v in enumerate(norm) if v >= frac), n - 1)
    # changepoint (reference)
    best, best_k = np.inf, 1
    for k in range(1, n):
        sse = ((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum()
        if sse < best:
            best, best_k = sse, k
    steps = np.diff(norm)
    return {"location_t": int(positions_t[cross]), "cross_index": cross,
            "changepoint_t": int(positions_t[best_k]),
            "sharpness": float(steps.max()) if steps.size else 0.0,
            "range": rng}
