"""Convergence curves, the null baseline, ground-truth check, and plots.

The rollout-convergence curve and the phase-transition matrix are two reads of the
same activations. The convergence curve at position t is R²(t) = ruler(states at t
-> states at the final analyzed position): how well the state at t already maps to
the converged state. A point only counts as "converged" if it sits ABOVE the null
band (Section 5d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from activations import pooled_matrix, rollout_matrix, source_matrix
from ruler import RulerConfig, ruler


def _mats(arr, meta, layer, axis, source_id):
    pos = meta["positions_t"]
    if axis == "pooled":
        return pos, [pooled_matrix(arr, meta, t, layer) for t in pos]
    if axis == "source":
        return pos, [source_matrix(arr, meta, t, layer) for t in pos]
    return pos, [rollout_matrix(arr, meta, source_id, t, layer) for t in pos]


# --------------------------------------------------------------------------- #
# Convergence curve: state@t -> converged (final) state                        #
# --------------------------------------------------------------------------- #
def convergence_curve(arr, meta, layer, ruler_cfg: RulerConfig,
                      rng: np.random.Generator, axis: str = "pooled",
                      source_id: int = 0) -> dict:
    pos, mats = _mats(arr, meta, layer, axis, source_id)
    ref = mats[-1]                                   # final (separated) arrangement
    r2 = np.array([ruler(m, ref, ruler_cfg, rng) for m in mats])
    return {"positions_t": pos, "r2": r2}


# --------------------------------------------------------------------------- #
# Null baseline (Section 5d): shuffled rollouts, other sources, matched noise  #
# --------------------------------------------------------------------------- #
def null_band(arr, meta, layer, ruler_cfg: RulerConfig, rng: np.random.Generator,
              axis: str = "pooled", source_id: int = 0, n_draws: int = 12) -> dict:
    """Null ruler values per position against the converged reference.

    Null types: (a) shuffled row correspondence (breaks the source/rollout match),
    (c) dimension/scale-matched Gaussian clouds. A point counts as "converged"
    only if it sits ABOVE this band.
    """
    pos, mats = _mats(arr, meta, layer, axis, source_id)
    ref = mats[-1]
    n, h = ref.shape
    vals_per_pos = []
    for m in mats:
        draws = []
        for _ in range(n_draws):
            perm = rng.permutation(n)                                  # (a) shuffle match
            draws.append(ruler(m[perm], ref, ruler_cfg, rng))
            g1 = rng.standard_normal((n, h)) * m.std() + m.mean()      # (c) matched noise
            g2 = rng.standard_normal((n, h)) * ref.std() + ref.mean()
            draws.append(ruler(g1, g2, ruler_cfg, rng))
        vals_per_pos.append(draws)
    vals = np.array(vals_per_pos)
    return {"positions_t": pos, "mean": vals.mean(1),
            "lo": np.percentile(vals, 5, axis=1), "hi": np.percentile(vals, 95, axis=1)}


# --------------------------------------------------------------------------- #
# Ground-truth check (Section 5c): analytic posterior convergence              #
# --------------------------------------------------------------------------- #
def analytic_convergence(rollouts, sources, positions_t) -> np.ndarray:
    """Mean posterior mass on the true source at each analyzed position t."""
    out = []
    for t in positions_t:
        vals = [r.posterior[t, r.source_id] for r in rollouts]
        out.append(np.mean(vals))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def save_convergence_plot(curve, null, out_dir: Path, prefix: str,
                          analytic=None, metric="R²"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pos = curve["positions_t"]
    x = np.arange(len(pos))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(x, curve["r2"], "-o", color="C0", label=f"ruler {metric}(t) → converged state")
    ax.fill_between(x, null["lo"], null["hi"], color="grey", alpha=0.25, label="null band (5–95%)")
    ax.plot(x, null["mean"], "--", color="grey", lw=1)
    if analytic is not None:
        ax2 = ax.twinx()
        ax2.plot(x, analytic, "-^", color="C3", alpha=0.7, label="analytic P(true source)")
        ax2.set_ylabel("analytic P(true source)", color="C3"); ax2.set_ylim(0.4, 1.02)
        ax2.legend(loc="lower right")
    ax.set_xticks(x); ax.set_xticklabels(pos)
    ax.set_xlabel("context length t"); ax.set_ylabel(f"ruler {metric}")
    ax.set_title("Rollout convergence vs context length (above null = converged)")
    ax.legend(loc="center right"); ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140); plt.close(fig)
    return png
