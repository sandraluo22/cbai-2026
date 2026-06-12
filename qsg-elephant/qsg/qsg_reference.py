"""Quantized Simplex Gossip (QSG) — pure-numpy reference / null model.

This module is the analytic ground truth for the experiment harness:

  * It defines the QSG *communication channel* (Hard / Top-m / Soft) used by BOTH
    the numeric simulator here AND the LLM-driven engine (``engine.py`` imports
    ``sample_message`` / ``qsg_update`` from here so there is exactly one
    implementation of the math).
  * It provides a fast, LLM-free simulator so the sweep / plotting / analysis
    pipeline can be validated before burning any GPU time, and so LLM dynamics
    can be compared against the neutral-drift null model.

Model recap
-----------
Each agent ``i`` holds a belief ``x_i`` on the simplex Δ^(K-1) over K candidate
answers.  On each interaction an ordered pair (speaker S, listener L), S != L, is
drawn uniformly from the N(N-1) ordered pairs (well-mixed topology).  The speaker
emits a quantized message ``y`` sampled from ``x_S`` and the listener relaxes
toward it::

    x_L <- (1 - alpha) * x_L + alpha * y

Communication bandwidth ``m``:
  * m == 1            (Hard) : y = e_{k*},   k* ~ Cat(x_S)
  * 1 < m < inf       (Top-m): y = (1/m) sum_j e_{k_j}, k_j ~ Cat(x_S) iid
  * m == inf / None   (Soft) : y = x_S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Sentinel for the Soft channel (m = infinity).
SOFT = float("inf")


# --------------------------------------------------------------------------- #
# Channel                                                                      #
# --------------------------------------------------------------------------- #
def sample_message(x_s: np.ndarray, m: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a quantized QSG message ``y`` from speaker belief ``x_s``.

    Parameters
    ----------
    x_s : (K,) float array on the simplex (will be renormalized defensively).
    m   : bandwidth. ``1`` -> Hard, finite ``>1`` -> Top-m (m iid draws averaged),
          ``float('inf')`` / ``None`` -> Soft (return the full distribution).
    rng : seeded ``np.random.Generator`` *dedicated to QSG sampling* (kept
          separate from any model-sampling RNG for reproducibility).

    Returns
    -------
    y : (K,) float array on the simplex.
    """
    p = np.asarray(x_s, dtype=np.float64)
    p = p / p.sum()
    K = p.shape[0]

    if m is None or m == SOFT:
        return p.copy()

    m_int = int(m)
    if m_int < 1:
        raise ValueError(f"bandwidth m must be >= 1 or inf, got {m!r}")

    draws = rng.choice(K, size=m_int, p=p)
    y = np.bincount(draws, minlength=K).astype(np.float64) / m_int
    return y


def qsg_update(x_l: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Convex QSG adaptation update on the public simplex."""
    return (1.0 - alpha) * np.asarray(x_l, dtype=np.float64) + alpha * np.asarray(
        y, dtype=np.float64
    )


def draw_pair(n: int, rng: np.random.Generator) -> tuple[int, int]:
    """Uniform ordered (speaker, listener) pair, speaker != listener."""
    s = int(rng.integers(n))
    l = int(rng.integers(n - 1))
    if l >= s:
        l += 1
    return s, l


# --------------------------------------------------------------------------- #
# Order parameters                                                            #
# --------------------------------------------------------------------------- #
def polarization(beliefs: np.ndarray) -> float:
    """U = ||x_bar||_2^2  (mean-belief polarization order parameter)."""
    x_bar = beliefs.mean(axis=0)
    return float(np.dot(x_bar, x_bar))


def disagreement_energy(beliefs: np.ndarray) -> float:
    """V = sum_i ||x_i - x_bar||_2^2  (disagreement / spread order parameter)."""
    x_bar = beliefs.mean(axis=0)
    d = beliefs - x_bar
    return float(np.sum(d * d))


# --------------------------------------------------------------------------- #
# Reference simulator                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceResult:
    """Trajectory output of the numeric QSG simulator."""

    beliefs: np.ndarray            # (rounds + 1, N, K)
    U: np.ndarray                  # (rounds + 1,)  polarization per round
    V: np.ndarray                  # (rounds + 1,)  disagreement energy per round
    mean_traj: np.ndarray          # (rounds + 1, K) population mean per round
    config: dict = field(default_factory=dict)

    @property
    def final_mean(self) -> np.ndarray:
        return self.mean_traj[-1]


def random_simplex(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """N iid samples from a symmetric Dirichlet(1) over Δ^(K-1)."""
    x = rng.dirichlet(np.ones(K), size=n)
    return x


def run_reference(
    n: int,
    K: int,
    alpha: float,
    m: float,
    rounds: int,
    seed: int,
    *,
    x0: Optional[np.ndarray] = None,
    interactions_per_round: Optional[int] = None,
    ground_truth: Optional[int] = None,
    selection_strength: float = 0.0,
) -> ReferenceResult:
    """Run the pure-numpy QSG null model.

    One "round" performs ``interactions_per_round`` speaker/listener interactions
    (default: N, i.e. on average one update per agent per round) so the round
    index is comparable to the LLM engine, where each round every agent listens
    once.

    ``ground_truth`` / ``selection_strength`` add an optional weak selection
    nudge toward the true label each round (a crude analytic stand-in for the
    LLM's persistent private observation in two-layer mode); leave
    ``selection_strength=0`` for the neutral-drift null.
    """
    rng = np.random.default_rng(seed)
    if x0 is None:
        beliefs = random_simplex(n, K, rng)
    else:
        beliefs = np.array(x0, dtype=np.float64)
        beliefs = beliefs / beliefs.sum(axis=1, keepdims=True)

    if interactions_per_round is None:
        interactions_per_round = n

    U = np.empty(rounds + 1)
    V = np.empty(rounds + 1)
    mean_traj = np.empty((rounds + 1, K))
    traj = np.empty((rounds + 1, n, K))

    def record(t: int) -> None:
        traj[t] = beliefs
        U[t] = polarization(beliefs)
        V[t] = disagreement_energy(beliefs)
        mean_traj[t] = beliefs.mean(axis=0)

    record(0)
    for t in range(1, rounds + 1):
        for _ in range(interactions_per_round):
            s, l = draw_pair(n, rng)
            y = sample_message(beliefs[s], m, rng)
            beliefs[l] = qsg_update(beliefs[l], y, alpha)
        if ground_truth is not None and selection_strength > 0.0:
            sel = np.zeros(K)
            sel[ground_truth] = 1.0
            beliefs = qsg_update(beliefs, sel, selection_strength)
            beliefs /= beliefs.sum(axis=1, keepdims=True)
        record(t)

    return ReferenceResult(
        beliefs=traj,
        U=U,
        V=V,
        mean_traj=mean_traj,
        config=dict(
            n=n, K=K, alpha=alpha, m=m, rounds=rounds, seed=seed,
            interactions_per_round=interactions_per_round,
            ground_truth=ground_truth, selection_strength=selection_strength,
        ),
    )


def consensus_time(U: np.ndarray, threshold: float) -> Optional[int]:
    """First round index at which polarization U crosses ``threshold``."""
    idx = np.where(U >= threshold)[0]
    return int(idx[0]) if idx.size else None
