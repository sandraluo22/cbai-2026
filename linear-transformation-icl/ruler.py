"""The ruler: linear-transformability (and alternatives) between activation sets.

Given two matrices ``A, B`` each ``[N, hidden]`` whose rows are *matched* samples
(e.g. the same rollouts at two positions, or different sources at one position),
we ask how well one set maps onto the other.

  * Primary score: **out-of-sample R²** of the best ridge linear map A -> B, fit on
    a train split and scored on held-out rows. (In-sample R² is ~1 in high
    dimensions and is meaningless — we never report it.)
  * Alternatives for cross-checking whether structure is metric-dependent:
    linear CKA, RBF CKA, and (normalized) orthogonal Procrustes distance.

All scores are symmetric-ish similarity rulers except Procrustes (a distance);
we report 1 - normalized-Procrustes as a similarity where convenient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Train/test split                                                             #
# --------------------------------------------------------------------------- #
def train_test_split(n: int, train_frac: float, rng: np.random.Generator):
    idx = rng.permutation(n)
    k = max(1, int(round(train_frac * n)))
    k = min(k, n - 1)                      # always leave >=1 test row
    return idx[:k], idx[k:]


# --------------------------------------------------------------------------- #
# Linear-map out-of-sample R²                                                  #
# --------------------------------------------------------------------------- #
def _fit_ridge(A: np.ndarray, B: np.ndarray, ridge: float) -> np.ndarray:
    """Closed-form ridge solution W for A_aug W ≈ B, with a bias column on A.

    Returns W of shape (hidden_A + 1, hidden_B). Bias (last row) is unregularized.
    """
    n, d = A.shape
    A_aug = np.concatenate([A, np.ones((n, 1))], axis=1)        # bias column
    G = A_aug.T @ A_aug
    reg = ridge * np.eye(d + 1)
    reg[-1, -1] = 0.0                                           # don't regularize bias
    W = np.linalg.solve(G + reg, A_aug.T @ B)
    return W


def _apply(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    return A_aug @ W


def r2_score(B_true: np.ndarray, B_pred: np.ndarray) -> float:
    """Aggregated R² across all output dims (1 - SSE/SST about the test mean)."""
    sse = np.sum((B_true - B_pred) ** 2)
    sst = np.sum((B_true - B_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else 0.0


def linear_transform_r2(A: np.ndarray, B: np.ndarray, *, ridge: float = 1.0,
                        train_frac: float = 0.7, rng: np.random.Generator,
                        n_splits: int = 5) -> float:
    """Out-of-sample R² of the best ridge linear map A -> B, averaged over splits."""
    n = A.shape[0]
    if n < 4:
        return float("nan")
    scores = []
    for _ in range(n_splits):
        tr, te = train_test_split(n, train_frac, rng)
        W = _fit_ridge(A[tr], B[tr], ridge)
        scores.append(r2_score(B[te], _apply(A[te], W)))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# CKA                                                                          #
# --------------------------------------------------------------------------- #
def _center_gram(G: np.ndarray) -> np.ndarray:
    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H


def cka_linear(A: np.ndarray, B: np.ndarray) -> float:
    Ka, Kb = A @ A.T, B @ B.T
    Kac, Kbc = _center_gram(Ka), _center_gram(Kb)
    hsic = np.sum(Kac * Kbc)
    denom = np.sqrt(np.sum(Kac * Kac) * np.sum(Kbc * Kbc))
    return float(hsic / denom) if denom > 0 else 0.0


def cka_rbf(A: np.ndarray, B: np.ndarray, sigma_frac: float = 0.5) -> float:
    def rbf(X):
        sq = np.sum(X * X, 1)
        d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / (2 * (sigma_frac ** 2) * med + 1e-12))
    Kac, Kbc = _center_gram(rbf(A)), _center_gram(rbf(B))
    hsic = np.sum(Kac * Kbc)
    denom = np.sqrt(np.sum(Kac * Kac) * np.sum(Kbc * Kbc))
    return float(hsic / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Procrustes                                                                   #
# --------------------------------------------------------------------------- #
def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized orthogonal Procrustes distance in [0, ~2]. 0 = identical up to rotation."""
    A0 = A - A.mean(0)
    B0 = B - B.mean(0)
    A0 /= np.linalg.norm(A0) + 1e-12
    B0 /= np.linalg.norm(B0) + 1e-12
    # pad to common width
    d = max(A0.shape[1], B0.shape[1])
    A0 = np.pad(A0, ((0, 0), (0, d - A0.shape[1])))
    B0 = np.pad(B0, ((0, 0), (0, d - B0.shape[1])))
    M = B0.T @ A0
    U, s, Vt = np.linalg.svd(M)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * s.sum())))


# --------------------------------------------------------------------------- #
# Unified interface                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class RulerConfig:
    method: str = "r2"             # "r2" | "cka" | "cka_rbf" | "procrustes_sim"
    ridge: float = 1.0
    train_frac: float = 0.7
    n_splits: int = 5


def ruler(A: np.ndarray, B: np.ndarray, cfg: RulerConfig, rng: np.random.Generator) -> float:
    if cfg.method == "r2":
        return linear_transform_r2(A, B, ridge=cfg.ridge, train_frac=cfg.train_frac,
                                   rng=rng, n_splits=cfg.n_splits)
    if cfg.method == "cka":
        return cka_linear(A, B)
    if cfg.method == "cka_rbf":
        return cka_rbf(A, B)
    if cfg.method == "procrustes_sim":
        return 1.0 - procrustes_distance(A, B)
    raise ValueError(f"unknown ruler method {cfg.method!r}")
