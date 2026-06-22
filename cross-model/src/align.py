"""Cross-model alignment fitting and metrics.

Given paired per-occurrence activations X_A (Llama, d_A) and X_B (Gemma, d_B)
over the SAME (walk_id, step) occurrences, fit a map A->B and score it. Hidden
sizes differ (4096 vs 3584) so the map is RECTANGULAR; true orthogonal
Procrustes is unavailable in the full space. Two approaches:

  (a) ridge-regularized linear regression A->B in full space
  (b) reduce each to a shared top-k PCA subspace, then orthogonal Procrustes

Splits are BY walk_id so train/test walks never overlap. A well-posedness guard
asserts n_samples vs the map's degrees of freedom and warns loudly otherwise --
this is the failure mode we most want to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np

from config import Config
from models import CaptureResult


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def pair_occurrences(
    cap_a: CaptureResult, cap_b: CaptureResult, layer_a: int, layer_b: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return X_A, X_B aligned row-for-row by (walk_id, step), plus the shared
    metadata for those rows. Captures are produced from the same walks in the
    same order, so we assert identity rather than join, then return float64."""
    ma, mb = cap_a.meta, cap_b.meta
    assert np.array_equal(ma["walk_id"], mb["walk_id"]), "walk_id mismatch"
    assert np.array_equal(ma["step"], mb["step"]), "step mismatch"
    X_A = cap_a.acts[layer_a].astype(np.float64)
    X_B = cap_b.acts[layer_b].astype(np.float64)
    return X_A, X_B, {k: v.copy() for k, v in ma.items()}


def split_by_walk(meta: Dict[str, np.ndarray], test_frac: float, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks; whole walks go to one side or the other."""
    walk_ids = np.unique(meta["walk_id"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(walk_ids)
    n_test = max(1, int(round(len(walk_ids) * test_frac)))
    test_walks = set(perm[:n_test].tolist())
    test_mask = np.array([w in test_walks for w in meta["walk_id"]])
    return ~test_mask, test_mask


# ---------------------------------------------------------------------------
# Well-posedness guard
# ---------------------------------------------------------------------------
def check_wellposed(n_samples: int, n_params: int, ratio: float, tag: str) -> Dict:
    ok = n_samples >= ratio * n_params
    msg = (f"[wellposed:{tag}] n_samples={n_samples:,} vs n_params={n_params:,} "
           f"(ratio={n_samples / max(1, n_params):.2f}, need >= {ratio})")
    if not ok:
        warnings.warn("UNDER-DETERMINED " + msg, stacklevel=2)
    return {"tag": tag, "n_samples": n_samples, "n_params": n_params,
            "ratio": n_samples / max(1, n_params), "ok": bool(ok), "msg": msg}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall R^2 (variance-weighted across dims): 1 - SS_res/SS_tot."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Basis-free linear CKA. Handles differing widths; centered features."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    nx = np.linalg.norm(Xc.T @ Xc, "fro")
    ny = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(hsic / (nx * ny)) if nx > 0 and ny > 0 else float("nan")


# ---------------------------------------------------------------------------
# (a) full-space ridge regression A -> B
# ---------------------------------------------------------------------------
@dataclass
class RidgeMap:
    W: np.ndarray            # [d_A, d_B]
    mean_a: np.ndarray
    mean_b: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_a) @ self.W + self.mean_b


def fit_ridge(X_A: np.ndarray, X_B: np.ndarray, alpha: float) -> RidgeMap:
    mean_a, mean_b = X_A.mean(0), X_B.mean(0)
    Xc, Yc = X_A - mean_a, X_B - mean_b
    d_a = Xc.shape[1]
    G = Xc.T @ Xc + alpha * np.eye(d_a)
    W = np.linalg.solve(G, Xc.T @ Yc)
    return RidgeMap(W=W, mean_a=mean_a, mean_b=mean_b)


# ---------------------------------------------------------------------------
# (b) shared top-k PCA subspace + orthogonal Procrustes
# ---------------------------------------------------------------------------
@dataclass
class ProcrustesMap:
    mean_a: np.ndarray
    comps_a: np.ndarray      # [k, d_A]
    mean_b: np.ndarray
    comps_b: np.ndarray      # [k, d_B]
    R: np.ndarray            # [k, k] orthogonal, maps Z_A -> Z_B

    def project_a(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_a) @ self.comps_a.T

    def project_b(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_b) @ self.comps_b.T

    def predict_subspace(self, X_A: np.ndarray) -> np.ndarray:
        return self.project_a(X_A) @ self.R


def _randomized_components(Xc: np.ndarray, k: int, n_iter: int = 4,
                           oversample: int = 10, seed: int = 0) -> np.ndarray:
    """Top-k right singular vectors of centered Xc via randomized SVD
    (Halko et al.). Orders of magnitude faster than a full SVD when k << dim,
    which is our case (k~100, dim~4000, n~150k). Seeded for reproducibility."""
    m, n = Xc.shape
    r = min(k + oversample, min(m, n))
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(Xc @ rng.standard_normal((n, r)))      # m x r
    for _ in range(n_iter):                                    # power iterations
        Q, _ = np.linalg.qr(Xc.T @ Q)                          # n x r
        Q, _ = np.linalg.qr(Xc @ Q)                            # m x r
    B = Q.T @ Xc                                               # r x n (small)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt[:k]


def _pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, top-k principal axes [k, d]). Uses an exact SVD for small
    inputs (e.g. the 16-node reproduce check) and randomized SVD for the large
    per-occurrence matrices."""
    mean = X.mean(0)
    Xc = X - mean
    m, n = Xc.shape
    if min(m, n) < 50 or min(m, n) <= k + 10:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)      # exact
        return mean, Vt[:k]
    return mean, _randomized_components(Xc, k)


def fit_procrustes(X_A: np.ndarray, X_B: np.ndarray, k: int) -> ProcrustesMap:
    mean_a, comps_a = _pca(X_A, k)
    mean_b, comps_b = _pca(X_B, k)
    Z_A = (X_A - mean_a) @ comps_a.T
    Z_B = (X_B - mean_b) @ comps_b.T
    # orthogonal Procrustes: min ||Z_A R - Z_B||, R = U V^T from SVD(Z_A^T Z_B)
    U, _, Vt = np.linalg.svd(Z_A.T @ Z_B)
    R = U @ Vt
    return ProcrustesMap(mean_a, comps_a, mean_b, comps_b, R)


def procrustes_residual(pm: ProcrustesMap, X_A: np.ndarray, X_B: np.ndarray) -> float:
    """Normalized residual ||Z_A R - Z_B||_F^2 / ||Z_B||_F^2 in the subspace."""
    Z_B = pm.project_b(X_B)
    pred = pm.predict_subspace(X_A)
    return float(((pred - Z_B) ** 2).sum() / (Z_B ** 2).sum())


# ---------------------------------------------------------------------------
# Per-context-length selection
# ---------------------------------------------------------------------------
def context_bin_mask(meta: Dict[str, np.ndarray], checkpoint: int,
                     window: float) -> np.ndarray:
    lo, hi = checkpoint * (1 - window), checkpoint * (1 + window)
    cl = meta["context_length"]
    return (cl >= lo) & (cl <= hi)


# ---------------------------------------------------------------------------
# Orchestrated analysis
# ---------------------------------------------------------------------------
@dataclass
class AlignmentReport:
    wellposed: List[Dict] = field(default_factory=list)
    ridge: Dict = field(default_factory=dict)
    procrustes: Dict = field(default_factory=dict)
    cka_overall: float = float("nan")
    by_context: List[Dict] = field(default_factory=list)
    matched_vs_mismatched: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return self.__dict__


def run_alignment(cap_a: CaptureResult, cap_b: CaptureResult, cfg: Config
                  ) -> AlignmentReport:
    X_A, X_B, meta = pair_occurrences(cap_a, cap_b, cfg.align_layer_a, cfg.align_layer_b)
    d_a, d_b = X_A.shape[1], X_B.shape[1]
    train, test = split_by_walk(meta, cfg.test_frac, cfg.seed)
    rep = AlignmentReport()

    # --- well-posedness for both maps (counted on the TRAIN set) -----------
    n_train = int(train.sum())
    rep.wellposed.append(check_wellposed(
        n_train, d_a * d_b + d_b, cfg.wellposed_ratio, "ridge_full"))
    rep.wellposed.append(check_wellposed(
        n_train, cfg.pca_k * (cfg.pca_k - 1) // 2, cfg.wellposed_ratio,
        "procrustes_subspace"))

    # --- (a) ridge ---------------------------------------------------------
    rm = fit_ridge(X_A[train], X_B[train], cfg.ridge_alpha)
    rep.ridge = {
        "alpha": cfg.ridge_alpha,
        "r2_train": r2(X_B[train], rm.predict(X_A[train])),
        "r2_test": r2(X_B[test], rm.predict(X_A[test])),
        "n_params": d_a * d_b + d_b,
    }

    # --- (b) PCA + Procrustes ---------------------------------------------
    pm = fit_procrustes(X_A[train], X_B[train], cfg.pca_k)
    rep.procrustes = {
        "k": cfg.pca_k,
        "residual_train": procrustes_residual(pm, X_A[train], X_B[train]),
        "residual_test": procrustes_residual(pm, X_A[test], X_B[test]),
        "r2_test_subspace": r2(pm.project_b(X_B[test]), pm.predict_subspace(X_A[test])),
    }

    # --- (3) CKA basis-free cross-check (held-out) -------------------------
    rep.cka_overall = linear_cka(X_A[test], X_B[test])

    # --- (4) trajectory: fit pooled, evaluate per context length ----------
    for C in cfg.context_checkpoints:
        m = test & context_bin_mask(meta, C, cfg.checkpoint_window)
        if m.sum() < 5:
            rep.by_context.append({"context": C, "n": int(m.sum()), "skipped": True})
            continue
        rep.by_context.append({
            "context": C,
            "n": int(m.sum()),
            "ridge_r2": r2(X_B[m], rm.predict(X_A[m])),
            "procrustes_residual": procrustes_residual(pm, X_A[m], X_B[m]),
            "cka": linear_cka(X_A[m], X_B[m]),
        })

    # --- (5) matched vs mismatched context control ------------------------
    rep.matched_vs_mismatched = _matched_control(rm, X_A, X_B, meta, test, cfg)
    return rep


def _matched_control(rm: RidgeMap, X_A: np.ndarray, X_B: np.ndarray,
                     meta: Dict[str, np.ndarray], test: np.ndarray, cfg: Config
                     ) -> Dict:
    """Evaluate the (pooled) map with A at context L and B at the SAME L
    (matched) vs B at a DIFFERENT L (mismatched). If matched is not better, the
    alignment tracks static geometry, not the in-context process.

    We compare on equal-sized random pairings within the held-out set so the
    only difference is whether the B rows share A's context bin.
    """
    rng = np.random.default_rng(cfg.seed + 1)
    checkpoints = list(cfg.context_checkpoints)
    rows = []
    for i, C in enumerate(checkpoints):
        a_mask = test & context_bin_mask(meta, C, cfg.checkpoint_window)
        if a_mask.sum() < 5:
            continue
        # matched: same context bin for B
        b_match = a_mask
        # mismatched: B drawn from a different checkpoint bin
        C_other = checkpoints[(i + 1) % len(checkpoints)]
        b_mis = test & context_bin_mask(meta, C_other, cfg.checkpoint_window)
        if b_mis.sum() < 5:
            continue

        # predictions from A's context-C occurrences
        pred = rm.predict(X_A[a_mask])
        # matched target: the true B activations at the SAME occurrences
        r2_matched = r2(X_B[b_match], pred)
        # mismatched: compare A@C predictions against B@C_other targets of equal
        # count (random subset), measuring fit to a different-context geometry
        idx_mis = np.where(b_mis)[0]
        take = min(len(pred), len(idx_mis))
        sub = rng.choice(idx_mis, size=take, replace=False)
        r2_mismatched = r2(X_B[sub], pred[:take])
        rows.append({"context": C, "context_other": C_other,
                     "r2_matched": r2_matched, "r2_mismatched": r2_mismatched,
                     "matched_better": bool(r2_matched > r2_mismatched)})
    summary = {
        "per_context": rows,
        "matched_wins": int(sum(r["matched_better"] for r in rows)),
        "total": len(rows),
    }
    return summary
