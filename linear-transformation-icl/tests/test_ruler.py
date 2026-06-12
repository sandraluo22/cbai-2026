"""Unit tests for the ruler on synthetic clouds (build step 2).

Spec requirement: B = A·R + noise -> high R²/CKA;  unrelated clouds -> near null.
And out-of-sample R² (not in-sample).
"""

import numpy as np
import pytest

import ruler


def _cloud(n, d, rng):
    return rng.standard_normal((n, d))


def test_linear_related_high_r2():
    """B = A R + small noise => out-of-sample R² near 1."""
    rng = np.random.default_rng(0)
    A = _cloud(200, 16, rng)
    R = rng.standard_normal((16, 16))
    B = A @ R + 0.01 * rng.standard_normal((200, 16))
    score = ruler.linear_transform_r2(A, B, ridge=1e-3, rng=rng)
    assert score > 0.95


def test_unrelated_near_null_r2():
    """Independent clouds => out-of-sample R² near 0 (or negative)."""
    rng = np.random.default_rng(1)
    A = _cloud(200, 16, rng)
    B = _cloud(200, 16, rng)
    score = ruler.linear_transform_r2(A, B, ridge=1.0, rng=rng)
    assert score < 0.1


def test_out_of_sample_distinguishes_where_insample_cannot():
    """With more dims than train rows, in-sample R²≈1 even for unrelated clouds;
    out-of-sample R² must stay near null. This is why we use held-out scoring."""
    rng = np.random.default_rng(2)
    A = _cloud(60, 100, rng)         # d > n: in-sample fit is perfect
    B = _cloud(60, 100, rng)         # unrelated
    # in-sample (fit and score on all rows) is spuriously high:
    W = ruler._fit_ridge(A, B, 1e-6)
    insample = ruler.r2_score(B, ruler._apply(A, W))
    oos = ruler.linear_transform_r2(A, B, ridge=1.0, rng=rng)
    assert insample > 0.9            # the trap
    assert oos < 0.2                 # the ruler avoids it


def test_cka_related_vs_unrelated():
    """CKA is invariant to ORTHOGONAL transforms (rotation) -> ~1; unrelated -> ~0."""
    rng = np.random.default_rng(3)
    A = _cloud(150, 20, rng)
    Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    B = A @ Q + 0.01 * rng.standard_normal((150, 20))
    C = _cloud(150, 20, rng)
    assert ruler.cka_linear(A, B) > 0.9
    assert ruler.cka_linear(A, C) < 0.3


def test_cka_not_invariant_to_general_linear_but_r2_is():
    """Key metric-dependence: a random invertible linear map preserves R² (~1) but
    NOT CKA (<1). This is why we report both rulers side by side."""
    rng = np.random.default_rng(33)
    A = _cloud(200, 20, rng)
    R = rng.standard_normal((20, 20))         # general (non-orthogonal) linear map
    B = A @ R
    assert ruler.linear_transform_r2(A, B, ridge=1e-4, rng=rng) > 0.95
    assert ruler.cka_linear(A, B) < 0.9       # CKA sees the geometric distortion


def test_procrustes_aligned_vs_unrelated():
    rng = np.random.default_rng(4)
    A = _cloud(150, 12, rng)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))   # orthogonal
    B = A @ Q                                            # pure rotation
    C = _cloud(150, 12, rng)
    assert ruler.procrustes_distance(A, B) < 0.05        # rotation-invariant ~ 0
    assert ruler.procrustes_distance(A, C) > ruler.procrustes_distance(A, B)


def test_r2_increases_with_signal():
    """Monotone: more signal (less noise) => higher out-of-sample R²."""
    rng = np.random.default_rng(5)
    A = _cloud(200, 16, rng)
    R = rng.standard_normal((16, 16))
    base = A @ R
    scores = []
    for noise in [2.0, 1.0, 0.3, 0.05]:
        B = base + noise * rng.standard_normal((200, 16))
        scores.append(ruler.linear_transform_r2(A, B, ridge=1e-2, rng=rng))
    assert scores == sorted(scores)      # increasing as noise decreases


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
