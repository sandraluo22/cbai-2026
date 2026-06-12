"""Tests for the LLM-free analysis primitives (synthetic data)."""
import numpy as np
import pytest

import analyze as A


def test_ridge_r2_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    w = rng.standard_normal(10)
    y = X @ w + 0.05 * rng.standard_normal(200)
    assert A.ridge_r2_oos(X, y, ridge=1e-2) > 0.95


def test_ridge_r2_null_for_noise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 10)); y = rng.standard_normal(200)
    assert A.ridge_r2_oos(X, y, ridge=1.0) < 0.2


def test_revealed_weight_recovers_known_reliance():
    """Synthetic trials where choice depends mostly on SOCIAL -> lambda_hat near 1."""
    rng = np.random.default_rng(2)
    rows = []
    for _ in range(300):
        priv = rng.standard_normal(4); soc = rng.standard_normal(4)
        tgt = int(rng.integers(4))
        # choice driven 90% by social, 10% by private at the target
        score = 0.1 * priv + 0.9 * soc
        chosen = int(np.argmax(score))
        rows.append({"lmbda": 0.5, "w": 0.0, "tau": 0.0, "target": tgt,
                     "private_implied": priv.tolist(), "social_implied": soc.tolist(),
                     "parsed": {"company": chosen}, "rational_eff_social_weight": 0.9})
    rw = A.revealed_social_weight(rows)
    (v,) = rw.values()
    assert v["lambda_hat"] > 0.7          # social-dominant reliance recovered


def test_patching_localization_aggregates():
    rows = [{"pair_id": "p", "layer": L, "position_set": "marker",
             "flipped_to_low": (L >= 16)} for L in range(32)]
    loc = A.patching_localization(rows)
    fr = loc["marker"]["flip_rate"]
    assert fr[0] == 0.0 and fr[-1] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
