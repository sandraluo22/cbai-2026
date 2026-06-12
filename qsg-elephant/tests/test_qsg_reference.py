"""Sanity checks (spec §8) for the numeric QSG null model."""

from __future__ import annotations

import numpy as np
import pytest

from qsg.qsg_reference import (
    SOFT,
    consensus_time,
    qsg_update,
    run_reference,
    sample_message,
)


def test_message_on_simplex():
    rng = np.random.default_rng(0)
    x = rng.dirichlet(np.ones(5))
    for m in (1, 3, 10, SOFT):
        y = sample_message(x, m, rng)
        assert y.shape == (5,)
        assert np.isclose(y.sum(), 1.0)
        assert np.all(y >= 0)


def test_soft_message_is_identity():
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(6))
    np.testing.assert_allclose(sample_message(x, SOFT, rng), x)


def test_update_stays_on_simplex():
    rng = np.random.default_rng(2)
    x_l = rng.dirichlet(np.ones(4))
    y = rng.dirichlet(np.ones(4))
    out = qsg_update(x_l, y, 0.3)
    assert np.isclose(out.sum(), 1.0)


def test_soft_preserves_population_mean_in_expectation():
    """§8: Soft (m=inf) preserves the population mean in expectation (martingale).

    Monte Carlo over seeds: the mean of final population means equals the
    (fixed) initial population mean.
    """
    N, K, rounds, alpha = 6, 4, 40, 0.4
    base = np.random.default_rng(123).dirichlet(np.ones(K), size=N)
    init_mean = base.mean(0)

    finals = []
    for seed in range(400):
        res = run_reference(N, K, alpha, SOFT, rounds, seed=seed, x0=base.copy())
        finals.append(res.final_mean)
    mc_mean = np.mean(finals, axis=0)

    # Standard error shrinks like 1/sqrt(400); tolerance generous but meaningful.
    np.testing.assert_allclose(mc_mean, init_mean, atol=0.02)


def test_hard_drives_consensus():
    """§8: Hard (m=1) drives the population to a single-vertex consensus."""
    N, K, rounds, alpha = 8, 5, 400, 0.5
    res = run_reference(N, K, alpha, 1.0, rounds, seed=7)
    # Disagreement energy collapses and one label dominates the mean.
    assert res.V[-1] < 1e-2
    assert res.final_mean.max() > 0.95


def test_larger_N_slows_drift_to_consensus():
    """§8: under neutrality, larger N slows drift to consensus (hard channel)."""
    def mean_consensus_time(N: int) -> float:
        times = []
        for seed in range(20):
            res = run_reference(N, 4, 0.4, 1.0, rounds=2000, seed=seed)
            ct = consensus_time(res.U, 0.95)
            times.append(ct if ct is not None else 2000)
        return float(np.mean(times))

    t_small = mean_consensus_time(4)
    t_large = mean_consensus_time(16)
    assert t_large > t_small


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
