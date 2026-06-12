"""Unit tests for the closed-form Balls & Urns posterior (build step 1)."""

import numpy as np
import pytest

import dgp


def test_sources_are_distributions():
    s = dgp.binary_sources(0.9)
    assert np.allclose(s.W.sum(1), 1.0)
    assert np.allclose(s.prior.sum(), 1.0)
    assert s.K == 2 and s.S == 2


def test_posterior_is_valid_distribution():
    rng = np.random.default_rng(0)
    s = dgp.binary_sources(0.8)
    seq = dgp.sample_sequence(s.W[0], 64, rng)
    post = dgp.posterior_trajectory(seq, s)
    assert post.shape == (65, 2)
    assert np.allclose(post.sum(1), 1.0)
    assert np.all(post >= 0)


def test_posterior_concentrates_on_true_source():
    """With separated sources, the posterior on the true source -> 1 with more data."""
    rng = np.random.default_rng(1)
    s = dgp.binary_sources(0.85)
    # average over many sequences from source 0
    finals = []
    for _ in range(200):
        seq = dgp.sample_sequence(s.W[0], 256, rng)
        finals.append(dgp.posterior_over_sources(seq, s)[0])
    assert np.mean(finals) > 0.95


def test_separation_controls_convergence_speed():
    """Larger separation p => posterior on truth rises faster (earlier convergence)."""
    rng = np.random.default_rng(2)

    def mean_post_at(p, t, n=300):
        s = dgp.binary_sources(p)
        vals = []
        for _ in range(n):
            seq = dgp.sample_sequence(s.W[0], t, rng)
            vals.append(dgp.posterior_trajectory(seq, s)[t, 0])
        return np.mean(vals)

    easy = mean_post_at(0.95, 16)
    hard = mean_post_at(0.55, 16)
    assert easy > hard
    assert easy > 0.9          # easy world basically solved by t=16
    assert hard < 0.8          # hard world still uncertain


def test_predictive_approaches_true_source():
    """Posterior-predictive -> w_star as the posterior concentrates."""
    rng = np.random.default_rng(3)
    s = dgp.binary_sources(0.9)
    seq = dgp.sample_sequence(s.W[0], 512, rng)
    pred = dgp.predictive_trajectory(seq, s)[-1]
    assert np.allclose(pred, s.W[0], atol=0.05)


def test_incremental_matches_batch():
    """Incremental log-lik trajectory matches a direct count-based computation."""
    rng = np.random.default_rng(4)
    s = dgp.binary_sources(0.7)
    seq = dgp.sample_sequence(s.W[0], 100, rng)
    post = dgp.posterior_trajectory(seq, s)
    # direct: at t, use counts of seq[:t]
    for t in [1, 10, 50, 100]:
        counts = np.bincount(seq[:t], minlength=2)
        loglik = np.log(s.prior) + counts @ np.log(s.W).T
        direct = dgp._softmax(loglik)
        np.testing.assert_allclose(post[t], direct, atol=1e-10)


def test_log_spaced_positions():
    assert dgp.log_spaced_positions(256)[0] == 1
    assert dgp.log_spaced_positions(256)[-1] == 256
    assert dgp.log_spaced_positions(8) == [1, 2, 4, 8]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
