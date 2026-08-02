"""Unit tests for bayes.py — the learned-precision Bayesian baseline.

Key required test (per spec): the precision update + migrating trust curve flips
A -> B on a synthetic known-sigma stream.
"""
import numpy as np
import pytest

import env as E
import bayes as B


def test_precision_update_recovers_true_precision():
    # With many low-information-prior observations, pi_hat -> 1/sigma^2.
    cfg = E.EnvConfig(M=50, T=50, sigma_B=10.0, gap=3.0)  # sigma_A=30, sigma_B=10
    g = E.make_game(cfg, seed=3)
    # Weak prior so data dominates.
    prior = B.ReputationPrior(sig0_A=20.0, sig0_B=20.0, nu0_A=0.1, nu0_B=0.1)
    obs = B.BayesObserver(prior)
    for t in range(g.T):
        obs.observe(g.a[t], g.b[t], g.theta[t])
    pi_A, pi_B = obs.precisions()
    assert pi_A == pytest.approx(1.0 / 30.0 ** 2, rel=0.12)
    assert pi_B == pytest.approx(1.0 / 10.0 ** 2, rel=0.12)


def test_prior_starts_favoring_A():
    prior = B.ReputationPrior()
    assert prior.trust_B() < 0.5          # reputation favors A before any evidence


def test_trust_flips_A_to_B_on_known_sigma_stream():
    cfg = E.EnvConfig(M=3, T=24, sigma_B=12.0, gap=3.0)
    g = E.make_game(cfg, seed=0)
    traj = B.bayes_trajectory(g, B.ReputationPrior())
    assert traj["trust_pre"][0] < 0.5     # starts on A (reputation)
    assert traj["trust_pre"][-1] > 0.5    # migrates to B (accuracy)
    # and it should be heading toward the oracle asymptote
    assert traj["oracle_trust_B"] > 0.5


def test_trust_is_monotone_in_expectation():
    # Averaged over many games the normative curve should rise toward B.
    cfg = E.EnvConfig(M=3, T=20, sigma_B=12.0, gap=3.0)
    acc = np.zeros(cfg.T)
    n = 400
    for k in range(n):
        g = E.make_game(cfg, seed=1000 + k)
        acc += B.bayes_trajectory(g, B.ReputationPrior())["trust_pre"]
    mean = acc / n
    assert mean[0] < 0.5 < mean[-1]
    assert mean[-1] > mean[0]


def test_forecast_precision_weighted():
    # If A is treated as far more precise, the blend sits near A's estimate.
    obs = B.BayesObserver(B.ReputationPrior(sig0_A=1.0, sig0_B=100.0, nu0_A=50, nu0_B=50))
    a = np.array([100.0]); b = np.array([200.0])
    f = obs.forecast(a, b)
    assert f[0] < 110.0                   # pulled strongly toward A


def test_expected_flip_round_within_T_for_large_gap():
    cfg = E.EnvConfig(M=3, T=24, sigma_B=12.0, gap=3.0)
    flip = B.expected_flip_round(cfg, B.ReputationPrior(), n_sims=300)
    assert flip is not None and 1 <= flip <= cfg.T


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
