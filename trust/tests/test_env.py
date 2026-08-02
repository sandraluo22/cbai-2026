"""Unit tests for env.py (LLM-free synthetic environment)."""
import numpy as np
import pytest

import env as E


def test_game_shapes():
    cfg = E.EnvConfig(M=4, T=10)
    g = E.make_game(cfg, seed=0)
    assert g.theta.shape == g.a.shape == g.b.shape == (10, 4)
    assert g.T == 10 and g.M == 4


def test_gap_sets_sigma_A():
    cfg = E.EnvConfig(sigma_B=10.0, gap=3.0)
    assert cfg.sigma_A == pytest.approx(30.0)


def test_source_noise_levels_recovered():
    # B should be much tighter around the truth than A.
    cfg = E.EnvConfig(M=200, T=200, sigma_B=10.0, gap=3.0)
    g = E.make_game(cfg, seed=1)
    sd_A = (g.a - g.theta).std()
    sd_B = (g.b - g.theta).std()
    assert sd_A == pytest.approx(30.0, rel=0.05)
    assert sd_B == pytest.approx(10.0, rel=0.05)
    assert sd_A > sd_B


def test_fresh_theta_each_round():
    # New items every round => rows are not duplicated.
    cfg = E.EnvConfig(M=3, T=5)
    g = E.make_game(cfg, seed=2)
    for t in range(1, g.T):
        assert not np.allclose(g.theta[t], g.theta[t - 1])


def test_reproducible():
    cfg = E.EnvConfig(M=3, T=8)
    g1 = E.make_game(cfg, seed=7)
    g2 = E.make_game(cfg, seed=7)
    np.testing.assert_array_equal(g1.a, g2.a)
    np.testing.assert_array_equal(g1.b, g2.b)
    g3 = E.make_game(cfg, seed=8)
    assert not np.allclose(g1.a, g3.a)


def test_oracle_trust_matches_gap_formula():
    cfg = E.EnvConfig(sigma_B=12.0, gap=3.0)
    # w* = gap^2 / (gap^2 + 1)
    assert cfg.oracle_trust_B == pytest.approx(9.0 / 10.0)
    cfg2 = E.EnvConfig(sigma_B=12.0, gap=1.5)
    assert cfg2.oracle_trust_B == pytest.approx(2.25 / 3.25)


def test_larger_gap_favors_B_more():
    small = E.EnvConfig(sigma_B=10.0, gap=1.5).oracle_trust_B
    large = E.EnvConfig(sigma_B=10.0, gap=4.0).oracle_trust_B
    assert 0.5 < small < large < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
