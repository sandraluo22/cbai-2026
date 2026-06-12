"""Unit tests for env.py (LLM-free core) — single-agent synthetic design."""
import numpy as np
import pytest

import env as E
import prompt as P


def test_lambda_knob():
    # more social noise => less weight on social; symmetric noise => 0.5
    assert E.lambda_info(1.0, 1.0) == pytest.approx(0.5)
    assert E.lambda_info(1.0, 0.3) > 0.8          # social much tighter
    assert E.lambda_info(0.3, 1.0) < 0.2


def test_reward_decomposition():
    theta = np.array([1.0, 2.0]); c = np.array([5.0, 0.0])
    np.testing.assert_allclose(E.reward(theta, c, 0.0), theta)   # w=0 -> truth
    np.testing.assert_allclose(E.reward(theta, c, 1.0), c)       # w=1 -> crowd
    np.testing.assert_allclose(E.reward(theta, c, 0.5), 0.5 * (theta + c))


def test_tau0_collapses_to_single_latent():
    cfg = E.EnvConfig(n_companies=1, sigma_p=1.0, sigma_s=0.5, tau=0.0, prior_std=10.0)
    priv = np.array([[1.0, 1.0]]); soc = np.array([[3.0, 3.0]])
    et, ec = E.bayes_posteriors(priv, soc, cfg)
    np.testing.assert_allclose(et, ec)            # c==theta when tau=0
    # social tighter => posterior pulled toward social mean (3) above private (1)
    assert et[0] > 1.5


def test_tau_separates_theta_and_c():
    cfg = E.EnvConfig(n_companies=1, sigma_p=1.0, sigma_s=0.5, tau=2.0, prior_std=10.0)
    priv = np.array([[1.0, 1.0, 1.0]]); soc = np.array([[5.0, 5.0, 5.0]])
    et, ec = E.bayes_posteriors(priv, soc, cfg)
    # E[c] tracks social strongly; E[theta] stays closer to private (crowd may be biased)
    assert ec[0] > et[0]
    assert ec[0] > 3.0 and et[0] < 3.0


def test_expected_reward_w_shifts_target():
    """w=0 optimal follows theta; w=1 optimal follows c — and they can differ."""
    cfg0 = E.EnvConfig(n_companies=2, sigma_p=0.5, sigma_s=0.5, tau=2.0, w=0.0, prior_std=10.0)
    cfg1 = E.EnvConfig(n_companies=2, sigma_p=0.5, sigma_s=0.5, tau=2.0, w=1.0, prior_std=10.0)
    # company 0: high theta, low crowd; company 1: low theta, high crowd
    priv = np.array([[4, 4, 4], [0, 0, 0]], float)
    soc = np.array([[0, 0, 0], [4, 4, 4]], float)
    assert E.rational_action(priv, soc, cfg0) == 0     # truth game -> company 0
    assert E.rational_action(priv, soc, cfg1) == 1     # beauty contest -> company 1


def test_effective_social_weight_tracks_lambda_at_w0():
    """At w=0, tau=0 the rational social weight ~ lambda_info (single reading each)."""
    cfg = E.EnvConfig(sigma_p=1.0, sigma_s=0.5, tau=0.0, w=0.0, prior_std=1e6)
    esw = E.rational_effective_social_weight(cfg, kp=1, ks=1)
    assert esw == pytest.approx(E.lambda_info(1.0, 0.5), abs=0.02)


def test_effective_social_weight_rises_with_w():
    cfg_lo = E.EnvConfig(sigma_p=1.0, sigma_s=1.0, tau=1.0, w=0.0, prior_std=10.0)
    cfg_hi = E.EnvConfig(sigma_p=1.0, sigma_s=1.0, tau=1.0, w=1.0, prior_std=10.0)
    lo = E.rational_effective_social_weight(cfg_lo, 3, 3)
    hi = E.rational_effective_social_weight(cfg_hi, 3, 3)
    assert hi > lo                                  # paying for the crowd => weight social more


def test_counterfactual_pair_differs_only_in_social_target():
    cfg = E.EnvConfig(n_companies=4, T=6)
    lo, hi = E.counterfactual_pair(cfg, seed=1, target=2, delta=3.0)
    assert lo.pair_id == hi.pair_id
    np.testing.assert_array_equal(lo.state.private, hi.state.private)   # private fixed
    diff = hi.state.social - lo.state.social
    np.testing.assert_allclose(diff[2], 3.0)        # only target social shifted
    diff_others = np.delete(diff, 2, axis=0)
    np.testing.assert_allclose(diff_others, 0.0)


def test_prompt_blocks_symmetric_char_length():
    cfg = E.EnvConfig(n_companies=4, T=6)
    st = E.make_state(cfg, seed=0)
    lp, ls = P.block_char_lengths(st.private, st.social)
    assert lp == ls                                 # identical template => equal char length


def test_prompt_ends_with_marker():
    cfg = E.EnvConfig(n_companies=3, T=4)
    st = E.make_state(cfg, seed=0)
    s = P.render(st.private, st.social, cfg)
    assert s.rstrip().endswith("Decision :")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
