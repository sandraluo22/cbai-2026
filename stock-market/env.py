"""Environment: synthetic private vs. social evidence with two orthogonal knobs.

One focal agent, n companies, T rounds. Fully synthetic & controllable — no live
population. The two channels are presented in the SAME FORMAT (see prompt.py) so
they differ only by SOURCE LABEL, removing the rich-vs-scalar confound.

  PRIVATE:  p_i^(t) = theta_i + N(0, sigma_p^2)        (readings of the fundamental)
  SOCIAL:   s_i^(t) = c_i      + N(0, sigma_s^2)        (readings of the crowd target)
            c_i     = theta_i  + N(0, tau^2)            (crowd tracks theta up to tau)

Two knobs:
  KNOB 1  lambda = (1/sigma_s^2) / (1/sigma_p^2 + 1/sigma_s^2)   — how INFORMATIVE
          social is about theta (Bayes weight on social, in the tau=0 case).
  KNOB 2  reward(i) = (1-w)*theta_i + w*c_i                      — how much reward
          DIRECTLY tracks social. w=0 pure truth game; w=1 pure beauty contest.

The optimal choice maximizes expected reward (1-w)*E[theta_i] + w*E[c_i] given the
visible histories; E[theta], E[c] come from the joint linear-Gaussian posterior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    n_companies: int = 4
    T: int = 6
    sigma_p: float = 1.0             # private noise std
    sigma_s: float = 1.0             # social noise std
    tau: float = 0.0                 # crowd-vs-fundamental gap std (0 => c==theta)
    w: float = 0.0                   # reward weight on social (beauty-contest knob)
    theta_scale: float = 3.0         # std of hidden fundamentals
    prior_mean: float = 0.0
    prior_std: float = 5.0           # weak Gaussian prior over theta
    allow_withdraw: bool = False
    seed: int = 0

    @property
    def lambda_info(self) -> float:
        return lambda_info(self.sigma_p, self.sigma_s)


def lambda_info(sigma_p: float, sigma_s: float) -> float:
    """KNOB 1: Bayes weight on the social channel as an estimator of theta (tau=0)."""
    pp, ps = 1.0 / sigma_p ** 2, 1.0 / sigma_s ** 2
    return float(ps / (pp + ps))


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
def sample_theta(cfg: EnvConfig, rng) -> np.ndarray:
    return rng.normal(0.0, cfg.theta_scale, size=cfg.n_companies)


def sample_crowd_target(theta: np.ndarray, tau: float, rng) -> np.ndarray:
    """c_i = theta_i + N(0, tau^2)."""
    return theta + rng.normal(0.0, tau, size=theta.shape) if tau > 0 else theta.copy()


def sample_private(theta: np.ndarray, sigma_p: float, T: int, rng) -> np.ndarray:
    return theta[:, None] + rng.normal(0.0, sigma_p, size=(theta.shape[0], T))


def sample_social(c: np.ndarray, sigma_s: float, T: int, rng) -> np.ndarray:
    return c[:, None] + rng.normal(0.0, sigma_s, size=(c.shape[0], T))


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class State:
    theta: np.ndarray              # (n,)
    c: np.ndarray                  # (n,)
    private: np.ndarray            # (n, t)
    social: np.ndarray            # (n, t)
    t: int

    def copy(self) -> "State":
        return State(self.theta.copy(), self.c.copy(), self.private.copy(),
                     self.social.copy(), self.t)


def make_state(cfg: EnvConfig, seed: int, t: Optional[int] = None) -> State:
    t = cfg.T if t is None else t
    rng = np.random.default_rng(seed)
    theta = sample_theta(cfg, rng)
    c = sample_crowd_target(theta, cfg.tau, rng)
    private = sample_private(theta, cfg.sigma_p, t, rng)
    social = sample_social(c, cfg.sigma_s, t, rng)
    return State(theta, c, private, social, t)


# --------------------------------------------------------------------------- #
# Reward                                                                       #
# --------------------------------------------------------------------------- #
def reward(theta: np.ndarray, c: np.ndarray, w: float) -> np.ndarray:
    """Per-company realized reward: (1-w)*theta + w*c."""
    return (1.0 - w) * theta + w * c


# --------------------------------------------------------------------------- #
# Rational baseline: joint linear-Gaussian posterior over (theta_i, c_i)       #
# --------------------------------------------------------------------------- #
def bayes_posteriors(private: np.ndarray, social: np.ndarray, cfg: EnvConfig):
    """Return (E_theta[n], E_c[n]) given visible private & social histories.

    Per-company linear-Gaussian model:
        theta ~ N(mu0, sigma0^2);  c|theta ~ N(theta, tau^2)
        private readings ~ N(theta, sigma_p^2);  social readings ~ N(c, sigma_s^2)
    Solved in (theta, c) information form. tau=0 collapses c==theta (handled).
    """
    n = private.shape[0]
    kp, ks = private.shape[1], social.shape[1]
    a = kp / cfg.sigma_p ** 2                    # private precision about theta
    b = ks / cfg.sigma_s ** 2                    # social precision about c
    pbar = private.mean(1) if kp else np.zeros(n)
    sbar = social.mean(1) if ks else np.zeros(n)
    prior_prec = 1.0 / cfg.prior_std ** 2

    if cfg.tau <= 0:                             # c == theta: social informs theta directly
        post_prec = prior_prec + a + b
        e_theta = (prior_prec * cfg.prior_mean + a * pbar + b * sbar) / post_prec
        return e_theta, e_theta.copy()

    inv_tau2 = 1.0 / cfg.tau ** 2
    E_theta = np.empty(n); E_c = np.empty(n)
    for i in range(n):
        Lam = np.array([[prior_prec + inv_tau2 + a, -inv_tau2],
                        [-inv_tau2, inv_tau2 + b]])
        eta = np.array([prior_prec * cfg.prior_mean + a * pbar[i], b * sbar[i]])
        mu = np.linalg.solve(Lam, eta)
        E_theta[i], E_c[i] = mu[0], mu[1]
    return E_theta, E_c


def expected_reward(private, social, cfg: EnvConfig):
    E_theta, E_c = bayes_posteriors(private, social, cfg)
    return (1.0 - cfg.w) * E_theta + cfg.w * E_c, E_theta, E_c


def rational_action(private, social, cfg: EnvConfig) -> int:
    er, _, _ = expected_reward(private, social, cfg)
    return int(np.argmax(er))


def channel_estimates(private: np.ndarray, social: np.ndarray):
    """Single-channel point estimates (per-company means) for the revealed-reliance
    regression. Returns (private_implied[n], social_implied[n])."""
    priv = private.mean(axis=1) if private.shape[1] else np.zeros(private.shape[0])
    soc = social.mean(axis=1) if social.shape[1] else np.zeros(social.shape[0])
    return priv, soc


def rational_effective_social_weight(cfg: EnvConfig, kp: int, ks: int) -> float:
    """Sensitivity of expected reward to the SOCIAL vs PRIVATE sufficient statistic.

    Since the posterior means are linear in (pbar, sbar), perturbing each gives the
    exact partials. Returns d(er)/d(sbar) / (d(er)/d(sbar) + d(er)/d(pbar)) — the
    weight the OPTIMAL decision puts on social evidence for this (lambda, w, tau).
    """
    n = 1
    base_p = np.zeros((n, max(kp, 1))); base_s = np.zeros((n, max(ks, 1)))
    eps = 1.0

    def er_of(pbar, sbar):
        p = np.full((1, kp), pbar) if kp else np.zeros((1, 0))
        s = np.full((1, ks), sbar) if ks else np.zeros((1, 0))
        er, _, _ = expected_reward(p, s, cfg)
        return er[0]

    d_priv = (er_of(eps, 0) - er_of(0, 0)) if kp else 0.0
    d_soc = (er_of(0, eps) - er_of(0, 0)) if ks else 0.0
    denom = abs(d_priv) + abs(d_soc)
    return float(abs(d_soc) / denom) if denom > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# Trial modes + counterfactual pairs                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    state: State
    mode: str                       # "neutral" | "disagreement" | "counterfactual"
    target: int
    delta: float
    pair_id: Optional[str] = None
    arm: Optional[str] = None       # "low"/"high" within a counterfactual pair
    note: str = ""


def counterfactual_pair(cfg: EnvConfig, seed: int, target: int, delta: float,
                        t: Optional[int] = None) -> tuple[Trial, Trial]:
    """Two trials IDENTICAL except the SOCIAL evidence for `target` is shifted by
    delta (private held fixed). The behavioral/activation difference isolates the
    social channel's causal effect."""
    base = make_state(cfg, seed, t)
    lo, hi = base.copy(), base.copy()
    hi.social[target] = hi.social[target] + delta        # shift all social readings of target
    pid = f"s{seed}_t{target}_d{delta:+.2f}"
    return (Trial(lo, "counterfactual", target, 0.0, pid, "low"),
            Trial(hi, "counterfactual", target, delta, pid, "high"))


def disagreement_trial(cfg: EnvConfig, seed: int, target: int, delta: float,
                       t: Optional[int] = None) -> Trial:
    """Private-implied and social-implied estimates for `target` conflict by delta
    (others neutral). Edits both channels' target readings symmetrically around 0."""
    base = make_state(cfg, seed, t)
    base.private[target] = base.private[target] - base.private[target].mean() + (+delta / 2)
    base.social[target] = base.social[target] - base.social[target].mean() + (-delta / 2)
    return Trial(base, "disagreement", target, delta, note="private vs social conflict")


def neutral_trial(cfg: EnvConfig, seed: int, target: int, t: Optional[int] = None) -> Trial:
    return Trial(make_state(cfg, seed, t), "neutral", target, 0.0)


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #
def save_config(cfg: EnvConfig, path: str | Path):
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))
