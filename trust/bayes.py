"""Learned-precision Bayesian baseline — the normative trust trajectory.

The rational observer does NOT know sigma_A, sigma_B. It ESTIMATES each source's
precision from the errors it has observed so far, starting from a REPUTATION-DERIVED
prior that favors A (A is *believed* more accurate, even though B truly is). As truth
accrues round by round, the estimated precisions correct and trust migrates A -> B.

Conjugate model (per source s)
------------------------------
Errors e = estimate - truth are modeled e ~ N(0, sigma_s^2) with unknown precision
pi_s = 1/sigma_s^2 and a Gamma prior expressed through two interpretable knobs:

    nu0_s   : prior pseudo-count  (how many "track-record" observations back the belief;
              this is the REPUTATION STRENGTH for the normative prior — larger = stickier)
    sig0_s  : prior believed std  (reputation's claim about the source's accuracy;
              the prior favors A by setting sig0_A < sig0_B)

After observing n_s real errors with sum-of-squares S_s, the posterior-mean precision
has the clean closed form

    pi_hat_s = (nu0_s + n_s) / (nu0_s * sig0_s^2 + S_s).

Trust weight on B:  w = pi_hat_B / (pi_hat_A + pi_hat_B).
Oracle asymptote:   w* uses the TRUE sigmas (see env.oracle_trust_B).
Bayesian forecast:  precision-weighted blend  (pi_A a + pi_B b) / (pi_A + pi_B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import env as E


# --------------------------------------------------------------------------- #
# Reputation-derived prior                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class ReputationPrior:
    """Gamma prior on each source's precision, parameterised by pseudo-count + believed
    std. Defaults favor A (sig0_A < sig0_B) — the prior is WRONG, and evidence corrects
    it. `nu0` is the reputation strength for the normative observer (stickiness)."""
    sig0_A: float = 15.0        # reputation believes A is accurate (small std)
    sig0_B: float = 45.0        # reputation believes B is unestablished (large std)
    nu0_A: float = 6.0          # pseudo-count backing A's reputation
    nu0_B: float = 2.0          # weaker prior on the newcomer B

    def trust_B(self) -> float:
        """Prior trust weight on B BEFORE any evidence (should be < 0.5: A favored)."""
        pa = 1.0 / self.sig0_A ** 2
        pb = 1.0 / self.sig0_B ** 2
        return float(pb / (pa + pb))


# --------------------------------------------------------------------------- #
# Running observer                                                             #
# --------------------------------------------------------------------------- #
class BayesObserver:
    """Accumulates per-source sum-of-squared errors and reports learned precisions."""

    def __init__(self, prior: ReputationPrior | None = None):
        p = prior or ReputationPrior()
        self.nu0_A, self.s0sq_A = p.nu0_A, p.sig0_A ** 2
        self.nu0_B, self.s0sq_B = p.nu0_B, p.sig0_B ** 2
        self.nA = 0
        self.nB = 0
        self.SA = 0.0
        self.SB = 0.0

    def precisions(self) -> tuple[float, float]:
        pi_A = (self.nu0_A + self.nA) / (self.nu0_A * self.s0sq_A + self.SA)
        pi_B = (self.nu0_B + self.nB) / (self.nu0_B * self.s0sq_B + self.SB)
        return float(pi_A), float(pi_B)

    def trust_B(self) -> float:
        pi_A, pi_B = self.precisions()
        return pi_B / (pi_A + pi_B)

    def forecast(self, a_row: np.ndarray, b_row: np.ndarray) -> np.ndarray:
        """Precision-weighted blend using CURRENT (pre-truth) precisions."""
        pi_A, pi_B = self.precisions()
        return (pi_A * a_row + pi_B * b_row) / (pi_A + pi_B)

    def observe(self, a_row: np.ndarray, b_row: np.ndarray, theta_row: np.ndarray):
        """Fold in one round's revealed truths."""
        self.SA += float(np.sum((a_row - theta_row) ** 2))
        self.SB += float(np.sum((b_row - theta_row) ** 2))
        self.nA += int(theta_row.size)
        self.nB += int(theta_row.size)


# --------------------------------------------------------------------------- #
# Trajectory over a game                                                       #
# --------------------------------------------------------------------------- #
def bayes_trajectory(game: E.Game, prior: ReputationPrior | None = None) -> dict:
    """Run the observer through `game`. For each round t we record the trust weight and
    forecast computed BEFORE round t's truth is revealed (using rounds 1..t-1) — this is
    the apples-to-apples comparison for the model's round-t estimate — then we observe
    round t. `trust_pre[t]` is the normative trust curve; it should migrate A -> B."""
    obs = BayesObserver(prior)
    T, M = game.T, game.M
    trust_pre = np.empty(T)
    trust_post = np.empty(T)
    pi_A_pre = np.empty(T)
    pi_B_pre = np.empty(T)
    forecast = np.empty((T, M))

    for t in range(T):
        pa, pb = obs.precisions()
        pi_A_pre[t], pi_B_pre[t] = pa, pb
        trust_pre[t] = pb / (pa + pb)
        forecast[t] = obs.forecast(game.a[t], game.b[t])
        obs.observe(game.a[t], game.b[t], game.theta[t])
        trust_post[t] = obs.trust_B()

    return {
        "trust_pre": trust_pre,
        "trust_post": trust_post,
        "pi_A_pre": pi_A_pre,
        "pi_B_pre": pi_B_pre,
        "forecast": forecast,
        "oracle_trust_B": E.oracle_trust_B(game.sigma_A, game.sigma_B),
        "prior_trust_B": (prior or ReputationPrior()).trust_B(),
    }


def expected_flip_round(cfg: E.EnvConfig, prior: ReputationPrior | None = None,
                        n_sims: int = 2000, base_seed: int = 10_000) -> int | None:
    """Round budget helper. Monte-Carlo the MEAN normative trust curve and return the
    first round (1-indexed) where mean trust_pre crosses 0.5. None if it never flips
    within T. Use to set T comfortably beyond the flip so failure-to-flip is meaningful."""
    acc = np.zeros(cfg.T)
    for k in range(n_sims):
        g = E.make_game(cfg, seed=base_seed + k)
        acc += bayes_trajectory(g, prior)["trust_pre"]
    mean_curve = acc / n_sims
    crossed = np.where(mean_curve >= 0.5)[0]
    return int(crossed[0] + 1) if crossed.size else None


# --------------------------------------------------------------------------- #
# Demo: show the rational trajectory flipping A -> B on a synthetic stream      #
# --------------------------------------------------------------------------- #
def _demo():
    cfg = E.EnvConfig(M=3, T=24, sigma_B=12.0, gap=3.0, seed=0)
    prior = ReputationPrior()
    g = E.make_game(cfg, seed=0)
    traj = bayes_trajectory(g, prior)

    print("=== DEMO: learned-precision Bayesian baseline (NO LLM) ===")
    print(f"sigma_A={cfg.sigma_A:.1f}  sigma_B={cfg.sigma_B:.1f}  gap={cfg.gap}")
    print(f"reputation prior trust on B = {traj['prior_trust_B']:.3f}  (A favored => < 0.5)")
    print(f"oracle (true-sigma) trust on B = {traj['oracle_trust_B']:.3f}")
    flip = expected_flip_round(cfg, prior, n_sims=2000)
    print(f"expected flip round (mean curve crosses 0.5 over 2000 games) = {flip}")
    print(f"\n round   pi_A      pi_B     trust_B(pre)")
    for t in range(cfg.T):
        mark = "   <-- crosses 0.5" if (t > 0 and traj["trust_pre"][t] >= 0.5
                                        > traj["trust_pre"][t - 1]) else ""
        print(f"  {t + 1:>3}   {traj['pi_A_pre'][t]:.5f}  {traj['pi_B_pre'][t]:.5f}   "
              f"{traj['trust_pre'][t]:.3f}{mark}")

    assert traj["trust_pre"][0] < 0.5, "prior should start favoring A"
    assert traj["trust_pre"][-1] > 0.5, "trust should migrate to B by the final round"
    print("\nDemo OK: trust starts on A (< 0.5) and migrates to B (> 0.5).")


if __name__ == "__main__":
    _demo()
