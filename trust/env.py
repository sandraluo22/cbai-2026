"""Environment for the Trust experiment: in-context trust-learning under a
reputation-vs-accuracy conflict.

Each round t presents M companies, each with a hidden true value drawn FRESH (new
items every round => the only stable, learnable thing is which SOURCE is reliable):

    theta_i^(t) ~ N(mu, theta_scale^2)
    a_i^(t)     = theta_i^(t) + N(0, sigma_A^2)     # Source A — NOISIER
    b_i^(t)     = theta_i^(t) + N(0, sigma_B^2)     # Source B — ACCURATE

sigma_A, sigma_B are FIXED across the whole game (reliability is a stable property to
be learned). The estimate IS the recommendation; there is no separate decision layer.
Truth is revealed after each round, so accuracy evidence accumulates round by round.

DIFFICULTY is set by the accuracy GAP = sigma_A / sigma_B (the primary swept axis):
small gap => sources hard to tell apart (reputation prior dominates for many rounds);
large gap => obviously B is better.

The environment is deliberately LLM-free and knows NOTHING about reputation — reputation
is a purely verbal framing layer applied in prompt.py. This module only owns the numbers
and is fully seeded/reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    M: int = 3                  # companies per round (more => more reliability evidence/round)
    T: int = 24                 # rounds per game
    mu: float = 300.0           # prior mean of hidden true values
    theta_scale: float = 120.0  # std of hidden true values (fresh each round)
    sigma_B: float = 12.0       # ACCURATE source noise std (Source B)
    gap: float = 3.0            # accuracy gap: sigma_A = gap * sigma_B  (gap > 1)
    seed: int = 0

    @property
    def sigma_A(self) -> float:
        """Source A noise std — derived from the accuracy gap (A is noisier)."""
        return self.gap * self.sigma_B

    @property
    def oracle_trust_B(self) -> float:
        """Accuracy-justified trust weight on B if the TRUE sigmas were known:
        w* = pi_B / (pi_A + pi_B), pi = 1/sigma^2  ==  gap^2 / (gap^2 + 1)."""
        return oracle_trust_B(self.sigma_A, self.sigma_B)


def oracle_trust_B(sigma_A: float, sigma_B: float) -> float:
    pa, pb = 1.0 / sigma_A ** 2, 1.0 / sigma_B ** 2
    return float(pb / (pa + pb))


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Game:
    """A full T-round game. All arrays are shape (T, M)."""
    theta: np.ndarray           # hidden true values
    a: np.ndarray               # Source A estimates (noisier)
    b: np.ndarray               # Source B estimates (accurate)
    sigma_A: float
    sigma_B: float
    seed: int

    @property
    def T(self) -> int:
        return self.theta.shape[0]

    @property
    def M(self) -> int:
        return self.theta.shape[1]


def make_game(cfg: EnvConfig, seed: int | None = None) -> Game:
    """Sample one full game. Fresh theta every round; FIXED source noise levels."""
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    sa, sb = cfg.sigma_A, cfg.sigma_B
    theta = rng.normal(cfg.mu, cfg.theta_scale, size=(cfg.T, cfg.M))
    a = theta + rng.normal(0.0, sa, size=(cfg.T, cfg.M))
    b = theta + rng.normal(0.0, sb, size=(cfg.T, cfg.M))
    return Game(theta=theta, a=a, b=b, sigma_A=sa, sigma_B=sb, seed=seed)


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #
def save_config(cfg: EnvConfig, path: str | Path):
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))
