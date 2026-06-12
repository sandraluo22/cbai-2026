"""Data-generating process: latent-source sequences ("Balls & Urns").

A *source* (urn) is a categorical distribution ``w`` over ``K`` symbols. To make a
run we fix one true source ``w*`` and draw ``C`` tokens iid from ``Categorical(w*)``.
A model reading the sequence should sharpen its belief about *which source* it is
seeing as more tokens arrive.

The point of this DGP is that the optimal belief is available in **closed form**,
so it can serve as ground truth to check the "ruler" against:

  * posterior over sources given a prefix:  p(s | x_{1:t}) ∝ prior(s) · Π_k w_s[k]^{n_k}
  * posterior-predictive over the next token: p(x_{t+1}=k | x_{1:t}) = Σ_s p(s|x_{1:t}) w_s[k]

The K=2 binary case is the "Cat/Dog" belief state: Cat=(p,1-p), Dog=(1-p,p).
``p`` is the *source-separation* sweep knob — large p ⇒ easy/sharp convergence,
near 0.5 ⇒ hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Sources                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class SourceSet:
    """A set of ``S`` categorical sources over ``K`` symbols, with a prior."""

    W: np.ndarray              # (S, K) rows are categorical distributions
    prior: np.ndarray          # (S,)   prior over sources
    meta: dict

    @property
    def S(self) -> int:
        return self.W.shape[0]

    @property
    def K(self) -> int:
        return self.W.shape[1]


def binary_sources(p: float) -> SourceSet:
    """K=2 Cat/Dog worlds. Cat=(p,1-p), Dog=(1-p,p). ``p`` is the separation knob."""
    W = np.array([[p, 1.0 - p], [1.0 - p, p]], dtype=np.float64)
    return SourceSet(W=W, prior=np.array([0.5, 0.5]), meta={"mode": "binary", "p": p, "K": 2})


def dirichlet_sources(K: int, n_sources: int, diversity: float, rng: np.random.Generator) -> SourceSet:
    """Draw ``n_sources`` sources from a symmetric Dirichlet.

    ``diversity`` D is the Dirichlet concentration: small D ⇒ peaky, well-separated
    sources (easy); large D ⇒ near-uniform, overlapping sources (hard).
    """
    W = rng.dirichlet(diversity * np.ones(K), size=n_sources)
    prior = np.ones(n_sources) / n_sources
    return SourceSet(W=W, prior=prior,
                     meta={"mode": "dirichlet", "K": K, "n_sources": n_sources, "diversity": diversity})


def make_sources(cfg: dict, rng: np.random.Generator) -> SourceSet:
    mode = cfg.get("mode", "binary")
    if mode == "binary":
        return binary_sources(cfg.get("p", 0.95))
    if mode == "dirichlet":
        return dirichlet_sources(cfg["K"], cfg.get("n_sources", cfg["K"]),
                                 cfg.get("diversity", 1.0), rng)
    raise ValueError(f"unknown source mode {mode!r}")


# --------------------------------------------------------------------------- #
# Sequences                                                                    #
# --------------------------------------------------------------------------- #
def sample_sequence(w_star: np.ndarray, C: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``C`` tokens iid from Categorical(w_star). Returns int array (C,)."""
    K = w_star.shape[0]
    return rng.choice(K, size=C, p=w_star)


# --------------------------------------------------------------------------- #
# Closed-form optimal belief                                                   #
# --------------------------------------------------------------------------- #
def posterior_trajectory(seq: np.ndarray, sources: SourceSet) -> np.ndarray:
    """Posterior over sources after each prefix length t = 0..C.

    Returns (C+1, S). Row 0 is the prior; row t is p(s | x_{1:t}). Computed
    incrementally in log space for numerical stability.
    """
    C = seq.shape[0]
    logW = np.log(np.clip(sources.W, 1e-300, None))     # (S, K)
    logprior = np.log(np.clip(sources.prior, 1e-300, None))
    out = np.empty((C + 1, sources.S))
    loglik = logprior.copy()
    out[0] = _softmax(loglik)
    for t in range(C):
        loglik = loglik + logW[:, seq[t]]               # add this token's log-likelihood
        out[t + 1] = _softmax(loglik)
    return out


def predictive_trajectory(seq: np.ndarray, sources: SourceSet) -> np.ndarray:
    """Posterior-predictive over the next symbol after each prefix length t = 0..C.

    Returns (C+1, K). Row t = Σ_s p(s | x_{1:t}) · w_s  — the OPTIMAL next-token
    distribution after reading t tokens (what the model's anchor readout should match).
    """
    post = posterior_trajectory(seq, sources)           # (C+1, S)
    return post @ sources.W                             # (C+1, K)


def posterior_over_sources(seq: np.ndarray, sources: SourceSet) -> np.ndarray:
    """Final posterior over sources given the whole sequence. (S,)"""
    return posterior_trajectory(seq, sources)[-1]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# --------------------------------------------------------------------------- #
# Convenience: a single rollout bundle                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    seq: np.ndarray                 # (C,)
    source_id: int
    posterior: np.ndarray           # (C+1, S)
    predictive: np.ndarray          # (C+1, K)


def make_rollout(sources: SourceSet, source_id: int, C: int, rng: np.random.Generator) -> Rollout:
    seq = sample_sequence(sources.W[source_id], C, rng)
    return Rollout(seq=seq, source_id=source_id,
                   posterior=posterior_trajectory(seq, sources),
                   predictive=predictive_trajectory(seq, sources))


def log_spaced_positions(C: int) -> list[int]:
    """Positions t in {1,2,4,...,C} (log-spaced), the analyzed prefix lengths."""
    ps, t = [], 1
    while t < C:
        ps.append(t)
        t *= 2
    ps.append(C)
    return ps
