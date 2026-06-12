"""Belief readout: soft (faithful) + hard (validation) + agreement (§3).

Every round, each agent's belief is read TWO ways and BOTH are logged:

  * soft  (§3a): next-token logits at the fixed anchor, restricted to the
                 candidate vocabulary, renormalized. Scored two ways:
                   (i)  first-token logit            (biased toward common first tokens)
                   (ii) full-sequence length-normalized logprob  (default canonical)
  * hard  (§3b): model emits an explicit JSON distribution; parsed robustly.

The canonical x_i (default: length-normalized soft) is flagged in the logs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .agents import Agent, ModelBackend
from .config import ReadoutCanonical, RunConfig


@dataclass
class SoftReadout:
    first_token: np.ndarray        # (K,) distribution
    length_norm: np.ndarray        # (K,) distribution
    canonical: np.ndarray          # whichever is configured canonical
    canonical_name: str
    anchor_token_id: int
    activations: Optional[np.ndarray] = None   # (n_layers, hidden_dim)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def soft_readout(
    backend: ModelBackend, agent: Agent, cfg: RunConfig, hooks=None
) -> SoftReadout:
    vocab = backend.vocab
    prompt = agent.soft_prompt(cfg, vocab)

    logits_last, anchor_id = backend.soft_forward(prompt, images=agent.images, hooks=hooks)
    logits_last = logits_last.float().cpu().numpy()

    # (i) first-token scoring over candidate first tokens
    first_logits = np.array([logits_last[tid] for tid in vocab.first_token_ids])
    first_dist = _softmax(first_logits)

    # (ii) length-normalized full-sequence scoring
    ln_logprobs = backend.candidate_length_norm_logprobs(prompt, images=agent.images)
    ln_dist = _softmax(ln_logprobs)

    if cfg.readout.canonical == ReadoutCanonical.FIRST_TOKEN:
        canonical, cname = first_dist, "first_token"
    else:
        canonical, cname = ln_dist, "length_norm"

    acts = None
    if hooks is not None and getattr(hooks, "_buffers", None):
        acts = hooks.last_token_per_layer()

    return SoftReadout(
        first_token=first_dist, length_norm=ln_dist, canonical=canonical,
        canonical_name=cname, anchor_token_id=anchor_id, activations=acts,
    )


# --------------------------------------------------------------------------- #
# Hard readout                                                                #
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_distribution(text: str, candidate_names: list[str]) -> np.ndarray:
    """Robustly parse a JSON candidate->probability map. Tolerates fences/trailing text."""
    K = len(candidate_names)
    cleaned = text.strip()
    cleaned = re.sub(r"```(json)?", "", cleaned)
    obj = None
    for match in [cleaned] + _JSON_RE.findall(cleaned):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                break
        except (json.JSONDecodeError, TypeError):
            obj = None
    dist = np.zeros(K)
    if isinstance(obj, dict):
        lower = {k.lower(): v for k, v in obj.items()}
        for i, name in enumerate(candidate_names):
            v = lower.get(name.lower(), 0.0)
            try:
                dist[i] = float(v)
            except (TypeError, ValueError):
                dist[i] = 0.0
    if dist.sum() <= 0:
        # Salvage: JSON failed -> count candidate-name mentions in the free text
        # (e.g. "I think it's an elephant") before giving up to uniform. This stops
        # parse failures from masquerading as a genuinely uniform belief.
        counts = np.array([cleaned.lower().count(n.lower()) for n in candidate_names], float)
        dist = counts if counts.sum() > 0 else np.ones(K)
    return dist / dist.sum()


def hard_readout(backend: ModelBackend, agent: Agent, cfg: RunConfig) -> tuple[np.ndarray, str]:
    """Return (parsed distribution, raw model generation) for verbatim transcripts."""
    prompt = agent.hard_prompt(cfg, backend.vocab)
    raw = backend.generate(prompt, images=agent.images, max_new_tokens=160)
    return parse_distribution(raw, backend.vocab.names), raw


# --------------------------------------------------------------------------- #
# 3c. Agreement                                                               #
# --------------------------------------------------------------------------- #
def kl_l1(soft: np.ndarray, hard: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = soft + eps
    q = hard + eps
    p, q = p / p.sum(), q / q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    l1 = float(np.sum(np.abs(soft - hard)))
    return kl, l1


# --------------------------------------------------------------------------- #
# §8 degenerate-behavior detector                                            #
# --------------------------------------------------------------------------- #
def is_degenerate(dist: np.ndarray, uniform_tol: float = 0.02) -> bool:
    """Flag near-uniform output (model confusion vs. genuine drift)."""
    K = dist.shape[0]
    return bool(np.max(np.abs(dist - 1.0 / K)) < uniform_tol)
