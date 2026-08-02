"""Lightweight text embeddings + convergence metrics for the conversational games.

Prefers sentence-transformers (all-MiniLM-L6-v2, small & ungated). Falls back to a
hashing bag-of-words vector if unavailable, so the harness still runs (metrics are
coarser but the pipeline is intact).
"""
from __future__ import annotations

from typing import List
import numpy as np

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _MODEL = "hash"
    return _MODEL


def embed(texts: List[str]) -> np.ndarray:
    m = _get_model()
    if m == "hash":
        return _hash_embed(texts)
    v = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=float)


def _hash_embed(texts: List[str], dim: int = 256) -> np.ndarray:
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out / n


def pairwise_cosine_mean(vecs: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity of a set of (normalized) row vectors.
    High -> the set is semantically converged."""
    if len(vecs) < 2:
        return float("nan")
    S = vecs @ vecs.T
    iu = np.triu_indices(len(vecs), k=1)
    return float(S[iu].mean())


def centroid_spread(vecs: np.ndarray) -> float:
    """Mean distance of vectors to their centroid (lower -> tighter cluster)."""
    if len(vecs) < 2:
        return float("nan")
    c = vecs.mean(0, keepdims=True)
    return float(np.linalg.norm(vecs - c, axis=1).mean())
