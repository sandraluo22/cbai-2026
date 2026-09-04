"""Deterministic seed derivation.

Every stochastic operation in the pipeline derives its seed from a stable
tuple of identifying fields via BLAKE2b. Paired causal branches use *common
random numbers*: the generation seed for a (world, replicate, agent, round,
slot) tuple deliberately excludes the branch identifier, so corresponding
generation slots in a baseline episode and any of its branches consume
identical RNG streams. Branches therefore differ only through their
interventions (steering, message clamping/replay), never through fresh
sampling noise. Where a branch requires an *independent* stream (never for
paired comparisons), ``derive_seed`` is called with the branch id appended.
"""

from __future__ import annotations

from hashlib import blake2b

import numpy as np

_SEP = "\x1f"  # unit separator: cannot occur in our identifier strings


def derive_seed(*parts: object) -> int:
    """Derive a stable 31-bit seed from an arbitrary tuple of parts."""
    payload = _SEP.join(str(p) for p in parts).encode("utf-8")
    h = blake2b(payload, digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


def generation_seed(
    world_id: str, replicate_seed: int, agent_id: int, round_idx: int, slot: str
) -> int:
    """Seed for one generation slot.

    The branch identifier is intentionally absent (common random numbers);
    see module docstring.
    """
    return derive_seed("gen", world_id, replicate_seed, agent_id, round_idx, slot)


def rng(*parts: object) -> np.random.Generator:
    """NumPy generator seeded from a derived seed."""
    return np.random.default_rng(derive_seed(*parts))
