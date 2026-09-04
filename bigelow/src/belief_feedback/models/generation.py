"""Generation result records and prompt hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

Message = dict[str, str]  # {"role": ..., "content": ...}


def prompt_hash(messages: list[Message]) -> str:
    payload = "\x1e".join(f"{m['role']}\x1f{m['content']}" for m in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class GenerationResult:
    text: str
    seed: int
    prompt_hash: str
    context_token_count: int
    generation_token_count: int
    wall_time: float
    peak_gpu_memory: float  # bytes; 0 when no GPU


@dataclass
class ScoreResult:
    """Summed and length-normalized conditional sequence log probabilities."""

    logps: list[float]
    logps_normalized: list[float]
    token_counts: list[int]
