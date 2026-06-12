"""Config (dataclass). Seeds non-LLM randomness; holds model-per-seat + protocol knobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameConfig:
    n_players: int = 7
    seed: int = 0

    # model per seat (default all seats to current Opus); lets you mix later
    default_model: str = "claude-opus-4-8"
    models_per_seat: Optional[dict[int, str]] = None     # seat -> model id

    # terse private chain-of-thought (god-view only; NEVER shown to other players)
    capture_cot: bool = True
    cot_sentences: int = 2                               # cap CoT to a few sentences

    # discussion protocol
    discussion_ready_threshold: int = 4                  # >= this many ready -> end discussion
    max_discussion_rounds: int = 3
    max_sentences_per_statement: int = 3
    word_backstop: int = 80                              # hard char/word cap on public statement

    # API
    max_retries: int = 5
    request_timeout: float = 120.0
    max_tokens: int = 1024

    def model_for(self, seat: int) -> str:
        if self.models_per_seat and seat in self.models_per_seat:
            return self.models_per_seat[seat]
        return self.default_model
