"""Common backend interface for the mock and Hugging Face models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..world.schema import World
from .generation import GenerationResult, Message, ScoreResult
from .steering import SteeringSpec


class Backend(ABC):
    """Every pipeline operation goes through this interface.

    Implementations must be deterministic given (messages, seed, steering).
    """

    n_layers: int
    hidden_size: int
    model_id: str

    def register_world(self, world: World) -> None:  # noqa: B027 - optional hook
        """Give the backend visibility into a world's registry.

        Used only by the mock backend to look up hidden ground truth for its
        synthetic belief dynamics. The HF backend ignores it: a real model
        sees nothing but the prompt text.
        """

    @abstractmethod
    def score_choices(
        self,
        messages: list[Message],
        choices: list[str],
        steering: SteeringSpec | None = None,
    ) -> ScoreResult:
        """Exact conditional sequence log probabilities of each completion.

        Handles multi-token completions; returns summed and
        length-normalized scores.
        """

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        seed: int,
        steering: SteeringSpec | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResult:
        """Sample one completion. Malformed output is returned as-is."""

    @abstractmethod
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int] | None = None,
        steering: SteeringSpec | None = None,
    ) -> np.ndarray:
        """Final-prompt-token residual-stream activations, [n_layers, hidden].

        ``layers=None`` returns all layers (index 0 = embeddings output).
        """

    # ---- batched paths ----------------------------------------------------
    # Defaults loop over the per-item methods, which preserves the exact
    # per-slot seeding contract. The HF backend overrides these with true
    # batched forwards: rows within a round are generated from frozen,
    # independent contexts, so batching them is protocol-equivalent; per-row
    # sampling streams are independent given a fixed row order (documented
    # in METHODS.md).

    def generate_batch(
        self,
        messages_list: list[list[Message]],
        seeds: list[int],
        steerings: list[SteeringSpec | None] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        steerings = steerings or [None] * len(messages_list)
        return [
            self.generate(m, seed=s, steering=sp, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p)
            for m, s, sp in zip(messages_list, seeds, steerings)
        ]

    def score_choices_batch(
        self,
        messages_list: list[list[Message]],
        choices: list[str],
        steerings: list[SteeringSpec | None] | None = None,
    ) -> list[ScoreResult]:
        steerings = steerings or [None] * len(messages_list)
        return [
            self.score_choices(m, choices, steering=sp)
            for m, sp in zip(messages_list, steerings)
        ]

    def get_selected_activations_batch(
        self,
        messages_list: list[list[Message]],
        layer: int,
        steerings: list[SteeringSpec | None] | None = None,
    ) -> list[np.ndarray]:
        """Final-prompt-token activation at one layer, per context."""
        steerings = steerings or [None] * len(messages_list)
        out = []
        for m, sp in zip(messages_list, steerings):
            acts = self.get_activations(m, steering=sp)
            out.append(acts[min(layer, acts.shape[0] - 1)])
        return out
