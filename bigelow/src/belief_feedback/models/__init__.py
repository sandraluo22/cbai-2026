"""Model backends: a deterministic mock and a Hugging Face implementation."""

from __future__ import annotations

from ..config import Config
from .base import Backend


def make_backend(cfg: Config) -> Backend:
    """Instantiate the configured backend.

    A scientific run never silently substitutes a model: the HF backend
    loads exactly ``cfg.model.model_id`` and any deliberate fallback must be
    configured explicitly (and lands in every manifest).
    """
    if cfg.model.backend == "mock":
        from .mock_backend import MockBackend

        return MockBackend(cfg)
    from .hf_backend import HuggingFaceBackend

    return HuggingFaceBackend(cfg)
