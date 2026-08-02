"""Factory: build a roster of chat agents for either backend.

backend="open"  -> HFChatAgent on the GPU (Llama/Gemma/Qwen instruct)
backend="api"   -> APIChatAgent via the Anthropic API (Claude Opus/Sonnet/Haiku)

The conversational games (convergence, volunteer's dilemma, attractor states) are
backend-agnostic: they only call `.say(system, transcript)` on each agent.
"""
from __future__ import annotations

import os
from typing import List
from .modelreg import OPEN_MODELS, API_MODELS


def build_chat_agents(tags: List[str], backend: str, device: str = "cuda"):
    agents = []
    if backend == "open":
        from .hf_agent import HFChatAgent
        for t in tags:
            spec = OPEN_MODELS[t]
            agents.append(HFChatAgent(
                t, spec["chat"], no_system_role=spec.get("no_system_role", False),
                device=device))
    elif backend == "api":
        from .api_agent import APIChatAgent
        from . import io_utils
        io_utils.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        for t in tags:
            agents.append(APIChatAgent(t, API_MODELS[t]))
    else:
        raise ValueError(f"unknown backend {backend!r} (use 'open' or 'api')")
    return agents


def default_roster(backend: str, n: int) -> List[str]:
    from .modelreg import OPEN_ROSTER_2, OPEN_ROSTER_3, API_ROSTER_2, API_ROSTER_3
    if backend == "open":
        return OPEN_ROSTER_3[:n] if n >= 3 else OPEN_ROSTER_2[:n]
    return API_ROSTER_3[:n] if n >= 3 else API_ROSTER_2[:n]
