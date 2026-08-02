"""Model registry for the games/ experiments.

Two families of agents are used across the four games:

- Open-weight models on the GPU pod (default). We use *ungated mirrors* so no HF
  token is needed (the pod already caches the three base mirrors). BASE models are
  used for the mechanistic game (random-walk ping-pong, which needs residual
  activations); INSTRUCT/chat models are used for the three conversational games.
- Anthropic API models (Claude) as a separate, GPU-free version of the three
  conversational games. See api_agent.py.

The three open families mirror the cross-model project (Llama / Gemma / Qwen).
"""
from __future__ import annotations

# tag -> {base HF id (for activation capture), chat HF id (for turn-taking)}
# All ids are ungated public mirrors so `huggingface-cli login` is not required.
OPEN_MODELS = {
    "Llama": {
        "base": "NousResearch/Meta-Llama-3.1-8B",
        "chat": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "n_layers": 32,       # decoder blocks (0..31); grid peaks ~L31 (cross-model)
    },
    "Gemma": {
        "base": "unsloth/gemma-2-9b",
        "chat": "unsloth/gemma-2-9b-it",
        "n_layers": 42,
        # gemma-2 chat template rejects a `system` role -> fold system into user turn
        "no_system_role": True,
    },
    "Qwen": {
        "base": "Qwen/Qwen3-8B-Base",
        "chat": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 36,
    },
}

# Claude models for the API version of the conversational games.
# Deliberately diverse tiers so multi-LLM games have genuinely different agents.
API_MODELS = {
    "Opus":   "claude-opus-4-8",
    "Sonnet": "claude-sonnet-4-6",
    "Haiku":  "claude-haiku-4-5",
}

# Default rosters (2-agent then 3-agent) for the multi-LLM games.
OPEN_ROSTER_2 = ["Llama", "Gemma"]
OPEN_ROSTER_3 = ["Llama", "Gemma", "Qwen"]
API_ROSTER_2 = ["Opus", "Haiku"]
API_ROSTER_3 = ["Opus", "Sonnet", "Haiku"]
