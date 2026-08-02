"""Anthropic (Claude) chat agent — the GPU-free version of the conversational games.

Same turn-taking contract as HFChatAgent: a transcript of (speaker, text) lines
is mapped to Claude message roles from this agent's perspective (its own lines are
`assistant`, everyone else's are `user`, prefixed with the speaker tag).

The API key is read from the environment (ANTHROPIC_API_KEY). Load it from an
untracked .env — never hardcode it. See games/.env.example.
"""
from __future__ import annotations

import os
from typing import List, Tuple


class APIChatAgent:
    def __init__(self, tag: str, model_id: str, api_key: str | None = None):
        import anthropic
        self.tag = tag
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _build_messages(self, transcript: List[Tuple[str, str]]):
        msgs = []
        for speaker, text in transcript:
            if speaker == self.tag:
                msgs.append({"role": "assistant", "content": text})
            else:
                msgs.append({"role": "user", "content": f"[{speaker}]: {text}"})
        # Claude requires the first message to be `user` and the last to be `user`.
        if not msgs or msgs[0]["role"] != "user":
            msgs.insert(0, {"role": "user", "content": "(begin)"})
        if msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": "(your turn)"})
        return msgs

    def say(self, system: str, transcript: List[Tuple[str, str]],
            max_new_tokens: int = 400) -> str:
        msgs = self._build_messages(transcript)
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_new_tokens,
            system=system,
            messages=msgs,
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def free(self):
        pass
