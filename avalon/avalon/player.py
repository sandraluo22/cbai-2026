"""Player seat = one stateless Anthropic API call per action.

Security: the API key is read by the SDK from ANTHROPIC_API_KEY in the
environment. It is NEVER read into a variable we log, never printed, never
written anywhere. We only ever check presence (a bool).

Each machine action returns STRUCTURED JSON (parsed defensively, one re-prompt on
malformed output, then fail loudly). Only the public statement is free text, and
it is length-capped. Token usage is returned per call for cost accounting.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class CallResult:
    parsed: dict
    raw_text: str
    input_tokens: int
    output_tokens: int
    reasoning: str = ""          # terse PRIVATE chain-of-thought (god-view only)


class ApiError(RuntimeError):
    pass


SYSTEM = (
    "You are playing a single game of The Resistance: Avalon as one seat at the "
    "table. There are 7 players (seats 0-6): 4 GOOD (Merlin, Percival, two Loyal "
    "Servants) and 3 EVIL (Assassin, Morgana, Minion of Mordred). There is NO "
    "night phase and NO eliminations. Good wins by succeeding 3 missions (unless "
    "the Assassin then identifies Merlin); Evil wins by failing 3 missions, by 5 "
    "consecutive rejected team proposals, or by a correct Assassin guess of Merlin.\n"
    "You only know what is in YOUR secret knowledge plus the public state and the "
    "public discussion. Play to win for your team. When asked for a machine action, "
    "respond with ONLY a single JSON object and nothing else."
)


class PlayerClient:
    def __init__(self, cfg):
        # SDK auto-reads ANTHROPIC_API_KEY from the env; we never touch the value.
        self.client = anthropic.Anthropic()
        self.cfg = cfg

    # -- low-level call with retry/backoff -------------------------------- #
    def _raw_call(self, model: str, user: str) -> tuple[str, int, int]:
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                msg = self.client.messages.create(
                    model=model, max_tokens=self.cfg.max_tokens, system=SYSTEM,
                    messages=[{"role": "user", "content": user}],
                    timeout=self.cfg.request_timeout)
                text = "".join(b.text for b in msg.content if b.type == "text")
                return text, msg.usage.input_tokens, msg.usage.output_tokens
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as e:
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise ApiError(f"API call failed after {self.cfg.max_retries} retries: {type(last).__name__}")

    # -- structured action with one re-prompt ---------------------------- #
    def act(self, seat: int, model: str, view_text: str, instruction: str,
            required_keys: list[str]) -> CallResult:
        cot_note = ""
        if getattr(self.cfg, "capture_cot", False):
            cot_note = (f'\nAlso include a "reasoning" field: at most {self.cfg.cot_sentences} '
                        f"sentences of your PRIVATE reasoning. This is never shown to other players.")
        prompt = f"{view_text}\n\n=== YOUR TASK ===\n{instruction}{cot_note}"
        in_tok = out_tok = 0
        for repromot in range(2):
            text, it, ot = self._raw_call(model, prompt if repromot == 0 else
                                          prompt + "\n\nYour previous reply was not valid JSON with "
                                          f"keys {required_keys}. Respond with ONLY that JSON object.")
            in_tok += it; out_tok += ot
            parsed = _extract_json(text)
            if parsed is not None and all(k in parsed for k in required_keys):
                reasoning = str(parsed.get("reasoning", "")).strip()
                if reasoning:                  # cap to a terse few sentences
                    reasoning, _ = enforce_statement_length(
                        reasoning, self.cfg.cot_sentences, self.cfg.word_backstop)
                return CallResult(parsed, text, in_tok, out_tok, reasoning=reasoning)
        raise ApiError(f"seat {seat}: malformed structured output after re-prompt "
                       f"(needed keys {required_keys})")


def _extract_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(json)?", "", text).strip()
    for cand in [cleaned] + re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Public statement length enforcement                                          #
# --------------------------------------------------------------------------- #
def enforce_statement_length(text: str, max_sentences: int, word_backstop: int):
    """Return (capped_text, was_truncated). Caps sentences then a hard word backstop."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    truncated = False
    if len(sentences) > max_sentences:
        text = " ".join(sentences[:max_sentences]); truncated = True
    words = text.split()
    if len(words) > word_backstop:
        text = " ".join(words[:word_backstop]) + " …"; truncated = True
    return text.strip(), truncated
