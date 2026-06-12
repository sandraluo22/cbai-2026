"""LLM runner: present a scripted urn history + own signal, elicit choice + posterior.

Uses the Anthropic SDK with structured output (messages.parse). Surface variation
(paraphrase template + red/blue swap) averages out prompt-surface effects; results
are mapped back to the CANONICAL frame (probability the urn is RED-majority).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from urn import RED, BLUE


# --------------------------------------------------------------------------- #
def load_env():
    """Populate ANTHROPIC_API_KEY from the project .env if not already set.
    Never prints the value."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    envf = Path(__file__).with_name(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class UrnAnswer(BaseModel):
    reasoning: str
    prob_red_majority: float          # P(urn is RED-majority) in the PROMPT's color frame
    choice: Literal["red", "blue"]


# --------------------------------------------------------------------------- #
# Prompt surfaces                                                              #
# --------------------------------------------------------------------------- #
def _system(template_idx: int, q: float) -> str:
    qpct = f"{q:.3f}".rstrip("0").rstrip(".")
    base_rules = (
        f"One of two urns was selected at random with equal probability and is used "
        f"for everyone: a RED-majority urn or a BLUE-majority urn. Participants act "
        f"one at a time. Before deciding, each participant privately draws ONE ball; "
        f"that ball matches the urn's majority color with probability {qpct} and the "
        f"other color with probability {1-q:.3f}. So a private ball is informative but "
        f"not decisive. When it is your turn you are shown the CHOICES (RED or BLUE) of "
        f"the participants before you, in order — you do NOT see the balls they drew. "
        f"Reason carefully about what those prior choices reveal, then combine that with "
        f"your own ball to judge which urn is more likely.")
    if template_idx == 0:
        return "You are a rational participant in a sequential urn-guessing experiment.\n" + base_rules
    if template_idx == 1:
        return ("You are an expert Bayesian reasoner playing a sequential social-learning "
                "game.\n" + base_rules +
                "\nWeigh the public history and your private evidence like an optimal "
                "Bayesian agent.")
    return (base_rules +
            "\nYou are participant in this game and must decide as accurately as possible "
            "which urn is in use.")


def _history_str(history, swap: bool) -> str:
    def show(c):
        c2 = (BLUE if c == RED else RED) if swap else c
        return c2.upper()
    if not history:
        return "(you are the first to decide — no prior choices)"
    return ", ".join(f"#{i+1} chose {show(c)}" for i, c in enumerate(history))


def build_messages(scenario, variant_idx: int, n_templates: int):
    """Return (system, user, swap). swap=True means red/blue are flipped in the prompt."""
    template_idx = variant_idx % n_templates
    swap = (variant_idx // n_templates) % 2 == 1
    sysmsg = _system(template_idx, scenario.q)
    own = (BLUE if scenario.own_signal == RED else RED) if swap else scenario.own_signal
    k = len(scenario.history)
    user = (
        f"{k} participant(s) decided before you. Their choices, in order:\n"
        f"  {_history_str(scenario.history, swap)}\n\n"
        f"Your own private ball is {own.upper()}.\n\n"
        f"State your probability that the urn is the RED-majority urn (a number between "
        f"0 and 1), and your final choice (red or blue). Give one or two sentences of "
        f"reasoning.")
    return sysmsg, user, swap


def _to_canonical(ans: UrnAnswer, swap: bool) -> dict:
    p = max(0.0, min(1.0, float(ans.prob_red_majority)))
    if swap:
        p = 1.0 - p
        choice = RED if ans.choice == BLUE else BLUE
    else:
        choice = ans.choice
    return {"prob_red_majority": p, "choice": choice, "reasoning": ans.reasoning}


# --------------------------------------------------------------------------- #
def ask(client, scenario, variant_idx: int, cfg: dict) -> dict:
    sysmsg, user, swap = build_messages(scenario, variant_idx, cfg["n_templates"])
    kwargs = dict(model=cfg["model"], max_tokens=cfg.get("max_tokens", 2000),
                  system=sysmsg, messages=[{"role": "user", "content": user}],
                  output_format=UrnAnswer)
    if cfg.get("thinking", "adaptive") == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": cfg.get("effort", "medium")}
    resp = client.messages.parse(**kwargs)
    canon = _to_canonical(resp.parsed_output, swap)
    rat = scenario.rational(); nai = scenario.naive()
    return {
        **{k: scenario.meta[k] for k in ("scenario_id", "L", "arm", "phase", "run_color", "onset")},
        "q": scenario.q, "family": scenario.family,
        "history_len": len(scenario.history), "own_signal": scenario.own_signal,
        "true_state": scenario.true_state, "variant_idx": variant_idx, "swap": swap,
        "template_idx": variant_idx % cfg["n_templates"],
        # model (canonical frame)
        "model_choice": canon["choice"], "model_prob_red": canon["prob_red_majority"],
        "model_follows_private": canon["choice"] == scenario.own_signal,
        "model_reasoning": canon["reasoning"][:600],
        # benchmarks
        "rational_choice": rat["choice"], "rational_prob_red": rat["posterior_red"],
        "rational_in_cascade": rat["in_cascade"], "rational_follows_private": rat["follows_private"],
        "naive_choice": nai["choice"], "naive_prob_red": nai["posterior_red"],
        "request_id": getattr(resp, "_request_id", None),
    }


def ask_comprehension(client, q: float, cfg: dict) -> dict:
    """Ask the model to restate the task — a comprehension check."""
    sysmsg = _system(0, q)
    user = ("Before we play: briefly confirm your understanding. (1) With what probability "
            "does your private ball match the urn's true majority color? (2) What information "
            "about earlier participants can you see, and what can you NOT see? (3) In one "
            "sentence, what is the optimal decision rule?")
    resp = client.messages.create(
        model=cfg["model"], max_tokens=cfg.get("max_tokens", 2000),
        system=sysmsg, messages=[{"role": "user", "content": user}])
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {"q": q, "text": text[:1200],
            "states_q": (f"{q:.2f}".rstrip("0").rstrip(".") in text or f"{q:.3f}" in text
                         or f"{round(q*100)}%" in text)}
