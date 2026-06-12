"""Game-state renderer with SYMMETRIC private/social blocks.

The two evidence blocks use the IDENTICAL template and differ only by their source
label — same number of rows, same per-reading formatting, same length. This is
deliberate: it removes the rich-vs-scalar confound, so any reliance difference is
attributable to the SOURCE, not to format/length/dimensionality. (A token-length
symmetry assert lives in model_runner/run_experiment, where a tokenizer exists.)

The prompt ends with the MARKER line "Decision :" — the trailing colon (its own
token, thanks to the space) is the default activation-capture anchor.
"""

from __future__ import annotations

import re

import numpy as np

MARKER = "Decision :"
# Equal-length labels so the two blocks match char-for-char (token symmetry is
# additionally asserted at runtime). PERSONAL = the agent's own analysis (private);
# EXTERNAL = other analysts / the market (social).
PRIVATE_LABEL = "PERSONAL"
SOCIAL_LABEL = "EXTERNAL"


def _company_label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"C{i}"


def _readings_block(label: str, hist: np.ndarray, n: int) -> str:
    """Identical template for either channel; only `label` changes."""
    t = hist.shape[1]
    lines = [f"{label} readings (rounds 1..{t}):"]
    for i in range(n):
        vals = "  ".join(f"r{r+1}={hist[i, r]:+.2f}" for r in range(t))
        lines.append(f"  Company {_company_label(i)}: {vals}")
    return "\n".join(lines)


def render(private: np.ndarray, social: np.ndarray, cfg, *, marker: str = MARKER,
           private_label: str = PRIVATE_LABEL, social_label: str = SOCIAL_LABEL) -> str:
    """Build the full prompt. private/social: (n, t) with equal t."""
    n = private.shape[0]
    companies = ", ".join(_company_label(i) for i in range(n))
    act = f"invest in ONE company ({companies})"
    if getattr(cfg, "allow_withdraw", False):
        act += f", or withdraw from one"
    # Two signals with DISTINCT meaning (private value estimate vs. market trend),
    # rendered in the identical block format. We deliberately do NOT state how the
    # payoff is computed (the reward knob w stays hidden), so the model's reliance
    # on social is spontaneous/uninstructed, not a response to stated incentives.
    framing = (
        f"You are an investor choosing where to invest among {n} companies: "
        f"{companies}. Each company has a hidden underlying value. For each one you "
        f"have two signals, shown in the same format and on the same scale "
        f"(higher is better):\n"
        f"  - {private_label}: your own private analysis of the company's value.\n"
        f"  - {social_label}: the market trend — how strongly other investors are "
        f"currently favoring the company.\n"
    )
    pblock = _readings_block(private_label, private, n)
    sblock = _readings_block(social_label, social, n)
    instruct = (
        f"\nWeigh the two sources and {act}. Respond with ONLY the single company "
        f"letter you choose (one of {companies}) — no other words.\n"
    )
    return f"{framing}\n{pblock}\n\n{sblock}\n{instruct}\n{marker}"


def block_char_lengths(private: np.ndarray, social: np.ndarray) -> tuple[int, int]:
    """Char-length symmetry check (token check happens with a tokenizer)."""
    n = private.shape[0]
    return (len(_readings_block(PRIVATE_LABEL, private, n)),
            len(_readings_block(SOCIAL_LABEL, social, n)))


def parse_action(text: str, n: int, allow_withdraw: bool = False):
    t = text.strip()
    withdraw = allow_withdraw and "withdraw" in t.lower()
    for tok in re.findall(r"\b([A-Z])\b", t.upper()):
        idx = ord(tok) - ord("A")
        if 0 <= idx < n:
            return {"company": idx, "withdraw": withdraw}
    return {"company": None, "withdraw": withdraw}
