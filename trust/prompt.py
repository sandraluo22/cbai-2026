"""Prompt construction for the trust-learning runner.

Each round the context is RECONSTRUCTED from scratch (in-context learning; frozen
weights). It always contains:
  1. a PERSISTENT reputation header (top of context, never buried),
  2. a RAW running recap of past rounds (advisor record only by default), and
  3. the current round's source estimates, with a request for one numeric estimate
     per company.

Reputation is purely VERBAL track-record framing on neutral labels (Source A / B) —
never loaded identities, never bare numeric ratings. The environment's accurate source
is `b` and noisy source is `a`; which DISPLAY letter carries the reputation vs. the
accuracy is controlled by `PromptStyle` (so we can swap labels for the robustness
check). Logged variables are always in ENV terms (a=noisy, b=accurate); the display
swap only changes the text the model sees.

The model answers via STRUCTURED OUTPUT (see ESTIMATE_SCHEMA) — a clean numeric
readout per company, which also keeps reasoning out of the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Style knobs                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class PromptStyle:
    rep_strength: str = "moderate"     # faint | moderate | strong
    info_level: str = "lean"           # lean | self_history | summary | full | no_advisor
    # robustness / surface knobs (defaults = canonical condition):
    rep_letter: str = "A"              # which DISPLAY letter is the reputable one
    order: tuple = ("A", "B")          # display order of the two sources each round
    paraphrase: int = 0                # reputation-wording variant index

    @property
    def acc_letter(self) -> str:       # display letter carrying the ACCURATE source (env b)
        return "B" if self.rep_letter == "A" else "A"

    @property
    def show_self(self) -> bool:
        return self.info_level in ("self_history", "full")

    @property
    def show_summary(self) -> bool:
        return self.info_level in ("summary", "full")

    @property
    def no_advisor(self) -> bool:
        return self.info_level == "no_advisor"


# --------------------------------------------------------------------------- #
# Reputation header (verbal track-record framing)                              #
# --------------------------------------------------------------------------- #
# Reputation is established by SITUATING the task: a concrete analyst role plus each
# provider's standing within that world. Bullets are listed in fixed letter order
# (A then B); only the DESCRIPTION moves with `rep_letter`, so the label-swap robustness
# check flips reputation without moving anything else. `paraphrase` selects a wording.
_REP_VARIANTS = [
    {"intro": "You are the lead valuation analyst at an investment firm. Each round the "
              "firm needs your best estimate of the true value of several companies, and "
              "you can draw on two research providers:",
     "estab": "the firm's established provider. It has supplied these valuations for years "
              "and the investment committee has long relied on it.",
     "new":   "a provider the firm brought on only recently. It has no track record with "
              "the firm yet.",
     "closer": "Issue the firm's official value estimate for each company."},
    {"intro": "You work as the valuation analyst on an investment desk. Each round you must "
              "value several companies, drawing on two outside research houses the firm "
              "subscribes to:",
     "estab": "a research house the firm has subscribed to for many years; its valuations "
              "are well regarded and the desk has long trusted them.",
     "new":   "a research house the firm signed up with only this quarter; its reliability "
              "here is not yet established.",
     "closer": "Record your own value estimate for each company."},
]


def reputation_header(style: PromptStyle) -> str:
    v = _REP_VARIANTS[style.paraphrase % len(_REP_VARIANTS)]
    bullets = [f"  - Source {L}: {v['estab'] if L == style.rep_letter else v['new']}"
               for L in ("A", "B")]
    return v["intro"] + "\n" + "\n".join(bullets) + "\n" + v["closer"]


def _rep_tag(letter: str, style: PromptStyle) -> str:
    """Inline reputation tag for the STRONG rep-strength condition."""
    if style.rep_strength != "strong":
        return ""
    return " (long-established, widely trusted)" if letter == style.rep_letter \
        else " (recently added, unproven)"


# --------------------------------------------------------------------------- #
# Per-round value lookup respecting the display swap                           #
# --------------------------------------------------------------------------- #
def _display_value(letter: str, a_row: np.ndarray, b_row: np.ndarray, i: int,
                   style: PromptStyle) -> float:
    """Value shown next to DISPLAY `letter` for company i. The reputable display
    letter is fed the NOISY source (env a); the other gets the ACCURATE source (env b)."""
    return float(a_row[i]) if letter == style.rep_letter else float(b_row[i])


def _fmt(x: float) -> str:
    return f"{x:.0f}"


# --------------------------------------------------------------------------- #
# Recap + current round                                                        #
# --------------------------------------------------------------------------- #
def _recap_block(game, t: int, style: PromptStyle, self_hist: np.ndarray | None,
                 running_abs_err) -> str:
    """RAW recap of rounds 0..t-1. Advisor record only by default; info-ladder adds
    self-history and/or running per-source accuracy. NOT annotated with who was closer."""
    if t == 0:
        return "No past rounds yet — this is the first round.\n"
    lines = ["History of past rounds (raw):"]
    for r in range(t):
        parts = []
        for i in range(game.M):
            seg = f"Co.{i + 1}: "
            if not style.no_advisor:
                cells = []
                for letter in style.order:
                    v = _display_value(letter, game.a[r], game.b[r], i, style)
                    cells.append(f"{letter}={_fmt(v)}")
                seg += ", ".join(cells) + ", "
                if style.show_self and self_hist is not None:
                    seg += f"you={_fmt(self_hist[r, i])}, "
            seg += f"true={_fmt(game.theta[r, i])}"
            parts.append(seg)
        lines.append(f"Round {r + 1} — " + "  ".join(parts))
    if style.show_summary and not style.no_advisor and running_abs_err is not None:
        mae_rep, mae_acc = running_abs_err
        lines.append(
            f"Running average error so far — Source {style.rep_letter}: {mae_rep:.1f}; "
            f"Source {style.acc_letter}: {mae_acc:.1f}.")
    return "\n".join(lines) + "\n"


def _current_block(game, t: int, style: PromptStyle) -> str:
    if style.no_advisor:
        cos = "  ".join(f"Co.{i + 1}" for i in range(game.M))
        return (f"Round {t + 1} (current). No source estimates are available this round. "
                f"Give your best numeric estimate of the true value for each company: {cos}.")
    lines = [f"Round {t + 1} (current). Two sources give estimates below. Give your best "
             f"numeric estimate of each company's true value."]
    for i in range(game.M):
        cells = []
        for letter in style.order:
            v = _display_value(letter, game.a[t], game.b[t], i, style)
            cells.append(f"Source {letter}{_rep_tag(letter, style)} = {_fmt(v)}")
        lines.append(f"Co.{i + 1}: " + ", ".join(cells))
    return "\n".join(lines)


SYSTEM = (
    "You are completing a sequence of valuation rounds. New companies appear every round, "
    "so the only thing that carries over is what you learn about the two providers' "
    "reliability from the revealed outcomes. Weigh the providers using that running "
    "record. Respond only with your numeric estimates in the required format."
)


def build_prompt_parts(game, t: int, style: PromptStyle, self_hist: np.ndarray | None = None,
                       running_abs_err=None) -> tuple[str, str]:
    """Return (stable_prefix, volatile_suffix). The prefix = reputation header + the raw
    recap, which only GROWS round to round (append-only) and is therefore the natural
    prompt-cache breakpoint; the suffix = this round's source estimates, which change
    every call. (Caching only actually engages once the prefix exceeds the model's minimum
    cacheable length — 2048 tokens for Sonnet 4.6 — so it is a no-op for short games.)"""
    header = reputation_header(style)
    restate = ""
    if style.rep_strength == "moderate" and t > 0:
        restate = ("\nReminder: Source {R} is the firm's long-trusted provider; Source {U} "
                   "is the recent addition.\n").format(R=style.rep_letter, U=style.acc_letter)
    recap = _recap_block(game, t, style, self_hist, running_abs_err)
    current = _current_block(game, t, style)
    return f"{header}\n{restate}\n{recap}", current


def build_prompt(game, t: int, style: PromptStyle, self_hist: np.ndarray | None = None,
                 running_abs_err=None) -> str:
    """Full reconstructed user prompt for round t (0-indexed)."""
    prefix, suffix = build_prompt_parts(game, t, style, self_hist, running_abs_err)
    return f"{prefix}\n{suffix}"


# --------------------------------------------------------------------------- #
# Structured-output schema + parsing                                           #
# --------------------------------------------------------------------------- #
def estimate_schema(M: int) -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "estimates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company": {"type": "integer"},
                            "value": {"type": "number"},
                        },
                        "required": ["company", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["estimates"],
            "additionalProperties": False,
        },
    }


def parse_estimates(obj: dict, M: int) -> np.ndarray:
    """Map the structured response to an (M,) array ordered by company 1..M."""
    out = np.full(M, np.nan)
    for item in obj.get("estimates", []):
        c = int(item["company"]) - 1
        if 0 <= c < M:
            out[c] = float(item["value"])
    return out


COMPREHENSION_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "reputable_source": {"type": "string"},      # which label is described as long-trusted
            "what_is_revealed": {"type": "string"},        # what they see after each round
            "fresh_items_each_round": {"type": "boolean"},
        },
        "required": ["reputable_source", "what_is_revealed", "fresh_items_each_round"],
        "additionalProperties": False,
    },
}
