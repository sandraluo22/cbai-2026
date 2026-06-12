"""Information-cascade urn paradigm (Bikhchandani-Hirshleifer-Welch / Anderson-Holt)
and the EXACT Bayesian solver. No LLM here.

Setup
-----
Two equally-likely states: the urn's majority color is "red" or "blue".
Each agent draws ONE private signal (a ball) that matches the true majority color
with probability q in (1/2, 1). Agents act in sequence; each sees the ORDERED
history of prior CHOICES (not signals) plus its own signal, then chooses a color.

Bayesian inference over a choice history
----------------------------------------
Work in log-likelihood-ratio (LLR) units of L = log(q/(1-q)) for state=red vs blue.
Each *informative* choice reveals one signal: +1 net for red, -1 for blue. The key
BHW insight: once the public net signal count reaches |d| >= 2, the next agent's
own signal (worth +/-1) can no longer flip the public belief, so that agent
*cascades* — its choice is uninformative and does NOT update the public count.

Consequences (all verified in test_urn.py):
  * The cascade threshold is |net informative signals| >= 2, INDEPENDENT of q.
    (q sets the strength of belief, not the count threshold.)
  * A uniform run of 2 same-color choices makes agent #3 the first to cascade.
  * A rational agent CAPS the public evidence at +/-2 signals; a naive
    majority-counter (which treats every choice as informative) does not.

Tie convention: an indifferent agent (total LLR == 0) follows its own signal
(the standard Anderson-Holt convention), which keeps |d|=1 agents informative and
places the cascade onset cleanly at |d|=2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RED, BLUE = "red", "blue"


def signal_llr(q: float) -> float:
    """L = log(q/(1-q)); the LLR contribution of one revealed signal."""
    if not (0.5 < q < 1.0):
        raise ValueError("q must be in (0.5, 1.0)")
    return math.log(q / (1.0 - q))


def _pm(color: str) -> int:
    return +1 if color == RED else -1


def infer_history(choices: list[str]) -> dict:
    """Rational forward inference over a history of prior CHOICES.

    Returns:
      public_count   : net informative signals d (red +1 / blue -1), capped by cascades
      informative    : per-position bool (did this choice reveal a signal?)
      in_cascade     : per-position bool (was this agent cascading when it chose?)
      onset          : 1-based position of the FIRST agent that cascades (or None if
                       no agent within/just-after the history would cascade)
      offpath        : per-position bool (choice contradicts an active cascade ->
                       impossible under rational play; flagged, treated as uninformative)
    """
    d = 0
    informative, in_cascade, offpath = [], [], []
    onset = None
    for c in choices:
        casc = abs(d) >= 2
        in_cascade.append(casc)
        if casc:
            if onset is None:
                onset = len(in_cascade)            # 1-based index of this position
            informative.append(False)
            # on-path a cascader chooses sign(d); a contrary scripted choice is off-path
            offpath.append(_pm(c) != (1 if d > 0 else -1))
        else:
            informative.append(True)
            offpath.append(False)
            d += _pm(c)
    if onset is None and abs(d) >= 2:
        onset = len(choices) + 1                    # the NEXT agent would be first to cascade
    return {"public_count": d, "informative": informative,
            "in_cascade": in_cascade, "onset": onset, "offpath": offpath}


def cascade_onset(choices: list[str]) -> int | None:
    """1-based position of the first agent that would cascade given this prefix."""
    return infer_history(choices)["onset"]


def _decision_from_units(t: int, d_public: int, own_signal: str, q: float,
                         in_cascade: bool) -> dict:
    L = signal_llr(q)
    post_red = 1.0 / (1.0 + math.exp(-t * L))
    if in_cascade:
        choice = RED if d_public > 0 else BLUE      # public dominates; own signal ignored
    elif t > 0:
        choice = RED
    elif t < 0:
        choice = BLUE
    else:
        choice = own_signal                          # tie -> follow own signal
    return {"posterior_red": post_red, "choice": choice, "total_units": t}


def rational_decision(history: list[str], own_signal: str, q: float) -> dict:
    """Exact Bayesian focal decision given prior choices + own signal.

    Returns posterior P(majority=red), the rational choice, whether the focal agent
    is in a cascade (own signal cannot change its choice), and the public net count.
    """
    info = infer_history(history)
    d = info["public_count"]
    in_casc = abs(d) >= 2
    t = d + _pm(own_signal)
    out = _decision_from_units(t, d, own_signal, q, in_casc)
    out.update({"in_cascade": in_casc, "public_count": d,
                "follows_private": out["choice"] == own_signal,
                "onset": info["onset"]})
    return out


def naive_decision(history: list[str], own_signal: str, q: float) -> dict:
    """Naive majority-counter: treats EVERY prior choice as an informative signal
    (ignores cascade-uninformativeness). Diagnostic baseline, not rational."""
    d = sum(_pm(c) for c in history)
    t = d + _pm(own_signal)
    out = _decision_from_units(t, d, own_signal, q, in_cascade=False)
    out.update({"public_count": d, "follows_private": out["choice"] == own_signal})
    return out


@dataclass
class Scenario:
    """A scripted decision point for the focal LLM."""
    history: list[str]            # ordered prior choices
    own_signal: str               # focal agent's private signal
    q: float
    true_state: str = ""          # the actual majority color (for welfare/wrong-cascade)
    family: str = ""              # scenario family tag
    meta: dict = field(default_factory=dict)

    @property
    def position(self) -> int:
        return len(self.history) + 1

    def rational(self) -> dict:
        return rational_decision(self.history, self.own_signal, self.q)

    def naive(self) -> dict:
        return naive_decision(self.history, self.own_signal, self.q)
