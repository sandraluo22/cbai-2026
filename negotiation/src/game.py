"""The repeated-negotiation game loop.

Round structure (fixed n_rounds per episode, no early termination):
  1. B proposes a split of `pie` points:      OFFER: me=<x> you=<y>
  2. A replies:                               ACCEPT | COUNTER: me=<y> you=<x>
  3. If A countered, B replies:               ACCEPT | REJECT
     (REJECT -> both get 0 this round)
Running totals are injected into both players' round headers.

After step 2 of every round -- i.e. right after A has acted, with A's own
reply as the final message in A's context -- we run one capture pass over A's
context and keep the last-token residual at ALL layers. That per-(episode,
round) tensor is the raw material for every probe in the study.

Parsing and fallbacks: replies are regex-parsed; on failure we retry once with
a terse format reminder, then fall back to scripted behavior and set a
`fallback` flag on the turn (probes can filter fallback-heavy episodes; a high
fallback rate on a real instruct model is a red flag for the run). B's
scripted fallback offer is drawn from an alpha-dependent noisy distribution --
deliberately, so the SMOKE preset (whose stub model can't follow formats)
still produces a learnable alpha signal to sanity-check the pipeline. On real
runs fallbacks should be rare enough (<~2%) that this channel is negligible;
`episodes.py` prints the rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from config import Config
from modeling import Steering, capture_last_token, generate
from prompts import (FORMAT_A_REPLY, FORMAT_B_OFFER, FORMAT_B_VERDICT,
                     VERBALIZE_QUESTION, b_private_reminder, round_header,
                     system_a, system_b_tier1, system_b_tier2)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_OFFER_RE = re.compile(r"(?:OFFER|COUNTER)\s*:\s*me\s*=\s*(\d+)\s*,?\s*you\s*=\s*(\d+)",
                       re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\bACCEPT\b", re.IGNORECASE)
_REJECT_RE = re.compile(r"\bREJECT\b", re.IGNORECASE)


def parse_split(text: str, pie: int) -> Optional[int]:
    """Return the speaker's own share ("me") if a valid split is present."""
    m = _OFFER_RE.search(text)
    if m:
        me, you = int(m.group(1)), int(m.group(2))
        if me + you == pie and 0 <= me <= pie:
            return me
    # lenient fallback: first two integers in the text that sum to the pie
    ints = [int(s) for s in re.findall(r"\d+", text)]
    for i in range(len(ints) - 1):
        if ints[i] + ints[i + 1] == pie and 0 <= ints[i] <= pie:
            return ints[i]
    return None


def parse_a_reply(text: str, pie: int):
    """Return ("accept", None) or ("counter", a_share) or (None, None)."""
    counter = parse_split(text, pie)
    accepted = _ACCEPT_RE.search(text) is not None
    if counter is not None and not accepted:
        return "counter", counter
    if accepted:
        return "accept", None
    return None, None


def parse_b_verdict(text: str) -> Optional[str]:
    acc, rej = _ACCEPT_RE.search(text), _REJECT_RE.search(text)
    if acc and not rej:
        return "accept"
    if rej and not acc:
        return "reject"
    return None


# ---------------------------------------------------------------------------
# Scripted fallbacks (see module docstring)
# ---------------------------------------------------------------------------
def fallback_b_offer(alpha: float, pie: int, rng: np.random.Generator) -> int:
    mean = pie / 2 + (alpha - 0.5) * 0.8 * pie
    return int(np.clip(round(rng.normal(mean, 0.12 * pie)), 1, pie - 1))


def fallback_a_reply(b_keep: int, pie: int, rng: np.random.Generator):
    if pie - b_keep >= 0.4 * pie:
        return "accept", None
    return "counter", int(np.clip(round(rng.normal(0.55 * pie, 0.05 * pie)),
                                  1, pie - 1))


def fallback_b_verdict(a_share: int, alpha: float, pie: int,
                       rng: np.random.Generator) -> str:
    # more greed -> demands more from a counter before accepting
    threshold = 0.25 * pie + alpha * 0.4 * pie
    return "accept" if pie - a_share >= threshold + rng.normal(0, 0.05 * pie) \
        else "reject"


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    rnd: int
    b_offer_keep: int              # B's own share in B's opening offer
    b_offer_text: str
    a_action: str                  # "accept" | "counter"
    a_counter_share: Optional[int]  # A's own share if countered
    a_text: str
    b_verdict: Optional[str]       # "accept" | "reject" | None (A accepted)
    b_verdict_text: str
    a_points: int
    b_points: int
    fallbacks: int                 # how many of the moves this round were scripted


@dataclass
class Episode:
    episode: int
    alpha: float
    tier: int
    turns: List[Turn] = field(default_factory=list)
    a_total: int = 0
    b_total: int = 0
    verbalized: List[dict] = field(default_factory=list)  # {round, text, guess}
    # acts[t] : fp16 [n_layers+1, d], A's last-token residual after round t+1
    acts: List[np.ndarray] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "episode": self.episode,
            "alpha": self.alpha,
            "tier": self.tier,
            "a_total": self.a_total,
            "b_total": self.b_total,
            "verbalized": self.verbalized,
            "turns": [vars(t) for t in self.turns],
        }


def _gen_with_retry(model, tok, messages, cfg, parse_fn, reminder, steer=None):
    """Generate; if parse_fn fails, retry once with a format reminder appended
    to the last user message. Returns (text, parsed, n_fallbacks_incurred)."""
    text = generate(model, tok, messages, cfg, steer=steer)
    parsed = parse_fn(text)
    if _parse_ok(parsed):
        return text, parsed, 0
    retry_messages = [dict(m) for m in messages]
    retry_messages[-1]["content"] += f"\n(Format reminder: {reminder})"
    text2 = generate(model, tok, retry_messages, cfg, steer=steer)
    parsed2 = parse_fn(text2)
    if _parse_ok(parsed2):
        return text2, parsed2, 0
    return text, parsed, 1


def _parse_ok(parsed) -> bool:
    if isinstance(parsed, tuple):
        return parsed[0] is not None
    return parsed is not None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def play_episode(model, tok, cfg: Config, episode_idx: int, alpha: float,
                 steer_vecs: Optional[np.ndarray] = None,
                 a_steer: Optional[Steering] = None,
                 capture: bool = True,
                 verbalize: bool = True) -> Episode:
    """Play one full episode.

    steer_vecs : tier-2 greed direction [n_layers+1, d]; B's generations run
                 under Steering(coef = steer_scale * alpha). None for tier 1.
    a_steer    : optional Steering applied to A's generations (the causal
                 step injects the probe direction here). Capture passes are
                 NEVER steered -- probes must read A's unmanipulated stream...
                 except that for the causal experiment we only use behavior,
                 and capture is typically off.
    """
    rng = np.random.default_rng(cfg.seed * 1_000_003 + episode_idx)
    torch.manual_seed(cfg.seed * 1_000_003 + episode_idx)

    pie, n_rounds = cfg.pie, cfg.n_rounds
    if cfg.tier == 1:
        b_system = system_b_tier1(alpha, pie, n_rounds)
    else:
        assert steer_vecs is not None, "tier 2 requires steer_vecs"
        b_system = system_b_tier2(pie, n_rounds)

    def b_steer_for_round():
        """Tier 2: B's effective greed this round is a noisy draw around
        alpha (see Config.steer_noise_sd) -- the latent sets the distribution
        of B's behavior, not any single offer."""
        if cfg.tier != 2:
            return None
        coef = cfg.steer_coef(alpha) + rng.normal(0.0, cfg.steer_noise_sd)
        coef = float(np.clip(coef, -cfg.steer_coef_max, cfg.steer_coef_max))
        return Steering(model, steer_vecs, cfg.steer_layers, coef=coef)

    a_msgs = [{"role": "system", "content": system_a(pie, n_rounds)}]
    b_msgs = [{"role": "system", "content": b_system}]

    ep = Episode(episode=episode_idx, alpha=alpha, tier=cfg.tier)

    for rnd in range(1, n_rounds + 1):
        n_fb = 0
        b_steer = b_steer_for_round()   # fresh noisy draw each round (tier 2)

        # ---- 1. B proposes ------------------------------------------------
        reminder = b_private_reminder(alpha) if cfg.tier == 1 else ""
        b_msgs.append({"role": "user", "content":
                       round_header(rnd, n_rounds, ep.b_total, ep.a_total)
                       + reminder + " Make your offer."})
        b_text, b_keep, fb = _gen_with_retry(
            model, tok, b_msgs, cfg, lambda t: parse_split(t, pie),
            FORMAT_B_OFFER.format(pie=pie), steer=b_steer)
        n_fb += fb
        if b_keep is None:
            b_keep = fallback_b_offer(alpha, pie, rng)
            b_text = f"OFFER: me={b_keep} you={pie - b_keep}"
        b_msgs.append({"role": "assistant", "content": b_text})

        # ---- 2. A accepts or counters --------------------------------------
        a_msgs.append({"role": "user", "content":
                       round_header(rnd, n_rounds, ep.a_total, ep.b_total)
                       + f" Opponent says: {b_text}"})
        a_text, (a_action, a_share), fb = _gen_with_retry(
            model, tok, a_msgs, cfg, lambda t: parse_a_reply(t, pie),
            FORMAT_A_REPLY.format(pie=pie), steer=a_steer)
        n_fb += fb
        if a_action is None:
            a_action, a_share = fallback_a_reply(b_keep, pie, rng)
            a_text = "ACCEPT" if a_action == "accept" \
                else f"COUNTER: me={a_share} you={pie - a_share}"
        a_msgs.append({"role": "assistant", "content": a_text})

        # ---- capture: A's residual stream right after A's action -----------
        if capture:
            ep.acts.append(capture_last_token(model, tok, a_msgs))

        # ---- verbalized-guess fork (side branch, never appended back) ------
        if verbalize and rnd in cfg.verbalize_rounds:
            fork = a_msgs + [{"role": "user", "content": VERBALIZE_QUESTION}]
            v_text = generate(model, tok, fork, cfg, max_new_tokens=8)
            m = re.search(r"\d+", v_text)
            guess = min(100, int(m.group())) / 100.0 if m else None
            ep.verbalized.append({"round": rnd, "text": v_text, "guess": guess})

        # ---- 3. B's verdict on a counter ------------------------------------
        b_verdict, b_verdict_text = None, ""
        if a_action == "accept":
            a_pts, b_pts = pie - b_keep, b_keep
            b_msgs.append({"role": "user",
                           "content": "Opponent says: ACCEPT"})
            b_msgs.append({"role": "assistant", "content": "Noted."})
        else:
            b_msgs.append({"role": "user", "content":
                           f"Opponent says: {a_text}\nDo you accept their "
                           f"counter (you would get {pie - a_share})?"})
            b_verdict_text, b_verdict, fb = _gen_with_retry(
                model, tok, b_msgs, cfg, parse_b_verdict,
                FORMAT_B_VERDICT, steer=b_steer)
            n_fb += fb
            if b_verdict is None:
                b_verdict = fallback_b_verdict(a_share, alpha, pie, rng)
                b_verdict_text = b_verdict.upper()
            b_msgs.append({"role": "assistant", "content": b_verdict_text})
            if b_verdict == "accept":
                a_pts, b_pts = a_share, pie - a_share
            else:
                a_pts, b_pts = 0, 0
            # tell A the outcome so its next-round header totals make sense
            a_msgs.append({"role": "user", "content":
                           f"Opponent replied: {b_verdict_text}"})
            a_msgs.append({"role": "assistant", "content": "Noted."})

        ep.a_total += a_pts
        ep.b_total += b_pts
        ep.turns.append(Turn(
            rnd=rnd, b_offer_keep=b_keep, b_offer_text=b_text,
            a_action=a_action, a_counter_share=a_share, a_text=a_text,
            b_verdict=b_verdict, b_verdict_text=b_verdict_text,
            a_points=a_pts, b_points=b_pts, fallbacks=n_fb,
        ))

    return ep


# ---------------------------------------------------------------------------
# Transcript rendering (single canonical form, used by the shadow observer
# and the text-only baselines so all controls read the SAME surface text)
# ---------------------------------------------------------------------------
def render_transcript(turns: List[dict], upto_round: int, pie: int) -> str:
    lines = []
    a_tot = b_tot = 0
    for t in turns[:upto_round]:
        lines.append(f"Round {t['rnd']}:")
        lines.append(f"  B: {t['b_offer_text']}")
        lines.append(f"  A: {t['a_text']}")
        if t["b_verdict"] is not None:
            lines.append(f"  B: {t['b_verdict_text']}")
        a_tot += t["a_points"]
        b_tot += t["b_points"]
        lines.append(f"  [Totals: A={a_tot}, B={b_tot}]")
    return "\n".join(lines)
