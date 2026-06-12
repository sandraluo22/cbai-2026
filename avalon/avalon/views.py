"""View construction — the ONLY place a player's prompt context is assembled.

A view = the seat's LEGAL private knowledge + the current PUBLIC game state + the
PUBLIC transcript so far. It must NEVER contain another seat's role or any hidden
info the seat isn't entitled to, and indistinguishable sets must be rendered
UNORDERED (no which-is-which). The leak tests in tests/test_views.py assert this,
including the strong invariance property: permuting roles WITHIN a set the seat
can't disambiguate must leave that seat's view byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .roles import PrivateKnowledge, Role


# --------------------------------------------------------------------------- #
# Public state + public transcript (NO hidden info)                            #
# --------------------------------------------------------------------------- #
@dataclass
class PublicEvent:
    """One public, append-only transcript entry (a statement or an explicit pass)."""
    round_idx: int
    seat: int
    passed: bool
    text: str            # "" if passed


@dataclass
class PublicState:
    mission_idx: int = 0
    successes: int = 0
    fails: int = 0
    leader: int = 0
    proposed_team: Optional[list[int]] = None
    rejected_count: int = 0                       # consecutive rejects this mission round
    mission_results: list[dict] = field(default_factory=list)   # [{mission_idx, fail_count, failed}]
    public_votes: list[dict] = field(default_factory=list)      # [{proposal, votes: {seat:approve/reject}}]
    phase: str = "proposal"


# --------------------------------------------------------------------------- #
# Private-knowledge rendering (UNORDERED, no which-is-which)                    #
# --------------------------------------------------------------------------- #
def render_private_knowledge(pk: PrivateKnowledge) -> str:
    lines = [f"You are seat {pk.seat}. Your secret role is {pk.role.value} "
             f"({pk.alignment.upper()})."]
    if pk.evil_set_seen is not None:                      # Merlin
        lines.append(f"You secretly perceive that the EVIL players occupy seats "
                     f"{sorted(pk.evil_set_seen)}. You do NOT know which evil role each holds.")
    if pk.evil_team is not None:                          # any evil seat
        lines.append(f"Your EVIL team occupies seats {sorted(pk.evil_team)} (this includes you). "
                     f"You do NOT know who Merlin is.")
    if pk.merlin_morgana_pair is not None:                # Percival
        lines.append(f"You secretly perceive that seats {sorted(pk.merlin_morgana_pair)} are "
                     f"Merlin and Morgana, in some unknown order. You do NOT know which is which.")
    if pk.role == Role.LOYAL:
        lines.append("You have no special knowledge about any other player.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public state + transcript rendering                                          #
# --------------------------------------------------------------------------- #
def render_public_state(ps: PublicState) -> str:
    parts = [
        f"Mission {ps.mission_idx + 1} of 5. Score: GOOD {ps.successes} / EVIL {ps.fails} "
        f"(3 to win). Current leader: seat {ps.leader}.",
    ]
    if ps.proposed_team is not None:
        parts.append(f"Proposed team: seats {sorted(ps.proposed_team)}.")
    if ps.rejected_count:
        parts.append(f"Consecutive rejected proposals this round: {ps.rejected_count} "
                     f"(5 rejections => Evil wins).")
    for m in ps.mission_results:
        parts.append(f"  Mission {m['mission_idx'] + 1}: {'FAILED' if m['failed'] else 'succeeded'} "
                     f"({m['fail_count']} fail card(s) revealed; identities hidden).")
    return "\n".join(parts)


def render_transcript(events: list[PublicEvent]) -> str:
    if not events:
        return "(no discussion yet)"
    out = []
    for e in events:
        if e.passed:
            out.append(f"[round {e.round_idx + 1}] seat {e.seat}: (passed)")
        else:
            out.append(f"[round {e.round_idx + 1}] seat {e.seat}: {e.text}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# THE view                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class PlayerView:
    seat: int
    private_text: str
    public_text: str
    transcript_text: str

    def render(self) -> str:
        return (f"=== YOUR SECRET KNOWLEDGE ===\n{self.private_text}\n\n"
                f"=== PUBLIC GAME STATE ===\n{self.public_text}\n\n"
                f"=== PUBLIC DISCUSSION SO FAR ===\n{self.transcript_text}")


def build_view(seat: int, knowledge: dict[int, PrivateKnowledge],
               public_state: PublicState, transcript: list[PublicEvent]) -> PlayerView:
    """Build seat's legal view FROM SCRATCH. The ONLY entry point for prompt context.
    `knowledge` is the full per-seat knowledge map but we read ONLY knowledge[seat]."""
    pk = knowledge[seat]                                  # <-- only this seat's knowledge is read
    return PlayerView(
        seat=seat,
        private_text=render_private_knowledge(pk),
        public_text=render_public_state(public_state),
        transcript_text=render_transcript(transcript),
    )
