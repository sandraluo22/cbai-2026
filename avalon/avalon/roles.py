"""Roles, alignment, role assignment, and the LEAK-CRITICAL setup reveal.

7-player Avalon (no night phase, no eliminations).
  GOOD: Merlin, Percival, Loyal Servant x2
  EVIL: Assassin, Morgana, Minion of Mordred

The reveal is the single most important piece for research validity: each seat's
private knowledge must contain ONLY what the role legally permits, and sets that
are "indistinguishable" must be stored UNORDERED with no which-is-which label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Role(str, Enum):
    MERLIN = "Merlin"
    PERCIVAL = "Percival"
    LOYAL = "Loyal Servant of Arthur"
    ASSASSIN = "Assassin"
    MORGANA = "Morgana"
    MINION = "Minion of Mordred"


GOOD_ROLES = (Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL)
EVIL_ROLES = (Role.ASSASSIN, Role.MORGANA, Role.MINION)
ALL_ROLES = GOOD_ROLES + EVIL_ROLES                       # exactly 7


def alignment(role: Role) -> str:
    return "evil" if role in (Role.ASSASSIN, Role.MORGANA, Role.MINION) else "good"


# --------------------------------------------------------------------------- #
# Assignment                                                                   #
# --------------------------------------------------------------------------- #
def assign_roles(n_players: int, rng: np.random.Generator) -> dict[int, Role]:
    """Shuffle the 7 fixed roles onto seats 0..6. Seeded for reproducibility."""
    assert n_players == 7, "this pilot is fixed at 7 players"
    roles = list(ALL_ROLES)
    perm = rng.permutation(n_players)
    return {int(seat): roles[i] for i, seat in enumerate(perm)}


def seats_with_role(assignment: dict[int, Role], role: Role) -> list[int]:
    return sorted(s for s, r in assignment.items() if r == role)


def evil_seats(assignment: dict[int, Role]) -> frozenset[int]:
    return frozenset(s for s, r in assignment.items() if alignment(r) == "evil")


# --------------------------------------------------------------------------- #
# Private knowledge (the reveal)                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrivateKnowledge:
    """Exactly what a seat legally knows after setup. UNLABELED sets only — the
    fields below never encode which-is-which for indistinguishable pairs/sets."""

    seat: int
    role: Role
    alignment: str
    # Merlin: the set of all evil seats (NOT which is which; no Mordred/Oberon here)
    evil_set_seen: Optional[frozenset[int]] = None
    # Evil: the full evil team (includes self) — all evil know each other
    evil_team: Optional[frozenset[int]] = None
    # Percival: the {Merlin, Morgana} pair as an unordered set (NOT which is which)
    merlin_morgana_pair: Optional[frozenset[int]] = None

    def as_public_safe_dict(self) -> dict:
        """Serialization that preserves the unordered (sorted-list) sets."""
        d = {"seat": self.seat, "role": self.role.value, "alignment": self.alignment}
        if self.evil_set_seen is not None:
            d["evil_set_seen"] = sorted(self.evil_set_seen)
        if self.evil_team is not None:
            d["evil_team"] = sorted(self.evil_team)
        if self.merlin_morgana_pair is not None:
            d["merlin_morgana_pair"] = sorted(self.merlin_morgana_pair)
        return d


def compute_knowledge(assignment: dict[int, Role]) -> dict[int, PrivateKnowledge]:
    """THE reveal function: given the full assignment, return each seat's LEGAL
    private knowledge. This is the only place setup knowledge is derived."""
    evils = evil_seats(assignment)
    merlin_seat = seats_with_role(assignment, Role.MERLIN)[0]
    morgana_seat = seats_with_role(assignment, Role.MORGANA)[0]

    out: dict[int, PrivateKnowledge] = {}
    for seat, role in assignment.items():
        align = alignment(role)
        evil_set_seen = merlin_morgana = team = None
        if role == Role.MERLIN:
            evil_set_seen = frozenset(evils)               # sees evils as a SET only
        elif role == Role.PERCIVAL:
            merlin_morgana = frozenset({merlin_seat, morgana_seat})  # unordered pair
        elif align == "evil":
            team = frozenset(evils)                        # full evil team (incl. self)
        # Loyal Servants: nothing beyond own role
        out[seat] = PrivateKnowledge(
            seat=seat, role=role, alignment=align,
            evil_set_seen=evil_set_seen, evil_team=team, merlin_morgana_pair=merlin_morgana)
    return out
