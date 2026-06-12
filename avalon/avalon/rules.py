"""Pure, deterministic Avalon rules (no LLMs, no state). Auditable rule tables.

7-player standard: 5 missions, team sizes [2,3,3,4,4]; mission 4 (index 3) needs
TWO fail cards to fail, all others fail on one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .roles import Role, alignment

# Auditable mission table (mission index 0..4)
TEAM_SIZES = (2, 3, 3, 4, 4)
FAILS_REQUIRED = (1, 1, 1, 2, 1)        # mission 4 (index 3) needs 2 fails
N_MISSIONS = 5
MISSIONS_TO_WIN = 3
MAX_REJECTS_PER_ROUND = 5               # 5 consecutive rejects -> EVIL wins (hammer)


def team_size(mission_idx: int) -> int:
    return TEAM_SIZES[mission_idx]


def fails_required(mission_idx: int) -> int:
    return FAILS_REQUIRED[mission_idx]


# --------------------------------------------------------------------------- #
# Team vote                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class VoteResult:
    approves: int
    rejects: int
    approved: bool


def tally_votes(votes: dict[int, str]) -> VoteResult:
    """votes: seat -> 'approve'/'reject'. Strict majority Approve passes (>=4 of 7).
    Tie or majority Reject => rejected."""
    approves = sum(1 for v in votes.values() if v == "approve")
    rejects = len(votes) - approves
    return VoteResult(approves, rejects, approved=approves > rejects)


def hammer_triggered(consecutive_rejects: int) -> bool:
    return consecutive_rejects >= MAX_REJECTS_PER_ROUND


# --------------------------------------------------------------------------- #
# Mission resolution                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class MissionResult:
    mission_idx: int
    fail_count: int
    failed: bool


def enforce_good_play_success(card: str, role: Role) -> tuple[str, bool]:
    """GOOD players MUST play Success. If a good seat tries to Fail, override to
    Success and flag it (logged by caller). Returns (effective_card, was_overridden)."""
    if alignment(role) == "good" and card == "fail":
        return "success", True
    return card, False


def resolve_mission(mission_idx: int, cards: dict[int, str]) -> MissionResult:
    """cards: seat -> 'success'/'fail' (already good-enforced). Mission fails if
    fail_count >= fails_required for that mission."""
    fail_count = sum(1 for c in cards.values() if c == "fail")
    failed = fail_count >= fails_required(mission_idx)
    return MissionResult(mission_idx, fail_count, failed)


# --------------------------------------------------------------------------- #
# Win conditions                                                               #
# --------------------------------------------------------------------------- #
def mission_winner(successes: int, fails: int):
    """Returns 'good_provisional' (3 successes -> assassin phase), 'evil' (3 fails),
    or None (game continues)."""
    if fails >= MISSIONS_TO_WIN:
        return "evil"
    if successes >= MISSIONS_TO_WIN:
        return "good_provisional"
    return None


def assassin_outcome(guess_seat: int, merlin_seat: int) -> str:
    """Correct Merlin guess -> EVIL wins; wrong -> GOOD wins."""
    return "evil" if guess_seat == merlin_seat else "good"
