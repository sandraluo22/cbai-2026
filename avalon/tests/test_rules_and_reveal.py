"""Gate 1: rules + the leak-critical reveal (no LLMs)."""
import numpy as np
import pytest

from avalon import roles as R
from avalon import rules as RU


# --------------------------------------------------------------------------- #
# Assignment                                                                   #
# --------------------------------------------------------------------------- #
def test_assignment_is_4_good_3_evil():
    a = R.assign_roles(7, np.random.default_rng(0))
    assert len(a) == 7
    aligns = [R.alignment(r) for r in a.values()]
    assert aligns.count("good") == 4 and aligns.count("evil") == 3
    # exactly the canonical role multiset
    assert sorted(r.value for r in a.values()) == sorted(r.value for r in R.ALL_ROLES)


def test_assignment_reproducible():
    a1 = R.assign_roles(7, np.random.default_rng(42))
    a2 = R.assign_roles(7, np.random.default_rng(42))
    assert a1 == a2


# --------------------------------------------------------------------------- #
# Reveal correctness                                                           #
# --------------------------------------------------------------------------- #
def _fixed_assignment():
    # seats: 0 Merlin,1 Percival,2 Loyal,3 Loyal,4 Assassin,5 Morgana,6 Minion
    return {0: R.Role.MERLIN, 1: R.Role.PERCIVAL, 2: R.Role.LOYAL, 3: R.Role.LOYAL,
            4: R.Role.ASSASSIN, 5: R.Role.MORGANA, 6: R.Role.MINION}


def test_merlin_sees_evil_set_unlabeled():
    k = R.compute_knowledge(_fixed_assignment())
    m = k[0]
    assert m.evil_set_seen == frozenset({4, 5, 6})       # all evil, as a SET
    assert m.evil_team is None and m.merlin_morgana_pair is None
    # no which-is-which: it's a frozenset, carries no role labels
    assert not hasattr(m.evil_set_seen, "items")


def test_all_evil_know_full_evil_team_not_merlin():
    k = R.compute_knowledge(_fixed_assignment())
    for s in (4, 5, 6):
        assert k[s].evil_team == frozenset({4, 5, 6})    # includes self + others
        assert k[s].evil_set_seen is None                # not Merlin's knowledge
        # evil never learns Merlin's seat anywhere in their knowledge
        assert 0 not in (k[s].evil_team or set())


def test_percival_sees_merlin_morgana_pair_unlabeled():
    k = R.compute_knowledge(_fixed_assignment())
    p = k[1]
    assert p.merlin_morgana_pair == frozenset({0, 5})    # {Merlin, Morgana}, unordered
    assert p.evil_set_seen is None and p.evil_team is None


def test_loyal_servants_know_nothing_extra():
    k = R.compute_knowledge(_fixed_assignment())
    for s in (2, 3):
        assert k[s].evil_set_seen is None
        assert k[s].evil_team is None
        assert k[s].merlin_morgana_pair is None


def test_morgana_gets_only_evil_team():
    k = R.compute_knowledge(_fixed_assignment())
    assert k[5].evil_team == frozenset({4, 5, 6})
    assert k[5].merlin_morgana_pair is None              # Morgana is not Percival


# --------------------------------------------------------------------------- #
# Mission table + rules                                                        #
# --------------------------------------------------------------------------- #
def test_mission_table():
    assert RU.TEAM_SIZES == (2, 3, 3, 4, 4)
    assert RU.FAILS_REQUIRED == (1, 1, 1, 2, 1)          # mission 4 needs 2 fails


def test_vote_strict_majority():
    approve_all = {i: "approve" for i in range(7)}
    assert RU.tally_votes(approve_all).approved
    # 3 approve, 4 reject -> rejected; tie impossible with 7 but check 3/4
    mixed = {0: "approve", 1: "approve", 2: "approve", 3: "reject", 4: "reject", 5: "reject", 6: "reject"}
    assert not RU.tally_votes(mixed).approved
    # 4 approve, 3 reject -> approved
    four = {0: "approve", 1: "approve", 2: "approve", 3: "approve", 4: "reject", 5: "reject", 6: "reject"}
    assert RU.tally_votes(four).approved


def test_hammer():
    assert not RU.hammer_triggered(4)
    assert RU.hammer_triggered(5)


def test_good_play_success_enforced():
    card, ov = RU.enforce_good_play_success("fail", R.Role.LOYAL)
    assert card == "success" and ov is True
    card, ov = RU.enforce_good_play_success("fail", R.Role.ASSASSIN)
    assert card == "fail" and ov is False                 # evil may fail


def test_mission4_needs_two_fails():
    # mission index 3: one fail -> still success
    assert not RU.resolve_mission(3, {0: "fail", 1: "success", 2: "success", 3: "success"}).failed
    assert RU.resolve_mission(3, {0: "fail", 1: "fail", 2: "success", 3: "success"}).failed
    # mission index 0: one fail -> fails
    assert RU.resolve_mission(0, {0: "fail", 1: "success"}).failed


def test_win_conditions_and_assassin():
    assert RU.mission_winner(2, 1) is None
    assert RU.mission_winner(3, 1) == "good_provisional"
    assert RU.mission_winner(1, 3) == "evil"
    assert RU.assassin_outcome(0, 0) == "evil"            # correct Merlin guess
    assert RU.assassin_outcome(2, 0) == "good"            # wrong guess


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
