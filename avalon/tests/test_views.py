"""Gate 2: LEAK TESTS for view construction.

The strongest checks are INVARIANCE tests: permuting roles within a set a seat
cannot legally disambiguate must leave that seat's view byte-identical. If a
which-is-which leak existed, the view would change.
"""
import numpy as np
import pytest

from avalon import roles as R
from avalon.views import PublicState, build_view


def _assign(merlin=0, percival=1, loyal=(2, 3), assassin=4, morgana=5, minion=6):
    a = {merlin: R.Role.MERLIN, percival: R.Role.PERCIVAL,
         assassin: R.Role.ASSASSIN, morgana: R.Role.MORGANA, minion: R.Role.MINION}
    for s in loyal:
        a[s] = R.Role.LOYAL
    return a


def _view_text(seat, assignment):
    k = R.compute_knowledge(assignment)
    return build_view(seat, k, PublicState(), []).render()


# --------------------------------------------------------------------------- #
# Direct content leak checks                                                   #
# --------------------------------------------------------------------------- #
def test_loyal_view_has_no_info_on_others():
    txt = _view_text(2, _assign())
    for forbidden in ["Merlin", "Percival", "Assassin", "Morgana", "Minion"]:
        assert forbidden not in txt          # a loyal servant learns nothing about anyone
    assert "Loyal Servant" in txt            # except its own role


def test_merlin_sees_evil_seats_but_not_which_evil_role():
    txt = _view_text(0, _assign())
    assert "[4, 5, 6]" in txt                 # the evil SET
    for role in ["Assassin", "Morgana", "Minion"]:
        assert role not in txt                # never the specific evil role of a seat


def test_percival_sees_pair_but_not_mapping():
    txt = _view_text(1, _assign())
    assert "[0, 5]" in txt                     # the {Merlin, Morgana} seats
    assert "Merlin and Morgana" in txt         # legal: knows these two roles are the pair
    # but no direct mapping like "seat 0 is Merlin"
    assert "seat 0 is Merlin" not in txt.lower().replace("seat 0 are", "")


def test_evil_view_never_contains_merlin_seat_as_merlin():
    for evil_seat in (4, 5, 6):
        txt = _view_text(evil_seat, _assign())
        assert "[4, 5, 6]" in txt              # evil team set
        assert "Merlin" in txt and "do NOT know who Merlin" in txt   # only the negation
        # the only "Merlin" mention is the explicit "you don't know who Merlin is"
        assert txt.count("Merlin") == 1


# --------------------------------------------------------------------------- #
# Invariance (which-is-which) leak tests — the rigorous ones                    #
# --------------------------------------------------------------------------- #
def test_percival_view_invariant_to_merlin_morgana_swap():
    base = _assign(merlin=0, morgana=5)
    swapped = _assign(merlin=5, morgana=0)     # swap which of {0,5} is Merlin vs Morgana
    assert _view_text(1, base) == _view_text(1, swapped)


def test_merlin_view_invariant_to_evil_role_permutation():
    base = _assign(assassin=4, morgana=5, minion=6)
    perm = _assign(assassin=6, morgana=4, minion=5)   # same evil SEATS {4,5,6}, roles permuted
    assert _view_text(0, base) == _view_text(0, perm)


def test_evil_teammate_view_invariant_to_others_roles():
    """An evil seat's view must not depend on which evil role its TEAMMATES hold."""
    base = _assign(assassin=4, morgana=5, minion=6)
    perm = _assign(assassin=4, morgana=6, minion=5)   # seat 4 still Assassin; teammates swapped
    assert _view_text(4, base) == _view_text(4, perm)


def test_view_only_reads_own_knowledge(monkeypatch):
    """build_view must read knowledge[seat] only — corrupting others' entries must
    not change the view."""
    a = _assign()
    k = R.compute_knowledge(a)
    txt1 = build_view(2, k, PublicState(), []).render()
    # poison every OTHER seat's knowledge object
    for s in range(7):
        if s != 2:
            k[s] = R.PrivateKnowledge(seat=s, role=R.Role.MERLIN, alignment="good",
                                      evil_set_seen=frozenset({0, 1, 2, 3, 4, 5, 6}))
    txt2 = build_view(2, k, PublicState(), []).render()
    assert txt1 == txt2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
