"""Unit tests for the BHW Bayesian solver against known closed-form predictions."""
import math
import pytest

from urn import (signal_llr, infer_history, cascade_onset, rational_decision,
                 naive_decision, RED, BLUE)

Q = 2.0 / 3.0


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# --- single-signal posterior --------------------------------------------- #
def test_single_signal_posterior():
    # empty history, own red -> P(red)=q
    r = rational_decision([], RED, Q)
    assert approx(r["posterior_red"], Q)
    assert r["choice"] == RED and not r["in_cascade"] and r["follows_private"]
    r = rational_decision([], BLUE, Q)
    assert approx(r["posterior_red"], 1 - Q) and r["choice"] == BLUE


# --- the classic two-in-a-row cascade ------------------------------------ #
def test_two_same_triggers_cascade_at_position_3():
    assert cascade_onset([RED, RED]) == 3
    assert cascade_onset([BLUE, BLUE]) == 3
    # focal at position 3 with OPPOSING signal still herds (cascade)
    r = rational_decision([RED, RED], BLUE, Q)
    assert r["in_cascade"] and r["choice"] == RED and not r["follows_private"]
    assert r["public_count"] == 2
    # posterior still incorporates own signal: t = 2 + (-1) = 1 -> sigma(L) = q
    assert approx(r["posterior_red"], Q)


def test_no_cascade_before_threshold():
    # one prior red, own blue -> indifferent, follows own signal, NOT cascade
    r = rational_decision([RED], BLUE, Q)
    assert not r["in_cascade"] and r["choice"] == BLUE
    assert approx(r["posterior_red"], 0.5) and r["total_units"] == 0
    # mixed [R,B] dissolves the count -> no cascade, next agent free
    assert cascade_onset([RED, BLUE]) is None
    assert infer_history([RED, BLUE])["public_count"] == 0


# --- cascade threshold is q-INVARIANT ------------------------------------ #
@pytest.mark.parametrize("q", [0.55, 0.60, 2 / 3, 0.75, 0.9])
def test_cascade_threshold_independent_of_q(q):
    # onset depends only on the count, not q
    assert cascade_onset([RED, RED]) == 3
    r = rational_decision([RED, RED], BLUE, q)
    assert r["in_cascade"] and r["choice"] == RED
    # but the posterior magnitude DOES scale with q
    assert approx(r["posterior_red"], 1 / (1 + math.exp(-1 * signal_llr(q))))


# --- rational caps evidence; naive does not ------------------------------ #
def test_rational_caps_but_naive_counts():
    hist = [RED] * 5
    info = infer_history(hist)
    assert info["public_count"] == 2                      # capped at threshold
    assert info["informative"][:2] == [True, True]
    assert all(not x for x in info["informative"][2:])    # cascaders uninformative
    assert info["in_cascade"][2:] == [True, True, True]
    naive = naive_decision(hist, BLUE, Q)
    assert naive["public_count"] == 5                      # counts them all
    # rational posterior (t=1) much weaker than naive (t=4)
    rat = rational_decision(hist, BLUE, Q)
    assert rat["posterior_red"] < naive["posterior_red"]
    assert approx(rat["posterior_red"], 1 / (1 + math.exp(-1 * signal_llr(Q))))
    assert approx(naive["posterior_red"], 1 / (1 + math.exp(-4 * signal_llr(Q))))


# --- wrong cascade: rational herds into the error ------------------------ #
def test_wrong_cascade_rational_herds():
    # true state BLUE, but first two drew red -> red cascade; focal has correct blue
    r = rational_decision([RED, RED], BLUE, Q)
    assert r["choice"] == RED            # rational follows the (wrong) cascade
    assert not r["follows_private"]      # breaking it would be NON-Bayesian (the human move)


# --- offpath detection --------------------------------------------------- #
def test_offpath_flagged():
    # an agent choosing against an established cascade is impossible under rational play
    info = infer_history([RED, RED, BLUE])   # 3rd agent in red cascade chose blue
    assert info["in_cascade"][2] and info["offpath"][2]
    assert info["public_count"] == 2          # off-path choice does not update belief


# --- onset for longer / blue runs ---------------------------------------- #
def test_onset_variants():
    assert cascade_onset([RED]) is None
    assert cascade_onset([RED, RED, RED]) == 3
    assert cascade_onset([BLUE, BLUE, BLUE, BLUE]) == 3
    assert cascade_onset([RED, BLUE, RED]) is None        # count oscillates, never hits 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
