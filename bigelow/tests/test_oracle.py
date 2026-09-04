"""Oracle exactness, duplicate handling, and balance (Part 24: 1, 2, 4)."""

from __future__ import annotations

import math

from belief_feedback.world.generator import build_ordinary_world
from belief_feedback.world.oracle import blind_oracle_for_reports, oracle_for_reports
from belief_feedback.world.schema import UPSTREAM


def test_exact_posterior(world):
    agent_reports = world.assignments[0]
    res = oracle_for_reports(world, agent_reports)
    expected = sum(
        world.event(eid).llr for eid in {world.report(r).event_id for r in agent_reports}
    )
    assert math.isclose(res.oracle_log_odds, expected, rel_tol=1e-12)
    assert math.isclose(
        res.oracle_probability_upstream, 1 / (1 + math.exp(-expected)), rel_tol=1e-12
    )


def test_duplicates_counted_once_by_aware_oracle(recycling_pair):
    _, recycled = recycling_pair
    focal = [r.report_id for r in recycled.reports if "-F" in r.report_id]
    assert len(focal) == 3
    one = oracle_for_reports(recycled, focal[:1])
    three = oracle_for_reports(recycled, focal)
    assert math.isclose(one.oracle_log_odds, three.oracle_log_odds, rel_tol=1e-12)
    assert three.repeated_report_count == 2
    assert three.unique_event_count == 1


def test_duplicates_do_alter_blind_oracle(recycling_pair):
    _, recycled = recycling_pair
    focal = [r.report_id for r in recycled.reports if "-F" in r.report_id]
    blind = blind_oracle_for_reports(recycled, focal)
    aware = oracle_for_reports(recycled, focal)
    assert math.isclose(blind.oracle_log_odds, 3 * aware.oracle_log_odds, rel_tol=1e-9)


def test_label_counterbalancing_preserves_semantic_log_odds(cfg):
    """Two worlds identical except for label mapping give equal semantic ell."""
    w = build_ordinary_world(cfg, "w_cb_0000", "exogenous_train", 0)
    ell = oracle_for_reports(w, w.assignments[0]).oracle_log_odds
    flipped = w.model_copy(deep=True)
    flipped.alpha_is_upstream = not w.alpha_is_upstream
    ell_flipped = oracle_for_reports(flipped, flipped.assignments[0]).oracle_log_odds
    assert math.isclose(ell, ell_flipped)  # semantic quantities never move
    # but the visible representation flips
    assert math.isclose(w.semantic_to_visible(ell), -flipped.semantic_to_visible(ell))


def test_truth_balance_within_one(cfg):
    worlds = [build_ordinary_world(cfg, f"w_bal_{i:04d}", "exogenous_train", i) for i in range(6)]
    n_up = sum(w.true_hypothesis == UPSTREAM for w in worlds)
    assert abs(2 * n_up - len(worlds)) <= 1
    n_alpha = sum(w.alpha_is_upstream for w in worlds)
    assert abs(2 * n_alpha - len(worlds)) <= 1
