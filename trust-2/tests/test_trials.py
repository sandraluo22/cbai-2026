"""Unit tests for trial generation (no model / API)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conditions import CONDITIONS, DOSE_LEVELS
from trials import generate_trials, render_prompt


def test_determinism_same_seed():
    a = generate_trials("labels", 5, seed=0)
    b = generate_trials("labels", 5, seed=0)
    assert [t.trial_hash() for t in a] == [t.trial_hash() for t in b]


def test_different_seed_differs():
    a = generate_trials("labels", 5, seed=0)
    b = generate_trials("labels", 5, seed=1)
    assert [t.trial_hash() for t in a] != [t.trial_hash() for t in b]


def test_correctness_counts_match_plan():
    # 90%/30% over 10 claims → 9 and 3 correct.
    trials = generate_trials("labels", 4, seed=3, n_claims=10)
    for t in trials:
        for s in t.sources:
            if s.key == "high_acc":
                assert s.n_correct == 9 and s.n_claims == 10
            if s.key == "low_acc":
                assert s.n_correct == 3 and s.n_claims == 10
            if s.demonstrated_accuracy is not None:
                assert abs(s.demonstrated_accuracy - s.n_correct / s.n_claims) < 1e-9


def test_asserted_values_consistent_with_correctness():
    trials = generate_trials("order", 3, seed=5, n_claims=8)
    for t in trials:
        for c in t.claims:
            if c.correct:
                assert c.asserted_value == c.item.true_value
            else:
                assert c.asserted_value != c.item.true_value


def test_final_item_is_novel():
    # The contested entity must never appear among the early-claim entities.
    trials = generate_trials("labels", 6, seed=7)
    for t in trials:
        early = {c.item.entity for c in t.claims}
        assert t.final_entity not in early


def test_names_distinct_and_positions_counterbalanced():
    trials = generate_trials("labels", 20, seed=2)
    for t in trials:
        names = [s.name for s in t.sources]
        assert len(names) == len(set(names))            # distinct names
        positions = sorted(s.position for s in t.sources)
        assert positions == list(range(len(t.sources)))  # a permutation
    # high_acc should not always sit in position 0 across trials
    first_positions = {t.trial_id: next(s.position for s in t.sources
                                        if s.key == "high_acc") for t in trials}
    assert len(set(first_positions.values())) > 1


def test_baseline_has_no_track_record():
    trials = generate_trials("baseline", 4, seed=0)
    for t in trials:
        assert t.claims == []
        for s in t.sources:
            assert s.demonstrated_accuracy is None
            assert s.label in ("peer-reviewed lab", "anonymous forum poster")


def test_dose_sweeps_levels():
    trials = generate_trials("dose", 8, seed=0)
    seen = {t.sources[0].n_claims for t in trials}
    assert seen.issubset(set(DOSE_LEVELS))
    assert len(seen) > 1


def test_all_conditions_generate_and_render():
    for cond in CONDITIONS:
        trials = generate_trials(cond, 2, seed=1)
        assert len(trials) == 2
        for t in trials:
            system, user = render_prompt(t)
            assert "FINAL CONTESTED CLAIM" in user
            assert "trust" in user
            for s in t.sources:
                assert s.name in user
