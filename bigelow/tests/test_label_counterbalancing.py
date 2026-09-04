"""ALPHA/BETA counterbalancing and template holdout (Part 24: 4, 5)."""

from __future__ import annotations

from belief_feedback.config import TEST_SPLITS
from belief_feedback.world.generator import build_ordinary_world
from belief_feedback.world.schema import LOCAL, UPSTREAM
from belief_feedback.world.templates import HELD_OUT_VARIANT


def test_visible_label_mapping(world):
    up_label = world.visible_label(UPSTREAM)
    local_label = world.visible_label(LOCAL)
    assert {up_label, local_label} == {"ALPHA", "BETA"}
    assert world.semantic_hypothesis(up_label) == UPSTREAM
    assert world.semantic_hypothesis(local_label) == LOCAL


def test_semantic_visible_roundtrip(world):
    for ell in (-2.5, 0.0, 1.75):
        assert world.visible_to_semantic(world.semantic_to_visible(ell)) == ell


def test_mapping_balanced_across_worlds(cfg):
    worlds = [build_ordinary_world(cfg, f"w_map_{i:04d}", "exogenous_train", i) for i in range(8)]
    n_alpha_up = sum(w.alpha_is_upstream for w in worlds)
    assert n_alpha_up == 4
    # mapping is crossed with truth: all four (truth, mapping) cells occur
    cells = {(w.true_hypothesis, w.alpha_is_upstream) for w in worlds}
    assert len(cells) == 4


def test_heldout_variants_only_in_test_splits(cfg):
    train_world = build_ordinary_world(cfg, "w_hv_tr_0000", "exogenous_train", 0)
    test_world = build_ordinary_world(cfg, "w_hv_te_0000", "endogenous_test", 0)
    assert "endogenous_test" in TEST_SPLITS
    assert all(r.template_variant != HELD_OUT_VARIANT for r in train_world.reports)
    assert all(r.template_variant == HELD_OUT_VARIANT for r in test_world.reports)
