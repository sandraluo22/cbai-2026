"""Branch routing follows the intended causal graph (Part 24: 11, 12)."""

from __future__ import annotations

import numpy as np

from belief_feedback.agents.protocol import BranchSpec, run_episode
from belief_feedback.seeds import generation_seed


def _run(cfg, backend, world, steer_ctx, spec, baseline=None):
    return run_episode(cfg, backend, world, 11, spec, steer_ctx, baseline=baseline)


def test_paired_branches_share_rng_streams():
    s1 = generation_seed("w_x", 11, 3, 2, "memo")
    s2 = generation_seed("w_x", 11, 3, 2, "memo")
    assert s1 == s2  # branch id is absent from the tuple by design


def test_unsteered_agents_reproduce_baseline_memos(cfg, backend, test_world, steer_ctx):
    base = _run(cfg, backend, test_world, steer_ctx, BranchSpec(name="baseline"))
    imp = _run(
        cfg, backend, test_world, steer_ctx,
        BranchSpec(name="positive_impulse", condition="positive_impulse",
                   branch_parent="baseline", steering={(0, 1): 1.0}),
        baseline=base,
    )
    # round 1 is generated from identical frozen contexts: only agent 0 differs
    for a in range(1, test_world.n_agents):
        assert imp.memos[(a, 1)] == base.memos[(a, 1)]


def test_one_hop_has_exactly_one_altered_source_memo(cfg, backend, test_world, steer_ctx):
    base = _run(cfg, backend, test_world, steer_ctx, BranchSpec(name="baseline"))
    oh = _run(
        cfg, backend, test_world, steer_ctx,
        BranchSpec(name="positive_one_hop", condition="positive_one_hop",
                   branch_parent="baseline", steering={(0, 1): 1.0}, one_hop_from_round=2),
        baseline=base,
    )
    altered = {
        (d["source_agent"], d["round"])
        for d in oh.delivery_rows
        if d["status"] == "live" and d["intervention_path_status"] == "altered"
    }
    assert altered == {(0, 1)}
    # from round 2 on, everything is clamped to baseline
    late = [d for d in oh.delivery_rows if d["round"] >= 2]
    assert late and all(d["status"] == "clamped" for d in late)


def test_no_return_blocks_altered_messages_into_source(cfg, backend, test_world, steer_ctx):
    base = _run(cfg, backend, test_world, steer_ctx, BranchSpec(name="baseline"))
    nr = _run(
        cfg, backend, test_world, steer_ctx,
        BranchSpec(name="positive_no_return", condition="positive_no_return",
                   branch_parent="baseline", steering={(0, 1): 1.0}, no_return_agent=0),
        baseline=base,
    )
    incoming_to_source = [
        d for d in nr.delivery_rows
        if d["recipient_agent"] == 0 and d["source_agent"] != 0 and d["round"] >= 2
    ]
    assert incoming_to_source
    assert all(d["status"] == "clamped" for d in incoming_to_source)
    # everyone else stays live
    others = [
        d for d in nr.delivery_rows
        if d["recipient_agent"] != 0 and d["round"] >= 2 and not d["is_self_history"]
    ]
    assert all(d["status"] == "live" for d in others)


def test_full_text_clamp_exposes_baseline_memo_everywhere(cfg, backend, test_world, steer_ctx):
    base = _run(cfg, backend, test_world, steer_ctx, BranchSpec(name="baseline"))
    ftc = _run(
        cfg, backend, test_world, steer_ctx,
        BranchSpec(name="positive_full_text_clamp", condition="positive_full_text_clamp",
                   branch_parent="baseline", steering={(0, 1): 1.0},
                   full_text_clamp=[(0, 1)]),
        baseline=base,
    )
    r1_from_source = [d for d in ftc.delivery_rows if d["source_agent"] == 0 and d["round"] == 1]
    assert r1_from_source
    assert all(d["status"] == "clamped" for d in r1_from_source)  # self-history included
    # downstream trajectories therefore equal baseline for all other agents
    assert np.allclose(ftc.beliefs[:, 1:], base.beliefs[:, 1:])


def test_fixed_replay_delivers_entirely_baseline_stream(cfg, backend, test_world, steer_ctx):
    base = _run(cfg, backend, test_world, steer_ctx, BranchSpec(name="baseline"))
    fr = _run(
        cfg, backend, test_world, steer_ctx,
        BranchSpec(name="fixed_replay_positive", condition="fixed_replay_positive",
                   branch_parent="baseline", steering={(0, 1): 1.0}, fixed_replay=True),
        baseline=base,
    )
    assert fr.delivery_rows
    assert all(d["status"] == "replayed" for d in fr.delivery_rows)
    assert all(d["actual_generated_branch"] == "baseline" for d in fr.delivery_rows)
