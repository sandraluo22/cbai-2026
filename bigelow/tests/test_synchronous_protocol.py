"""Synchrony and probe privacy (Part 24: 6, 7)."""

from __future__ import annotations

from belief_feedback.agents.prompts import PROBE_QUESTION
from belief_feedback.agents.protocol import BranchSpec, run_episode
from belief_feedback.models.mock_backend import MockBackend


class RecordingBackend(MockBackend):
    """Mock backend that records every generation and scoring context."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.generation_contexts = []
        self.scoring_contexts = []

    def generate(self, messages, seed, steering=None, **kw):
        self.generation_contexts.append(messages)
        return super().generate(messages, seed, steering=steering, **kw)

    def score_choices(self, messages, choices, steering=None):
        self.scoring_contexts.append(messages)
        return super().score_choices(messages, choices, steering=steering)


def test_no_same_round_leakage(cfg, test_world, steer_ctx):
    be = RecordingBackend(cfg)
    res = run_episode(cfg, be, test_world, 11, BranchSpec(name="baseline"), steer_ctx)
    # Every memo generated in round r must be absent from every other
    # generation context of round r (contexts were frozen pre-round).
    n = test_world.n_agents
    for r in range(1, cfg.network.rounds + 1):
        round_contexts = be.generation_contexts[(r - 1) * n : r * n]
        for i, ctx in enumerate(round_contexts):
            text = "\n".join(m["content"] for m in ctx)
            for j in range(n):
                if j == i:
                    continue
                memo_j = res.memos[(j, r)]
                assert memo_j[:80] not in text, f"agent {i} saw agent {j}'s round-{r} memo"


def test_probes_never_enter_transcript(cfg, test_world, steer_ctx):
    be = RecordingBackend(cfg)
    run_episode(cfg, be, test_world, 11, BranchSpec(name="baseline"), steer_ctx)
    # probe question appears in scoring contexts only, never in generation contexts
    assert all(
        PROBE_QUESTION in "\n".join(m["content"] for m in ctx) for ctx in be.scoring_contexts
    )
    for ctx in be.generation_contexts:
        assert PROBE_QUESTION not in "\n".join(m["content"] for m in ctx)


def test_probe_is_separate_forward_pass(cfg, test_world, steer_ctx):
    be = RecordingBackend(cfg)
    run_episode(cfg, be, test_world, 11, BranchSpec(name="baseline"), steer_ctx)
    n = test_world.n_agents
    # one probe per agent per time index (t = 0..rounds)
    assert len(be.scoring_contexts) == n * (cfg.network.rounds + 1)
