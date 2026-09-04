"""Branch specifications for the endogenous network experiment (Part 7)."""

from __future__ import annotations

from ..agents.protocol import BranchSpec

SOURCE_AGENT = 0


def make_branch_specs(delta: float, rounds: int, conditions: list[str]) -> list[BranchSpec]:
    """Paired causal branches sharing the baseline's RNG streams."""
    persistent_rounds = list(range(1, min(3, rounds) + 1))
    catalog: dict[str, BranchSpec] = {
        "baseline": BranchSpec(name="baseline", condition="baseline"),
        "positive_impulse": BranchSpec(
            name="positive_impulse",
            condition="positive_impulse",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): +delta},
        ),
        "negative_impulse": BranchSpec(
            name="negative_impulse",
            condition="negative_impulse",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): -delta},
        ),
        "positive_persistent": BranchSpec(
            name="positive_persistent",
            condition="positive_persistent",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, r): +delta for r in persistent_rounds},
        ),
        "negative_persistent": BranchSpec(
            name="negative_persistent",
            condition="negative_persistent",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, r): -delta for r in persistent_rounds},
        ),
        "positive_one_hop": BranchSpec(
            name="positive_one_hop",
            condition="positive_one_hop",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): +delta},
            one_hop_from_round=2,
        ),
        "negative_one_hop": BranchSpec(
            name="negative_one_hop",
            condition="negative_one_hop",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): -delta},
            one_hop_from_round=2,
        ),
        "positive_no_return": BranchSpec(
            name="positive_no_return",
            condition="positive_no_return",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): +delta},
            no_return_agent=SOURCE_AGENT,
        ),
        "negative_no_return": BranchSpec(
            name="negative_no_return",
            condition="negative_no_return",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): -delta},
            no_return_agent=SOURCE_AGENT,
        ),
        "positive_full_text_clamp": BranchSpec(
            name="positive_full_text_clamp",
            condition="positive_full_text_clamp",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): +delta},
            full_text_clamp=[(SOURCE_AGENT, 1)],
        ),
        "negative_full_text_clamp": BranchSpec(
            name="negative_full_text_clamp",
            condition="negative_full_text_clamp",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): -delta},
            full_text_clamp=[(SOURCE_AGENT, 1)],
        ),
        "fixed_replay_positive": BranchSpec(
            name="fixed_replay_positive",
            condition="fixed_replay_positive",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): +delta},
            fixed_replay=True,
        ),
        "fixed_replay_negative": BranchSpec(
            name="fixed_replay_negative",
            condition="fixed_replay_negative",
            branch_parent="baseline",
            steering={(SOURCE_AGENT, 1): -delta},
            fixed_replay=True,
        ),
    }
    return [catalog[c] for c in conditions]
