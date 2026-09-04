"""Exact Bayesian oracles.

The provenance-aware oracle counts each latent event once regardless of how
many reports mention it. The deliberately provenance-blind oracle counts
every report separately; the gap between the two quantifies the normative
cost of evidence recycling.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .schema import World


@dataclass
class OracleResult:
    oracle_log_odds: float
    oracle_probability_upstream: float
    unique_event_count: int
    repeated_report_count: int
    event_ids_seen: list[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def oracle_for_reports(world: World, report_ids: Iterable[str]) -> OracleResult:
    """Provenance-aware posterior over reports visible to one agent.

    Prior log odds are zero. Each unique event contributes its LLR exactly
    once, no matter how many primary or secondary reports repeat it.
    """
    counts: Counter[str] = Counter()
    for rid in report_ids:
        counts[world.report(rid).event_id] += 1
    log_odds = sum(world.event(eid).llr for eid in counts)
    repeated = sum(c - 1 for c in counts.values())
    return OracleResult(
        oracle_log_odds=log_odds,
        oracle_probability_upstream=_sigmoid(log_odds),
        unique_event_count=len(counts),
        repeated_report_count=repeated,
        event_ids_seen=sorted(counts),
    )


def blind_oracle_for_reports(world: World, report_ids: Iterable[str]) -> OracleResult:
    """Provenance-blind posterior: every report counts as independent."""
    rids = list(report_ids)
    counts: Counter[str] = Counter(world.report(rid).event_id for rid in rids)
    log_odds = sum(world.event(world.report(rid).event_id).llr for rid in rids)
    repeated = sum(c - 1 for c in counts.values())
    return OracleResult(
        oracle_log_odds=log_odds,
        oracle_probability_upstream=_sigmoid(log_odds),
        unique_event_count=len(counts),
        repeated_report_count=repeated,
        event_ids_seen=sorted(counts),
    )


def agent_initial_oracle(world: World, agent_id: int) -> OracleResult:
    return oracle_for_reports(world, world.assignments.get(agent_id, []))


def network_oracle(world: World) -> OracleResult:
    all_reports = [rid for rids in world.assignments.values() for rid in rids]
    return oracle_for_reports(world, all_reports)
