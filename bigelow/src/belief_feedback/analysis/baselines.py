"""Normative oracle baselines for belief-trajectory prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..world.oracle import blind_oracle_for_reports, oracle_for_reports
from ..world.schema import World


def oracle_trajectory_prediction(world: World, belief_rows: pd.DataFrame) -> pd.DataFrame:
    """Predict each agent-round belief with the provenance-aware oracle.

    The oracle sees exactly the events the agent could access at that time
    (already recorded in the belief rows); this is the normative reference
    for the model's behavioral beliefs.
    """
    out = belief_rows[["world_id", "agent_id", "round", "semantic_log_odds", "oracle_log_odds"]].copy()
    out["predicted"] = out["oracle_log_odds"]
    out["error"] = out["predicted"] - out["semantic_log_odds"]
    return out


def recycling_oracle_gains(world: World, focal_reports: list[str]) -> dict[str, float]:
    aware = oracle_for_reports(world, focal_reports)
    blind = blind_oracle_for_reports(world, focal_reports)
    one = oracle_for_reports(world, focal_reports[:1])
    return {
        "aware_gain_three": aware.oracle_log_odds,
        "blind_gain_three": blind.oracle_log_odds,
        "gain_one": one.oracle_log_odds,
        "aware_multiplier": abs(aware.oracle_log_odds) / max(abs(one.oracle_log_odds), 1e-9),
        "blind_multiplier": abs(blind.oracle_log_odds) / max(abs(one.oracle_log_odds), 1e-9),
    }


def interval_coverage(observed: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = (observed >= lo) & (observed <= hi)
    return float(np.mean(ok)) if len(observed) else float("nan")
