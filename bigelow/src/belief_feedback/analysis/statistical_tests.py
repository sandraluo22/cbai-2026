"""Primary-hypothesis tests and Benjamini-Hochberg correction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import Config
from .bootstrap import paired_world_test


def hypothesis_row(
    cfg: Config,
    name: str,
    values_by_world: pd.Series,
    description: str,
    secondary: bool = False,
) -> dict[str, Any]:
    res = paired_world_test(
        values_by_world,
        n_resamples=cfg.analysis.bootstrap_samples,
        seed_parts=("hyp", name),
    )
    return {
        "hypothesis": name,
        "description": description,
        "effect": res.estimate,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "standardized_effect": res.standardized_effect,
        "p_value": res.p_value,
        "n_worlds": res.n_worlds,
        "secondary": secondary,
    }


def benjamini_hochberg(df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    """BH correction applied only across rows flagged secondary."""
    out = df.copy()
    out["p_adjusted"] = out[p_col]
    sec = out[out["secondary"] & out[p_col].notna()].sort_values(p_col)
    m = len(sec)
    if m:
        adj = []
        prev = 1.0
        for rank, (_, row) in enumerate(reversed(list(sec.iterrows())), start=0):
            i = m - rank
            val = min(prev, row[p_col] * m / i)
            adj.append((row.name, val))
            prev = val
        for idx, val in adj:
            out.loc[idx, "p_adjusted"] = val
    out["significant_05"] = out["p_adjusted"] < alpha
    return out
