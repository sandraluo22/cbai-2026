"""Result-table schemas: required identifying columns for every result row."""

from __future__ import annotations

import pandas as pd

ID_COLUMNS = [
    "config_hash",
    "git_commit",
    "model_id",
    "tokenizer_id",
    "replicate_seed",
    "world_id",
    "agent_id",
    "round",
    "condition",
    "branch_parent",
    "seed",
]


def check_id_columns(df: pd.DataFrame, table_name: str) -> list[str]:
    """Return the identifying columns missing from a result table."""
    return [c for c in ID_COLUMNS if c not in df.columns]
