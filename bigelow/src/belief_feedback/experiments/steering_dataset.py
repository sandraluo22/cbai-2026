"""Contrastive dataset for the CAA steering direction.

Each item pairs one incident context (from steering-split worlds) with two
matched assistant conclusions — one favoring each semantic hypothesis. The
visible label attached to each semantic conclusion follows the world's
counterbalanced ALPHA/BETA mapping, so the resulting vector cannot be a bare
ALPHA-vs-BETA token direction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents.prompts import system_prompt
from ..agents.roles import role_for_agent
from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models.generation import Message
from ..seeds import rng as make_rng
from ..world.generator import load_worlds, worlds_in_split
from ..world.schema import LOCAL, UPSTREAM, World
from .common import outputs_exist, save_df

log = get_logger(__name__)

QUESTION = (
    "Given only the records above, state your current conclusion about the "
    "more likely root cause in one sentence, beginning with 'Current assessment:'."
)

UP_CONCLUSION = (
    "Current assessment: {label}. The available evidence currently favors the "
    "upstream-material explanation."
)
LOCAL_CONCLUSION = (
    "Current assessment: {label}. The available evidence currently favors the "
    "station-calibration explanation."
)


def caa_item_messages(world: World, report_ids: list[str], completion: str) -> list[Message]:
    docs = "\n\n---\n\n".join(world.report(rid).text for rid in report_ids)
    return [
        {"role": "system", "content": system_prompt(world, 0, role_for_agent(0))},
        {"role": "user", "content": f"Your private records:\n\n{docs}\n\n{QUESTION}"},
        {"role": "assistant", "content": completion},
    ]


def build_caa_items(cfg: Config, split: str, n_items: int) -> list[dict[str, Any]]:
    worlds = worlds_in_split(load_worlds(cfg), split)
    if not worlds:
        raise RuntimeError(f"no worlds in split {split}")
    items = []
    for i in range(n_items):
        world = worlds[i % len(worlds)]
        r = make_rng("caa_item", split, i)
        n_reports = int(r.integers(1, 5))
        rids = list(r.choice([rep.report_id for rep in world.reports], size=min(n_reports, len(world.reports)), replace=False))
        up_label = world.visible_label(UPSTREAM)
        local_label = world.visible_label(LOCAL)
        items.append(
            {
                "item_id": f"caa_{split}_{i:05d}",
                "world_id": world.world_id,
                "split": split,
                "report_ids": rids,
                "up_completion": UP_CONCLUSION.format(label=up_label),
                "local_completion": LOCAL_CONCLUSION.format(label=local_label),
                "up_visible_label": up_label,
                "local_visible_label": local_label,
                "alpha_is_upstream": world.alpha_is_upstream,
            }
        )
    return items


def run(cfg: Config) -> pd.DataFrame:
    out = cfg.paths.runs / "caa_dataset.parquet"
    if outputs_exist([out]):
        log.info("steering dataset exists; skipping")
        return pd.read_parquet(out)
    started = now_iso()
    train = build_caa_items(cfg, "steering_train", cfg.steering.caa_train_pairs)
    val = build_caa_items(cfg, "steering_validation", cfg.steering.caa_val_pairs)
    rows = []
    for role_split, items in (("train", train), ("validation", val)):
        for it in items:
            rows.append({**it, "report_ids": ",".join(it["report_ids"]), "usage": role_split})
    df = pd.DataFrame(rows)
    save_df(df, out)
    n_alpha_up = int(df["alpha_is_upstream"].sum())
    write_manifest(
        cfg,
        "steering_dataset",
        started=started,
        artifact_paths=[str(out)],
        completed_jobs=len(df),
        extra={"visible_label_balance": {"alpha_is_upstream": n_alpha_up, "total": len(df)}},
    )
    return df
