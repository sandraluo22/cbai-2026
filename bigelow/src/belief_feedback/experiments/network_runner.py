"""Endogenous network experiment (Part 7): baseline plus paired branches.

Per (world, replicate seed): run the baseline episode once, then every
branch against the same baseline and RNG streams. Results are written as
per-world part files, so interrupted runs resume without duplicating rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents.protocol import BranchSpec, EpisodeResult, SteeringContext, run_episode
from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..models.activations import ActivationStore
from ..world.generator import load_worlds, worlds_in_split
from ..world.schema import World
from .branches import make_branch_specs
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)

TABLES = ["belief_states", "public_messages", "deliveries", "interventions", "episodes"]


def episode_summary(cfg: Config, world: World, result: EpisodeResult, replicate_seed: int) -> dict[str, Any]:
    from ..analysis.metrics import episode_metrics

    return episode_metrics(cfg, world, result, replicate_seed)


def run_world(
    cfg: Config,
    backend,
    world: World,
    replicate_seed: int,
    branches: list[BranchSpec],
    steer_ctx: SteeringContext,
    act_store: ActivationStore | None,
    rounds: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLES}
    baseline_spec = next(b for b in branches if b.name == "baseline")
    baseline = run_episode(
        cfg, backend, world, replicate_seed, baseline_spec, steer_ctx, rounds=rounds, act_store=act_store
    )
    results = {"baseline": baseline}
    for spec in branches:
        if spec.name == "baseline":
            continue
        results[spec.name] = run_episode(
            cfg,
            backend,
            world,
            replicate_seed,
            spec,
            steer_ctx,
            baseline=baseline,
            rounds=rounds,
            act_store=act_store,
        )
    for res in results.values():
        rows["belief_states"].extend(res.belief_rows)
        rows["public_messages"].extend(res.message_rows)
        rows["deliveries"].extend(res.delivery_rows)
        rows["interventions"].extend(res.intervention_rows)
        rows["episodes"].append(episode_summary(cfg, world, res, replicate_seed))
    return rows


def run(cfg: Config) -> None:
    out_dir = cfg.paths.runs / "network"
    finals = [cfg.paths.runs / f"{t}.parquet" for t in TABLES]
    if outputs_exist(finals):
        log.info("network run exists; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)
    branches = make_branch_specs(delta, cfg.network.rounds, cfg.endogenous_conditions)
    act_store = ActivationStore(cfg.paths.activations / "network.npz")

    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for world in worlds_in_split(worlds, "endogenous_test"):
        for rep in cfg.network.replicate_seeds:
            part = parts_dir / f"{world.world_id}__s{rep}.parquet.done"
            if part.exists():
                completed += 1
                continue
            rows = run_world(cfg, backend, world, rep, branches, steer_ctx, act_store)
            for t in TABLES:
                save_df(pd.DataFrame(rows[t]), parts_dir / f"{world.world_id}__s{rep}__{t}.parquet")
            part.touch()
            completed += 1
            log.info("network: %s seed %d done", world.world_id, rep)
    act_store.save()

    for t in TABLES:
        files = sorted(parts_dir.glob(f"*__{t}.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        save_df(df, cfg.paths.runs / f"{t}.parquet")
    write_manifest(
        cfg,
        "run_network",
        started=started,
        artifact_paths=[str(p) for p in finals],
        completed_jobs=completed,
        extra={"delta": delta, "conditions": cfg.endogenous_conditions},
    )
