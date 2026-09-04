"""Hysteresis experiment (Part 11): early vs late steering schedules with
equal total dose and zero steering in the final two rounds, under live,
fixed-replay, and full-text-clamped communication."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..agents.protocol import BranchSpec, run_episode
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)

SOURCE = 0


def _schedules(rounds: int) -> dict[str, list[int]]:
    early = [1, 2]
    late = [min(3, rounds - 1), min(4, rounds)]
    return {"early": early, "late": sorted(set(late))}


def run(cfg: Config) -> None:
    out = cfg.paths.runs / "hysteresis_results.parquet"
    traj_out = cfg.paths.runs / "hysteresis_trajectories.parquet"
    if outputs_exist([out, traj_out]):
        log.info("hysteresis results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)
    rounds = cfg.network.hysteresis_rounds
    scheds = _schedules(rounds)

    rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    for world in worlds_in_split(worlds, "hysteresis_test"):
        for rep in cfg.network.replicate_seeds:
            baseline = run_episode(
                cfg, backend, world, rep, BranchSpec(name="baseline", condition="baseline"),
                steer_ctx, rounds=rounds,
            )
            finals: dict[tuple[str, str, str], float] = {}
            final_stds: dict[tuple[str, str, str], float] = {}
            trajs: dict[tuple[str, str, str], np.ndarray] = {}
            for sign, mag in (("positive", delta), ("negative", -delta)):
                for sched_name, sched_rounds in scheds.items():
                    steering = {(SOURCE, r): mag for r in sched_rounds}
                    for comm in ("live", "replay", "clamp"):
                        spec = BranchSpec(
                            name=f"hyst_{sign}_{sched_name}_{comm}",
                            condition=f"{sign}_{sched_name}_{comm}",
                            branch_parent="baseline",
                            steering=steering,
                            fixed_replay=comm == "replay",
                            full_text_clamp=(
                                [(SOURCE, r) for r in sched_rounds] if comm == "clamp" else []
                            ),
                        )
                        res = run_episode(
                            cfg, backend, world, rep, spec, steer_ctx,
                            baseline=baseline, rounds=rounds,
                        )
                        mean_traj = res.beliefs.mean(axis=1)
                        finals[(sign, sched_name, comm)] = float(mean_traj[-1])
                        final_stds[(sign, sched_name, comm)] = float(res.beliefs[-1].std())
                        trajs[(sign, sched_name, comm)] = mean_traj
                        for t, v in enumerate(mean_traj):
                            traj_rows.append(
                                {
                                    **id_fields(cfg),
                                    "world_id": world.world_id,
                                    "split": world.split,
                                    "condition": spec.condition,
                                    "branch": spec.name,
                                    "branch_parent": "baseline",
                                    "replicate_seed": rep,
                                    "agent_id": -1,
                                    "round": t,
                                    "seed": 0,
                                    "sign": sign,
                                    "schedule": sched_name,
                                    "comm": comm,
                                    "mean_ell": float(v),
                                }
                            )
            for sign in ("positive", "negative"):
                for comm in ("live", "replay", "clamp"):
                    e = trajs[(sign, "early", comm)]
                    line = trajs[(sign, "late", comm)]
                    disagreement = 0.5 * (
                        final_stds[(sign, "early", comm)] + final_stds[(sign, "late", comm)]
                    )
                    rows.append(
                        {
                            **id_fields(cfg),
                            "world_id": world.world_id,
                            "split": world.split,
                            "condition": f"{sign}_{comm}",
                            "branch": f"hyst_{sign}_{comm}",
                            "branch_parent": "baseline",
                            "replicate_seed": rep,
                            "agent_id": -1,
                            "round": rounds,
                            "seed": 0,
                            "sign": sign,
                            "comm": comm,
                            "final_early": finals[(sign, "early", comm)],
                            "final_late": finals[(sign, "late", comm)],
                            "hysteresis_gap": finals[(sign, "early", comm)] - finals[(sign, "late", comm)],
                            "trajectory_area": float(np.trapz(np.abs(e - line))),
                            "final_disagreement": disagreement,
                        }
                    )
    save_df(pd.DataFrame(rows), out)
    save_df(pd.DataFrame(traj_rows), traj_out)
    write_manifest(
        cfg, "run_hysteresis", started=started, artifact_paths=[str(out), str(traj_out)],
        completed_jobs=len(rows),
    )
