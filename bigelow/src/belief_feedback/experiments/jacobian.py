"""Empirical network Jacobian (Part 13): paired one-round finite differences."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents.protocol import BranchSpec, run_episode
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)


def run(cfg: Config) -> None:
    out = cfg.paths.runs / "jacobian_results.parquet"
    if outputs_exist([out]):
        log.info("jacobian results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)
    subset = worlds_in_split(worlds, "endogenous_test")[: cfg.analysis.jacobian_worlds]
    rep = cfg.network.replicate_seeds[0]
    rounds = cfg.network.rounds

    rows: list[dict[str, Any]] = []
    for world in subset:
        baseline = run_episode(
            cfg, backend, world, rep, BranchSpec(name="baseline", condition="baseline"), steer_ctx
        )
        max_t = rounds if cfg.analysis.jacobian_rounds is None else min(
            cfg.analysis.jacobian_rounds, rounds
        )
        for j in range(world.n_agents):
            for t in range(1, max_t + 1):
                plus = run_episode(
                    cfg, backend, world, rep,
                    BranchSpec(
                        name=f"jac_p_{j}_{t}", condition="jacobian_plus",
                        branch_parent="baseline", steering={(j, t): +delta},
                    ),
                    steer_ctx, baseline=baseline,
                )
                minus = run_episode(
                    cfg, backend, world, rep,
                    BranchSpec(
                        name=f"jac_m_{j}_{t}", condition="jacobian_minus",
                        branch_parent="baseline", steering={(j, t): -delta},
                    ),
                    steer_ctx, baseline=baseline,
                )
                for i in range(world.n_agents):
                    # J_t[i, j] from ell at the round following the intervention
                    if t >= plus.beliefs.shape[0]:
                        continue
                    val = (plus.beliefs[t, i] - minus.beliefs[t, i]) / (2 * delta)
                    # multi-round propagation of the round-1 impulse
                    rows.append(
                        {
                            **id_fields(cfg),
                            "world_id": world.world_id,
                            "split": world.split,
                            "condition": "jacobian",
                            "branch": f"jac_{j}_{t}",
                            "branch_parent": "baseline",
                            "replicate_seed": rep,
                            "seed": 0,
                            "round": t,
                            "agent_id": i,
                            "source_agent": j,
                            "jacobian_value": float(val),
                            "delta": delta,
                        }
                    )
                if t == 1:
                    for tt in range(1, rounds + 1):
                        for i in range(world.n_agents):
                            rows.append(
                                {
                                    **id_fields(cfg),
                                    "world_id": world.world_id,
                                    "split": world.split,
                                    "condition": "impulse_response",
                                    "branch": f"jacimp_{j}",
                                    "branch_parent": "baseline",
                                    "replicate_seed": rep,
                                    "seed": 0,
                                    "round": tt,
                                    "agent_id": i,
                                    "source_agent": j,
                                    "jacobian_value": float(
                                        (plus.beliefs[tt, i] - minus.beliefs[tt, i]) / (2 * delta)
                                    ),
                                    "delta": delta,
                                }
                            )
    save_df(pd.DataFrame(rows), out)
    write_manifest(
        cfg, "run_jacobian", started=started, artifact_paths=[str(out)], completed_jobs=len(rows)
    )
