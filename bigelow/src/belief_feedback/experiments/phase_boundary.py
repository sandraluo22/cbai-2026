"""Phase-boundary experiment (Part 12): evidence bins x persistent steering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..agents.protocol import BranchSpec, run_episode
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from ..world.oracle import network_oracle
from .common import load_steering, outputs_exist, save_df

log = get_logger(__name__)

SOURCE = 0


def run(cfg: Config) -> None:
    out = cfg.paths.runs / "phase_boundary_results.parquet"
    if outputs_exist([out]):
        log.info("phase boundary results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    m_max = float(meta["m_max"])
    persistent = list(range(1, min(3, cfg.network.rounds) + 1))

    rows: list[dict[str, Any]] = []
    for world in worlds_in_split(worlds, "phase_boundary_test"):
        oracle = network_oracle(world).oracle_log_odds
        for rep in cfg.network.replicate_seeds[:2]:
            baseline = run_episode(
                cfg, backend, world, rep, BranchSpec(name="baseline", condition="baseline"), steer_ctx
            )
            for frac in cfg.analysis.phase_steering_fracs:
                if frac == 0.0:
                    res = baseline
                else:
                    spec = BranchSpec(
                        name=f"phase_{frac:+.1f}",
                        condition=f"phase_{frac:+.1f}",
                        branch_parent="baseline",
                        steering={(SOURCE, r): frac * m_max for r in persistent},
                    )
                    res = run_episode(cfg, backend, world, rep, spec, steer_ctx, baseline=baseline)
                final = res.beliefs[-1]
                initial = res.beliefs[0]
                n_up = int((final > 0).sum())
                rows.append(
                    {
                        **id_fields(cfg),
                        "world_id": world.world_id,
                        "split": world.split,
                        "condition": f"phase_{frac:+.1f}",
                        "branch": res.branch,
                        "branch_parent": None if frac == 0.0 else "baseline",
                        "replicate_seed": rep,
                        "agent_id": -1,
                        "round": cfg.network.rounds,
                        "seed": 0,
                        "phase_bin": float(world.tags.get("phase_bin", "nan")),
                        "network_oracle_log_odds": oracle,
                        "steering_frac": frac,
                        "steering_magnitude": frac * m_max,
                        "final_mean_ell": float(final.mean()),
                        "initial_beliefs": ",".join(f"{x:.4f}" for x in initial),
                        "n_agents_upstream": n_up,
                        "upstream_majority": n_up >= int(np.ceil(0.625 * world.n_agents)),
                        "strong_upstream_consensus": (
                            n_up >= int(np.ceil(0.875 * world.n_agents)) and final.mean() >= 1.0
                        ),
                    }
                )
    save_df(pd.DataFrame(rows), out)
    write_manifest(
        cfg, "run_phase_boundary", started=started, artifact_paths=[str(out)], completed_jobs=len(rows)
    )
