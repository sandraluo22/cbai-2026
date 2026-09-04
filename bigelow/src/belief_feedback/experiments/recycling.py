"""Evidence-recycling experiment (Part 10).

Two measurements per matched independent/recycled world pair:

1. Single-context belief gains: one report vs three independent vs three
   recycled reports, against the provenance-aware and provenance-blind
   oracle gains, under neutral and provenance-aware instructions.
2. Live network episodes for the interaction with a source-agent impulse.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents.prompts import PROBE_CHOICES, probe_messages, system_prompt
from ..agents.protocol import BranchSpec, run_episode
from ..agents.roles import role_for_agent
from ..agents.transcript import Transcript
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from ..world.oracle import blind_oracle_for_reports, oracle_for_reports
from ..world.schema import World
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)


def _focal_reports(world: World) -> list[str]:
    return sorted(
        rep.report_id for rep in world.reports if rep.report_id.split("-")[-1].startswith("F")
    )


def _probe_ell(cfg: Config, backend, world: World, report_ids: list[str], provenance_aware: bool) -> float:
    docs = "\n\n---\n\n".join(world.report(rid).text for rid in report_ids) or "(no private records)"
    tr = Transcript(
        system=system_prompt(world, 0, role_for_agent(0), provenance_aware=provenance_aware),
        private_records=f"Your private records:\n\n{docs}",
    )
    score = backend.score_choices(probe_messages(tr.context_messages()), PROBE_CHOICES)
    return world.visible_to_semantic(score.logps[0] - score.logps[1])


def run(cfg: Config) -> None:
    out1 = cfg.paths.runs / "recycling_results.parquet"
    out2 = cfg.paths.runs / "recycling_network.parquet"
    if outputs_exist([out1, out2]):
        log.info("recycling results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)
    pairs: dict[str, dict[str, World]] = {}
    for w in worlds_in_split(worlds, "recycling_test"):
        pairs.setdefault(w.tags["pair_id"], {})[w.tags["recycling_role"]] = w

    gain_rows: list[dict[str, Any]] = []
    net_rows: list[dict[str, Any]] = []
    for pair_id, pair in sorted(pairs.items()):
        for role_name, world in pair.items():
            backend.register_world(world)
            focal = _focal_reports(world)
            for prov in (False, True):
                ell0 = _probe_ell(cfg, backend, world, [], prov)
                ell1 = _probe_ell(cfg, backend, world, focal[:1], prov)
                ell3 = _probe_ell(cfg, backend, world, focal, prov)
                aware3 = oracle_for_reports(world, focal).oracle_log_odds
                blind3 = blind_oracle_for_reports(world, focal).oracle_log_odds
                aware1 = oracle_for_reports(world, focal[:1]).oracle_log_odds
                gain1 = ell1 - ell0
                gain3 = ell3 - ell0
                gain_rows.append(
                    {
                        **id_fields(cfg),
                        "pair_id": pair_id,
                        "world_id": world.world_id,
                        "split": world.split,
                        "condition": role_name,
                        "branch": "single_context",
                        "branch_parent": None,
                        "replicate_seed": 0,
                        "agent_id": 0,
                        "round": 0,
                        "seed": 0,
                        "provenance_aware_prompt": prov,
                        "gain_one_report": gain1,
                        "gain_three_reports": gain3,
                        "oracle_gain_one": aware1,
                        "oracle_aware_gain_three": aware3,
                        "oracle_blind_gain_three": blind3,
                        "multiplier": abs(gain3) / max(abs(gain1), 1e-6),
                        "double_counting_gap": abs(gain3) - abs(aware3),
                    }
                )
            for prov in (False, True):
                base_spec = BranchSpec(
                    name=f"recycling_base_p{int(prov)}",
                    condition="baseline",
                    provenance_aware=prov,
                )
                baseline = run_episode(
                    cfg, backend, world, cfg.network.replicate_seeds[0], base_spec, steer_ctx
                )
                for steer_mag in (0.0, delta):
                    if steer_mag == 0.0:
                        res = baseline
                        cond = "unsteered"
                    else:
                        spec = BranchSpec(
                            name=f"recycling_impulse_p{int(prov)}",
                            condition="impulse",
                            branch_parent=base_spec.name,
                            provenance_aware=prov,
                            steering={(0, 1): steer_mag},
                        )
                        res = run_episode(
                            cfg,
                            backend,
                            world,
                            cfg.network.replicate_seeds[0],
                            spec,
                            steer_ctx,
                            baseline=baseline,
                        )
                        cond = "impulse"
                    net_rows.append(
                        {
                            **id_fields(cfg),
                            "pair_id": pair_id,
                            "world_id": world.world_id,
                            "split": world.split,
                            "provenance_role": role_name,
                            "condition": cond,
                            "branch": res.branch,
                            "branch_parent": None if cond == "unsteered" else "baseline",
                            "replicate_seed": cfg.network.replicate_seeds[0],
                            "agent_id": -1,
                            "round": cfg.network.rounds,
                            "seed": 0,
                            "provenance_aware_prompt": prov,
                            "steering_magnitude": steer_mag,
                            "final_mean_ell": float(res.beliefs[-1].mean()),
                        }
                    )
    save_df(pd.DataFrame(gain_rows), out1)
    save_df(pd.DataFrame(net_rows), out2)
    write_manifest(
        cfg,
        "run_recycling",
        started=started,
        artifact_paths=[str(out1), str(out2)],
        completed_jobs=len(gain_rows) + len(net_rows),
    )
