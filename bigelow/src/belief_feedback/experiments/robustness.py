"""Robustness experiments (Parts 15-16): prompt variants, memory policy,
topology, asynchronous order, steering scope, presentation order, and the
communication-channel ablations. Never pooled with primary estimates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents.protocol import BranchSpec, SteeringContext, run_episode
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)

CHANNEL_MODES = ["full", "header_only", "body_citations", "citations_only", "paraphrase", "role_swap"]


def _variants(cfg: Config) -> list[tuple[str, str, dict[str, Any]]]:
    v: list[tuple[str, str, dict[str, Any]]] = [
        ("prompt", "variant_0", {"prompt_variant": 0}),
        ("prompt", "variant_1", {"prompt_variant": 1}),
        ("prompt", "variant_2", {"prompt_variant": 2}),
        ("memory", "full_transcript", {"memory_rounds": None}),
        ("memory", "last_two_rounds", {"memory_rounds": 2}),
        ("topology", "ring", {"topology": "ring"}),
        ("topology", "star", {"topology": "star"}),
        ("topology", "complete", {"topology": "complete"}),
        ("schedule", "synchronous", {"async_order_seed": None}),
        ("schedule", "asynchronous_seeded", {"async_order_seed": 1234}),
    ]
    for mode in CHANNEL_MODES:
        v.append(("channel", mode, {"channel_transform": mode}))
    return v


def run(cfg: Config) -> None:
    out = cfg.paths.runs / "robustness_results.parquet"
    if outputs_exist([out]):
        log.info("robustness results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)
    subset = worlds_in_split(worlds, "robustness_test")[: cfg.analysis.robustness_worlds]
    rep = cfg.network.replicate_seeds[0]

    rows: list[dict[str, Any]] = []

    def run_pair(dimension: str, variant: str, kwargs: dict[str, Any], ctx: SteeringContext) -> None:
        for world in subset:
            base = BranchSpec(name=f"rb_{dimension}_{variant}_base", condition="baseline", **kwargs)
            baseline = run_episode(cfg, backend, world, rep, base, ctx)
            imp = BranchSpec(
                name=f"rb_{dimension}_{variant}_imp",
                condition="positive_impulse",
                branch_parent=base.name,
                steering={(0, 1): +delta},
                **kwargs,
            )
            res = run_episode(cfg, backend, world, rep, imp, ctx, baseline=baseline)
            diff = res.beliefs - baseline.beliefs
            invalid = sum(1 for pm in res.parsed.values() if not pm.format_valid)
            halluc = sum(len(pm.hallucinated_report_ids) for pm in res.parsed.values())
            rows.append(
                {
                    **id_fields(cfg),
                    "world_id": world.world_id,
                    "split": world.split,
                    "condition": "robustness",
                    "branch": imp.name,
                    "branch_parent": base.name,
                    "replicate_seed": rep,
                    "agent_id": -1,
                    "round": -1,
                    "seed": 0,
                    "dimension": dimension,
                    "variant": variant,
                    "total_effect": float(diff.sum()),
                    "mean_effect": float(diff.mean()),
                    "final_mean_effect": float(diff[-1].mean()),
                    "malformed_rate": invalid / max(len(res.parsed), 1),
                    "hallucinated_citation_rate": halluc / max(len(res.parsed), 1),
                }
            )

    for dimension, variant, kwargs in _variants(cfg):
        run_pair(dimension, variant, kwargs, steer_ctx)
    # steering-scope robustness: all-token intervention
    run_pair("steering_scope", "final_token_and_generation", {}, steer_ctx)
    all_tok = SteeringContext(vector=steer_ctx.vector, layer=steer_ctx.layer, scope="all_tokens")
    run_pair("steering_scope", "all_tokens", {}, all_tok)

    save_df(pd.DataFrame(rows), out)
    write_manifest(
        cfg, "run_robustness", started=started, artifact_paths=[str(out)], completed_jobs=len(rows)
    )
