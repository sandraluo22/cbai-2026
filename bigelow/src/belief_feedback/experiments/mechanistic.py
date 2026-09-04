"""Mechanistic analysis (Part 14): linear probes, projection trajectories,
belief-component patching, and the text-mediation check."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..agents.prompts import PROBE_CHOICES, private_records_message, probe_messages, system_prompt
from ..agents.protocol import BranchSpec, run_episode
from ..agents.roles import role_for_agent
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..world.generator import load_worlds, worlds_in_split
from ..world.schema import World
from .common import delta_from_meta, load_steering, outputs_exist, save_df

log = get_logger(__name__)


def _single_agent_context(world: World, agent: int):
    return [
        {"role": "system", "content": system_prompt(world, agent, role_for_agent(agent))},
        {"role": "user", "content": private_records_message(world, agent)},
    ]


def _collect(cfg: Config, backend, worlds: list[World]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activations [n, layers, dim], behavioral ell [n], sign labels [n]."""
    acts, ells = [], []
    for w in worlds:
        backend.register_world(w)
        for agent in range(w.n_agents):
            ctx = _single_agent_context(w, agent)
            score = backend.score_choices(probe_messages(ctx), PROBE_CHOICES)
            ell = w.visible_to_semantic(score.logps[0] - score.logps[1])
            acts.append(backend.get_activations(ctx))
            ells.append(ell)
    a = np.stack(acts)
    e = np.array(ells)
    return a, e, (e > 0).astype(int)


def run(cfg: Config) -> None:
    probe_out = cfg.paths.runs / "probe_results.parquet"
    patch_out = cfg.paths.runs / "mechanistic_patching.parquet"
    if outputs_exist([probe_out, patch_out]):
        log.info("mechanistic results exist; skipping")
        return
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)

    # ---- linear probes (exogenous train/val, endogenous eval) ------------
    train_a, _, train_y = _collect(cfg, backend, worlds_in_split(worlds, "exogenous_train"))
    val_a, _, val_y = _collect(cfg, backend, worlds_in_split(worlds, "exogenous_validation"))
    endo_a, endo_ell, endo_y = _collect(cfg, backend, worlds_in_split(worlds, "endogenous_test"))
    unit = steer_ctx.vector / (np.linalg.norm(steer_ctx.vector) + 1e-9)

    probe_rows: list[dict[str, Any]] = []
    for layer in range(train_a.shape[1]):
        best_c, best_score = 1.0, -np.inf
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, max_iter=2000).fit(train_a[:, layer], train_y)
            s = clf.score(val_a[:, layer], val_y)
            if s > best_score:
                best_c, best_score = c, s
        clf = LogisticRegression(C=best_c, max_iter=2000).fit(train_a[:, layer], train_y)
        pred = clf.predict_proba(endo_a[:, layer])[:, 1]
        acc = float(((pred > 0.5).astype(int) == endo_y).mean())
        try:
            auroc = float(roc_auc_score(endo_y, pred))
        except ValueError:
            auroc = float("nan")
        with np.errstate(divide="ignore"):
            probe_ell = np.log(np.clip(pred, 1e-9, 1)) - np.log(np.clip(1 - pred, 1e-9, 1))
        corr = float(np.corrcoef(probe_ell, endo_ell)[0, 1]) if endo_ell.std() > 0 else float("nan")
        bins = np.clip((pred * 5).astype(int), 0, 4)
        ece = float(
            np.nanmean([abs(pred[bins == b].mean() - endo_y[bins == b].mean()) for b in range(5) if (bins == b).any()])
        )
        w = clf.coef_[0]
        cos = float(w @ unit / (np.linalg.norm(w) + 1e-9))
        probe_rows.append(
            {
                **id_fields(cfg),
                "split": "endogenous_test",
                "condition": "linear_probe",
                "branch": "probe",
                "branch_parent": None,
                "replicate_seed": 0,
                "world_id": "-",
                "agent_id": -1,
                "round": -1,
                "seed": 0,
                "layer": layer,
                "C": best_c,
                "val_accuracy": float(best_score),
                "accuracy": acc,
                "auroc": auroc,
                "behavior_correlation": corr,
                "calibration_ece": ece,
                "cosine_with_caa": cos,
            }
        )
    save_df(pd.DataFrame(probe_rows), probe_out)

    # ---- belief-component patching + text mediation ----------------------
    patch_rows: list[dict[str, Any]] = []
    subset = worlds_in_split(worlds, "endogenous_test")[: cfg.analysis.mechanistic_worlds]
    rep = cfg.network.replicate_seeds[0]
    for world in subset:
        baseline = run_episode(
            cfg, backend, world, rep, BranchSpec(name="baseline", condition="baseline"), steer_ctx
        )
        pos = run_episode(
            cfg, backend, world, rep,
            BranchSpec(name="pos", condition="positive_impulse", branch_parent="baseline",
                       steering={(0, 1): +delta}),
            steer_ctx, baseline=baseline,
        )
        neg = run_episode(
            cfg, backend, world, rep,
            BranchSpec(name="neg", condition="negative_impulse", branch_parent="baseline",
                       steering={(0, 1): -delta}),
            steer_ctx, baseline=baseline,
        )
        pos_clamped = run_episode(
            cfg, backend, world, rep,
            BranchSpec(name="pos_clamp", condition="positive_full_text_clamp",
                       branch_parent="baseline", steering={(0, 1): +delta},
                       full_text_clamp=[(0, 1)]),
            steer_ctx, baseline=baseline,
        )
        proj_source = next(
            r["caa_projection"] for r in pos.belief_rows if r["agent_id"] == 0 and r["round"] == 1
        )
        for clamp in (False, True):
            patch = run_episode(
                cfg, backend, world, rep,
                BranchSpec(
                    name=f"patch_clamp{int(clamp)}",
                    condition="projection_patch" + ("_text_clamp" if clamp else ""),
                    branch_parent="neg",
                    projection_patch={(0, 1): float(proj_source)},
                    full_text_clamp=[(0, 1)] if clamp else [],
                ),
                steer_ctx, baseline=baseline,
            )
            d_belief = patch.beliefs - neg.beliefs
            d_downstream = d_belief[:, 1:].mean()
            stance_change = sum(
                patch.parsed[(0, r)].parsed_assessment != neg.parsed[(0, r)].parsed_assessment
                for r in range(1, cfg.network.rounds + 1)
            )
            cites_change = sum(
                set(patch.parsed[(0, r)].cited_ids) != set(neg.parsed[(0, r)].cited_ids)
                for r in range(1, cfg.network.rounds + 1)
            )
            patch_rows.append(
                {
                    **id_fields(cfg),
                    "world_id": world.world_id,
                    "split": world.split,
                    "condition": "projection_patch",
                    "branch": f"patch_clamp{int(clamp)}",
                    "branch_parent": "neg",
                    "replicate_seed": rep,
                    "agent_id": 0,
                    "round": -1,
                    "seed": 0,
                    "text_clamped": clamp,
                    "proj_target": float(proj_source),
                    "delta_source_belief_t1": float(d_belief[1, 0]),
                    "delta_downstream_mean": float(d_downstream),
                    "n_stance_changes": int(stance_change),
                    "n_citation_changes": int(cites_change),
                    "steer_live_downstream": float((pos.beliefs - baseline.beliefs)[:, 1:].mean()),
                    "steer_clamp_downstream": float(
                        (pos_clamped.beliefs - baseline.beliefs)[:, 1:].mean()
                    ),
                }
            )
    save_df(pd.DataFrame(patch_rows), patch_out)
    write_manifest(
        cfg, "run_mechanistic", started=started,
        artifact_paths=[str(probe_out), str(patch_out)],
        completed_jobs=len(probe_rows) + len(patch_rows),
    )
