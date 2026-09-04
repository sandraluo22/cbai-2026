"""CAA vector computation, layer scan, and magnitude calibration.

Layer selection maximizes the held-out median behavioral slope
(d semantic log odds / d magnitude) subject to coherence constraints;
``m_max`` is the largest symmetric magnitude keeping generation coherent and
the response approximately monotonic. ``delta = 0.5 * m_max`` is the
finite-difference impulse used everywhere downstream.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..agents.memo_parser import parse_memo, repeated_4gram_rate
from ..agents.prompts import (
    MEMO_REQUEST,
    PROBE_CHOICES,
    private_records_message,
    probe_messages,
    system_prompt,
)
from ..agents.roles import role_for_agent
from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..models.base import Backend
from ..models.generation import Message
from ..models.steering import SteeringSpec
from ..seeds import derive_seed
from ..world.generator import load_worlds, worlds_in_split
from ..world.schema import World
from .common import outputs_exist, save_df, save_steering, steering_paths
from .steering_dataset import caa_item_messages

log = get_logger(__name__)

NEUTRAL_PROMPTS = [
    "In one or two sentences, describe the purpose of an incident review.",
    "Briefly explain what a calibration check verifies.",
]


def _val_contexts(cfg: Config, worlds: dict[str, World], k: int = 8) -> list[tuple[World, list[Message]]]:
    """Single-agent held-out contexts for behavioral evaluation."""
    out = []
    for w in worlds_in_split(worlds, "steering_validation"):
        for agent in range(min(2, w.n_agents)):
            ctx: list[Message] = [
                {"role": "system", "content": system_prompt(w, agent, role_for_agent(agent))},
                {"role": "user", "content": private_records_message(w, agent)},
            ]
            out.append((w, ctx))
            if len(out) >= k:
                return out
    return out


def _behavioral_ell(backend: Backend, world: World, ctx: list[Message], spec: SteeringSpec | None) -> float:
    score = backend.score_choices(probe_messages(ctx), PROBE_CHOICES, steering=spec)
    return world.visible_to_semantic(score.logps[0] - score.logps[1])


def _coherence(backend: Backend, world: World, ctx: list[Message], spec: SteeringSpec | None, seed: int) -> dict[str, float]:
    gen = backend.generate(ctx + [{"role": "user", "content": MEMO_REQUEST}], seed=seed, steering=spec)
    pm = parse_memo(gen.text, world)
    return {
        "valid": float(pm.format_valid),
        "rep4": repeated_4gram_rate(gen.text),
        "length": float(len(gen.text.split())),
    }


def compute_caa_vectors(cfg: Config, backend: Backend, caa_df: pd.DataFrame) -> dict[str, Any]:
    """Per-layer difference-in-means vectors on the training items."""
    worlds = load_worlds(cfg)
    train = caa_df[caa_df["usage"] == "train"]
    acts_up, acts_local = [], []
    proj_rows = []
    for _, it in train.iterrows():
        w = worlds[it["world_id"]]
        backend.register_world(w)
        rids = it["report_ids"].split(",")
        a_up = backend.get_activations(caa_item_messages(w, rids, it["up_completion"]))
        a_lo = backend.get_activations(caa_item_messages(w, rids, it["local_completion"]))
        acts_up.append(a_up)
        acts_local.append(a_lo)
        proj_rows.append({"item_id": it["item_id"], "alpha_is_upstream": bool(it["alpha_is_upstream"])})
    up = np.stack(acts_up)  # [n, layers, dim]
    lo = np.stack(acts_local)
    d = up.mean(axis=0) - lo.mean(axis=0)  # [layers, dim]
    # per-example projections on each layer's unit direction
    units = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    proj_up = np.einsum("nld,ld->nl", up, units)
    proj_lo = np.einsum("nld,ld->nl", lo, units)
    return {
        "vectors": d,
        "norms": np.linalg.norm(d, axis=1),
        "proj_up": proj_up,
        "proj_lo": proj_lo,
        "proj_rows": proj_rows,
    }


def _layer_slope_and_coherence(
    cfg: Config,
    backend: Backend,
    worlds: dict[str, World],
    vector: np.ndarray,
    layer: int,
    magnitudes: list[float],
) -> dict[str, Any]:
    contexts = _val_contexts(cfg, worlds)
    for w, _ in contexts:
        backend.register_world(w)
    ells: dict[float, list[float]] = {m: [] for m in magnitudes}
    cohs: dict[float, list[dict[str, float]]] = {m: [] for m in magnitudes}
    for ci, (w, ctx) in enumerate(contexts):
        for m in magnitudes:
            spec: SteeringSpec | None = (
                SteeringSpec(vector=vector, layer=layer, magnitude=m, scope=cfg.steering.scope)
                if m != 0
                else None
            )
            ells[m].append(_behavioral_ell(backend, w, ctx, spec))
            cohs[m].append(_coherence(backend, w, ctx, spec, seed=derive_seed("layer_scan", layer, m, ci)))
    lo, hi = min(magnitudes), max(magnitudes)
    slopes = [(a - b) / (hi - lo) for a, b in zip(ells[hi], ells[lo])]
    med = {m: float(np.median(ells[m])) for m in magnitudes}
    base = cohs[0.0] if 0.0 in cohs else cohs[min(magnitudes, key=abs)]
    base_valid = float(np.mean([c["valid"] for c in base])) or 1e-9
    base_rep4 = float(np.median([c["rep4"] for c in base]))
    base_len = float(np.median([c["length"] for c in base]))
    worst_valid = min(float(np.mean([c["valid"] for c in cohs[m]])) for m in magnitudes)
    worst_rep4 = max(float(np.median([c["rep4"] for c in cohs[m]])) for m in magnitudes)
    worst_len = max(float(np.median([c["length"] for c in cohs[m]])) for m in magnitudes)
    monotone = all(med[a] <= med[b] + 0.15 for a, b in zip(sorted(med), sorted(med)[1:]))
    coherent = (
        worst_valid >= 0.95 * base_valid
        and worst_rep4 <= max(base_rep4 * 1.2, base_rep4 + 0.02)
        and worst_len <= base_len * 1.2
        and monotone
    )
    return {
        "layer": layer,
        "median_slope": float(np.median(slopes)),
        "median_ells": med,
        "valid_rate": worst_valid,
        "rep4": worst_rep4,
        "length": worst_len,
        "coherent": coherent,
    }


def run(cfg: Config) -> dict[str, Any]:
    vec_path, meta_path = steering_paths(cfg)
    scan_out = cfg.paths.runs / "steering_calibration.parquet"
    if outputs_exist([vec_path, meta_path, scan_out]):
        log.info("steering calibration exists; skipping")
        return json.loads(meta_path.read_text())
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    caa_df = pd.read_parquet(cfg.paths.runs / "caa_dataset.parquet")

    caa = compute_caa_vectors(cfg, backend, caa_df)
    vectors, norms = caa["vectors"], caa["norms"]
    n_layers = vectors.shape[0]

    # ---- layer scan: every second layer at magnitudes -1, 0, +1 ----------
    coarse_layers = list(range(0, n_layers, 2))
    scan_rows = []
    for line in coarse_layers:
        res = _layer_slope_and_coherence(
            cfg, backend, worlds, vectors[line], line, cfg.steering.layer_scan_magnitudes
        )
        res["phase"] = "coarse"
        scan_rows.append(res)
    coherent_rows = [r for r in scan_rows if r["coherent"]] or scan_rows
    best = max(coherent_rows, key=lambda r: r["median_slope"])["layer"]
    fine_layers = sorted({max(0, best - 1), best, min(n_layers - 1, best + 1)})
    for line in fine_layers:
        res = _layer_slope_and_coherence(
            cfg, backend, worlds, vectors[line], line, cfg.steering.layer_scan_magnitudes
        )
        res["phase"] = "fine"
        scan_rows.append(res)
    fine = [r for r in scan_rows if r["phase"] == "fine" and r["coherent"]]
    fine = fine or [r for r in scan_rows if r["phase"] == "fine"]
    selected = max(fine, key=lambda r: r["median_slope"])["layer"]
    if cfg.steering.layer is not None:
        selected = cfg.steering.layer

    # ---- magnitude calibration at the selected layer ---------------------
    contexts = _val_contexts(cfg, worlds)
    mag_rows = []
    med_by_m: dict[float, float] = {}
    coh_by_m: dict[float, dict[str, float]] = {}
    for m in cfg.steering.magnitude_scan:
        spec = (
            SteeringSpec(vector=vectors[selected], layer=selected, magnitude=m, scope=cfg.steering.scope)
            if m != 0
            else None
        )
        ells, cohs = [], []
        for ci, (w, ctx) in enumerate(contexts):
            ells.append(_behavioral_ell(backend, w, ctx, spec))
            cohs.append(_coherence(backend, w, ctx, spec, seed=derive_seed("mag_scan", m, ci)))
        med_by_m[m] = float(np.median(ells))
        coh_by_m[m] = {
            "valid": float(np.mean([c["valid"] for c in cohs])),
            "rep4": float(np.median([c["rep4"] for c in cohs])),
            "length": float(np.median([c["length"] for c in cohs])),
        }
        mag_rows.append({"magnitude": m, "median_ell": med_by_m[m], **coh_by_m[m]})

    base = coh_by_m[0.0]
    base_valid = base["valid"] or 1e-9

    def ok(m: float) -> bool:
        c = coh_by_m[m]
        return (
            c["valid"] >= 0.95 * base_valid
            and c["rep4"] <= max(base["rep4"] * 1.2, base["rep4"] + 0.02)
            and c["length"] <= base["length"] * 1.2
        )

    candidates = sorted({abs(m) for m in cfg.steering.magnitude_scan if m != 0})
    m_max = 0.0
    for m in candidates:
        window = [x for x in cfg.steering.magnitude_scan if -m - 1e-9 <= x <= m + 1e-9]
        mono = all(
            med_by_m[a] <= med_by_m[b] + 0.25 for a, b in zip(sorted(window), sorted(window)[1:])
        )
        if ok(m) and ok(-m) and mono:
            m_max = m
        else:
            break
    if m_max == 0.0:
        m_max = candidates[0]
        log.warning("no symmetric magnitude passed coherence; defaulting m_max=%s", m_max)

    # neutral-prompt sanity check
    neutral_ok = True
    for i, p in enumerate(NEUTRAL_PROMPTS):
        g = backend.generate(
            [{"role": "user", "content": p}],
            seed=derive_seed("neutral", i),
            steering=SteeringSpec(vector=vectors[selected], layer=selected, magnitude=m_max, scope=cfg.steering.scope),
        )
        if repeated_4gram_rate(g.text) > 0.5:
            neutral_ok = False

    scan_df = pd.DataFrame(
        [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in scan_rows]
    )
    save_df(scan_df, scan_out)
    save_df(pd.DataFrame(mag_rows), cfg.paths.runs / "steering_magnitude_scan.parquet")

    balance = {
        "semantic_pairs": int(len(caa["proj_rows"])),
        "alpha_is_upstream": int(sum(r["alpha_is_upstream"] for r in caa["proj_rows"])),
    }
    meta = {
        "layer": int(selected),
        "m_max": float(m_max),
        "delta": float(0.5 * m_max),
        "scope": cfg.steering.scope,
        "vector_norms_by_layer": [float(x) for x in norms],
        "layer_scan": [{k: v for k, v in r.items() if k != "median_ells"} for r in scan_rows],
        "magnitude_scan": mag_rows,
        "semantic_label_balance": balance,
        "visible_label_balance": balance,
        "neutral_prompt_ok": neutral_ok,
        "model_id": cfg.model.model_id,
    }
    save_steering(cfg, vectors[selected], meta)
    np.save(cfg.paths.vectors / cfg.model.slug / "caa_vectors_all_layers.npy", vectors)
    np.savez(
        cfg.paths.vectors / cfg.model.slug / "caa_projections.npz",
        proj_up=caa["proj_up"],
        proj_lo=caa["proj_lo"],
    )
    write_manifest(
        cfg,
        "steering_calibration",
        started=started,
        artifact_paths=[str(vec_path), str(meta_path), str(scan_out)],
        completed_jobs=len(scan_rows),
        extra={"selected_layer": int(selected), "m_max": float(m_max)},
    )
    log.info("selected layer %d, m_max=%.2f", selected, m_max)
    return meta
