"""Sweep entry point (build step 4): separation/diversity sweep + transition extraction.

For each (diversity D, seed): train a tiny model on a Dirichlet(D) continuum,
evaluate on a fixed source set, capture anchor activations, build the source-axis
similarity matrix (R² + CKA) + convergence curve + null band, and extract the
transition location & sharpness. Aggregate: transition-location vs D and
sharpness vs D.

    python sweep.py --config configs/sweep.yaml --dry-run
    python sweep.py --config configs/sweep.yaml --override train.device=cpu seeds=[0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analysis
import matrix
from activations import (ActMeta, ActStore, capture_rollouts, estimate_disk_gb,
                         load_activations)
from dgp import dirichlet_sources, log_spaced_positions, make_rollout
from model import TinyGPTConfig, TrainConfig, train_tiny
from ruler import RulerConfig


# --------------------------------------------------------------------------- #
# Config: YAML + dotted CLI overrides                                          #
# --------------------------------------------------------------------------- #
def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        k, raw = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


# --------------------------------------------------------------------------- #
# One run                                                                      #
# --------------------------------------------------------------------------- #
def run_one(cfg, diversity, seed) -> dict:
    K, C = cfg["K"], cfg["C"]
    nS, nR = cfg["n_sources"], cfg["n_rollouts"]
    positions = log_spaced_positions(C)
    run_id = f"D{diversity}_seed{seed}".replace(".", "p")
    out = Path(cfg["output_root"]) / run_id
    out.mkdir(parents=True, exist_ok=True)

    rng_src = np.random.default_rng(seed)
    rng_roll = np.random.default_rng(seed + 100)
    rng_ruler = np.random.default_rng(seed + 200)

    eval_sources = dirichlet_sources(K, nS, diversity, rng_src)
    mcfg = TinyGPTConfig(vocab=K, block_size=C, **cfg["model"])
    tcfg = TrainConfig(seed=seed, online_dirichlet={"K": K, "diversity": diversity}, **cfg["train"])
    model, ckpts, losses = train_tiny(None, mcfg, tcfg)

    meta = ActMeta(run_id=run_id, model_name="TinyGPT", n_sources=nS, n_rollouts=nR,
                   positions_t=positions, n_layers=mcfg.n_layer, hidden_dim=mcfg.d_model,
                   dtype="float32", anchor_rule="balls_urns: t-1", template_hash="none",
                   store_format="memmap",
                   shape=[nS, nR, len(positions), mcfg.n_layer, mcfg.d_model])
    store = ActStore(out / "activations", meta, max_disk_gb=cfg["max_disk_gb"])
    rollouts = {}
    for sid in range(nS):
        rolls = [make_rollout(eval_sources, sid, C, rng_roll) for _ in range(nR)]
        rollouts[sid] = rolls
        acts, _ = capture_rollouts(model, np.stack([r.seq for r in rolls]), positions,
                                   device=cfg["train"]["device"])
        for r in range(nR):
            store.put(sid, r, acts[r])
    store.flush()

    arr, meta_d = load_activations(out)
    layer = mcfg.n_layer - 1 if cfg["layer"] == -1 else cfg["layer"]
    rcfg = RulerConfig(**cfg["ruler"])

    for method, label in [("r2", "out-of-sample R²"), ("cka", "linear CKA")]:
        S = matrix.build_similarity_matrix(arr, meta_d, layer, RulerConfig(method=method),
                                           rng_ruler, axis="pooled")
        matrix.save_heatmap(S, positions, out / "plots", f"simmat_{method}", label)

    curve = analysis.convergence_curve(arr, meta_d, layer, rcfg, rng_ruler, axis="pooled")
    null = analysis.null_band(arr, meta_d, layer, rcfg, rng_ruler, axis="pooled")
    analytic = np.array([np.mean([rollouts[s][r].posterior[t, s]
                                  for s in range(nS) for r in range(nR)]) for t in positions])
    analysis.save_convergence_plot(curve, null, out / "plots", "convergence_r2", analytic=analytic)

    # transition on the ruler curve AND on analytic (for the ground-truth comparison)
    tr_ruler = matrix.detect_transition(curve["r2"], positions)
    tr_analytic = matrix.detect_transition(analytic, positions)
    (out / "transitions.json").write_text(json.dumps(
        {"ruler": tr_ruler, "analytic": tr_analytic}, indent=2))

    manifest = {"run_id": run_id, "diversity": diversity, "seed": seed,
                "final_loss": float(np.mean(losses[-50:])),
                "transition_t_ruler": tr_ruler["location_t"], "sharpness_ruler": tr_ruler["sharpness"],
                "transition_t_analytic": tr_analytic["location_t"],
                "r2_curve": curve["r2"].tolist(), "analytic": analytic.tolist()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "diversity": diversity, "seed": seed}))
    print(f"  {run_id}: loss={manifest['final_loss']:.3f}  "
          f"transition_t(ruler)={tr_ruler['location_t']}  sharpness={tr_ruler['sharpness']:.2f}")
    return manifest


# --------------------------------------------------------------------------- #
# Aggregate                                                                    #
# --------------------------------------------------------------------------- #
def aggregate(cfg, manifests):
    out = Path(cfg["output_root"])
    Ds = sorted(set(m["diversity"] for m in manifests))
    def stat(key):
        m = [[x[key] for x in manifests if x["diversity"] == D and x[key] is not None] for D in Ds]
        mean = [np.mean(v) if v else np.nan for v in m]
        sem = [np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in m]
        return np.array(mean), np.array(sem)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tl, tls = stat("transition_t_ruler")
    ta, _ = stat("transition_t_analytic")
    axes[0].errorbar(Ds, tl, yerr=tls, marker="o", capsize=3, label="ruler R²")
    axes[0].plot(Ds, ta, "^--", color="C3", alpha=0.7, label="analytic")
    axes[0].set(title="Transition location vs separation", xlabel="Dirichlet diversity D",
                ylabel="transition t*"); axes[0].set_yscale("log", base=2); axes[0].legend()
    sh, shs = stat("sharpness_ruler")
    axes[1].errorbar(Ds, sh, yerr=shs, marker="o", capsize=3, color="C2")
    axes[1].set(title="Transition sharpness vs separation", xlabel="Dirichlet diversity D",
                ylabel="sharpness (norm. level jump)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "aggregate_transition_vs_separation.png", dpi=140); plt.close(fig)
    (out / "sweep_manifests.json").write_text(json.dumps(manifests, indent=2))
    print(f"\nAggregate -> {out}/aggregate_transition_vs_separation.png")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    jobs = [(D, s) for D in cfg["diversity_values"] for s in cfg["seeds"]]
    P = len(log_spaced_positions(cfg["C"]))
    gb = len(jobs) * estimate_disk_gb(cfg["n_sources"], cfg["n_rollouts"], P,
                                      cfg["model"]["n_layer"], cfg["model"]["d_model"], "float32")
    print(f"=== sweep: {len(jobs)} runs (D×seed), est. activation disk {gb:.3f} GB ===")
    for D, s in jobs:
        print(f"  D={D} seed={s}")
    if args.dry_run:
        print("--dry-run: exiting before training."); return

    manifests = [run_one(cfg, D, s) for D, s in jobs]
    aggregate(cfg, manifests)


if __name__ == "__main__":
    main()
