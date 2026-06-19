"""Hook EVERY layer of both models, then analyze across all layers.

Full per-occurrence activations for all layers (~120GB) exceed both the volume
quota and local disk, so we hold them in RAM (the pod has ~2TB) and write only
small derived results:
  - grid_per_layer.json : grid-recovery distance-corr at every layer, both models
  - cka_heatmap.npy/.png: linear CKA for all Gemma-layer x Qwen-layer pairs
                          (high-context occurrences, where the signal lives)
  - acts_sub_{gemma,qwen}.npz : a 15k-occurrence ALL-LAYER subsample, saved so
                          any later alignment can re-run without re-inference.

Capture order is Qwen-first (its weights are already cached on the volume); we
free Qwen before downloading Gemma so the ~35GB fp32 Gemma fits the quota.
"""
from __future__ import annotations
import os, json, gc, shutil
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
import align as A
import reproduce as R


def cka32(X: np.ndarray, Y: np.ndarray) -> float:
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(hsic / den) if den > 0 else float("nan")


def hf_cache_dir(name: str) -> str:
    hf = os.environ.get("HF_HOME", "/workspace/hf")
    return os.path.join(hf, "hub", "models--" + name.replace("/", "--"))


def main():
    base = get_config("gemma_qwen")
    # fewer walks than the headline run keeps all-layer RAM ~60GB and the run
    # snappy; still ample occurrences for per-node means and CKA.
    cfg = replace(base, n_walks=100, name="gemma_qwen_all", out_dir="/root/cmrun")
    GEMMA, QWEN = cfg.model_a, cfg.model_b
    gl, ql = tuple(range(42)), tuple(range(36))
    run_dir = f"{cfg.out_dir}/{cfg.name}"
    os.makedirs(run_dir, exist_ok=True)

    graph = G.build_grid_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    n_occ = sum(len(w.nodes) for w in walks)
    print(f"[all] {cfg.n_walks} walks, {n_occ:,} occ | Gemma {len(gl)} layers, "
          f"Qwen {len(ql)} layers", flush=True)

    # --- 1) Qwen first (cached on volume) ---
    print("[all] capturing Qwen, all layers ...", flush=True)
    mq, tq = M.load_model(QWEN, cfg)
    capq = M.capture(mq, tq, walks, ql, cfg)
    del mq, tq; gc.collect(); torch.cuda.empty_cache()
    print("[all] freeing Qwen weights to make room for Gemma", flush=True)
    shutil.rmtree(hf_cache_dir(QWEN), ignore_errors=True)

    # --- 2) Gemma (downloads ~35GB; xet disabled via env) ---
    print("[all] capturing Gemma, all layers (downloads first) ...", flush=True)
    mg, tg = M.load_model(GEMMA, cfg)
    capg = M.capture(mg, tg, walks, gl, cfg)
    del mg, tg; gc.collect(); torch.cuda.empty_cache()

    meta = capg.meta
    assert np.array_equal(meta["walk_id"], capq.meta["walk_id"])
    assert np.array_equal(meta["step"], capq.meta["step"])

    # --- grid recovery at every layer ---
    hi = cfg.context_checkpoints[-1]

    def grid_curve(cap, layers):
        rows = []
        for L in layers:
            nm = R.per_node_means(cap, L, graph, hi)
            rows.append({"layer": int(L),
                         "grid_corr": R.grid_recovery_score(nm, graph)["distance_corr"]})
        return rows

    grid_g, grid_q = grid_curve(capg, gl), grid_curve(capq, ql)
    json.dump({"gemma": grid_g, "qwen": grid_q},
              open(f"{run_dir}/grid_per_layer.json", "w"), indent=2)
    bg = max(grid_g, key=lambda r: r["grid_corr"])
    bq = max(grid_q, key=lambda r: r["grid_corr"])
    print(f"[all] Gemma best grid layer: L{bg['layer']} = {bg['grid_corr']:.3f}", flush=True)
    print(f"[all] Qwen  best grid layer: L{bq['layer']} = {bq['grid_corr']:.3f}", flush=True)

    # --- CKA heatmap over all layer pairs (high-context occurrences) ---
    himask = A.context_bin_mask(meta, hi, cfg.checkpoint_window)
    hidx = np.where(himask)[0]
    rng = np.random.default_rng(cfg.seed)
    if len(hidx) > 4000:
        hidx = np.sort(rng.choice(hidx, 4000, replace=False))
    print(f"[all] CKA heatmap on {len(hidx)} high-ctx occ, "
          f"{len(gl)}x{len(ql)} pairs ...", flush=True)
    Xg = {L: capg.acts[L][hidx] for L in gl}
    Yq = {L: capq.acts[L][hidx] for L in ql}
    H = np.zeros((len(gl), len(ql)))
    for i, gi in enumerate(gl):
        for j, qj in enumerate(ql):
            H[i, j] = cka32(Xg[gi], Yq[qj])
    np.save(f"{run_dir}/cka_heatmap.npy", H)
    bi, bj = np.unravel_index(int(np.argmax(H)), H.shape)
    print(f"[all] max CKA {H[bi, bj]:.3f} at Gemma L{gl[bi]} <-> Qwen L{ql[bj]}", flush=True)
    json.dump({"gemma_layers": list(gl), "qwen_layers": list(ql),
               "best": {"gemma": int(gl[bi]), "qwen": int(ql[bj]),
                        "cka": float(H[bi, bj])}},
              open(f"{run_dir}/cka_heatmap_meta.json", "w"), indent=2)

    # --- save a 15k all-layer subsample for later re-runs (uncompressed=fast) ---
    sidx = np.sort(rng.choice(n_occ, min(15000, n_occ), replace=False))
    np.savez(f"{run_dir}/acts_sub_gemma.npz",
             **{f"layer_{L}": capg.acts[L][sidx] for L in gl},
             **{f"meta_{k}": v[sidx] for k, v in meta.items()},
             _hidden_size=np.array([capg.hidden_size]), _layers=np.array(gl))
    np.savez(f"{run_dir}/acts_sub_qwen.npz",
             **{f"layer_{L}": capq.acts[L][sidx] for L in ql},
             **{f"meta_{k}": v[sidx] for k, v in meta.items()},
             _hidden_size=np.array([capq.hidden_size]), _layers=np.array(ql))
    print("[all] saved 15k all-layer subsample", flush=True)

    # --- heatmap plot ---
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Qwen layer"); ax.set_ylabel("Gemma layer")
        ax.set_title("linear CKA, high-context (Gemma x Qwen, all layers)")
        ax.set_xticks(range(len(ql))); ax.set_xticklabels(ql, fontsize=5)
        ax.set_yticks(range(len(gl))); ax.set_yticklabels(gl, fontsize=5)
        fig.colorbar(im, label="CKA")
        fig.tight_layout(); fig.savefig(f"{run_dir}/cka_heatmap.png", dpi=130)
        print("[all] wrote cka_heatmap.png", flush=True)
    except Exception as e:
        print("[all] plot skipped:", e, flush=True)

    print("[all] DONE", flush=True)


if __name__ == "__main__":
    main()
