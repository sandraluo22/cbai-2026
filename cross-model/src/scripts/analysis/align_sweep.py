"""Fit Gemma layer 32 -> every captured Qwen layer (offline, from saved acts).

For each Qwen layer we report:
  - held-out ridge R^2 and Procrustes residual (Gemma L32 -> Qwen L_k)
  - linear CKA pooled across context, and within the highest context bin
  - the Qwen layer's own grid-recovery distance-corr (does structure live there?)

No GPU / no re-capture; reuses runs/v1/<preset>/acts_model_{a,b}.npz.
Usage: python align_sweep.py [gemma_layer]   (default 32)
"""
from __future__ import annotations
import sys, json
import numpy as np

from config import get_config
import graph as G
import models as M
import align as A
import reproduce as R


def main(gemma_layer: int = 32):
    cfg = get_config("gemma_qwen")
    run_dir = f"{cfg.out_dir}/{cfg.name}"
    cap_a = M.load_capture(f"{run_dir}/acts_model_a.npz")   # Gemma
    cap_b = M.load_capture(f"{run_dir}/acts_model_b.npz")   # Qwen
    graph = G.build_grid_graph(cfg)

    assert gemma_layer in cap_a.acts, f"Gemma layer {gemma_layer} not captured ({sorted(cap_a.acts)})"
    qlayers = sorted(cap_b.acts)
    hi = cfg.context_checkpoints[-1]

    print(f"Gemma L{gemma_layer} (d={cap_a.hidden_size}) -> Qwen layers {qlayers} "
          f"(d={cap_b.hidden_size})")
    print(f"{'Qlayer':>6} {'ridge_R2':>9} {'proc_resid':>11} {'CKA_pooled':>11} "
          f"{'CKA_hi':>8} {'Qwen_grid':>10}")
    rows = []
    for ql in qlayers:
        X_A, X_B, meta = A.pair_occurrences(cap_a, cap_b, gemma_layer, ql)
        train, test = A.split_by_walk(meta, cfg.test_frac, cfg.seed)

        rm = A.fit_ridge(X_A[train], X_B[train], cfg.ridge_alpha)
        pm = A.fit_procrustes(X_A[train], X_B[train], cfg.pca_k)
        r2_test = A.r2(X_B[test], rm.predict(X_A[test]))
        resid = A.procrustes_residual(pm, X_A[test], X_B[test])
        cka_pool = A.linear_cka(X_A[test], X_B[test])

        m = test & A.context_bin_mask(meta, hi, cfg.checkpoint_window)
        cka_hi = A.linear_cka(X_A[m], X_B[m]) if m.sum() >= 5 else float("nan")

        nm = R.per_node_means(cap_b, ql, graph, min_context=hi)
        grid = R.grid_recovery_score(nm, graph)["distance_corr"]

        rows.append({"qwen_layer": ql, "ridge_r2_test": r2_test,
                     "procrustes_residual_test": resid, "cka_pooled": cka_pool,
                     "cka_high_ctx": cka_hi, "qwen_grid_corr": grid})
        print(f"{ql:>6} {r2_test:>9.3f} {resid:>11.3f} {cka_pool:>11.3f} "
              f"{cka_hi:>8.3f} {grid:>10.3f}")

    out = {"gemma_layer": gemma_layer, "rows": rows}
    with open(f"{run_dir}/sweep_gemmaL{gemma_layer}_to_qwen.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {run_dir}/sweep_gemmaL{gemma_layer}_to_qwen.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
