"""End-to-end numeric (LLM-free) demo of the QSG harness.

Runs the pure-numpy null model across the three communication channels and emits
the full analysis stack (similarity heatmaps + U/V/accuracy curves), exercising
exactly the plotting pipeline the LLM runs use. Run this FIRST to validate
analysis/plotting before any GPU time.

    python run_reference_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qsg import analysis
from qsg.qsg_reference import SOFT, consensus_time, run_reference


def main() -> None:
    out = Path("results/_reference_demo")
    out.mkdir(parents=True, exist_ok=True)

    N, K, rounds, alpha = 8, 7, 60, 0.3
    ground_truth = 0  # "elephant" index in the ontology candidate list

    channels = {"hard_m1": 1.0, "topm_m4": 4.0, "soft_minf": SOFT}

    # Shared initial belief: weak nudge toward ground truth (mimics partial info).
    rng = np.random.default_rng(0)
    x0 = rng.dirichlet(np.ones(K), size=N)
    x0[:, ground_truth] += 0.4
    x0 /= x0.sum(1, keepdims=True)

    for name, m in channels.items():
        res = run_reference(
            N, K, alpha, m, rounds, seed=0, x0=x0.copy(),
            ground_truth=ground_truth, selection_strength=0.02,
        )
        cdir = out / name
        paths = analysis.save_similarity_heatmaps(res.beliefs, cdir)
        analysis.save_diagnostic_curves(
            res.beliefs, cdir, ground_truth=ground_truth, consensus_threshold=0.95
        )
        ct = consensus_time(res.U, 0.95)
        print(
            f"[{name:9s}] U: {res.U[0]:.3f} -> {res.U[-1]:.3f}  "
            f"V: {res.V[0]:.3f} -> {res.V[-1]:.3f}  "
            f"consensus@U>=0.95: {ct}  "
            f"final argmax={int(np.argmax(res.final_mean))} (truth={ground_truth})"
        )
        print(f"           heatmaps -> {paths['cosine_png'].parent}")

    print(f"\nDemo complete. See {out}/<channel>/ for PNGs + .npy matrices.")


if __name__ == "__main__":
    main()
