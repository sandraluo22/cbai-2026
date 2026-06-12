"""Build step 3 (sources axis) — end-to-end on CPU before scaling.

Train one tiny model on a CONTINUUM of K-ary Dirichlet sources, then evaluate on a
fixed set of sources. N-axis = sources: each row is a source's mean activation. As
context grows, distinct sources SEPARATE, so the cloud spreads into stable
structure (it does not collapse like the rollout cloud). We build ONE similarity
matrix (R² and CKA) + ONE convergence curve + the null band, and overlay the
analytic source-separability ground truth.

    python run_step3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analysis
import matrix
from activations import ActMeta, ActStore, capture_rollouts, load_activations
from dgp import dirichlet_sources, log_spaced_positions, make_rollout
from model import TinyGPTConfig, TrainConfig, train_tiny
from ruler import RulerConfig

OUT = Path("results/step3_sources")
K = 4
DIVERSITY = 0.3            # Dirichlet concentration: low = peaky/well-separated sources
C = 256
N_SOURCES = 32            # the N-axis
N_ROLLOUTS = 16           # per source (averaged -> source mean state)
LAYER = -1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng_src = np.random.default_rng(0)
    rng_roll = np.random.default_rng(1)
    rng_ruler = np.random.default_rng(2)

    positions = log_spaced_positions(C)
    eval_sources = dirichlet_sources(K, N_SOURCES, DIVERSITY, rng_src)   # fixed hypothesis set
    print(f"K={K}, D={DIVERSITY}, {N_SOURCES} eval sources, positions t={positions}")

    # --- train on a Dirichlet CONTINUUM (general source inference) ---
    mcfg = TinyGPTConfig(vocab=K, d_model=64, n_layer=2, n_head=2, block_size=C)
    tcfg = TrainConfig(steps=2000, batch=128, lr=3e-3, seed=0,
                       online_dirichlet={"K": K, "diversity": DIVERSITY})
    print("training tiny GPT on Dirichlet continuum (CPU) ...")
    model, ckpts, losses = train_tiny(None, mcfg, tcfg)
    print(f"  final train loss {np.mean(losses[-50:]):.4f}")

    # --- capture: each eval source, N_ROLLOUTS rollouts, anchor activations ---
    meta = ActMeta(
        run_id="step3_sources", model_name="TinyGPT", n_sources=N_SOURCES, n_rollouts=N_ROLLOUTS,
        positions_t=positions, n_layers=mcfg.n_layer, hidden_dim=mcfg.d_model, dtype="float32",
        anchor_rule="balls_urns: anchor = position t-1 (last token of prefix length t)",
        template_hash="none", store_format="memmap",
        shape=[N_SOURCES, N_ROLLOUTS, len(positions), mcfg.n_layer, mcfg.d_model],
    )
    store = ActStore(OUT / "activations", meta)
    rollouts_by_src = {}
    blog = (OUT / "beliefs.jsonl").open("w")
    for sid in range(N_SOURCES):
        rolls = [make_rollout(eval_sources, sid, C, rng_roll) for _ in range(N_ROLLOUTS)]
        rollouts_by_src[sid] = rolls
        seqs = np.stack([r.seq for r in rolls])
        acts, beliefs = capture_rollouts(model, seqs, positions)
        for r in range(N_ROLLOUTS):
            store.put(sid, r, acts[r])
            for pi, t in enumerate(positions):
                blog.write(json.dumps({"source": sid, "rollout": r, "t": t,
                    "model_belief": beliefs[r, pi].tolist(),
                    "analytic_post_true": float(rolls[r].posterior[t, sid])}) + "\n")
    blog.close(); store.flush()

    arr, meta_d = load_activations(OUT)
    layer = mcfg.n_layer - 1 if LAYER == -1 else LAYER

    # --- ONE similarity matrix, R² and CKA side by side (N-axis = sources) ---
    for method, label in [("r2", "out-of-sample R²"), ("cka", "linear CKA")]:
        S = matrix.build_similarity_matrix(arr, meta_d, layer, RulerConfig(method=method),
                                           rng_ruler, axis="source")
        matrix.save_heatmap(S, positions, OUT / "plots", f"simmat_{method}", label)

    # --- ONE convergence curve + null band + analytic separability overlay ---
    rcfg = RulerConfig(method="r2")
    curve = analysis.convergence_curve(arr, meta_d, layer, rcfg, rng_ruler, axis="source")
    null = analysis.null_band(arr, meta_d, layer, rcfg, rng_ruler, axis="source")
    # analytic: mean posterior mass on the true source (separability of the 32 sources)
    analytic = np.array([np.mean([rollouts_by_src[s][r].posterior[t, s]
                                  for s in range(N_SOURCES) for r in range(N_ROLLOUTS)])
                         for t in positions])
    analysis.save_convergence_plot(curve, null, OUT / "plots", "convergence_r2", analytic=analytic)

    print("\nR²(t) source-arrangement -> final:", np.round(curve["r2"], 3))
    print("null mean                         :", np.round(null["mean"], 3))
    print("analytic source separability      :", np.round(analytic, 3))
    print(f"\nStep 3 (sources) done -> {OUT}/")


if __name__ == "__main__":
    main()
