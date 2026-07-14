# induction-head/ — the two-circuit investigation

Artifacts for the two-circuit finding: **QK prefix-match ("induction") heads carry
the in-context next-step behaviour; a separate late-layer set of DLA "writer"
heads writes the decodable grid *geometry* into the residual stream.** The two can
be knocked out independently — a double dissociation — and that dissociation holds
not just for next-token prediction but across a long autoregressive rollout.

Leaf dir names are unchanged from the original flat layout, so they still match
each script's default `OUTDIR` basename. The **pod** keeps the flat layout
(scripts write there); this local tree is the tidy pulled mirror. Pull future
outputs into the matching group below. `reorg.sh` reproduces this grouping (and
documents the move mapping).

## The dissociation at a glance

Two circuits × two questions × two horizons. Each cell says what happens to the
metric when that head-group is ablated (K=15 heads/group, Llama-3.1-8B, square grid):

|                          | **behaviour** (neighbour mass / validity) | **geometry** (coord-probe R²) |
|--------------------------|-------------------------------------------|-------------------------------|
| ablate **QK / induction**| **collapses** (val 0.79→0.21)             | mostly survives (0.60→0.35)   |
| ablate **DLA writers**   | mostly spared (val 0.79→0.59)             | **destroyed** (0.60→ −0.2)    |
| ablate random (control)  | mild drop (val →0.64)                      | mild drop (R² →0.52)          |

Read the two off-diagonal cells: QK ablation is behaviour-specific, DLA ablation
is geometry-specific — and the random control shows neither effect is just "remove
15 heads." **The dissociation replicates on the ring graph** (`gen_head_ablation_ring`):
ablate-QK drives validity 0.88→0.31 while geometry stays 0.70 (≈ clean 0.73);
ablate-DLA leaves validity 0.73 (≈ clean) while geometry falls 0.73→0.22. So it is
not a grid-specific artifact. Where to see it, by horizon:

- **next-token / teacher-forced:** `3_ablations/ablation_probe_qk` and
  `3_ablations/ablation_probe_dla` — coord-probe R² under each ablation vs a
  clean/random control, evaluated one step ahead.
- **long-horizon / autoregressive:** `6_generation/gen_head_ablation` — the same
  two ablations, but seed with context then generate freely and track behaviour
  **and** per-layer geometry across generation windows. This is the experiment the
  table above is drawn from; see its summary page for the three-panel figure.

## Layout

```
1_circuits/          WHERE the heads are
  induction_heads/     induction.json, induction_heads.pdf, qk_histogram.pdf   (QK heads)
  attribution/         head_attribution_*.json/.pdf, dla_histogram.pdf         (DLA writer heads)
  head_sweep/  copying/  sanity_ov/  atlas/  outlier/  node_output/
2_probes/            geometry decodability (the trusted representation)
  coord_decode/        leave-one-node-out linear coordinate probe (+ perm null)
  cross_model_sim/     CKA / ridge-LOO alignment across models
  cross_layer_heatmap/ layer_A x layer_B cross-model ridge-alignment R²
  basis_cossim/  cross_context/  injection/
3_ablations/         causal knockouts + RSA controls (NEXT-TOKEN horizon)
  ablation{,_allqk,_dla,_logit,_rsa}/  layer_ablation/  positional_ablation/
  posablation_after/  posablation_exact/  puncture_rsa/  rsa_shuffle/  context_traj/
  ablation_probe_qk/   coord-probe R² under induction-QK-head ablation vs clean/random
  ablation_probe_dla/  coord-probe R² under DLA-writer-head ablation (legend says
                       "induction" but red curve = the ablated writer heads)
4_patching/          activation / topological patching
  patch_swap/  topo_patch/
5_steering/          causal read-out of the map (single-token / teacher-forced)
  steer/  steer_isolate/  steer_x_ablate/  removal_probe/
  steer_probe/         steer ±dose along coord-probe readout dirs → Δ expected coordinate
6_generation/        LONG-HORIZON autoregressive rollout under intervention
  gen_head_ablation/   ★ ablate QK vs DLA vs random head-groups across a rollout; per window
                       track neighbour mass + validity (behaviour) AND per-layer coord-probe
                       R² of generated tokens (geometry). <graph>.json + .pdf (summary page +
                       one per-layer geometry page). Run: gen_head_ablation.py.
  removal_followup/    subspace removal on context, then free generation: does geometry
                       re-form and does behaviour hold (MODE=generate); plus all-layer
                       teacher-forced removal (MODE=alllayers)
_data/               loose sample activation caches (sample_<Model>.npz)
_logs/               battery / driver run logs, kept for provenance
```

## gen_head_ablation — method (the ★ experiment)

`src/scripts/analysis/gen_head_ablation.py`. For each condition
(`clean | ablate_induction | ablate_dla | ablate_random`) the chosen head group is
zeroed into `o_proj` at **all positions**; the model is seeded with `XCTX` context
steps and samples `GSTEPS` steps freely. Per generation window we record, from the
real output, **neighbour mass** (softmax mass on the last node's true graph
neighbours) and **validity** (fraction of sampled steps that are a true neighbour).
Then one final ablated forward over the generated tokens grabs residual-stream
node-means per layer, and a leave-one-node-out ridge **coord-probe R²** per
(layer, window) measures whether the grid still lives in the stream.

Head groups: top-K induction heads by `induction.json` "generic" score; top-K DLA
writers by `attribution/head_attribution_<graph>.json` "head_attr"; random = K heads
drawn from outside both sets (rank-matched control). It is the generate-mode
analogue of the teacher-forced `3_ablations` knockouts — so the two dissociation
axes (which circuit / next-token vs long-term) are directly comparable.

Run (from the pod, `/workspace/cross-model`, with the models cached):
```bash
PYTHONPATH=src HF_HOME=/workspace/hf \
GEN_MODEL=Llama GRAPH=square_grid XCTX=150 GSTEPS=150 NSEED=4 NWIN=6 KGROUP=15 \
  INDJSON=runs/induction-head/induction.json \
  DLAJSON=runs/induction-head/attribution/head_attribution_square_grid.json \
  OUTDIR=runs/induction-head/gen_head_ablation \
  python3 src/scripts/analysis/gen_head_ablation.py
```
`GRAPH ∈ {square_grid, ring, hex, days}`, `GEN_MODEL ∈ {Llama, Gemma, Qwen}`.
</content>
</invoke>
