#!/usr/bin/env bash
# Reorganize runs/induction-head into themed groups following the investigation's narrative:
#   circuits (where the heads are) -> probes (geometry decodability) -> ablations (causal knockouts)
#   -> patching -> steering.  Leaf directory names are preserved verbatim (greppable, reversible);
#   loose root files are foldered.  Run from runs/induction-head/.  Idempotent-ish: skips missing srcs.
set -euo pipefail
cd "$(dirname "$0")"

mv_safe () { [ -e "$1" ] && mv "$1" "$2/" && echo "  $1 -> $2/" || true; }

mkdir -p 1_circuits 2_probes 3_ablations 4_patching 5_steering 6_generation _data _logs

# --- 1. circuits: identifying / characterizing the induction + writer heads ---
mkdir -p 1_circuits/induction_heads 1_circuits/sanity_ov
mv_safe induction.json        1_circuits/induction_heads
mv_safe induction_heads.pdf   1_circuits/induction_heads
mv_safe qk_histogram.pdf      1_circuits/induction_heads
mv_safe sanity_ov.json        1_circuits/sanity_ov
for d in head_sweep attribution copying atlas outlier node_output; do mv_safe "$d" 1_circuits; done

# --- 2. probes: geometry decodability (the representation we trust) ---
for d in coord_decode cross_model_sim cross_layer_heatmap; do mv_safe "$d" 2_probes; done

# --- 3. ablations: causal knockouts + RSA controls ---
for d in ablation ablation_allqk ablation_dla ablation_logit ablation_rsa \
         layer_ablation positional_ablation posablation_after posablation_exact \
         puncture_rsa rsa_shuffle context_traj; do mv_safe "$d" 3_ablations; done

# --- 4. patching: activation patch / topological patch ---
for d in patch_swap topo_patch; do mv_safe "$d" 4_patching; done

# --- 5. steering: causal read-out of the map (single-token / teacher-forced) ---
for d in steer steer_isolate steer_x_ablate steer_probe removal_probe; do mv_safe "$d" 5_steering; done

# --- 6. generation: long-horizon autoregressive rollout under intervention ---
#   removal_followup: subspace removal + free generation; gen_head_ablation: induction/DLA
#   head-group ablation over a rollout, tracking behaviour (nbr mass/validity) + probe R².
for d in removal_followup gen_head_ablation; do mv_safe "$d" 6_generation; done

# --- data: loose sample activation caches ---
for f in sample_Gemma.npz sample_Llama.npz sample_Qwen.npz; do mv_safe "$f" _data; done

# --- logs: loose battery / driver run logs (kept for provenance, out of the way) ---
for f in battery2.log battery3.log battery4.log battery5.log battery6.log \
         probe_battery.log probe_battery.out; do mv_safe "$f" _logs; done

echo "reorg complete."
