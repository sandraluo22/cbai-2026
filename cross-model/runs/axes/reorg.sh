#!/usr/bin/env bash
# Reproduce the axes/ grouping: pull the divider-axis investigation out of induction-head/2_probes
# (where it was first written) into its own themed top-level tree. Run from runs/. Idempotent-ish.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p axes/1_decomposition axes/2_geometry axes/3_causal axes/4_circuits
P=induction-head/2_probes
mv_safe(){ [ -e "$P/$1" ] && mv "$P/$1" "$2/" && echo "  $1 -> $2/" || true; }
mv_safe divider_basis        axes/1_decomposition
mv_safe context_layer_probe  axes/1_decomposition
mv_safe axis_geometry        axes/2_geometry
for d in axis_under_ablation mode_ablate axis_steer axis_cut_sweep axis_cut_sweep_fine; do mv_safe "$d" axes/3_causal; done
mv_safe head_axis_sweep      axes/4_circuits
echo "axes/ reorg complete."
