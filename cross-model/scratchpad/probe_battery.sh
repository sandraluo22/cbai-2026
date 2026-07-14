#!/usr/bin/env bash
# Full probe battery on the pod. Two-pass HF_HOME (Gemma -> /root/hf, others -> /workspace/hf).
# Each experiment: Llama+Qwen pass then Gemma pass; scripts merge models into their JSON so the
# final PDF has all three. Detached-run friendly: START/END markers per step for polling.
cd /workspace/cross-model
export PYTHONPATH=src
ROOT=runs/induction-head
LOG=$ROOT/probe_battery.log
HFW=/workspace/hf; HFR=/root/hf
: > "$LOG"
step(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- Phase A: grid-probe on hex/days activations (cross-graph NULL) ----------
for G in hex days; do
  step "START coord_decode gridnull $G (Llama,Qwen)"
  HF_HOME=$HFW MODELS_FILTER=Llama,Qwen GRAPH=$G LABELS=gridlabel SUFFIX=_gridnull NWALKS=24 WLEN=300 \
    OUTDIR=$ROOT/coord_decode python3 src/scripts/analysis/coord_decode.py >>"$LOG" 2>&1
  step "START coord_decode gridnull $G (Gemma)"
  HF_HOME=$HFR MODELS_FILTER=Gemma GRAPH=$G LABELS=gridlabel SUFFIX=_gridnull NWALKS=24 WLEN=300 \
    OUTDIR=$ROOT/coord_decode python3 src/scripts/analysis/coord_decode.py >>"$LOG" 2>&1
  step "END coord_decode gridnull $G"
done

# ---------- Phase B: ablation x coord-probe (does knockout degrade the probe?) ----------
for MODE in qk dla; do
  step "START ablation_probe HEADS_MODE=$MODE (Llama,Qwen)"
  HF_HOME=$HFW MODELS_FILTER=Llama,Qwen HEADS_MODE=$MODE GRAPHS=square_grid,ring ABLATE_K=15 NWALKS=20 WLEN=300 \
    OUTDIR=$ROOT/ablation_probe_$MODE python3 src/scripts/analysis/induction_ablation.py >>"$LOG" 2>&1
  step "START ablation_probe HEADS_MODE=$MODE (Gemma)"
  HF_HOME=$HFR MODELS_FILTER=Gemma HEADS_MODE=$MODE GRAPHS=square_grid,ring ABLATE_K=15 NWALKS=20 WLEN=300 \
    OUTDIR=$ROOT/ablation_probe_$MODE python3 src/scripts/analysis/induction_ablation.py >>"$LOG" 2>&1
  step "END ablation_probe HEADS_MODE=$MODE"
done

# ---------- Phase C: steer along probe directions -> logit impact ----------
for G in square_grid ring; do
  step "START steer_probe $G (Llama,Qwen)"
  HF_HOME=$HFW MODELS_FILTER=Llama,Qwen GRAPH=$G NWALKS=16 WLEN=300 SCALES=2,4,8 \
    OUTDIR=$ROOT/steer_probe python3 src/scripts/analysis/steer_probe.py >>"$LOG" 2>&1
  step "START steer_probe $G (Gemma)"
  HF_HOME=$HFR MODELS_FILTER=Gemma GRAPH=$G NWALKS=16 WLEN=300 SCALES=2,4,8 \
    OUTDIR=$ROOT/steer_probe python3 src/scripts/analysis/steer_probe.py >>"$LOG" 2>&1
  step "END steer_probe $G"
done
step "ALL_DONE"
