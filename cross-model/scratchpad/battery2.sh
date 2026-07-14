#!/usr/bin/env bash
# Battery 2: Removal (necessity) + Cross-context capture. Two-pass HF_HOME for Gemma.
cd /workspace/cross-model
export PYTHONPATH=src
ROOT=runs/induction-head
LOG=$ROOT/battery2.log
HFW=/workspace/hf; HFR=/root/hf
: > "$LOG"
step(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- Removal: project probe subspace out ----------
for G in square_grid ring; do
  step "START removal_probe $G (Llama,Qwen)"
  HF_HOME=$HFW MODELS_FILTER=Llama,Qwen GRAPH=$G NWALKS=20 WLEN=300 \
    OUTDIR=$ROOT/removal_probe python3 src/scripts/analysis/removal_probe.py >>"$LOG" 2>&1
  step "START removal_probe $G (Gemma)"
  HF_HOME=$HFR MODELS_FILTER=Gemma GRAPH=$G NWALKS=20 WLEN=300 \
    OUTDIR=$ROOT/removal_probe python3 src/scripts/analysis/removal_probe.py >>"$LOG" 2>&1
  step "END removal_probe $G"
done

# ---------- Cross-context: capture per-bin means (combine done locally) ----------
for G in square_grid ring; do
  step "START cross_context capture $G (Llama,Qwen)"
  HF_HOME=$HFW MODE=capture MODELS_FILTER=Llama,Qwen GRAPH=$G NWALKS=24 WLEN=300 NBINS=8 \
    OUTDIR=$ROOT/cross_context python3 src/scripts/analysis/cross_context.py >>"$LOG" 2>&1
  step "START cross_context capture $G (Gemma)"
  HF_HOME=$HFR MODE=capture MODELS_FILTER=Gemma GRAPH=$G NWALKS=24 WLEN=300 NBINS=8 \
    OUTDIR=$ROOT/cross_context python3 src/scripts/analysis/cross_context.py >>"$LOG" 2>&1
  step "END cross_context capture $G"
done
step "ALL_DONE"
