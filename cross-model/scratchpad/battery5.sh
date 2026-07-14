#!/usr/bin/env bash
# Battery 5: injection heatmaps (align R² | injected nbr mass), removal_generate per-layer lens,
# dynamic per-depth 2x3 slideshow. Two-pass HF_HOME only for dynamic (3 models).
cd /workspace/cross-model
export PYTHONPATH=src
IH=runs/induction-head
LOG=$IH/battery5.log
HFW=/workspace/hf; HFR=/root/hf
: > "$LOG"
step(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for G in square_grid ring; do
  for PAIR in "Llama Qwen" "Qwen Llama"; do
    set -- $PAIR
    step "START injection $1->$2 $G"
    HF_HOME=$HFW A_TAG=$1 B_TAG=$2 GRAPH=$G NWALKS=8 NLA=10 NLB=14 \
      OUTDIR=$IH/injection python3 src/scripts/analysis/injection.py >>"$LOG" 2>&1
  done
  step "END injection $G"
done

for G in square_grid ring; do
  step "START removal_generate $G (Llama)"
  HF_HOME=$HFW MODE=generate GEN_MODEL=Llama GRAPH=$G XCTX=150 GSTEPS=150 NSEED=4 GWIN=60 \
    OUTDIR=$IH/removal_followup python3 src/scripts/analysis/removal_followup.py >>"$LOG" 2>&1
  step "END removal_generate $G"
done

for SW in remove swap; do
  step "START dynamic $SW (Llama,Qwen)"
  HF_HOME=$HFW SWITCH=$SW MODELS_FILTER=Llama,Qwen GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 NDEPTH=12 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "START dynamic $SW (Gemma)"
  HF_HOME=$HFR SWITCH=$SW MODELS_FILTER=Gemma GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 NDEPTH=12 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "END dynamic $SW"
done
step "ALL_DONE"
