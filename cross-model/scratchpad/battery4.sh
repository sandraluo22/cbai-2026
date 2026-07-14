#!/usr/bin/env bash
# Battery 4: injection L_A x L_B heatmaps + composition v2 (sampled, layer-wise) + dynamic v2
# (per-occurrence probe-axis scatter). Two-pass HF_HOME only for dynamic (3 models).
cd /workspace/cross-model
export PYTHONPATH=src
IH=runs/induction-head
LOG=$IH/battery4.log
HFW=/workspace/hf; HFR=/root/hf
: > "$LOG"
step(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- Injection heatmaps (Llama<->Qwen, single pass) ----------
for G in square_grid ring; do
  for PAIR in "Llama Qwen" "Qwen Llama"; do
    set -- $PAIR
    step "START injection $1->$2 $G"
    HF_HOME=$HFW A_TAG=$1 B_TAG=$2 GRAPH=$G NWALKS=8 NLA=6 NLB=8 \
      OUTDIR=$IH/injection python3 src/scripts/analysis/injection.py >>"$LOG" 2>&1
  done
  step "END injection $G"
done

# ---------- Composition v2 (sampled) ----------
for G in square_grid ring; do
  step "START composition $G"
  HF_HOME=$HFW A_TAG=Llama B_TAG=Qwen GRAPH=$G NWALKS=8 PREFIX=120 NMAX=128 WINDOW=96 \
    NS=0,8,16,32,64,128 CTXLO=40 TEMP=1.0 OUTDIR=runs/composition python3 src/scripts/analysis/composition.py >>"$LOG" 2>&1
  step "END composition $G"
done

# ---------- Dynamic v2 (3 models, two-pass) ----------
for SW in remove swap; do
  step "START dynamic $SW (Llama,Qwen)"
  HF_HOME=$HFW SWITCH=$SW MODELS_FILTER=Llama,Qwen GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "START dynamic $SW (Gemma)"
  HF_HOME=$HFR SWITCH=$SW MODELS_FILTER=Gemma GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "END dynamic $SW"
done
step "ALL_DONE"
