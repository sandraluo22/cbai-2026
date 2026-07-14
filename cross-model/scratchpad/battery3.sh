#!/usr/bin/env bash
# Battery 3: the full follow-up program.
#   removal_followup (alllayers 3-model two-pass; generate Llama) + injection + composition + dynamic.
# Two-pass HF_HOME only where Gemma is involved (alllayers, dynamic). injection/composition/generate
# are Llama<->Qwen only -> single pass on /workspace/hf.
cd /workspace/cross-model
export PYTHONPATH=src
IH=runs/induction-head
LOG=$IH/battery3.log
HFW=/workspace/hf; HFR=/root/hf
: > "$LOG"
step(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- Removal follow-up A: all-layer removal (3 models, two-pass) ----------
for G in square_grid ring; do
  step "START removal alllayers $G (Llama,Qwen)"
  HF_HOME=$HFW MODE=alllayers MODELS_FILTER=Llama,Qwen GRAPH=$G NWALKS=20 WLEN=300 \
    OUTDIR=$IH/removal_followup python3 src/scripts/analysis/removal_followup.py >>"$LOG" 2>&1
  step "START removal alllayers $G (Gemma)"
  HF_HOME=$HFR MODE=alllayers MODELS_FILTER=Gemma GRAPH=$G NWALKS=20 WLEN=300 \
    OUTDIR=$IH/removal_followup python3 src/scripts/analysis/removal_followup.py >>"$LOG" 2>&1
  step "END removal alllayers $G"
done

# ---------- Removal follow-up B: remove-on-context then Llama generates ----------
for G in square_grid ring; do
  step "START removal generate $G (Llama)"
  HF_HOME=$HFW MODE=generate GEN_MODEL=Llama GRAPH=$G XCTX=150 GSTEPS=150 NSEED=4 GWIN=60 \
    OUTDIR=$IH/removal_followup python3 src/scripts/analysis/removal_followup.py >>"$LOG" 2>&1
  step "END removal generate $G"
done

# ---------- Injection: ridge map A_LA -> B_LB (both directions) ----------
for G in square_grid ring; do
  for PAIR in "Llama Qwen" "Qwen Llama"; do
    set -- $PAIR
    step "START injection $1->$2 $G"
    HF_HOME=$HFW A_TAG=$1 B_TAG=$2 GRAPH=$G NWALKS=12 NLB=10 \
      OUTDIR=$IH/injection python3 src/scripts/analysis/injection.py >>"$LOG" 2>&1
  done
  step "END injection $G"
done

# ---------- Composition: A generates n steps then B continues ----------
for G in square_grid ring; do
  step "START composition Llama->Qwen $G"
  HF_HOME=$HFW A_TAG=Llama B_TAG=Qwen GRAPH=$G NWALKS=8 PREFIX=120 NMAX=128 WINDOW=96 \
    NS=0,8,16,32,64,128 CTXLO=40 OUTDIR=runs/composition python3 src/scripts/analysis/composition.py >>"$LOG" 2>&1
  step "END composition $G"
done

# ---------- Dynamic: mid-context graph switch (3 models, two-pass) ----------
for SW in remove swap; do
  step "START dynamic $SW square_grid (Llama,Qwen)"
  HF_HOME=$HFW SWITCH=$SW MODELS_FILTER=Llama,Qwen GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "START dynamic $SW square_grid (Gemma)"
  HF_HOME=$HFR SWITCH=$SW MODELS_FILTER=Gemma GRAPH=square_grid NW=12 T=150 POST=220 WIN=60 STRIDE=30 \
    OUTDIR=runs/dynamic python3 src/scripts/analysis/dynamic_switch.py >>"$LOG" 2>&1
  step "END dynamic $SW"
done
step "ALL_DONE"
