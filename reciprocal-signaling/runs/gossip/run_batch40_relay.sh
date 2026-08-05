#!/bin/bash
# batch40 (pod A): calibration->relay chains with provenance manipulation.
# N=8: P1 truth tower, P2 liar tower; calibration rounds 1-10 (all-contact, dense 72 steps),
# relay rounds 11-20 (P1->P3->P5->P7 truth chain, P2->P4->P6->P8 lie chain; 12 sweeps).
# PROV in sender|orig|none. Manipulation check (query + behavioral) logged at r10.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH38_DONE batch38.log 2>/dev/null; do sleep 180; done
T=11111111111111111111; L=00000000000000000000
for PROV in sender orig none; do
for SEED in 0 1; do
  PROV=$PROV MODEL=Qwen32 VAR=relay NAGENTS=8 FRESH=1 ROUNDS=20 STEPS=72 SWITCH_AT=10 \
    SCHED="$T;$L" SEED=$SEED OUT=relay/Qwen32_${PROV}_s$SEED \
    python qsg_gossip_fast.py 2>&1 | tee -a batch40.log
done
done
echo BATCH40_DONE | tee -a batch40.log
