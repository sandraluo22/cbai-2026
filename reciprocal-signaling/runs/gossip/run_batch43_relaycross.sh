#!/bin/bash
# batch43 (pod A): contested-node relay — N=6, P1->P3, P2->P4, BOTH P3,P4 -> P5, P5 -> P6.
# P5 faces a live choice between chains; provenance tags are the only record link.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH42_DONE batch42.log 2>/dev/null; do sleep 180; done
T=11111111111111111111; L=00000000000000000000
for PROV in sender orig; do
for SEED in 0 1; do
  PROV=$PROV MODEL=Qwen32 VAR=relaycross NAGENTS=6 FRESH=1 ROUNDS=20 STEPS=60 SWITCH_AT=10 \
    SCHED="$T;$L" SEED=$SEED OUT=relaycross/Qwen32_${PROV}_s$SEED \
    python qsg_gossip_fast.py 2>&1 | tee -a batch43.log
done
done
echo BATCH43_DONE | tee -a batch43.log
