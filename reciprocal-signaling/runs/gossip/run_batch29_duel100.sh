#!/bin/bash
# batch29 (pod .149): waits for all 4 Llama xmodel transcripts, then 100/0 duel — Llama70.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until [ "$(ls -d xmodel/Llama70_* 2>/dev/null | wc -l)" -ge 4 ] && ! pgrep -f "qsg_gossip_fas[t]" > /dev/null; do sleep 180; done
A=11111111111111111111; B=00000000000000000000
for SEED in 0 1; do
  MODEL=Llama70 VAR=curve FRESH=1 ROUNDS=20 STEPS=40 SCHED="$A;$B" SEED=$SEED \
    OUT=duel100/Llama70_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch29.log
done
echo BATCH29_DONE | tee -a batch29.log
