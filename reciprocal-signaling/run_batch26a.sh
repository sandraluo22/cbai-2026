#!/bin/bash
# batch26a (.149): cross-model — Llama70, harmonized protocol (STEPS=40, no earlystop)
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch25[a].sh' >/dev/null; do sleep 60; done
for SEED in 0 1; do
  FRESH=1 ROUNDS=20 STEPS=40 VAR=curve SCHED='11110111101111011110;01000010000100001000' MODEL=Llama70 SEED=$SEED OUT=xmodel/Llama70_none_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch26.log
  FRESH=1 ROUNDS=20 STEPS=40 VAR=curve SCHED='11110111101111011110;01000010000100001000' MODEL=Llama70 NOTES=append SEED=$SEED OUT=xmodel/Llama70_notes_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch26.log
done
echo BATCH26A_DONE
