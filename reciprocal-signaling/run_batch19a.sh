#!/bin/bash
# batch19a (.149): forced per-player notes x graded 80/20. Waits for batch18a.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch18[a].sh' >/dev/null; do sleep 60; done
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.8 P2REL=0.2 NOTES=peragent SEED=$SEED OUT=notes/Qwen32_g8020_peragent_s$SEED python qsg_gossip.py 2>&1 | tee -a batch19.log
done
echo BATCH19A_DONE
