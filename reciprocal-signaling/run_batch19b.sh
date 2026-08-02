#!/bin/bash
# batch19b (.140): forced per-player notes x betrayal+window3. Waits for batch18b.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch18[b].sh' >/dev/null; do sleep 60; done
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 WINDOW=3 SWITCH_AT=10 ROUNDS=20 VAR=betrayal NOTES=peragent SEED=$SEED OUT=notes/Qwen32_betrayal_w3_peragent_s$SEED python qsg_gossip.py 2>&1 | tee -a batch19.log
done
echo BATCH19B_DONE
