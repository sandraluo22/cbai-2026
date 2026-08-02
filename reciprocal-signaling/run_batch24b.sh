#!/bin/bash
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11110111101111011110 SEED=0 OUT=curve/Qwen32_p80_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11110111101111011110 SEED=1 OUT=curve/Qwen32_p80_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11110111101111011110 SEED=2 OUT=curve/Qwen32_p80_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111101111111110 SEED=0 OUT=curve/Qwen32_p90_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111101111111110 SEED=1 OUT=curve/Qwen32_p90_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111101111111110 SEED=2 OUT=curve/Qwen32_p90_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111111111 SEED=0 OUT=curve/Qwen32_p100_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111111111 SEED=1 OUT=curve/Qwen32_p100_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111111111 SEED=2 OUT=curve/Qwen32_p100_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00001111111111111111 SEED=0 OUT=curve/Qwen32_p80early_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00001111111111111111 SEED=1 OUT=curve/Qwen32_p80early_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00001111111111111111 SEED=2 OUT=curve/Qwen32_p80early_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111110000 SEED=0 OUT=curve/Qwen32_p80late_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111110000 SEED=1 OUT=curve/Qwen32_p80late_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111111111110000 SEED=2 OUT=curve/Qwen32_p80late_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111000011111111 SEED=0 OUT=curve/Qwen32_p80clust_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111000011111111 SEED=1 OUT=curve/Qwen32_p80clust_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11111111000011111111 SEED=2 OUT=curve/Qwen32_p80clust_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
echo BATCH24B_DONE
