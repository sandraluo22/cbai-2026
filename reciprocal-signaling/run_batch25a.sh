#!/bin/bash
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch24[a].sh' >/dev/null; do sleep 60; done
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' PRESENT=anon SEED=0 OUT=contrast/Qwen32_a80_b0_anon_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' PRESENT=anon SEED=1 OUT=contrast/Qwen32_a80_b0_anon_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' PRESENT=separate SEED=0 OUT=contrast/Qwen32_a80_b0_sep_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' PRESENT=separate SEED=1 OUT=contrast/Qwen32_a80_b0_sep_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' TALLY=true SEED=0 OUT=contrast/Qwen32_a80_b0_sum_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00000000000000000000' TALLY=true SEED=1 OUT=contrast/Qwen32_a80_b0_sum_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' PRESENT=anon SEED=0 OUT=contrast/Qwen32_a80_b40_anon_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' PRESENT=anon SEED=1 OUT=contrast/Qwen32_a80_b40_anon_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' PRESENT=separate SEED=0 OUT=contrast/Qwen32_a80_b40_sep_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' PRESENT=separate SEED=1 OUT=contrast/Qwen32_a80_b40_sep_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' TALLY=true SEED=0 OUT=contrast/Qwen32_a80_b40_sum_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;00101001010010100101' TALLY=true SEED=1 OUT=contrast/Qwen32_a80_b40_sum_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' PRESENT=anon SEED=0 OUT=contrast/Qwen32_a80_b80_anon_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' PRESENT=anon SEED=1 OUT=contrast/Qwen32_a80_b80_anon_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' PRESENT=separate SEED=0 OUT=contrast/Qwen32_a80_b80_sep_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' PRESENT=separate SEED=1 OUT=contrast/Qwen32_a80_b80_sep_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' TALLY=true SEED=0 OUT=contrast/Qwen32_a80_b80_sum_s0 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10111101111011110111' TALLY=true SEED=1 OUT=contrast/Qwen32_a80_b80_sum_s1 python qsg_gossip_fast.py 2>&1 | tee -a batch25.log
echo BATCH25A_DONE
