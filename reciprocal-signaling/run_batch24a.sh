#!/bin/bash
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python replay_identity.py earlystop/Qwen32_duel_s0/gossip_s0_transcript.jsonl earlystop/Qwen32_duel_s1/gossip_s1_transcript.jsonl 2>&1 | tee -a batch24.log
python replay_launder.py fresh50/Qwen32_misinformed_all/gossip_s0_transcript.jsonl earlystop/Qwen32_misinformed_all_s0/gossip_s0_transcript.jsonl 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00000000000000000000 SEED=0 OUT=curve/Qwen32_p0_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00000000000000000000 SEED=1 OUT=curve/Qwen32_p0_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=00000000000000000000 SEED=2 OUT=curve/Qwen32_p0_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10000100001000010000 SEED=0 OUT=curve/Qwen32_p20_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10000100001000010000 SEED=1 OUT=curve/Qwen32_p20_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10000100001000010000 SEED=2 OUT=curve/Qwen32_p20_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10100101001010010100 SEED=0 OUT=curve/Qwen32_p40_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10100101001010010100 SEED=1 OUT=curve/Qwen32_p40_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=10100101001010010100 SEED=2 OUT=curve/Qwen32_p40_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11010110101101011010 SEED=0 OUT=curve/Qwen32_p60_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11010110101101011010 SEED=1 OUT=curve/Qwen32_p60_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED=11010110101101011010 SEED=2 OUT=curve/Qwen32_p60_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000'  SEED=0 OUT=factorial/Qwen32_none_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000'  SEED=1 OUT=factorial/Qwen32_none_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000'  SEED=2 OUT=factorial/Qwen32_none_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=true SEED=0 OUT=factorial/Qwen32_tallyT_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=true SEED=1 OUT=factorial/Qwen32_tallyT_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=true SEED=2 OUT=factorial/Qwen32_tallyT_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=false SEED=0 OUT=factorial/Qwen32_tallyF_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=false SEED=1 OUT=factorial/Qwen32_tallyF_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' TALLY=false SEED=2 OUT=factorial/Qwen32_tallyF_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' NOTES=evalforce SEED=0 OUT=factorial/Qwen32_evalf_s0 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' NOTES=evalforce SEED=1 OUT=factorial/Qwen32_evalf_s1 python qsg_gossip.py 2>&1 | tee -a batch24.log
FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=curve SCHED='11110111101111011110;10000100001000010000' NOTES=evalforce SEED=2 OUT=factorial/Qwen32_evalf_s2 python qsg_gossip.py 2>&1 | tee -a batch24.log
echo BATCH24A_DONE
