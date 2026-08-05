#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export (ANTHROPIC|OPENAI)_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
D=11111111111111111111; Z=00000000000000000000
A=11110111101111011110; B=01000010000100001000
M=$1; LOG=api_words_$1.log
WORDS=1 MODEL=$M VAR=curve SCHED="$D;$Z" ROUNDS=20 STEPS=40 SEED=0 OUT=words/${M}_duel100w_s0 python3 qsg_api.py 2>&1 | tee -a $LOG
WORDS=1 MODEL=$M VAR=curve SCHED="$D;$Z" ROUNDS=20 STEPS=40 SEED=1 OUT=words/${M}_duel100w_s1 python3 qsg_api.py 2>&1 | tee -a $LOG
WORDS=1 MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=0 OUT=words/${M}_8020w_s0 python3 qsg_api.py 2>&1 | tee -a $LOG
WORDS=1 MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=1 OUT=words/${M}_8020w_s1 python3 qsg_api.py 2>&1 | tee -a $LOG
echo API_WORDS_${M}_DONE | tee -a $LOG
