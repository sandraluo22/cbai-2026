#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export ANTHROPIC_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
A=11111111110000000000; B=00000000001111111111
for SEED in 0 1; do
  MODEL=claude-sonnet-5 VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=duelrev/claude-sonnet-5_s$SEED python3 qsg_api.py 2>&1 | tee -a api_reversal.log
done
echo API_REVERSAL_DONE | tee -a api_reversal.log
