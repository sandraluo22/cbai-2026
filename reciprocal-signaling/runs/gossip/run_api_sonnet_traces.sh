#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export ANTHROPIC_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
T=11111111111111111111; L=00000000000000000000
until grep -q API_SCRIPTED_DONE api_scripted.log 2>/dev/null; do sleep 60; done
for SEED in 20 21; do
  SCRIPT=1 THINK=2048 MODEL=claude-sonnet-5 VAR=curve SCHED="$T;$L" ROUNDS=20 STEPS=40 \
    SEED=$SEED OUT=duel_think/claude-sonnet-5_traces_s$SEED \
    python3 qsg_api.py 2>&1 | tee -a api_traces.log
done
echo API_TRACES_DONE | tee -a api_traces.log
