#!/bin/bash
# batch32 (pod .140): trust-transfer suite. Phase 1 = 100/0 gibberish duel builds reputation
# (rounds 1-10), phase 2 switches to real content: weak-prior trivia (towers keep roles),
# strong-prior override (trusted tower pushes falsehoods), 2x2-digit products, and the
# elephant-style knower quiz (same riddle 10 rounds, no reveals) +/- reputation phase.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH31_DONE batch31.log 2>/dev/null; do sleep 120; done
for SEED in 0 1; do
  MODEL=Qwen32 VAR=trivia          FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED OUT=transfer/Qwen32_trivia_s$SEED    python qsg_gossip_fast.py 2>&1 | tee -a batch32.log
  MODEL=Qwen32 VAR=trivia_override FRESH=1 ROUNDS=15 STEPS=40 SWITCH_AT=10 SEED=$SEED OUT=transfer/Qwen32_override_s$SEED  python qsg_gossip_fast.py 2>&1 | tee -a batch32.log
  MODEL=Qwen32 VAR=mathsw          FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED OUT=transfer/Qwen32_mathsw_s$SEED    python qsg_gossip_fast.py 2>&1 | tee -a batch32.log
  MODEL=Qwen32 VAR=knower          FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED OUT=transfer/Qwen32_knower_s$SEED    python qsg_gossip_fast.py 2>&1 | tee -a batch32.log
  MODEL=Qwen32 VAR=knower          FRESH=1 ROUNDS=10 STEPS=40 SWITCH_AT=0  SEED=$SEED OUT=transfer/Qwen32_knower0_s$SEED   python qsg_gossip_fast.py 2>&1 | tee -a batch32.log
done
echo BATCH32_DONE | tee -a batch32.log
