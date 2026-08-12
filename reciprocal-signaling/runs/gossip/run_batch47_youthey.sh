#!/bin/bash
# batch47 (pod D): "You have said: ... They have said: ..." memory format (PRESENT=youthey),
# which also gives agents their OWN history for the first time.
# Baselines to beat: dyad liar 1.00 capture, dyad truth 1.00, 5-agent duel100 0.52.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=2 FRESH=1 ROUNDS=20 STEPS=8 SCHED="$L" \
    SEED=$SEED OUT=youthey/Qwen32_dyad_liar_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch47.log
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=2 FRESH=1 ROUNDS=20 STEPS=8 SCHED="$T" \
    SEED=$SEED OUT=youthey/Qwen32_dyad_truth_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch47.log
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=3 FRESH=1 ROUNDS=20 STEPS=12 SCHED="$T;$L" \
    SEED=$SEED OUT=youthey/Qwen32_duel_n1_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch47.log
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=5 FRESH=1 ROUNDS=20 STEPS=40 SCHED="$T;$L" \
    SEED=$SEED OUT=youthey/Qwen32_duel_n3_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch47.log
done
echo BATCH47_DONE | tee -a batch47.log
