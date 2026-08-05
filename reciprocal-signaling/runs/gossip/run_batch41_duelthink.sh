#!/bin/bash
# batch41 (pod C): THINKING duels — 1 truthteller + 1 liar + n neutrals, n = 1,2,3
# (NAGENTS 3,4,5), Qwen3-32B native thinking, 20 fresh-label rounds, 2 sweeps/round.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH37_DONE batch37.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for NA in 3 4 5; do
  STEPS=$((2 * NA * (NA - 1)))
  NAGENTS=$NA MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=$STEPS SCHED="$T;$L" SEED=0 \
    OUT=duel_think/Qwen32think_duel_n$((NA-2))_s0 python qsg_local_think.py 2>&1 | tee -a batch41.log
done
echo BATCH41_DONE | tee -a batch41.log
