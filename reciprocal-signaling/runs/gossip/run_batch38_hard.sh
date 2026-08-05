#!/bin/bash
# batch38 (pod A): HARD transfer rerun — genuinely uncertain trivia + hard knower question,
# with a solo-baseline arm (no towers) to measure the model's own priors per item.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH34_DONE batch34.log 2>/dev/null; do sleep 180; done
python - <<'PYEOF' 2>&1 | tee -a batch38.log
import json, random, numpy as np
import qsg_gossip_fast as Q
from run_games import load
model, tok, _ = load("Qwen32")
rng = random.Random(0)
out = {}
for q, opts, ans in Q.TRIVIA_HARD + Q.KNOWER_HARD:
    ids = [tok(o, add_special_tokens=False)["input_ids"][0] for o in opts]
    msg = Q.user_msg(2, list(opts), [], {}, 1, None, rng, False,
                     dict(q=q, fixed=False), "", "")
    prob = Q.belief(model, tok, msg, ids)
    out[q] = dict(opts=opts, ans=ans, p=[round(float(x), 3) for x in prob],
                  p_correct=round(float(prob[opts.index(ans)]), 3))
    print(f"[solo] p(correct)={out[q]['p_correct']:.2f}  {q[:60]}", flush=True)
json.dump(out, open("transfer/hard_solo_baseline.json", "w"), indent=1)
print(f"[solo] mean p(correct) = {np.mean([v['p_correct'] for v in out.values()]):.3f}", flush=True)
PYEOF
for SEED in 0 1; do
  HARD=1 MODEL=Qwen32 VAR=trivia FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED \
    OUT=transfer/Qwen32_triviahard_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch38.log
  HARD=1 MODEL=Qwen32 VAR=knower FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED \
    OUT=transfer/Qwen32_knowerhard_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch38.log
  HARD=1 MODEL=Qwen32 VAR=knower FRESH=1 ROUNDS=10 STEPS=40 SWITCH_AT=0 SEED=$SEED \
    OUT=transfer/Qwen32_knowerhard0_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch38.log
done
echo BATCH38_DONE | tee -a batch38.log
