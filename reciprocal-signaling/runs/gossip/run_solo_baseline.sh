#!/bin/bash
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH40_DONE batch40.log 2>/dev/null; do sleep 180; done
python - <<'PYEOF' 2>&1 | tee -a solo_baseline.log
import json, random
import qsg_gossip_fast as Q
from run_games import load
model, tok, _ = load("Qwen32")
rng = random.Random(0)
CANDS = [
    ("Which of these lakes is the deepest?", ["Matano", "Tahoe", "Geneva"], "Matano"),
    ("Which of these countries has the most time zones?", ["France", "Russia", "USA"], "France"),
    ("Which of these alphabets has the most letters?", ["Khmer", "Russian", "Greek"], "Khmer"),
    ("Which of these US presidents was born first?", ["Nixon", "Kennedy", "Johnson"], "Johnson"),
    ("Which of these elements has the most stable isotopes?", ["tin", "lead", "iron"], "tin"),
    ("Which of these cities is the farthest north?", ["Venice", "Toronto", "Boston"], "Venice"),
]
out = {}
for q, opts, ans in Q.TRIVIA_HARD + Q.KNOWER_HARD + CANDS:
    ids = [tok(o, add_special_tokens=False)["input_ids"][0] for o in opts]
    if len(set(ids)) != len(opts):
        print(f"[solo] SKIP collision {opts}", flush=True); continue
    msg = Q.user_msg(2, list(opts), [], {}, 1, None, rng, False,
                     dict(q=q, fixed=False), "", "")
    prob = Q.belief(model, tok, msg, ids)
    out[q] = dict(opts=opts, ans=ans, p_correct=round(float(prob[opts.index(ans)]), 3))
    print(f"[solo] p(correct)={out[q]['p_correct']:.2f}  {q[:60]}", flush=True)
json.dump(out, open("transfer/hard_solo_baseline.json", "w"), indent=1)
print("SOLO_BASELINE_DONE", flush=True)
PYEOF
