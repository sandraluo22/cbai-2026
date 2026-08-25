#!/bin/bash
# Pod A continuation: finish main-dim scaling (idempotent), wait for pod B's
# decoy sidecar, merge, then rebuild + validate as in run_newvec10.sh.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
DIMS=trust,comp,hon,rel DO_WARMTH=0 DO_ACCT=0 TARGET=200 \
  python src/scale_stories.py 2>&1 | tee nv21_scaleA.log
until grep -q SCALE_STORIES_DONE nv21b_sidecar.log 2>/dev/null; do sleep 60; done
python - <<'PY'
import json
main = json.load(open("out/stories.json"))
side = json.load(open("out/stories_sidecar.json"))
main["warmth"] = side["warmth"]
main["trust@acct"] = side["trust@acct"]
json.dump(main, open("out/stories.json", "w"))
print("merged:", {d: {c: len(v) for c, v in main[d].items()} for d in ("warmth", "trust@acct")})
PY
FAMS=story_trust,story_comp,story_hon,story_rel,story_warmth,story_trust@acct NITEM=200 \
  python src/build3.py 2>&1 | tee nv21_build.log
FAMS=storyend_x,storymid_x,story_all NITEM=200 python src/newvec_build.py 2>&1 | tee nv21_build2.log
SKIP_OPT=1 python src/newvec_opt.py 2>&1 | tee nv21_avg.log
python src/avg_variants.py 2>&1 | tee nv21_avgvar.log
DIRS_FILTER="story_trust,story_warmth,story_comp,story_hon,story_rel,warmth_b,FITTED trust,story_posavg" \
  LAYERS=45 NPROBE=6 python src/sweep_all.py 2>&1 | tee nv21_sweep.log
cp out/sweep_all.json out/newvec_sweep10.json
DIRS_FILTER="story_trust,story_warmth,story_comp,warmth_b,story_posavg,random" \
  python src/pushpull.py 2>&1 | tee nv21_pushpull.log
cp out/pushpull.json out/pushpull_200.json
DIRS="random,FITTED trust,optim,story_trust,story_warmth,story_comp,warmth_b" \
  python src/dissoc.py 2>&1 | tee nv21_dissoc.log
cp out/dissoc.json out/dissoc200.json
echo NEWVEC10_ALL_DONE | tee -a nv21_dissoc.log
