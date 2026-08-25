#!/bin/bash
# story_combx = avg(story_trust, storyend_x): the cross-structure story vector
# with CONTENT HELD FIXED (same stories under both structures) -- the
# confound-controlled version of Sandra's item (1). Build (no GPU work), then
# validate on the stated-trust sweep and push-pull next to story_comb.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SKIP_OPT=1 python src/newvec_opt.py 2>&1 | tee nv4_avg.log

DIRS_FILTER="story_comb,story_combx,story_trust,storyend,storyend_x,warmth_b" \
  LAYERS=45 NPROBE=6 python src/sweep_all.py 2>&1 | tee nv4_sweep.log
cp out/sweep_all.json out/newvec_sweep4.json

DIRS_FILTER="story_comb,story_combx,story_trust,storyend_x,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv4_pushpull.log
cp out/pushpull.json out/pushpull_combx.json

echo NEWVEC4_ALL_DONE | tee -a nv4_pushpull.log
