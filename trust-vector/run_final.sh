#!/bin/bash
# FINAL REBUILD. Priority order per Sandra: story vectors + story steering first
# (report lands as soon as that finishes), then the rest, then the two-name-pair
# advisor battery (held-out Ana/Bruno vs in-distribution Bob/Mira).
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# A. clean story bank (side-name controlled) and story vectors, then story steering
cp out/stories.json out/stories_pre_final.json 2>/dev/null
NSTORY=64 python src/gen_stories.py 2>&1 | tee f_gen.log
FAMS=story_trust,story_comp,story_hon,story_rel,story_trust@acct,story_trust@story,story_trust@acctnb,story_trust@storynb \
  NITEM=96 python src/build3.py 2>&1 | tee f_stories.log
DIRS_FILTER="story_trust,story_trust@acct,story_trust@story,story_trust@acctnb,story_trust@storynb,direct_b,warmth_b" \
  LAYERS=45,52 NPROBE=6 python src/sweep_all.py 2>&1 | tee f_storysteer.log
cp out/sweep_all.json out/story_steer_final.json
echo STORY_STEER_DONE | tee -a f_storysteer.log

# B. everything else
FAMS=direct_b,comp_b,hon_b,rel_b,warmth_b,benev,trustbehav,propensity,relational \
  NITEM=96 python src/build3.py 2>&1 | tee f_rest.log
python src/prior_src.py 2>&1 | tee f_prior.log
python src/prior_pooled.py 2>&1 | tee f_pooled.log
NITEM=96 python src/convo_derive.py 2>&1 | tee f_convo.log
LAYERS=35,45,52 NITEM=12 python src/fit2.py 2>&1 | tee f_fit.log
LAYERS=45,52 NPROBE=6 python src/sweep_all.py 2>&1 | tee f_sweep.log
python heat4.py 2>&1 | tee f_heat.log

# C. the two-name-pair battery
ADV_PAIR=Ana,Bruno ALPHA=0.5 python src/advisor_run.py 2>&1 | tee f_battery_heldout.log
cp out/advisor_battery.json out/battery_heldout.json
ADV_PAIR=Bob,Mira ALPHA=0.5 python src/advisor_run.py 2>&1 | tee f_battery_indist.log
cp out/advisor_battery.json out/battery_indist.json
echo FINAL_ALL_DONE | tee -a f_battery_indist.log
