#!/bin/bash
# FINAL REBUILD v2 — every derivation now reads at the literal appended-name token
# (fact-end for the nameless prior family). Order: quantify how much the defect
# mattered (readpos), then story vectors + story steering (Sandra's priority), then
# the rest, then both name-pair batteries.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python src/readpos.py 2>&1 | tee readpos.log
FAMS=story_trust,story_comp,story_hon,story_rel,story_trust@acct,story_trust@story,story_trust@acctnb,story_trust@storynb \
  NITEM=96 python src/build3.py 2>&1 | tee f2_stories.log
DIRS_FILTER="story_trust,story_trust@acct,story_trust@story,story_trust@acctnb,story_trust@storynb,direct_b,warmth_b" \
  LAYERS=45,52 NPROBE=6 python src/sweep_all.py 2>&1 | tee f2_storysteer.log
cp out/sweep_all.json out/story_steer_final2.json
echo STORY_STEER2_DONE | tee -a f2_storysteer.log

FAMS=direct_b,comp_b,hon_b,rel_b,warmth_b,benev,trustbehav,propensity,relational \
  NITEM=96 python src/build3.py 2>&1 | tee f2_rest.log
python src/prior_src.py 2>&1 | tee f2_prior.log
python src/prior_pooled.py 2>&1 | tee f2_pooled.log
NITEM=96 python src/convo_derive.py 2>&1 | tee f2_convo.log
LAYERS=35,45,52 NITEM=12 python src/fit2.py 2>&1 | tee f2_fit.log
LAYERS=45,52 NPROBE=6 python src/sweep_all.py 2>&1 | tee f2_sweep.log
python heat4.py 2>&1 | tee f2_heat.log

ADV_PAIR=Ana,Bruno ALPHA=0.5 python src/advisor_run.py 2>&1 | tee f2_battery_heldout.log
cp out/advisor_battery.json out/battery_heldout.json
ADV_PAIR=Bob,Mira ALPHA=0.5 python src/advisor_run.py 2>&1 | tee f2_battery_indist.log
cp out/advisor_battery.json out/battery_indist.json
echo FINAL2_ALL_DONE | tee -a f2_battery_indist.log
