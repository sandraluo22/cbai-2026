#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
DIRS_FILTER="FITTED trust,optim,optim_like,story_trust,story_warmth,story_comp,story_comb,story_combx,storyend_x,storyend,story_posavg,avg_all,nominate,nomfame,warmth_b,direct_b" LAYERS=45 NPROBE=6 python src/sweep_all.py 2>&1 | tee nv22_sweep.log
cp out/sweep_all.json out/newvec_sweep_final.json
DIRS_FILTER="FITTED trust,optim,optim_like,story_trust,story_warmth,story_comp,story_comb,story_combx,storyend_x,storyend,story_posavg,avg_all,nominate,nomfame,warmth_b,direct_b,random" python src/pushpull.py 2>&1 | tee nv22_pushpull.log
cp out/pushpull.json out/pushpull_final.json
DIRS_FILTER="storyend,storyend_x,nominate,nomfame,avg_all,optim,story_trust,story_warmth,warmth_b" \
  ARM_LAYERS=45 ADV_PAIR=Ana,Bruno ALPHA=0.5 python src/advisor_run.py 2>&1 | tee nv22_advisor.log
cp out/advisor_battery.json out/newvec_advisor_final.json
cp out/battery_heldout.json out/advisor_battery.json
CONDS="none,text+,FITTED trust+,FITTED trust-,optim+,optim-,optim_like+,story_trust+,story_warmth+,warmth_b+,random+" \
  OUTNAME=opengen2_final.json python src/opengen2.py 2>&1 | tee nv22_opengen2.log
echo RERUN_A_DONE | tee -a nv22_opengen2.log
