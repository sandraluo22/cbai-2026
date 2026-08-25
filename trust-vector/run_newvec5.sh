#!/bin/bash
# story_first: same name-throughout stories, vector read at the FIRST in-story
# mention (early position). Discriminates the two accounts of the second-slot
# steering gain: if an early-read vector still shows Bruno >> Ana in the plain
# battery, the gain is a property of the test bed, not of late derivation reads.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FAMS=story_first NITEM=96 python src/newvec_build.py 2>&1 | tee nv6_build.log

DIRS_FILTER="story_first,story_trust" ARM_LAYERS=45 ADV_PAIR=Ana,Bruno ALPHA=0.5 \
  python src/advisor_run.py 2>&1 | tee nv6_advisor.log
cp out/advisor_battery.json out/newvec_advisor6.json

echo NEWVEC5_ALL_DONE | tee -a nv6_advisor.log
