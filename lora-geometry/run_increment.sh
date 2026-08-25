#!/bin/bash
# Incremental re-run of concepts whose manipulation check failed and whose system
# prompt has since been strengthened. Scorers are UNCHANGED -- only the
# manipulation was strengthened, so a pass here is a pass on the original ruler.
#
# Safe to run after run_full.sh: gen_data and build_vecs merge instead of
# overwriting, the NEUTRAL reference is reused rather than regenerated (so these
# concepts share the main run's reference), and train_lora skips adapters that
# already exist.
#
# `direct` is NOT here and is not coming back: its scorer is -caveats and the
# neutral arm already emits ~0 caveats on this prompt pool, so it has no headroom
# below. That needs a different prompt pool, which would invalidate every other
# vector.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export MAX_NEW="${MAX_NEW:-700}" MAXLEN="${MAXLEN:-768}"
export INIT_SEEDS="${INIT_SEEDS:-0,1}" DATA_SEEDS="${DATA_SEEDS:-0,1}"
export ONLY="${ONLY:-formal,optimistic,overconfident}"

mkdir -p out logs
python src/gen_data.py    2>&1 | tee logs/21_gen_inc.log
# stop early if the strengthened prompts still do not clear the gate -- there is
# no point training adapters for a behaviour that did not happen
python - <<'PY'
import json, os, sys
rej = set(json.load(open("out/rejected.json")))
only = os.environ["ONLY"].split(",")
still = [n for n in only if n in rej]
print(f"[inc] still rejected: {still or 'none'}")
if len(still) == len(only):
    print("[inc] every re-run concept still fails the gate; stopping.")
    sys.exit(3)
PY
python src/build_vecs.py  2>&1 | tee logs/22_vec_inc.log
unset ONLY                      # train/analyse over everything that now passes
python src/train_lora.py  2>&1 | tee logs/23_lora_inc.log
python src/wspace.py      2>&1 | tee logs/24_wspace_inc.log
python src/induced.py     2>&1 | tee logs/25_induced_inc.log
for POS in response last; do
  for BLOCK in 0 1; do
    POS=$POS BLOCK=$BLOCK python src/compare.py 2>&1 \
      | tee "logs/26_compare_inc_${POS}_b${BLOCK}.log"
  done
done
echo INCREMENT_DONE
