#!/usr/bin/env bash
# Run the full negotiation pipeline for one preset.
#   bash scripts/run_all.sh smoke      # CPU plumbing check
#   bash scripts/run_all.sh default    # tier 1
#   bash scripts/run_all.sh tier2      # tier 2 (direction extracted below)
set -euo pipefail
cd "$(dirname "$0")/.."
PRESET="${1:-default}"

TIER2_PRESETS="tier2 gemma_tier2 smoke"   # smoke exercises the tier-2 path too
if [[ " $TIER2_PRESETS " == *" $PRESET "* ]]; then
  python src/steering.py --preset "$PRESET"            # 1. greed direction v
fi
python src/episodes.py  --preset "$PRESET"             # 2. corpus + A's acts
python src/shadow.py    --preset "$PRESET"             # 3. observer control
python src/probes.py    --preset "$PRESET"             # 4. headline R2(layer, turn)
python src/baselines.py --preset "$PRESET"             # 5. text-only baselines
python src/verbalize.py --preset "$PRESET"             # 6. introspection gap
python src/causal.py    --preset "$PRESET"             # 7. steer A along w
if [[ " $TIER2_PRESETS " == *" $PRESET "* ]]; then
  python src/samespace.py --preset "$PRESET"           # 8. cos(v, w)
fi
echo "all stages done -> runs/$PRESET/"
