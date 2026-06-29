#!/usr/bin/env bash
# Regenerate every version-aware figure for a given dataset version.
# Usage:  bash src/scripts/run_plots.sh v2     (or v1)
set -uo pipefail
V="${1:-v1}"
export CM_VERSION="$V"
export PYTHONPATH=src
mkdir -p "runs/$V/overview" "runs/$V/slides"

run() { echo "=== [$V] $1 ==="; python3 "$1" 2>&1 | grep -viE "warning|loading weights|fetching" | tail -20; }

# best-2D family
run src/scripts/viz/best_2d_report.py
run src/scripts/viz/best_2d_allmodels.py
run src/scripts/viz/best_2d_peroccurrence.py
run src/scripts/viz/best_2d_layersweep.py
run src/scripts/viz/best_2d_slideshows.py
# context-length RSA (by-layer + significance)
run src/scripts/analysis/context_rsa.py
# cross-model context-vs-context RSA (by-layer + significance)
run src/scripts/viz/cross_context_rsa.py
# cross-model RSA heatmaps + significance
run src/scripts/viz/heatmaps_with_null.py
run src/scripts/viz/significance_heatmaps.py
# PCA slideshows (per-occurrence + node-mean, 3 models, all graphs)
run src/scripts/viz/perocc_slideshows.py
run src/scripts/viz/pca_nodemean_slideshows.py
run src/scripts/viz/rebuild.py
run src/scripts/viz/make_pca_pdf.py
echo "=== [$V] ALL PLOTS DONE ==="
