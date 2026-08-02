#!/usr/bin/env bash
# Run the LLM extension (real Llama/Qwen spymaster<->guesser) on a GPU pod.
# The reference games (game1/game2) are CPU-only and don't need this.
#   HOST=root@1.2.3.4 PORT=22 KEY=~/.ssh/id_ed25519 bash remote/deploy.sh
set -euo pipefail
HOST="${HOST:?set HOST=root@<pod-ip>}"; PORT="${PORT:-19344}"; KEY="${KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE:-/workspace/mm}"; HF="${HF_HOME_REMOTE:-/workspace/hf}"; RUN_DIR="${RUN_DIR:-runs/llm}"
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
RSSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"
rs() { rsync -rlDvz --no-owner --no-group --no-perms -e "$RSSH" \
         --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' "$@"; }

echo "[1/3] sync games-2 + cross-model/src -> $HOST:$REMOTE"
$SSH "$HOST" "mkdir -p $REMOTE/cross-model $REMOTE/games-2 $HF"
rs --delete "$LOCAL/cross-model/src/" "$HOST:$REMOTE/cross-model/src/"
rs --delete --exclude 'runs/' "$LOCAL/games-2/" "$HOST:$REMOTE/games-2/"
$SSH "$HOST" "pip install -q --break-system-packages 'transformers>=4.56' numpy matplotlib scipy 2>/dev/null || pip install -q 'transformers>=4.56' numpy matplotlib scipy"

echo "[2/3] run Codenames between real LLMs (A=${A_MODEL:-Llama} B=${B_MODEL:-Qwen} L=${LEVEL:-2}) via ${SCRIPT:-src/game_llm.py}"
$SSH "$HOST" "cd $REMOTE/games-2 && HF_HOME=$HF PYTHONPATH=src \
  A_MODEL=${A_MODEL:-Llama} B_MODEL=${B_MODEL:-Qwen} LEVEL=${LEVEL:-2} RUN_DIR=$RUN_DIR \
  MODELS=${MODELS:-LlamaInst,QwenInst} M=${M:-4} ROUNDS=${ROUNDS:-4} GAMES=${GAMES:-8} SPY_MEMORY=${SPY_MEMORY:-0} SPY_SEES=${SPY_SEES:-remaining} \
  nohup python3 ${SCRIPT:-src/game_llm.py} > llm.log 2>&1 < /dev/null & echo launched \$!; sleep 5; tail -3 llm.log"

echo "[3/3] when llm.log shows DONE, pull:"
echo "  rsync -rlDvz --no-owner --no-group --no-perms -e \"$RSSH\" $HOST:$REMOTE/games-2/$RUN_DIR/ $LOCAL/games-2/$RUN_DIR/"
