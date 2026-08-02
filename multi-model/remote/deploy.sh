#!/usr/bin/env bash
# Sync the three sibling dirs (cross-model/src, emotion, multi-model) to the GPU
# pod PRESERVING the cbai-2026 relative layout, install deps, run all stages in
# the background (survives SSH drops), poll, then pull runs/ back.
#
# Usage:
#   HOST=root@1.2.3.4 PORT=22 KEY=~/.ssh/id_ed25519 bash remote/deploy.sh [PRESET]
# Defaults target the H200 pod provided for this run.
set -euo pipefail

HOST="${HOST:?set HOST=root@<pod-ip>}"
PORT="${PORT:-19344}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
PRESET="${1:-gemma_qwen}"
REMOTE="${REMOTE:-/workspace/mm}"                 # cbai-2026 root on the pod
HF_HOME_REMOTE="${HF_HOME_REMOTE:-/workspace/hf}"  # reuse the pod's model cache
RUN_DIR="${RUN_DIR:-runs/main}"

SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
RSSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"       # cbai-2026 (local)

echo "[1/5] sync code -> $HOST:$REMOTE"
# --no-owner/group/perms: the pod's network FS forbids chown; without these the
# Mac's rsync -a aborts with 'Operation not permitted' (code 23).
rs() { rsync -rlDvz --no-owner --no-group --no-perms -e "$RSSH" \
         --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' "$@"; }
$SSH "$HOST" "mkdir -p $REMOTE/cross-model $REMOTE/emotion $REMOTE/multi-model $HF_HOME_REMOTE"
rs --delete "$LOCAL/cross-model/src/" "$HOST:$REMOTE/cross-model/src/"
# emotion: only the helper + recipe are imported; skip its big results/*.dat
rs --exclude 'results/' "$LOCAL/emotion/" "$HOST:$REMOTE/emotion/"
rs --delete --exclude 'runs/' "$LOCAL/multi-model/" "$HOST:$REMOTE/multi-model/"

echo "[2/5] install deps"
$SSH "$HOST" "cd $REMOTE/multi-model && bash remote/setup.sh"

echo "[3/5] launch all stages DETACHED (preset=$PRESET) on the H200"
$SSH "$HOST" "cd $REMOTE/multi-model && HF_HOME=$HF_HOME_REMOTE PRESET=$PRESET RUN_DIR=$RUN_DIR \
  nohup bash src/run_all.sh > run.log 2>&1 < /dev/null & echo \$! > run.pid; \
  echo launched pid \$(cat run.pid)"

echo "[4/5] poll run.log until it exits (reconnects each tick)"
while true; do
  alive=$($SSH "$HOST" "kill -0 \$(cat $REMOTE/multi-model/run.pid 2>/dev/null) 2>/dev/null && echo 1 || echo 0" 2>/dev/null || echo ssherr)
  if [ "$alive" = "1" ]; then
    $SSH "$HOST" "tail -n 1 $REMOTE/multi-model/run.log" 2>/dev/null || true
    sleep 25
  elif [ "$alive" = "0" ]; then
    break
  else
    echo "(ssh poll blip — job still running on pod)"; sleep 25
  fi
done

echo "[5/5] job exited; last log lines:"
$SSH "$HOST" "tail -n 8 $REMOTE/multi-model/run.log"
if ! $SSH "$HOST" "grep -q 'ALL DONE' $REMOTE/multi-model/run.log"; then
  echo "!! did NOT reach ALL DONE — inspect $HOST:$REMOTE/multi-model/run.log. Not pulling." >&2
  exit 1
fi
echo "==> pulling results to $LOCAL/multi-model/$RUN_DIR/"
mkdir -p "$LOCAL/multi-model/$RUN_DIR"
rs "$HOST:$REMOTE/multi-model/$RUN_DIR/" "$LOCAL/multi-model/$RUN_DIR/"
echo "==> done."
