#!/usr/bin/env bash
# Drive the whole job on a runpod GPU from the Mac:
#   rsync code up -> install deps -> extract activations -> pull results back.
#
# Usage:
#   remote/run.sh <ssh-host> [extract args...]
# Examples:
#   remote/run.sh runpod-qwen                       # smoke (200 examples)
#   remote/run.sh runpod-qwen --split all --limit 0 --batch-size 32   # full
#
# <ssh-host> is an alias in ~/.ssh/config (e.g. runpod-qwen) or user@host.
set -euo pipefail

HOST="${1:?usage: run.sh <ssh-host> [extract args...]}"; shift || true
EXTRACT_ARGS="$*"
# Use the big /workspace volume on runpod for code, HF cache, and results —
# the root fs is typically tiny (~30G). Override with REMOTE_DIR / HF_HOME.
REMOTE_DIR="${REMOTE_DIR:-/workspace/emotion}"
REMOTE_HF="${HF_HOME:-/workspace/hf_cache}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> syncing code to $HOST:$REMOTE_DIR"
ssh "$HOST" "mkdir -p $REMOTE_DIR $REMOTE_HF"
rsync -rlDvz --no-owner --no-group --no-perms \
    --exclude results/ --exclude __pycache__/ --exclude '*.dat' \
    "$LOCAL_DIR"/ "$HOST:$REMOTE_DIR"/

echo "==> installing deps"
ssh "$HOST" "cd $REMOTE_DIR && bash remote/setup.sh"

echo "==> launching extraction DETACHED (survives SSH drops) — args: ${EXTRACT_ARGS:-<defaults>}"
# nohup + backgrounded with redirected stdio keeps the job alive even if our
# SSH session blips (runpod proxies reset long-lived connections). We capture
# python's OWN pid via $! (no setsid — setsid forks, so $! would be the wrong
# pid). We then poll the log/pid over fresh SSH connections.
ssh "$HOST" "cd $REMOTE_DIR && HF_HOME=$REMOTE_HF nohup \
    python -u extract_activations.py ${EXTRACT_ARGS} > run.log 2>&1 < /dev/null & \
    echo \$! > run.pid; echo launched pid \$(cat run.pid)"

echo "==> polling run.log until the job exits (reconnects each tick)"
while true; do
    alive=$(ssh "$HOST" "kill -0 \$(cat $REMOTE_DIR/run.pid 2>/dev/null) 2>/dev/null \
        && echo 1 || echo 0" 2>/dev/null || echo ssherr)
    if [ "$alive" = "1" ]; then
        ssh "$HOST" "tail -n 1 $REMOTE_DIR/run.log" 2>/dev/null || true
        sleep 20
    elif [ "$alive" = "0" ]; then
        break                     # confirmed not running
    else
        echo "(ssh poll failed — retrying, job still running on pod)"; sleep 20
    fi
done
echo "==> job exited; last log lines:"
ssh "$HOST" "tail -n 5 $REMOTE_DIR/run.log"
if ! ssh "$HOST" "grep -q '\[done\]' $REMOTE_DIR/run.log"; then
    echo "!! extraction did NOT reach [done] — see run.log on $HOST. Not pulling." >&2
    exit 1
fi

# run dir as reported by extract_activations.py's [done] line
RUN_DIR=$(ssh "$HOST" "grep '\[done\]' $REMOTE_DIR/run.log | sed 's/.*to //' | tr -d '\r'")
echo "==> run dir on pod: $RUN_DIR"

# REMOTE_PLOTS=1: generate plots ON the pod (needed when activations are too big
# to pull and plot locally). PULL_ACTS=0: skip the big *.dat activation files.
REMOTE_PLOTS="${REMOTE_PLOTS:-0}"
PULL_ACTS="${PULL_ACTS:-1}"

if [ "$REMOTE_PLOTS" = "1" ]; then
    echo "==> generating plots ON the pod"
    ssh "$HOST" "cd $REMOTE_DIR && HF_HOME=$REMOTE_HF python make_plots.py $RUN_DIR"
fi

echo "==> pulling results back to $LOCAL_DIR/results/  (PULL_ACTS=$PULL_ACTS)"
mkdir -p "$LOCAL_DIR/results"
EXCL=""
[ "$PULL_ACTS" = "0" ] && EXCL="--exclude *.dat"
rsync -rlDvz --no-owner --no-group --no-perms $EXCL \
    "$HOST:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"

if [ "$PULL_ACTS" = "0" ]; then
    echo "==> NOTE: raw activations (*.dat) kept on $HOST:$RUN_DIR (not pulled)."
fi
echo "==> done."
if [ "$REMOTE_PLOTS" != "1" ]; then
    echo "    Generate plots locally with: python make_plots.py results/<run_name>"
fi
