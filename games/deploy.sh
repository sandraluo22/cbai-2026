#!/usr/bin/env bash
# Deploy games/ to the H200 pod, run the open-weight experiments, and pull results.
# The pod already caches the base mirrors; instruct mirrors download on first use to
# /workspace/hf. Usage:
#   bash games/deploy.sh setup            # rsync code + install deps (once)
#   bash games/deploy.sh run <game_dir> [ENV=VAL ...]   # run one game in background
#   bash games/deploy.sh pull             # rsync all results back to the Mac
#   bash games/deploy.sh tail <game_dir>  # tail the running log
set -uo pipefail

HOST="${POD_HOST:?set POD_HOST=root@<pod-ip>}"
PORT=${POD_PORT:-19497}
KEY=~/.ssh/id_ed25519
REMOTE=/workspace/games
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
RSSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
LOCAL="$(cd "$(dirname "$0")" && pwd)"

cmd=${1:-help}
case "$cmd" in
  setup)
    $SSH "$HOST" "mkdir -p $REMOTE"
    rsync -az --delete -e "$RSSH" --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'results/' --exclude '.env' "$LOCAL/" "$HOST:$REMOTE/"
    $SSH "$HOST" "pip3 install -q 'transformers>=4.56' 'huggingface_hub>=0.24' sentence-transformers matplotlib numpy"
    echo "setup done"
    ;;
  run)
    game=${2:?game dir e.g. 02_convergence}; shift 2
    envs="$*"
    $SSH "$HOST" "cd $REMOTE && HF_HOME=/workspace/hf PYTHONPATH=$REMOTE $envs \
      nohup python3 -u $game/run.py > $REMOTE/$game.log 2>&1 & echo launched PID \$!; sleep 3; tail -5 $REMOTE/$game.log"
    ;;
  tail)
    game=${2:?game dir}; $SSH "$HOST" "tail -40 $REMOTE/$game.log"
    ;;
  pull)
    for g in 01_random_walk_pingpong 02_convergence 03_volunteers_dilemma 04_attractor_states; do
      mkdir -p "$LOCAL/$g/results"
      rsync -az -e "$RSSH" "$HOST:$REMOTE/$g/results/" "$LOCAL/$g/results/" 2>/dev/null && echo "pulled $g"
    done
    ;;
  *)
    echo "usage: $0 {setup|run <game_dir> [ENV=VAL...]|tail <game_dir>|pull}"; ;;
esac
