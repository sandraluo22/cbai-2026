#!/usr/bin/env bash
# Deploy + run the v2 capture on the remote H200, then pull runs/v2 back.
# Pod is bare (no transformers, no HF token, 30G root) so we use /workspace
# (46T network volume) for code, HF cache, and outputs, and ungated model
# mirrors (capture_v2.py falls back automatically). Usage:
#   bash src/scripts/capture/deploy_v2_remote.sh
set -euo pipefail

HOST=root@213.181.111.140
PORT=15042
KEY=~/.ssh/id_ed25519
REMOTE=/workspace/cross-model
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"

echo "[1/4] sync code -> $HOST:$REMOTE"
$SSH "$HOST" "mkdir -p $REMOTE"
rsync -az --delete -e "$RSYNC_SSH" --exclude '__pycache__' --exclude '*.pyc' \
  src/ "$HOST:$REMOTE/src/"
rsync -az -e "$RSYNC_SSH" requirements.txt "$HOST:$REMOTE/requirements.txt"

echo "[2/4] install deps (torch already present; don't touch it)"
$SSH "$HOST" "pip3 install -q 'transformers>=4.56' 'huggingface_hub>=0.24' numpy"

echo "[3/4] launch capture v2 in background (nohup) on /workspace"
$SSH "$HOST" "cd $REMOTE && HF_HOME=/workspace/hf PYTHONPATH=src \
  NWALKS=100 WLEN=2000 OUTDIR=$REMOTE/runs/v2 \
  nohup python3 -u src/scripts/capture/capture_v2.py > $REMOTE/v2_capture.log 2>&1 &
  echo launched; sleep 2; tail -5 $REMOTE/v2_capture.log"

echo "[4/4] when v2_capture.log shows ALL DONE, pull results:"
echo "  rsync -az -e \"$RSYNC_SSH\" $HOST:$REMOTE/runs/v2/ runs/v2/"
