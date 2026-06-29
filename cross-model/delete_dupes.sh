#!/usr/bin/env bash
# Phase 1: delete local .npz that are ALREADY on the volume (exact byte-size match).
set -uo pipefail
KEY=$HOME/.ssh/id_ed25519; HOST=root@213.181.111.140; PORT=18144; REMOTE=/workspace/cross-model
cd /Users/sandraluo/cbai-2026/cross-model
n=0
for rel in $(find runs -name '*.npz' | sort); do
  lsz=$(stat -f%z "$rel")
  rsz=$(ssh -p $PORT -i $KEY -o ConnectTimeout=30 "$HOST" "stat -c%s '$REMOTE/$rel' 2>/dev/null" 2>/dev/null || true)
  if [ "$rsz" = "$lsz" ]; then
    rm -f "$rel"; echo "[dupe rm] $rel ($lsz bytes)"; n=$((n+1))
  else
    echo "[non-dupe] $rel  local=$lsz remote=${rsz:-absent}"
  fi
done
echo "=== deleted $n dupes ==="; df -h "$HOME" | tail -1 | awk '{print $4" free on Mac"}'
