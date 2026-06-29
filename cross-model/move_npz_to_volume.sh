#!/usr/bin/env bash
# Move every local .npz to the /workspace volume with RESUMABLE rsync (the link
# drops ~every 500MB, so each attempt resumes from the partial). Deletes each
# local copy only after byte-exact size match. Resumable across restarts:
# partials are kept; only stale symlinks on the volume are removed.
set -uo pipefail
KEY=$HOME/.ssh/id_ed25519; HOST=root@213.181.111.140; PORT=18144; REMOTE=/workspace/cross-model
export RSYNC_RSH="ssh -p $PORT -i $KEY -o ConnectTimeout=30 -o ServerAliveInterval=20 -o ServerAliveCountMax=6"
cd /Users/sandraluo/cbai-2026/cross-model
sshq(){ ssh -p $PORT -i $KEY -o ConnectTimeout=30 "$HOST" "$@" 2>/dev/null; }

echo "to move: $(find runs -name '*.npz'|wc -l|tr -d ' ') files"
for rel in $(find runs -name '*.npz' | sort); do
  lsz=$(stat -f%z "$rel")
  rsz=$(sshq "stat -c%s '$REMOTE/$rel' 2>/dev/null" || true)
  if [ "$rsz" = "$lsz" ]; then rm -f "$rel"; echo "[skip+rm] $rel (already complete)"; continue; fi
  sshq "mkdir -p '$REMOTE/$(dirname "$rel")'; [ -L '$REMOTE/$rel' ] && rm -f '$REMOTE/$rel' || true"
  done=0
  for try in $(seq 1 60); do
    rsync --partial --inplace --timeout=120 "$rel" "$HOST:$REMOTE/$rel" 2>/dev/null || true
    rsz=$(sshq "stat -c%s '$REMOTE/$rel' 2>/dev/null" || echo 0)
    echo "[$(date +%H:%M:%S)] $rel try $try: $(( ${rsz:-0}/1000000 ))/$(( lsz/1000000 )) MB"
    [ "$rsz" = "$lsz" ] && { done=1; break; }
    sleep 3
  done
  if [ "$done" = 1 ]; then rm -f "$rel"; echo "[done+rm] $rel"; else echo "[INCOMPLETE] $rel (kept local)"; fi
done
echo "=== MOVE COMPLETE: local npz left $(find runs -name '*.npz'|wc -l|tr -d ' ') ==="
df -h "$HOME" | tail -1 | awk '{print $4" free on Mac"}'
