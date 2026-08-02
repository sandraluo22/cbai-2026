"""Control 1 -- the transcript shadow.

A third same-weights instance passively READS each saved transcript (framed
as an observer, not a participant) and we capture its last-token residuals at
the same per-round boundaries as A's. Probing these for alpha and comparing
against A's probes is the headline control: if the negotiating A encodes B's
latent better than a spectator, participation itself buys extra
opponent-information; if not, opponent modeling is transcript-general.

The observer's context after round r is:
  system: observer framing
  user  : canonical rendering of rounds 1..r (game.render_transcript)
  -> capture last token, all layers.

Outputs <run_dir>/shadow_acts/acts_XXXXX.npz, same schema as episodes.py.

Run:  python src/shadow.py --preset default
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config                              # noqa: E402
from episodes import acts_dir, save_shard, transcripts_dir  # noqa: E402
from game import render_transcript                         # noqa: E402
from modeling import capture_last_token, load_model        # noqa: E402
from prompts import system_observer                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()
    cfg = get_config(args.preset)

    tdir = transcripts_dir(cfg)
    files = sorted(f for f in os.listdir(tdir) if f.startswith("ep_"))
    if not files:
        raise SystemExit(f"no transcripts in {tdir}; run episodes.py first")

    torch.manual_seed(cfg.seed)
    model, tok = load_model(cfg)
    sdir = acts_dir(cfg, "shadow_acts")
    sys_msg = {"role": "system",
               "content": system_observer(cfg.pie, cfg.n_rounds)}

    shard_acts, shard_alpha, shard_ep, shard_fb = [], [], [], []
    shard_start = None

    for n, fname in enumerate(files):
        with open(os.path.join(tdir, fname)) as f:
            rec = json.load(f)
        i = rec["episode"]
        spath = os.path.join(
            sdir, f"acts_{(i // cfg.shard_size) * cfg.shard_size:05d}.npz")
        if os.path.exists(spath):
            continue

        per_round = []
        for r in range(1, cfg.n_rounds + 1):
            msgs = [sys_msg, {"role": "user", "content":
                              "Transcript so far:\n"
                              + render_transcript(rec["turns"], r, cfg.pie)}]
            per_round.append(capture_last_token(model, tok, msgs))

        if shard_start is None:
            shard_start = (i // cfg.shard_size) * cfg.shard_size
        shard_acts.append(np.stack(per_round))
        shard_alpha.append(rec["alpha"])
        shard_ep.append(i)
        shard_fb.append(sum(t["fallbacks"] for t in rec["turns"]))
        if len(shard_acts) == cfg.shard_size:
            save_shard(os.path.join(sdir, f"acts_{shard_start:05d}.npz"),
                       shard_acts, shard_alpha, shard_ep, shard_fb)
            print(f"[{n + 1}/{len(files)}] wrote shard acts_{shard_start:05d}")
            shard_acts, shard_alpha, shard_ep, shard_fb = [], [], [], []
            shard_start = None

    if shard_acts:
        save_shard(os.path.join(sdir, f"acts_{shard_start:05d}.npz"),
                   shard_acts, shard_alpha, shard_ep, shard_fb)
    print("shadow capture done.")


if __name__ == "__main__":
    main()
