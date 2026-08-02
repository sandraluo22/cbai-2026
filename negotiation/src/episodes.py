"""Episode-corpus generation: sample alpha ~ U(lo, hi) per episode, play the
game, save transcripts (JSON, one file per episode) and A's activations
(sharded npz). Resumable: existing episode files are skipped.

Outputs under <run_dir>/
  transcripts/ep_XXXXX.json
  acts/acts_XXXXX.npz     one shard per cfg.shard_size episodes:
      acts     fp16 [n_ep_in_shard, n_rounds, n_layers+1, d]
      alpha    f32  [n_ep_in_shard]
      episode  i64  [n_ep_in_shard]
      fallback i64  [n_ep_in_shard]   total scripted moves per episode

Run:  python src/episodes.py --preset default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config          # noqa: E402
from game import play_episode                  # noqa: E402
from modeling import load_model                # noqa: E402


def sample_alphas(cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    return rng.uniform(cfg.alpha_lo, cfg.alpha_hi, size=cfg.n_episodes)


def transcripts_dir(cfg: Config) -> str:
    d = os.path.join(cfg.run_dir(), "transcripts")
    os.makedirs(d, exist_ok=True)
    return d


def acts_dir(cfg: Config, sub: str = "acts") -> str:
    d = os.path.join(cfg.run_dir(), sub)
    os.makedirs(d, exist_ok=True)
    return d


def save_shard(path: str, acts, alphas, episodes, fallbacks) -> None:
    np.savez(path,
             acts=np.stack(acts).astype(np.float16),
             alpha=np.asarray(alphas, dtype=np.float32),
             episode=np.asarray(episodes, dtype=np.int64),
             fallback=np.asarray(fallbacks, dtype=np.int64))


def load_all_shards(d: str):
    """Concatenate every acts_*.npz in `d`. Returns (acts, alpha, episode,
    fallback) with acts fp16 [N, n_rounds, n_layers+1, dim]."""
    files = sorted(f for f in os.listdir(d) if f.startswith("acts_"))
    if not files:
        raise FileNotFoundError(f"no activation shards in {d}")
    parts = [np.load(os.path.join(d, f)) for f in files]
    return (np.concatenate([p["acts"] for p in parts]),
            np.concatenate([p["alpha"] for p in parts]),
            np.concatenate([p["episode"] for p in parts]),
            np.concatenate([p["fallback"] for p in parts]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--n-episodes", type=int, default=None,
                    help="override cfg.n_episodes (e.g. a short pilot)")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    n_episodes = args.n_episodes or cfg.n_episodes

    run_dir = cfg.run_dir()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())

    torch.manual_seed(cfg.seed)
    model, tok = load_model(cfg)

    steer_vecs = None
    if cfg.tier == 2:
        from steering import load_direction
        steer_vecs = load_direction(cfg)

    alphas = sample_alphas(cfg)
    tdir = transcripts_dir(cfg)
    adir = acts_dir(cfg)

    shard_acts, shard_alpha, shard_ep, shard_fb = [], [], [], []
    shard_start = None
    total_fb = total_moves = 0
    t0 = time.time()

    def flush_shard():
        nonlocal shard_acts, shard_alpha, shard_ep, shard_fb, shard_start
        if shard_acts:
            path = os.path.join(adir, f"acts_{shard_start:05d}.npz")
            save_shard(path, shard_acts, shard_alpha, shard_ep, shard_fb)
            print(f"  wrote {path} ({len(shard_acts)} episodes)")
        shard_acts, shard_alpha, shard_ep, shard_fb = [], [], [], []
        shard_start = None

    for i in range(n_episodes):
        tpath = os.path.join(tdir, f"ep_{i:05d}.json")
        spath = os.path.join(adir, f"acts_{(i // cfg.shard_size) * cfg.shard_size:05d}.npz")
        if os.path.exists(tpath) and os.path.exists(spath):
            continue    # resume: this episode is already in a finished shard

        ep = play_episode(model, tok, cfg, i, float(alphas[i]),
                          steer_vecs=steer_vecs)
        with open(tpath, "w") as f:
            json.dump(ep.to_json(), f, indent=1)

        fb = sum(t.fallbacks for t in ep.turns)
        total_fb += fb
        total_moves += 2 * cfg.n_rounds + sum(
            1 for t in ep.turns if t.a_action == "counter")

        if shard_start is None:
            shard_start = (i // cfg.shard_size) * cfg.shard_size
        shard_acts.append(np.stack(ep.acts))    # [n_rounds, L+1, d]
        shard_alpha.append(ep.alpha)
        shard_ep.append(i)
        shard_fb.append(fb)
        if (i + 1) % cfg.shard_size == 0:
            flush_shard()

        if (i + 1) % 10 == 0 or i == n_episodes - 1:
            rate = total_fb / max(total_moves, 1)
            print(f"[{i + 1}/{n_episodes}] alpha={alphas[i]:.2f} "
                  f"A={ep.a_total} B={ep.b_total} "
                  f"fallback_rate={rate:.3f} "
                  f"({(time.time() - t0) / (i + 1):.1f}s/ep)")

    flush_shard()
    rate = total_fb / max(total_moves, 1)
    print(f"done. corpus fallback rate {rate:.3f}"
          + (" -- WARNING: high; check the model follows the offer format"
             if rate > 0.05 and not cfg.use_stub_model else ""))


if __name__ == "__main__":
    main()
