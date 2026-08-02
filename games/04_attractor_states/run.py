"""Game 4 — 3-person attractor states (round-robin, 25 turns).

Following the attractor-states post (see seeds.py) but with THREE models instead of
two: start each of the six open-ended seed prompts and let three LLMs talk to each
other round-robin for 25 turns. Then check convergence — both WITHIN a conversation
(do consecutive turns get more self-similar, i.e. settle into an attractor) and
ACROSS the six seeds (do conversations from different starts drift toward a COMMON
attractor — the interesting claim).

Backends: BACKEND=open (Llama/Gemma/Qwen instruct) | BACKEND=api (Opus/Sonnet/Haiku)
Outputs (results/): attractor_<backend>.json  (transcripts + metrics) + attractor_<backend>.png
Env: BACKEND N_TURNS MAX_NEW TEMP DEVICE OUTDIR
Run: HF_HOME=/workspace/hf PYTHONPATH=games python games/04_attractor_states/run.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.agents import build_chat_agents, default_roster
from common import embed, io_utils
from seeds import SEEDS, SYSTEM

HERE = os.path.dirname(__file__)
BACKEND = os.environ.get("BACKEND", "open")
DEVICE = os.environ.get("DEVICE", "cuda")
N_TURNS = int(os.environ.get("N_TURNS", "25"))
# "no token limit": let each turn run to a natural stop. Open models generate up to
# MAX_NEW (they hit EOS well before); the API turn gets a generous ceiling.
MAX_NEW = int(os.environ.get("MAX_NEW", "1024"))
API_MAX = int(os.environ.get("API_MAX", "4096"))
TEMP = float(os.environ.get("TEMP", "0.9"))
OUTDIR = os.environ.get("OUTDIR", io_utils.results_dir(HERE))


def run_conversation(agents, seed_text):
    """Round-robin: seed is the opening 'host' message; agents rotate for N_TURNS."""
    transcript = [("host", seed_text)]
    turns = []                      # (speaker_tag, text) in turn order
    for t in range(N_TURNS):
        ag = agents[t % len(agents)]
        if BACKEND == "open":
            text = ag.say(SYSTEM, transcript, max_new_tokens=MAX_NEW, temperature=TEMP)
        else:
            text = ag.say(SYSTEM, transcript, max_new_tokens=API_MAX)
        transcript.append((ag.tag, text))
        turns.append((ag.tag, text))
        print(f"[attr] seed='{seed_text[:28]}...' turn {t} [{ag.tag}]: {text[:70]!r}", flush=True)
    return turns


def main():
    io_utils.seed_all(0)
    tags = default_roster(BACKEND, 3)
    agents = build_chat_agents(tags, BACKEND, DEVICE)
    conversations = []             # per seed: list of (tag, text)
    try:
        for si, seed in enumerate(SEEDS):
            print(f"\n=== attractor seed {si}: {seed!r} ({tags}, {BACKEND}) ===", flush=True)
            conversations.append(run_conversation(agents, seed))
    finally:
        for ag in agents:
            ag.free()

    # ---- metrics ----
    # embed every turn of every conversation
    n_seeds = len(SEEDS)
    emb = [embed.embed([txt for _, txt in conv]) for conv in conversations]  # [seeds][turns, d]

    # cross-seed spread at each turn (do the 6 conversations converge to one region?)
    cross_spread = []
    for t in range(N_TURNS):
        vt = np.stack([emb[s][t] for s in range(n_seeds)])
        cross_spread.append(embed.centroid_spread(vt))

    # within-conversation consecutive-turn similarity (settling into an attractor)
    within_sim = []
    for s in range(n_seeds):
        sims = [float(emb[s][t] @ emb[s][t + 1]) for t in range(N_TURNS - 1)]
        within_sim.append(sims)

    result = {
        "backend": BACKEND, "agents": tags, "n_turns": N_TURNS, "seeds": SEEDS,
        "cross_seed_spread_by_turn": cross_spread,
        "within_conv_consecutive_sim": within_sim,
        "cross_spread_early": float(np.mean(cross_spread[:3])),
        "cross_spread_late": float(np.mean(cross_spread[-3:])),
        "conversations": [[{"speaker": sp, "text": tx} for sp, tx in conv] for conv in conversations],
    }
    io_utils.dump_json(result, os.path.join(OUTDIR, f"attractor_{BACKEND}.json"))
    make_fig(result, emb, os.path.join(OUTDIR, f"attractor_{BACKEND}.png"))
    print(f"[attr] cross-seed spread early={result['cross_spread_early']:.3f} -> "
          f"late={result['cross_spread_late']:.3f} "
          f"({'CONVERGING' if result['cross_spread_late'] < result['cross_spread_early'] else 'not converging'})",
          flush=True)


def make_fig(result, emb, path):
    n_seeds = len(result["seeds"])
    N = result["n_turns"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    ax[0].plot(range(N), result["cross_seed_spread_by_turn"], "-o", color="tab:purple")
    ax[0].set_title("cross-seed spread per turn\n(↓ = the six conversations converge to a shared attractor)")
    ax[0].set_xlabel("turn"); ax[0].set_ylabel("mean dist to centroid (6 seeds)")

    # 2D PCA of all turn embeddings, colored by seed, opacity by turn
    allv = np.concatenate([emb[s] for s in range(n_seeds)], axis=0)
    allv = allv - allv.mean(0)
    _, _, Vt = np.linalg.svd(allv, full_matrices=False)
    proj = allv @ Vt[:2].T
    proj = proj.reshape(n_seeds, N, 2)
    cmap = plt.get_cmap("tab10")
    for s in range(n_seeds):
        alphas = np.linspace(0.25, 1.0, N)
        ax[1].plot(proj[s, :, 0], proj[s, :, 1], "-", color=cmap(s), alpha=0.4)
        ax[1].scatter(proj[s, :, 0], proj[s, :, 1], color=cmap(s), s=18,
                      alpha=list(alphas), label=f"seed {s}")
        ax[1].scatter(proj[s, -1, 0], proj[s, -1, 1], color=cmap(s), s=120,
                      marker="*", edgecolor="k", zorder=5)
    ax[1].set_title("conversation trajectories (PCA of turn embeddings)\n★ = final turn")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"3-person attractor states — {result['agents']} ({result['backend']})")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[io] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
