"""Game 1 — Random-walk relay (free generation) + graph-representation probe.

CORRECT mechanism (per the design): the sequence is SEEDED with a real random walk
over the 4x4 in-context grid; then models take turns. On its turn a model **freely
generates its own next word** (unconstrained — NOT scored/argmax'd over the 16 grid
words) and that word is **appended to the walk** and fed to the next model. We then
*measure* whether the free continuation lands on a valid grid node and whether it's
a legal neighbour move — we don't force it.

At each turn we also read the residual stream at the last token and project it
through a coordinate probe (leave-one-node-out ridge on teacher-forced node means)
to decode the model's internal grid position, so we can watch the graph
representation evolve as the relay grows.

Two configurations are run: a **2-person** and a **3-person** relay (default: that
many instances of Llama-3.1-8B base; override with PLAYERS=Llama,Gemma,Qwen).

Outputs (results/):
  pingpong_<k>p.json   per-turn records (words, node ids, valid-move, decoded coord) + walk
  pingpong_<k>p.png    (1) the walk mapped over the grid, (2) probe-decoded coord
                       trajectory, (3) on-grid / legal-move rate over turns

Env: PLAYERS SEED_LEN N_TURNS TEMP GEN_TOKENS N_FIT_WALKS FIT_LEN CTXLO LAYER DEVICE OUTDIR
Run: HF_HOME=/workspace/hf PYTHONPATH=games python games/01_random_walk_pingpong/run.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.modelreg import OPEN_MODELS
from common.grid import Grid, CoordProbe
from common.hf_agent import HFBaseLM
from common import io_utils

HERE = os.path.dirname(__file__)
DEVICE = os.environ.get("DEVICE", "cuda")
SEED_LEN = int(os.environ.get("SEED_LEN", "40"))
N_TURNS = int(os.environ.get("N_TURNS", "150"))
TEMP = float(os.environ.get("TEMP", "0.7"))          # sampling so >2 identical players differ
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "6"))
N_FIT_WALKS = int(os.environ.get("N_FIT_WALKS", "8"))
FIT_LEN = int(os.environ.get("FIT_LEN", "250"))
CTXLO = int(os.environ.get("CTXLO", "60"))
LAYER_ENV = os.environ.get("LAYER")
OUTDIR = os.environ.get("OUTDIR", io_utils.results_dir(HERE))
# configurations to run: default a 2-person and a 3-person relay (all Llama).
if os.environ.get("PLAYERS"):
    CONFIGS = [os.environ["PLAYERS"].split(",")]
else:
    CONFIGS = [["Llama", "Llama"], ["Llama", "Llama", "Llama"]]


def candidate_layers(tag):
    nL = OPEN_MODELS[tag]["n_layers"]
    band = sorted(set([int(nL * f) for f in (0.55, 0.7, 0.8, 0.9)] + [nL - 1]))
    return tuple(l for l in band if 0 <= l < nL)


def fit_probe(lm, grid, layers):
    n = grid.n
    sums = {L: np.zeros((n, lm.model.config.hidden_size)) for L in layers}
    cnt = np.zeros(n)
    for w in range(N_FIT_WALKS):
        nodes = grid.random_walk(FIT_LEN, start=w % n, seed=1000 + w)
        per_layer = lm.capture_words([grid.words[x] for x in nodes])
        for s, node in enumerate(nodes):
            if s + 1 >= CTXLO:
                for L in layers:
                    sums[L][node] += per_layer[L][s]
                cnt[node] += 1
    cnt = np.maximum(cnt, 1)
    means = {L: sums[L] / cnt[:, None] for L in layers}
    probe = CoordProbe(grid)
    scored = {L: probe.loo_r2(means[L]) for L in layers}
    best_L = int(LAYER_ENV) if LAYER_ENV else max(layers, key=lambda L: sum(scored[L]))
    probe.fit_full(means[best_L])
    return probe, best_L, {int(L): {"r2_row": scored[L][0], "r2_col": scored[L][1]} for L in layers}


def run_relay(players, lms, probes, layers, grid):
    """players: list of model tags. Free-generation relay; returns records + walk."""
    w2n = grid.word_to_node()
    seed_nodes = grid.random_walk(SEED_LEN, start=0, seed=7)
    words = [grid.words[x] for x in seed_nodes]
    node_seq = list(seed_nodes)                 # node id per word (>=0), or -1 off-grid

    rng = np.random.default_rng(123)
    rec = []
    for t in range(N_TURNS):
        tag = players[t % len(players)]
        lm, prb, layer = lms[tag], probes[tag], layers[tag]
        text = " ".join(words)
        # graph representation of the CURRENT last word
        _, resid = lm.forward_text(text)
        dec = prb.project(resid[layer]).ravel()
        prev_word = words[-1]
        prev_node = node_seq[-1]
        # FREE generation of the next word (unconstrained), then append it
        gen = lm.generate_word(text, max_new_tokens=GEN_TOKENS, temperature=TEMP)
        gnode = w2n.get(gen, -1)
        on_grid = gnode >= 0
        valid = bool(on_grid and prev_node >= 0 and gnode in grid.neighbors(prev_node))
        drow, dcol = grid.coords[prev_node] if prev_node >= 0 else (np.nan, np.nan)
        rec.append({
            "t": t, "player_idx": t % len(players), "player": tag,
            "prev_word": prev_word, "prev_node": prev_node,
            "gen_word": gen, "gen_node": gnode, "on_grid": on_grid, "valid_move": valid,
            "dec_row": float(dec[0]), "dec_col": float(dec[1]),
            "true_row": float(drow) if prev_node >= 0 else None,
            "true_col": float(dcol) if prev_node >= 0 else None,
        })
        words.append(gen)
        node_seq.append(gnode)

    on = np.array([r["on_grid"] for r in rec], float)
    # legal-move rate among steps where both endpoints are on the grid
    both = np.array([r["prev_node"] >= 0 and r["on_grid"] for r in rec], bool)
    valids = np.array([r["valid_move"] for r in rec], float)
    return {
        "players": players, "n_players": len(players),
        "seed_len": SEED_LEN, "n_turns": N_TURNS, "temp": TEMP, "gen_tokens": GEN_TOKENS,
        "on_grid_rate": float(on.mean()),
        "legal_move_rate_when_on_grid": float(valids[both].mean()) if both.any() else float("nan"),
        "seed_words": [grid.words[x] for x in seed_nodes],
        "walk_words": list(words),
        "walk_nodes": [int(x) for x in node_seq],   # -1 = off-grid word
        "records": rec,
    }


def main():
    io_utils.seed_all(0)
    grid = Grid(4, 4)
    tags = sorted({t for cfg in CONFIGS for t in cfg})
    lms, probes, layers, scores = {}, {}, {}, {}
    for tag in tags:
        print(f"[game1] loading {tag} (base) + fitting probe", flush=True)
        lm = HFBaseLM(tag, OPEN_MODELS[tag]["base"], candidate_layers(tag), DEVICE)
        prb, L, sc = fit_probe(lm, grid, candidate_layers(tag))
        lms[tag], probes[tag], layers[tag], scores[tag] = lm, prb, L, sc
        print(f"[game1] {tag} probe L{L} LOO R2 row/col="
              f"{sc[L]['r2_row']:.3f}/{sc[L]['r2_col']:.3f}", flush=True)

    for cfg in CONFIGS:
        k = len(cfg)
        print(f"[game1] === {k}-person relay: {cfg} ===", flush=True)
        res = run_relay(cfg, lms, probes, layers, grid)
        res["probe_layers"] = {t: layers[t] for t in set(cfg)}
        res["probe_layer_scores"] = {t: scores[t] for t in set(cfg)}
        io_utils.dump_json(res, os.path.join(OUTDIR, f"pingpong_{k}p.json"))
        make_fig(res, grid, os.path.join(OUTDIR, f"pingpong_{k}p.png"))
        print(f"[game1] {k}p: on-grid rate={res['on_grid_rate']:.3f}  "
              f"legal-move rate(on-grid)={res['legal_move_rate_when_on_grid']:.3f}", flush=True)


def _rolling(x, k=15):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x


def _draw_grid(ax, grid):
    for u in range(grid.n):
        for v in grid.adj[u]:
            if v > u:
                ax.plot([grid.coords[u][1], grid.coords[v][1]],
                        [grid.coords[u][0], grid.coords[v][0]], "-", color="0.85", zorder=1)
    for i, (r, c) in enumerate(grid.coords):
        ax.text(c, r, grid.words[i], ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85), zorder=4)
    ax.invert_yaxis(); ax.set_xlabel("col"); ax.set_ylabel("row")


def make_fig(res, grid, path):
    rec = res["records"]
    nodes = res["walk_nodes"]                 # includes seed; -1 = off-grid
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) the WALK mapped over the graph structure (path through visited nodes)
    _draw_grid(ax[0], grid)
    seg = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)
           if nodes[i] >= 0 and nodes[i + 1] >= 0]
    cmap = plt.get_cmap("plasma")
    for i, (a, b) in enumerate(seg):
        ra, ca = grid.coords[a]; rb, cb = grid.coords[b]
        jit = (np.random.default_rng(i).normal(0, 0.04, 4))    # tiny jitter to reveal repeats
        ax[0].annotate("", xy=(cb + jit[0], rb + jit[1]), xytext=(ca + jit[2], ra + jit[3]),
                       arrowprops=dict(arrowstyle="->", color=cmap(i / max(len(seg), 1)),
                                       alpha=0.5, lw=1.2), zorder=3)
    off = sum(1 for n in nodes[SEED_LEN:] if n < 0)
    ax[0].set_title(f"walk over the grid ({res['n_players']}-person relay)\n"
                    f"on-grid {res['on_grid_rate']:.2f}, off-grid words={off}")

    # (2) probe-decoded coordinate trajectory (internal representation)
    dec = np.array([[r["dec_row"], r["dec_col"]] for r in rec])
    ts = np.arange(len(rec))
    _draw_grid(ax[1], grid)
    sc = ax[1].scatter(dec[:, 1], dec[:, 0], c=ts, cmap="viridis", s=16, zorder=3)
    lyr = list(res["probe_layers"].values())[0]
    ax[1].set_title(f"probe-decoded coord over turns (L{lyr})")
    fig.colorbar(sc, ax=ax[1], label="turn")

    # (3) on-grid + legal-move rate over turns
    ax[2].plot(_rolling([r["on_grid"] for r in rec]), color="tab:blue", label="on-grid rate")
    both = [r["prev_node"] >= 0 for r in rec]
    lm = [r["valid_move"] for r in rec]
    ax[2].plot(_rolling(lm), color="tab:green", label="legal-move rate")
    ax[2].set_ylim(-0.05, 1.05); ax[2].set_xlabel("turn"); ax[2].set_ylabel("rate (rolling)")
    ax[2].set_title("does the free continuation stay on the grid?"); ax[2].legend(fontsize=8)

    fig.suptitle(f"Random-walk relay (free generation) — {res['players']}", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[io] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
