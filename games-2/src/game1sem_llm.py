"""GAME 1 (SEMANTIC) between two real LLMs -- Mind-Meld convergence on a BOUNDED
board, where semantic similarity drives coordination.

Fixes the critique that the plain convergence game is a pure Schelling/focal-point
task with no semantic content. Here the board is a fixed set of N words (so the
read-out stays a bounded N-simplex), but the DYNAMIC is semantic: each round both
players pick a board word whose MEANING bridges the two words last played, so they
converge toward a shared meaning (real "converge on a word" games work this way).

We measure, per round:
  * semantic distance between the two players' picks (1 - cosine of MiniLM
    embeddings) -> the headline "did they converge SEMANTICALLY" curve, should fall;
  * coupling (counterfactual swap of the partner's last pick) -> now semantically
    meaningful: changing the partner's word moves the bridge target;
  * convergence (same word) and turns-to-converge.

Reuses the transcript format, so analyze_transcript.py / step_kl_slides.py work on it.

Env: MODELS(LlamaInst,QwenInst) GAMES(5) SAFETY(16) DEVICE RUN_DIR
Out: <RUN_DIR>/game1sem_llm_<A>_vs_<B>.json + .pdf + _transcript.jsonl
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core as K
import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
GAMES = int(os.environ.get("GAMES", "5"))
SAFETY = int(os.environ.get("SAFETY", "16"))
TOPIC = os.environ.get("TOPIC", "")                 # e.g. "Minecraft block" -> vague topic bound (no list shown)
RUN_DIR = os.environ.get("RUN_DIR", "runs/llm_sem")

# explicit bounded semantic board (shown to the models)
BOARD = ["ocean", "river", "rain", "cloud", "storm", "sun", "desert", "forest",
         "tree", "flower", "garden", "city", "house", "road", "car", "engine"]

# HIDDEN candidate set for the topic bound (never shown; only the read-out basis).
MINECRAFT = ["stone", "dirt", "wood", "sand", "glass", "gold", "iron", "coal", "ice",
             "clay", "brick", "wool", "water", "lava", "snow", "grass", "log", "chest",
             "sponge", "cactus", "pumpkin", "gravel", "quartz", "obsidian", "diamond",
             "emerald", "bedrock", "leaves", "planks", "furnace"]
ANIMAL = ["dog", "cat", "lion", "tiger", "wolf", "fox", "bear", "deer", "horse", "cow",
          "pig", "sheep", "goat", "rabbit", "mouse", "bird", "eagle", "owl", "duck", "fish",
          "shark", "whale", "snake", "frog", "bee", "spider", "crab", "seal", "bat", "ant"]
TOPIC_CANDS = {"Minecraft block": MINECRAFT, "animal": ANIMAL}


def clean_candidates(cands, toks):
    """Keep only candidates with a DISTINCT first token under every tokenizer, so the
    first-token read-out has no collisions for either model."""
    seen = [set() for _ in toks]
    keep = []
    for w in cands:
        ids = [tk(" " + w, add_special_tokens=False)["input_ids"][0] for tk in toks]
        if all(ids[i] not in seen[i] for i in range(len(toks))):
            for i, x in enumerate(ids):
                seen[i].add(x)
            keep.append(w)
    return keep


def load_embedder(dev):
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tk = AutoTokenizer.from_pretrained(name)
    em = AutoModel.from_pretrained(name).to(dev).eval()
    return em, tk


def embed(em, tk, words, dev):
    import torch
    enc = tk(words, padding=True, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = em(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    v = torch.nn.functional.normalize(v, dim=1)
    return v.cpu().numpy()


def play(mA, tA, mB, tB, dev, tag, board_emb, board, style, topic):
    A = LA.ConvAgent(mA, tA, board, dev, "word", norepeat=False, style=style, topic=topic)
    B = LA.ConvAgent(mB, tB, board, dev, "word", norepeat=False, style=style, topic=topic)
    nameA, nameB = MODELS
    V = len(board)
    BOARD = board                                       # local alias used below
    tf = open(os.path.join(RUN_DIR, f"game1sem_llm_{tag}_transcript.jsonl"), "w")

    def semdist(i, j):
        return float(1.0 - board_emb[i] @ board_emb[j])

    conv, ttc = [], []
    coupA, coupB = [], []
    dist_by_turn = np.full((GAMES, SAFETY), np.nan)     # semantic distance between picks
    exemplar = []
    for gi in range(GAMES):
        rng = np.random.default_rng(gi)
        histA, histB = [], []
        agreed = None
        gcA, gcB, gdist = [], [], []
        for t in range(SAFETY):
            dA = A.dist(histA, set()); dB = B.dist(histB, set())
            a = int(rng.choice(V, p=dA)); b = int(rng.choice(V, p=dB))
            d = semdist(a, b); gdist.append(d)
            if t < SAFETY:
                dist_by_turn[gi, t] = d
            rec = {"game": gi, "turn": t, "mode": "semantic", "picks": {nameA: BOARD[a], nameB: BOARD[b]},
                   "agreed": bool(a == b), "semantic_dist": round(d, 4),
                   nameA: {"pick": BOARD[a], "dist": {BOARD[i]: round(float(dA[i]), 4) for i in range(V)},
                           "prompt": A._prompt(histA, set())},
                   nameB: {"pick": BOARD[b], "dist": {BOARD[i]: round(float(dB[i]), 4) for i in range(V)},
                           "prompt": B._prompt(histB, set())}}
            if t >= 1:
                altB = (histB[-1][0] + 1) % V; altA = (histA[-1][0] + 1) % V
                dB_swap = B.dist(histB[:-1] + [(altB, histB[-1][1])], set())
                dA_swap = A.dist(histA[:-1] + [(altA, histA[-1][1])], set())
                klB = K.kl(dB_swap, dB); klA = K.kl(dA_swap, dA)
                gcB.append(klB); gcA.append(klA)
                rec[nameB]["coupling"] = {"kl": round(klB, 4), "swap_other_from": BOARD[histB[-1][0]],
                                          "swap_other_to": BOARD[altB],
                                          "swap_dist": {BOARD[i]: round(float(dB_swap[i]), 4) for i in range(V)}}
                rec[nameA]["coupling"] = {"kl": round(klA, 4), "swap_other_from": BOARD[histA[-1][0]],
                                          "swap_other_to": BOARD[altA],
                                          "swap_dist": {BOARD[i]: round(float(dA_swap[i]), 4) for i in range(V)}}
            tf.write(json.dumps(rec) + "\n")
            if gi == 0:
                exemplar.append(rec)
            if a == b:
                agreed = t; break
            histA.append((b, a)); histB.append((a, b))
        conv.append(agreed is not None)
        if agreed is not None:
            ttc.append(agreed + 1)
        coupA.append(np.mean(gcA) if gcA else 0.0); coupB.append(np.mean(gcB) if gcB else 0.0)
        print(f"    game {gi}: {'converged at turn %d' % (agreed+1) if agreed is not None else 'no-converge'}"
              f"  start_dist={gdist[0]:.2f} end_dist={gdist[-1]:.2f}"
              f"  coupling {nameA}={coupA[-1]:.2f} {nameB}={coupB[-1]:.2f}", flush=True)
    tf.close()
    import kl_slides
    kl_slides.generate(os.path.join(RUN_DIR, f"game1sem_llm_{tag}_transcript.jsonl"))
    return {"converged_frac": float(np.mean(conv)),
            "turns_to_converge": float(np.mean(ttc)) if ttc else float("nan"),
            f"coupling_{nameA}": float(np.mean(coupA)), f"coupling_{nameB}": float(np.mean(coupB)),
            "sem_dist_by_turn": np.nanmean(dist_by_turn, 0).tolist()}, exemplar


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    loaded = {m: LA.load(m, dev) for m in MODELS}
    em, etk = load_embedder(dev)
    if TOPIC:                                            # vague topic bound: hidden candidate set
        toks = [loaded[MODELS[0]][1], loaded[MODELS[1]][1]]
        board = clean_candidates(TOPIC_CANDS[TOPIC], toks)
        style, topic = "topic", TOPIC
        print(f"[sem] topic='{TOPIC}': {len(board)}/{len(TOPIC_CANDS[TOPIC])} candidates survive "
              f"cross-tokenizer filtering (HIDDEN from models): {board}", flush=True)
    else:                                                # explicit shown board
        board, style, topic = BOARD, "semantic", None
        print(f"[sem] explicit board of {len(board)} words", flush=True)
    board_emb = embed(em, etk, board, dev)
    tag = "_vs_".join(MODELS)
    res, exemplar = play(*loaded[MODELS[0]], *loaded[MODELS[1]], dev, tag, board_emb, board, style, topic)
    print(f"[sem] converged={res['converged_frac']*100:.0f}% turns-to-converge={res['turns_to_converge']:.1f} "
          f"coupling {MODELS[0]}={res[f'coupling_{MODELS[0]}']:.2f} {MODELS[1]}={res[f'coupling_{MODELS[1]}']:.2f} "
          f"sem_dist {res['sem_dist_by_turn'][0]:.2f}->{[d for d in res['sem_dist_by_turn'] if d==d][-1]:.2f}", flush=True)
    out = {"models": MODELS, "board": board, "topic": TOPIC or None, "style": style, "games": GAMES, **res}
    json.dump(out, open(os.path.join(RUN_DIR, f"game1sem_llm_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"game1sem_llm_{tag}.pdf"))
    print(f"[sem] DONE -> {RUN_DIR}/game1sem_llm_{tag}.json", flush=True)


def make_fig(out, path):
    d = np.array(out["sem_dist_by_turn"]); turns = np.arange(1, len(d) + 1)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        ax[0].plot(turns, d, "-o", color="tab:green")
        ax[0].set_xlabel("round"); ax[0].set_ylabel("semantic distance between the two picks (1 - cos)")
        ax[0].set_title("Semantic convergence (should fall toward 0)", fontsize=10); ax[0].grid(alpha=.3)
        ax[1].bar([0, 1], [out[f"coupling_{MODELS[0]}"], out[f"coupling_{MODELS[1]}"]],
                  color=["tab:blue", "tab:orange"])
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(MODELS, fontsize=9)
        ax[1].set_ylabel("coupling KL"); ax[1].set_title("Coupling (partner-driven)", fontsize=10)
        bound = f"vague topic: '{out['topic']}' (hidden {len(out['board'])}-word candidate set)" if out.get("topic") else "explicit board"
        fig.suptitle(f"GAME 1 SEMANTIC meaning-bridge, {bound} — {' & '.join(MODELS)}. "
                     f"converged {out['converged_frac']*100:.0f}%, {out['turns_to_converge']:.1f} turns", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
