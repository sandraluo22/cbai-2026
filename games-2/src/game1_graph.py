"""GAME-1-GRAPH: two models on a HIDDEN grid try to meet at the same node.

Both players sit on a grid whose nodes are concept words. They are NOT told it's a
grid. Each round both players see (a) their own current word-node, (b) the OTHER
player's current word-node, and (c) the history; each outputs a DIRECTION
(up/down/left/right) read as a bounded 4-simplex over those tokens. We move each on the
grid (clamped at edges), reveal the new word-nodes, and they win when both are at the
SAME node. Tests whether they can internalize the grid geometry from experience AND
coordinate to meet (mutual navigation).

MODE:
  directions  : the game above, from scratch (must infer the grid while coordinating).
  walk_primed : first give EACH model a random walk over the grid in-context (the
                Park/Lubana 'In-Context Learning of Representations' setup, which
                induces a grid representation), THEN play the coordination game -- does
                pre-internalizing the graph help them meet faster?

Ground-truth metric: grid (Manhattan) distance between the two players per round ->
should fall to 0. Also convergence rate, rounds-to-meet, and the COUPLING instrument
(counterfactually move the OTHER player's revealed node -> does this player's direction
distribution shift? = does it navigate toward the partner).

Env: MODELS(QwenInst32,QwenInst32) MODE(directions|walk_primed|both) ROWS(5) COLS(5)
     GAMES(12) MAXR(20) WALK(40) TEMP(0.7) DEVICE RUN_DIR
Out: <RUN_DIR>/game1graph_<mode>_<A>_vs_<B>.json + .pdf + _transcript.jsonl/.json
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

MODELS = os.environ.get("MODELS", "QwenInst32,QwenInst32").split(",")
MODE = os.environ.get("MODE", "both")
ROWS = int(os.environ.get("ROWS", "5"))
COLS = int(os.environ.get("COLS", "5"))
GAMES = int(os.environ.get("GAMES", "12"))
MAXR = int(os.environ.get("MAXR", "100"))       # run until they meet, capped here
WALK = int(os.environ.get("WALK", "40"))
TEMP = float(os.environ.get("TEMP", "0.7"))
START_MODE = os.environ.get("START_MODE", "corners")   # corners | random
COUPLING = os.environ.get("COUPLING", "1") == "1"      # 0 -> skip the counterfactual coupling probe (faster)
DIR_TEMP = float(os.environ.get("DIR_TEMP", "1.0"))    # temperature on the 4-direction softmax before sampling
PREFILL = int(os.environ.get("PREFILL", "0"))          # force each model's first PREFILL moves to a random walk
GEN = os.environ.get("GEN", "0") == "1"                # 1 -> model GENERATES the move (free decode + parse) vs 4-token logit read
GEN_TEMP = float(os.environ.get("GEN_TEMP", "1.0"))    # sampling temperature when GEN=1
RUN_DIR = os.environ.get("RUN_DIR", "runs/game-1-graph")

# 64 distinct concrete nouns (supports up to 8x8); first 25 kept for back-compat with 5x5 runs.
WORDS = ["apple", "tiger", "river", "chair", "music", "cloud", "bread", "stone",
         "flame", "horse", "ocean", "paper", "glass", "wheel", "lemon", "clock",
         "table", "brush", "plant", "coin", "mirror", "candle", "garden", "planet", "rocket",
         "bottle", "window", "pillow", "ladder", "engine", "forest", "island", "bridge",
         "castle", "dragon", "feather", "guitar", "hammer", "jacket", "kettle", "needle",
         "orange", "pencil", "rabbit", "saddle", "turtle", "violin", "walnut", "zebra",
         "anchor", "button", "carpet", "dolphin", "eagle", "fabric", "grape", "helmet",
         "jungle", "kitten", "magnet", "onion", "pepper", "shovel", "wallet"]
DIRS = ["up", "down", "left", "right"]


def node(r, c):
    return r * COLS + c


def word_at(r, c):
    return WORDS[node(r, c)]


def move(r, c, d):
    if d == "up":
        r = max(0, r - 1)
    elif d == "down":
        r = min(ROWS - 1, r + 1)
    elif d == "left":
        c = max(0, c - 1)
    elif d == "right":
        c = min(COLS - 1, c + 1)
    return r, c


def random_walk(rng, k):
    r, c = rng.integers(ROWS), rng.integers(COLS)
    seq = [word_at(r, c)]
    for _ in range(k):                               # wall bumps clamp -> repeated words (kept: fine/authentic)
        d = DIRS[rng.integers(4)]; r, c = move(r, c, d); seq.append(word_at(r, c))
    return seq


def start_pair(gi):
    """Two distinct start nodes. corners: alternating opposite corners. random: two
    random nodes at least max(ROWS,COLS) apart (seeded per game)."""
    if START_MODE == "random":
        rng = np.random.default_rng(1000 + gi)
        for _ in range(200):
            na, nb = int(rng.integers(ROWS * COLS)), int(rng.integers(ROWS * COLS))
            ra, ca, rb, cb = na // COLS, na % COLS, nb // COLS, nb % COLS
            if abs(ra - rb) + abs(ca - cb) >= max(ROWS, COLS):
                return (ra, ca), (rb, cb)
        return (0, 0), (ROWS - 1, COLS - 1)
    return ((0, 0), (ROWS - 1, COLS - 1)) if gi % 2 == 0 else ((0, COLS - 1), (ROWS - 1, 0))


PROMPT = ("You and another player are each standing on a word. Every round you each choose one move "
          "(up, down, left, or right); after moving, your new words are revealed. You win only when you are "
          "both standing on the SAME word. Reply with just one move: up, down, left, or right.")

BLIND_PROMPT = ("You and another player are on a grid of words. The full map of moves is given above, and you "
                "each know BOTH starting words. Every round you both choose one move (up, down, left, or right). "
                "You are NOT told your new word after moving -- you must track your own position yourself using "
                "the map, and you get no feedback about the other player. You win only when you are both on the "
                "SAME word in the same round. Reply with just one move: up, down, left, or right.")


def transition_map():
    """Labeled action->word map for every node: 'On X: up=..,down=..,left=..,right=..'.
    This teaches the grid+action structure directly (the 'condition on the graph first,
    done right' primer)."""
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            t = {d: word_at(*move(r, c, d)) for d in DIRS}
            lines.append(f"On {word_at(r,c)}: up={t['up']}, down={t['down']}, left={t['left']}, right={t['right']}")
    return "Map of moves. " + "; ".join(lines) + ". "


def build_prompt(tok, mode, walk, hist, my_word, other_word, tmap=""):
    if mode == "blind_merge":                         # full map + both starts, NO position feedback
        moves = " ".join(f"Round {k+1}: you moved {mm}." for k, (mm, mw, ow) in enumerate(hist))
        body = (tmap + BLIND_PROMPT + f"\nYou start on {my_word}. The other player starts on {other_word}."
                + ((" Your moves so far: " + moves) if moves else ""))
        return LA._render(tok, body) + "\nMy move:"
    lines = " ".join(f"Round {k+1}: you moved {mm}, now you are on {mw}; the other player is on {ow}."
                     for k, (mm, mw, ow) in enumerate(hist))
    prime = ""
    if mode == "walk_primed" and walk:
        prime = " ".join(walk) + "\n"                 # bare walk (no framing) -> induce structure implicitly
    elif mode == "transition_primed":
        prime = tmap
    body = (f"{prime}{PROMPT}\nYou are on {my_word}. The other player is on {other_word}."
            + ((" " + lines) if lines else ""))
    return LA._render(tok, body) + "\nMy move:"


def make_reader(dev):
    import torch

    @torch.no_grad()
    def dir_dist(model, tok, prompt, dir_ids):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        lg = model(ids).logits[0, -1][dir_ids].float().cpu().numpy()
        return K.softmax(lg / DIR_TEMP)                 # temperature on the direction distribution
    return dir_dist


def make_gen(dev):
    """GEN mode: let the model actually DECODE its answer (free generation) and parse the
    first up/down/left/right out of the text, instead of reading the 4-token simplex. Returns
    (move|None, raw_text); None = no direction word emitted (parse fail -> caller falls back)."""
    import re
    import torch
    pat = re.compile(r"\b(up|down|left|right)\b", re.I)

    @torch.no_grad()
    def gen_move(model, tok, prompt, rng):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(int(rng.integers(1 << 31)))   # reproducible sampling, seeded off the game rng
        out = model.generate(ids, max_new_tokens=12, do_sample=True, temperature=GEN_TEMP,
                             top_p=1.0, pad_token_id=(tok.eos_token_id or tok.pad_token_id))
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        m = pat.search(text)
        return (m.group(1).lower() if m else None), text
    return gen_move


def play(mode, mA, tA, mB, tB, dev, dir_dist, tag, gen=None):
    dir_ids_A = __import__("torch").tensor(LA._first_ids(tA, DIRS), device=dev)
    dir_ids_B = __import__("torch").tensor(LA._first_ids(tB, DIRS), device=dev)
    nameA, nameB = (MODELS[0] + "_1", MODELS[1] + "_2") if MODELS[0] == MODELS[1] else (MODELS[0], MODELS[1])
    tf = open(os.path.join(RUN_DIR, f"game1graph_{mode}_{nameA}_vs_{nameB}_transcript.jsonl"), "w")

    dist_by_round = np.full((GAMES, MAXR), np.nan)
    coupA_by_round = np.full((GAMES, MAXR), np.nan)
    genfails = 0                                          # GEN mode: moves where no direction word was emitted
    met, rounds_to_meet = [], []
    for gi in range(GAMES):
        rng = np.random.default_rng(gi)
        (ra, ca), (rb, cb) = start_pair(gi)          # start nodes (corners or random)
        walkA = random_walk(np.random.default_rng(100 + gi), WALK)
        walkB = random_walk(np.random.default_rng(200 + gi), WALK)
        tmap = transition_map() if mode in ("transition_primed", "blind_merge") else ""
        startA, startB = word_at(ra, ca), word_at(rb, cb)   # record starts (for path plots)
        blind = mode == "blind_merge"
        histA, histB = [], []
        agreed = None
        for t in range(MAXR):
            wa, wb = word_at(ra, ca), word_at(rb, cb)
            if (ra, ca) == (rb, cb):
                agreed = t; break
            mwa, owa = (startA, startB) if blind else (wa, wb)   # blind: prompt shows fixed starts, no live nodes
            mwb, owb = (startB, startA) if blind else (wb, wa)
            rawA = rawB = None
            if t < PREFILL and not blind:                # forced random-walk prefill through the ACTION channel
                mvA, mvB = DIRS[rng.integers(4)], DIRS[rng.integers(4)]
            elif gen is not None:                        # GEN: model free-decodes its move, we parse the direction
                pA = build_prompt(tA, mode, walkA, histA, mwa, owa, tmap)
                pB = build_prompt(tB, mode, walkB, histB, mwb, owb, tmap)
                mvA, rawA = gen(mA, tA, pA, rng); mvB, rawB = gen(mB, tB, pB, rng)
                if mvA is None:
                    mvA = DIRS[rng.integers(4)]; genfails += 1
                if mvB is None:
                    mvB = DIRS[rng.integers(4)]; genfails += 1
            else:
                pA = build_prompt(tA, mode, walkA, histA, mwa, owa, tmap)
                pB = build_prompt(tB, mode, walkB, histB, mwb, owb, tmap)
                dA = dir_dist(mA, tA, pA, dir_ids_A); dB = dir_dist(mB, tB, pB, dir_ids_B)
                if COUPLING and not blind:               # counterfactually move the OTHER player -> does A shift?
                    alt = word_at((rb + 2) % ROWS, (cb + 2) % COLS)
                    dA_sw = dir_dist(mA, tA, build_prompt(tA, mode, walkA, histA, wa, alt, tmap), dir_ids_A)
                    coupA_by_round[gi, t] = K.kl(dA_sw, dA)
                mvA = DIRS[int(rng.choice(4, p=dA))]; mvB = DIRS[int(rng.choice(4, p=dB))]
            ra, ca = move(ra, ca, mvA); rb, cb = move(rb, cb, mvB)
            dist_by_round[gi, t] = abs(ra - rb) + abs(ca - cb)
            histA.append((mvA, word_at(ra, ca), word_at(rb, cb)))
            histB.append((mvB, word_at(rb, cb), word_at(ra, ca)))
            rec = {"mode": mode, "game": gi, "round": t, "startA": startA, "startB": startB,
                   "A_word": word_at(ra, ca), "B_word": word_at(rb, cb),
                   "A_move": mvA, "B_move": mvB,
                   "grid_dist": int(abs(ra - rb) + abs(ca - cb)),
                   "met": (ra, ca) == (rb, cb), "couplingA": round(float(coupA_by_round[gi, t]), 4)}
            if rawA is not None or rawB is not None:     # GEN mode: keep the raw decode for inspection
                rec["A_gen"], rec["B_gen"] = rawA, rawB
            tf.write(json.dumps(rec) + "\n")
            if (ra, ca) == (rb, cb):
                agreed = t; break
        met.append(agreed is not None)
        if agreed is not None:
            rounds_to_meet.append(agreed + 1)
        print(f"[graph/{mode}] game {gi}: {'MET @round %d' % (agreed+1) if agreed is not None else 'no-meet'}", flush=True)
    tf.close()
    if gen is not None:
        print(f"[graph/{mode}] GEN parse-fails (no direction word decoded): {genfails}", flush=True)
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, f"game1graph_{mode}_{nameA}_vs_{nameB}_transcript.jsonl"))
    except Exception:
        pass
    return {"mode": mode, "models": [nameA, nameB], "met_frac": float(np.mean(met)),
            "rounds_to_meet": float(np.mean(rounds_to_meet)) if rounds_to_meet else float("nan"),
            "grid_dist_by_round": np.nanmean(dist_by_round, 0).tolist(),
            "couplingA_by_round": np.nanmean(coupA_by_round, 0).tolist(),
            "rows": ROWS, "cols": COLS, "games": GAMES, "maxr": MAXR,
            "gen": GEN, "gen_temp": GEN_TEMP if GEN else None, "genfails": genfails,
            "dir_temp": DIR_TEMP, "prefill": PREFILL}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    base = {m: LA.load(m, dev) for m in set(MODELS)}
    dir_dist = make_reader(dev)
    gen = make_gen(dev) if GEN else None                 # GEN=1 -> model decodes+parses its move
    modes = ["directions", "walk_primed", "transition_primed"] if MODE == "both" else MODE.split(",")
    tag = "_vs_".join(m + str(i) for i, m in enumerate(MODELS)) if MODELS[0] == MODELS[1] else "_vs_".join(MODELS)
    out = {"models": MODELS, "modes": {}}
    for mode in modes:
        print(f"[graph] === {mode}{' (GEN temp=%.2g)' % GEN_TEMP if GEN else ''} ===", flush=True)
        r = play(mode, *base[MODELS[0]], *base[MODELS[1]], dev, dir_dist, tag, gen)
        out["modes"][mode] = r
        print(f"[graph] {mode}: met {r['met_frac']*100:.0f}%  rounds-to-meet {r['rounds_to_meet']:.1f}  "
              f"grid-dist {r['grid_dist_by_round'][0]:.1f}->{[d for d in r['grid_dist_by_round'] if d==d][-1]:.1f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, f"game1graph_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"game1graph_{tag}.pdf"))
    print(f"[graph] DONE -> {RUN_DIR}", flush=True)


def make_fig(out, path):
    cols = {"directions": "tab:blue", "walk_primed": "tab:green"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for mode, r in out["modes"].items():
            x = np.arange(1, len(r["grid_dist_by_round"]) + 1)
            ax[0].plot(x, r["grid_dist_by_round"], "-o", ms=3, color=cols.get(mode),
                       label=f"{mode} (met {r['met_frac']*100:.0f}%)")
            ax[1].plot(x, r["couplingA_by_round"], "-o", ms=3, color=cols.get(mode), label=mode)
        ax[0].set_xlabel("round"); ax[0].set_ylabel("grid (Manhattan) distance between players")
        ax[0].set_title("Do they meet? (distance → 0)", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("round"); ax[1].set_ylabel("coupling KL (A's move shift if partner moved)")
        ax[1].set_title("Does A navigate toward the partner?", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.suptitle(f"GAME-1-GRAPH — {' & '.join(out['models'])} on a hidden {out['modes'][list(out['modes'])[0]]['rows']}x"
                     f"{out['modes'][list(out['modes'])[0]]['cols']} grid", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
