"""GAME 1 between two real LLMs -- no-repeat convergence (agree on the same word /
number), measuring the mutual coupling per turn, with NEUTRAL instructions.

Each round both models read a DISTRIBUTION over the still-available tokens (logits
over the candidate token ids, used ones masked, softmaxed) and pick one; they win a
round by picking the SAME token; no already-chosen token may be reused. Per turn we
run the counterfactual COUPLING probe: from an identical state, fork the OTHER
player's last pick (real vs a swap) and take KL(dist_swap || dist_clean) -- how much
each model conditions on the other. No level is set; the KL is continuous.

FULL LOGGING: a per-(game,turn) transcript JSONL records both prompts, each model's
distribution over the candidates, the picks, agreement, and the counterfactual pair
behind every KL. A per-turn slideshow graphs each model's coupling KL turn by turn
with the underlying clean-vs-swap distribution shift.

Runs each game UNTIL the two models converge (pick the same token) -- there is no
turn cap; the no-repeat rule bounds a game by the pool size (>=2 tokens must remain
to make a genuine choice), so a game that never coordinates ends as "exhausted".

Env: MODELS(LlamaInst,QwenInst) MODE(words|numbers|both) VOCAB(28) GAMES(6) DEVICE RUN_DIR
Out: <RUN_DIR>/game1_llm_<A>_vs_<B>_<mode>_transcript.jsonl
     <RUN_DIR>/game1_llm_<A>_vs_<B>_<mode>_perturn.pdf
     <RUN_DIR>/game1_llm_<A>_vs_<B>.json  (+ combined .pdf)
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
MODE = os.environ.get("MODE", "both")
VOCAB = int(os.environ.get("VOCAB", "28"))         # word pool
NOREPEAT = os.environ.get("NOREPEAT", "1") == "1"  # if off, tokens may be reused (no exhaustion bound)
SAFETY = int(os.environ.get("SAFETY", "40"))       # backstop only when NOREPEAT is off (prevents infinite loop)
GAMES = int(os.environ.get("GAMES", "6"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/llm")

WORDS = ["apple", "tiger", "river", "chair", "music", "cloud", "bread", "stone",
         "flame", "horse", "ocean", "paper", "glass", "wheel", "lemon", "clock",
         "table", "brush", "plant", "coin", "mirror", "candle", "pencil", "garden",
         "planet", "button", "jacket", "rocket", "pillow", "anchor"]


def vocab_for(mode):
    # Llama/Qwen split multi-digit numbers per-digit (10,11,.. collide with 1's
    # first token), so numbers use single-digit 0-9 for a clean bounded read-out
    # (which caps a no-repeat numbers game at ~5 rounds before exhaustion).
    if mode == "numbers":
        return [str(i) for i in range(10)]
    return WORDS[:VOCAB]


def play(mA, tA, mB, tB, dev, mode, tag):
    vocab = vocab_for(mode)
    kind = "number" if mode == "numbers" else "word"
    A = LA.ConvAgent(mA, tA, vocab, dev, kind, norepeat=NOREPEAT)
    B = LA.ConvAgent(mB, tB, vocab, dev, kind, norepeat=NOREPEAT)
    nameA, nameB = MODELS[0], MODELS[1]
    tpath = os.path.join(RUN_DIR, f"game1_llm_{tag}_{mode}_transcript.jsonl")
    tf = open(tpath, "w")

    V = len(vocab)
    maxlen = V if NOREPEAT else SAFETY

    def pick_alt(last):                                 # a token != `last` to fork the other's pick
        if NOREPEAT:
            return next((j for j in range(V) if j not in used), (last + 1) % V)
        return (last + 1) % V

    agree_turns, coupA, coupB, ended = [], [], [], []
    per_turn_klA = np.full((GAMES, maxlen), np.nan)
    per_turn_klB = np.full((GAMES, maxlen), np.nan)
    exemplar = []                                       # game 0, for the slideshow
    for gi in range(GAMES):
        rng = np.random.default_rng(gi)
        histA, histB, used = [], [], set()
        agreed = None
        gcA, gcB = [], []
        outcome = None
        t = -1
        while True:
            if NOREPEAT and sum(i not in used for i in range(V)) < 2:   # exhausted the pool
                outcome = "exhausted"; break
            if (not NOREPEAT) and t + 1 >= SAFETY:                       # backstop only
                outcome = "cap"; break
            t += 1
            mask = used if NOREPEAT else set()          # NOREPEAT off -> nothing forbidden
            dA = A.dist(histA, mask); dB = B.dist(histB, mask)          # each = clean dist
            a = int(rng.choice(len(dA), p=dA)); b = int(rng.choice(len(dB), p=dB))
            rec = {"game": gi, "turn": t, "mode": mode,
                   "available": [vocab[i] for i in range(V) if i not in mask],
                   "picks": {nameA: vocab[a], nameB: vocab[b]}, "agreed": bool(a == b),
                   nameA: {"pick": vocab[a], "dist": {vocab[i]: round(float(dA[i]), 4) for i in range(V)},
                           "prompt": A._prompt(histA, mask)},
                   nameB: {"pick": vocab[b], "dist": {vocab[i]: round(float(dB[i]), 4) for i in range(V)},
                           "prompt": B._prompt(histB, mask)}}
            if t >= 1:
                altB = pick_alt(histB[-1][0]); altA = pick_alt(histA[-1][0])
                dB_swap = B.dist(histB[:-1] + [(altB, histB[-1][1])], mask)   # fork A's last pick
                dA_swap = A.dist(histA[:-1] + [(altA, histA[-1][1])], mask)   # fork B's last pick
                klB = K.kl(dB_swap, dB); klA = K.kl(dA_swap, dA)
                gcB.append(klB); gcA.append(klA)
                if t < maxlen:
                    per_turn_klA[gi, t] = klA; per_turn_klB[gi, t] = klB
                rec[nameB]["coupling"] = {"kl": round(klB, 4), "swap_other_from": vocab[histB[-1][0]],
                                          "swap_other_to": vocab[altB],
                                          "swap_dist": {vocab[i]: round(float(dB_swap[i]), 4) for i in range(V)}}
                rec[nameA]["coupling"] = {"kl": round(klA, 4), "swap_other_from": vocab[histA[-1][0]],
                                          "swap_other_to": vocab[altA],
                                          "swap_dist": {vocab[i]: round(float(dA_swap[i]), 4) for i in range(V)}}
            tf.write(json.dumps(rec) + "\n")
            if gi == 0:
                exemplar.append(rec)
            if a == b:                                  # CONVERGED -> stop this game
                agreed = t; outcome = "converged"; break
            histA.append((b, a)); histB.append((a, b))
            if NOREPEAT:
                used.add(a); used.add(b)
        agree_turns.append((agreed + 1) if agreed is not None else np.nan)   # 1-indexed turns-to-converge
        ended.append(outcome)
        coupA.append(np.mean(gcA) if gcA else 0.0); coupB.append(np.mean(gcB) if gcB else 0.0)
        print(f"    [{mode}] game {gi}: {outcome}"
              + (f" at turn {agreed+1}" if agreed is not None else "")
              + f"  coupling({nameA})={coupA[-1]:.2f} coupling({nameB})={coupB[-1]:.2f}", flush=True)
    tf.close()
    print(f"    [{mode}] transcript -> {tpath}", flush=True)

    import kl_slides
    kl_slides.generate(tpath)                           # every-turn coupling + step KL graphs
    conv = [o == "converged" for o in ended]
    return {"converged_frac": float(np.mean(conv)),
            "turns_to_converge": float(np.nanmean(agree_turns)) if any(conv) else float("nan"),
            "outcomes": ended,
            f"coupling_{MODELS[0]}": float(np.mean(coupA)),
            f"coupling_{MODELS[1]}": float(np.mean(coupB)),
            "per_turn_kl_" + MODELS[0]: np.nanmean(per_turn_klA, 0).tolist(),
            "per_turn_kl_" + MODELS[1]: np.nanmean(per_turn_klB, 0).tolist()}


def perturn_pdf(mode, tag, vocab, exemplar, klA_by_turn, klB_by_turn):
    """One slide per turn: each model's coupling KL + the clean-vs-swap distribution
    shift that produced it (exemplar game 0)."""
    nameA, nameB = MODELS
    path = os.path.join(RUN_DIR, f"game1_llm_{tag}_{mode}_perturn.pdf")
    x = np.arange(len(vocab))
    with PdfPages(path) as pdf:
        # summary page: per-turn KL for each model (mean over games)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        turns = np.arange(len(klA_by_turn))
        ax.plot(turns, klA_by_turn, "-o", color="tab:blue", label=nameA)
        ax.plot(turns, klB_by_turn, "-o", color="tab:orange", label=nameB)
        ax.set_xlabel("turn"); ax.set_ylabel("coupling KL (mean over games)")
        ax.set_title(f"GAME 1 [{mode}] — coupling KL per turn, per model", fontsize=11)
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for rec in exemplar:
            t = rec["turn"]
            fig = plt.figure(figsize=(13, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
            axk = fig.add_subplot(gs[0]); axA = fig.add_subplot(gs[1]); axB = fig.add_subplot(gs[2])
            klA = rec.get(nameA, {}).get("coupling", {}).get("kl", np.nan)
            klB = rec.get(nameB, {}).get("coupling", {}).get("kl", np.nan)
            axk.bar([0, 1], [klA if klA == klA else 0, klB if klB == klB else 0],
                    color=["tab:blue", "tab:orange"])
            axk.set_xticks([0, 1]); axk.set_xticklabels([nameA, nameB], fontsize=8, rotation=20)
            axk.set_ylabel("coupling KL"); axk.set_title(f"KL this turn", fontsize=9)
            for axm, name, c in ((axA, nameA, "tab:blue"), (axB, nameB, "tab:orange")):
                d = rec[name]["dist"]; clean = np.array([d[w] for w in vocab])
                cp = rec[name].get("coupling")
                axm.bar(x - 0.2, clean, 0.4, color=c, alpha=.85, label="clean (real other-pick)")
                title = f"{name}: pick={rec[name]['pick']}"
                if cp:
                    swap = np.array([cp["swap_dist"][w] for w in vocab])
                    axm.bar(x + 0.2, swap, 0.4, color="0.5", alpha=.8,
                            label=f"swap other {cp['swap_other_from']}→{cp['swap_other_to']}")
                    title += f"   KL={cp['kl']:.2f}"
                axm.set_xticks(x); axm.set_xticklabels(vocab, rotation=90, fontsize=6)
                axm.set_ylim(0, 1); axm.set_title(title, fontsize=9); axm.legend(fontsize=7)
            agreed = "  ★ AGREED" if rec["agreed"] else ""
            fig.suptitle(f"GAME 1 [{mode}] turn {t+1}: {nameA} chose {rec['picks'][nameA]}, "
                         f"{nameB} chose {rec['picks'][nameB]}{agreed}", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"    [{mode}] per-turn slides -> {path}", flush=True)


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    loaded = {m: LA.load(m, dev) for m in MODELS}
    print(f"[game1-llm] loaded {MODELS}", flush=True)
    tag = "_vs_".join(MODELS)
    modes = ["words", "numbers"] if MODE == "both" else [MODE]
    out = {"models": MODELS, "vocab": VOCAB, "games": GAMES, "norepeat": NOREPEAT, "modes": {}}
    for mode in modes:
        print(f"[game1-llm] === {mode} ===", flush=True)
        out["modes"][mode] = play(*loaded[MODELS[0]], *loaded[MODELS[1]], dev, mode, tag)
        r = out["modes"][mode]
        print(f"[game1-llm] {mode}: converged={r['converged_frac']*100:.0f}% "
              f"turns-to-converge={r['turns_to_converge']:.1f} "
              f"coupling {MODELS[0]}={r[f'coupling_{MODELS[0]}']:.2f} "
              f"{MODELS[1]}={r[f'coupling_{MODELS[1]}']:.2f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, f"game1_llm_{tag}.json"), "w"), indent=2)
    print(f"[game1-llm] DONE -> {RUN_DIR}/game1_llm_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
