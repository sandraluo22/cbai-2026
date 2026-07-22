"""Two real models play Codenames; MEASURE the mutual theory-of-mind (continuous).

Both models get only a NEUTRAL instruction (see llm_agents: "give a clue" /
"pick the likely target") -- never told to model each other. We then read off, from
behaviour, how much each conditions on the other:

  * coupling (guesser's ToM of the spymaster):
        KL(B_guess | swapped clue  ||  B_guess | real clue), from an identical state.
        How much a controlled change in A's clue moves B's guess.
  * adaptivity (spymaster's ToM of the guesser):
        KL(A_clue | guesser-found-X  ||  A_clue | guesser-found-nothing).
        How much A's clue moves when the guesser's revealed state changes.
  * recovery: B's target-posterior mass (does B actually infer A's hidden set).

Run BOTH role orderings of the two models, so we get the mutual picture (each model
as both spymaster and guesser). Nothing is a strict L1/L2 label -- the KLs are a
continuous "how much theory-of-mind did it show", calibrated by the reference agents.

Env: MODELS(Llama,Qwen) N(12) M(3) NCLUE(3) ROUNDS(6) GAMES(8) DEVICE RUN_DIR
Out: <RUN_DIR>/game_llm_<A>_vs_<B>.json + .pdf
"""
from __future__ import annotations

import os
import json
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core as K
import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
N = len(LA.BOARD_WORDS)                                 # semantic board = 12 words
M = int(os.environ.get("M", "3"))
NCLUE = int(os.environ.get("NCLUE", str(len(LA.CLUE_WORDS))))
ROUNDS = int(os.environ.get("ROUNDS", "6"))
GAMES = int(os.environ.get("GAMES", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/llm")


def play_pair(mA, tA, mB, tB, dev, words, sname, gname, tf=None):
    """A = spymaster (its adaptivity = A's ToM of B); B = guesser (its coupling = B's ToM of A).
    If `tf` is given, log a full per-round transcript (clue, belief, guess, targets, and the
    counterfactuals behind coupling & adaptivity)."""
    mass, coup, adapt = [], [], []
    for gi in range(GAMES):
        rng = np.random.default_rng(gi)
        targets = sorted(rng.choice(N, M, replace=False).tolist())
        S = LA.LLMSpeaker(mA, tA, words, targets, 1, dev)
        B = LA.LLMListener(mB, tB, words, 1, dev)
        gm, gc, ga = [], [], []
        for r in range(ROUNDS):
            c, _ = S.clue(B, rng=rng)
            cswap = (c + 1) % NCLUE
            clean = B.copy(); clean.update(c)
            swap = B.copy(); swap.update(cswap)
            kl_coup = K.kl(swap.guess_dist(), clean.guess_dist())
            gc.append(kl_coup)
            naive = LA.LLMListener(mB, tB, words, 1, dev)
            cd_real, cd_naive = S.clue_dist(B), S.clue_dist(naive)
            kl_adapt = K.kl(cd_real, cd_naive)
            ga.append(kl_adapt)
            belief = B.belief()
            B.update(c); g = B.pick_guess(); ok = g in S.remaining
            if tf is not None:
                tf.write(json.dumps({
                    "pair": f"{sname}->{gname}", "spymaster": sname, "guesser": gname,
                    "game": gi, "round": r, "targets": [words[i] for i in targets],
                    "clue": LA.CLUE_WORDS[c], "guess": words[g], "correct": bool(ok),
                    "found_so_far": [words[i] for i in sorted(B.known)],
                    "target_mass": round(float(K.target_mass(B.belief(), targets)), 4),
                    "belief": {words[i]: round(float(belief[i]), 4) for i in range(N)},
                    "coupling": {"kl": round(float(kl_coup), 4), "clue_swapped_to": LA.CLUE_WORDS[cswap],
                                 "guess_dist_clean": {words[i]: round(float(clean.guess_dist()[i]), 4) for i in range(N)},
                                 "guess_dist_swap": {words[i]: round(float(swap.guess_dist()[i]), 4) for i in range(N)}},
                    "adaptivity": {"kl": round(float(kl_adapt), 4),
                                   "clue_dist_real_guesser": {LA.CLUE_WORDS[j]: round(float(cd_real[j]), 4) for j in range(NCLUE)},
                                   "clue_dist_naive_guesser": {LA.CLUE_WORDS[j]: round(float(cd_naive[j]), 4) for j in range(NCLUE)}},
                }) + "\n")
            B.observe(g, ok); S.observe(g, ok)
            gm.append(K.target_mass(B.belief(), targets))
        mass.append(gm); coup.append(gc); adapt.append(ga)
        print(f"    game {gi}: recovery={gm[-1]:.2f} coupling={np.mean(gc):.2f} adaptivity={np.mean(ga):.2f}", flush=True)
    return (np.array(mass), np.array(coup), np.array(adapt))


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    os.environ["NCLUE"] = str(NCLUE)
    words = LA.BOARD_WORDS

    loaded = {}
    for m in MODELS:
        print(f"[llm] loading {m}", flush=True)
        loaded[m] = LA.load(m, dev)

    tf = open(os.path.join(RUN_DIR, f"game_llm_{'_vs_'.join(MODELS)}_transcript.jsonl"), "w")
    out = {"models": MODELS, "N": N, "M": M, "nclue": NCLUE, "rounds": ROUNDS, "games": GAMES, "pairs": {}}
    for A, B in itertools.permutations(MODELS, 2):     # both role orderings
        print(f"[llm] {A} = spymaster,  {B} = guesser", flush=True)
        mass, coup, adapt = play_pair(*loaded[A], *loaded[B], dev, words, A, B, tf)
        key = f"{A}->{B}"
        out["pairs"][key] = {
            "spymaster": A, "guesser": B,
            "recovery_mean": mass.mean(0).tolist(), "recovery_se": (mass.std(0) / np.sqrt(GAMES)).tolist(),
            "coupling_mean": coup.mean(0).tolist(),        # B's ToM of A
            "adaptivity_mean": adapt.mean(0).tolist(),      # A's ToM of B
            "coupling_overall": float(coup.mean()), "adaptivity_overall": float(adapt.mean()),
            "final_recovery": float(mass[:, -1].mean())}
        print(f"[llm] {key}: recovery={out['pairs'][key]['final_recovery']:.2f} "
              f"guesser-coupling={out['pairs'][key]['coupling_overall']:.2f} "
              f"spymaster-adaptivity={out['pairs'][key]['adaptivity_overall']:.2f}", flush=True)

    tf.close()
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, f"game_llm_{'_vs_'.join(MODELS)}_transcript.jsonl"))
    except Exception as e:
        print(f"[llm] json convert skipped: {e}", flush=True)
    tag = "_vs_".join(MODELS)
    json.dump(out, open(os.path.join(RUN_DIR, f"game_llm_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"game_llm_{tag}.pdf"))
    print(f"[llm] DONE -> {RUN_DIR}/game_llm_{tag}.json  (+ transcript.jsonl)", flush=True)


def make_fig(out, path):
    rounds = np.arange(1, ROUNDS + 1)
    pairs = list(out["pairs"])
    cols = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 2)))
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for i, k in enumerate(pairs):
            d = out["pairs"][k]
            ax[0].errorbar(rounds, d["recovery_mean"], yerr=d["recovery_se"], color=cols[i], label=k, capsize=2)
            ax[1].plot(rounds, d["coupling_mean"], "-o", color=cols[i], label=f"{k}  (guesser={d['guesser']})")
            ax[2].plot(rounds, d["adaptivity_mean"], "-o", color=cols[i], label=f"{k}  (spymaster={d['spymaster']})")
        ax[0].set_title("Recovery: guesser's target-posterior mass", fontsize=9); ax[0].set_ylim(0, 1)
        ax[1].set_title("Coupling = guesser's ToM of spymaster\nKL(guess|swap||guess|clean)", fontsize=9)
        ax[2].set_title("Adaptivity = spymaster's ToM of guesser\nKL(clue|B-found-X||clue|B-naive)", fontsize=9)
        for a in ax:
            a.set_xlabel("round"); a.legend(fontsize=7); a.grid(alpha=.3)
        fig.suptitle(f"Mutual theory of mind between {' & '.join(out['models'])} in Codenames "
                     f"(neutral instructions; both role orderings). KLs are continuous, not a label.", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
