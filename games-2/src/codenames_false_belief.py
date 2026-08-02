"""False-belief experiment: the spymaster THINKS the targets are T' (a set different
from the actual set T), and clues for T'. We ask whether the guesser recovers what the
spymaster BELIEVES/conveys (T') or the actual targets (T).

  actual   T  : the ground-truth target set (drives the public correct/incorrect reveal)
  believed T' : what the spymaster is told -> its prompt, its clues, its remaining set
                (in the 'false' condition T' is DISJOINT from T; in 'control' T'=T)

Both players see one public reveal, scored against ACTUAL T. The spymaster's belief is
fixed (it is never told it is wrong). Each round we log the guesser's belief mass on the
believed set and on the actual set. If the guesser tracks A's belief, recovery(T') >>
recovery(T) in the false condition.

Env: MODELS(LlamaInst,QwenInst) GAMES(10) ROUNDS(6) M(4) DEVICE RUN_DIR
Out: <RUN_DIR>/false_belief_<A>_vs_<B>.{json,pdf} + _transcript.jsonl
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
GAMES = int(os.environ.get("GAMES", "10"))
CAP = int(os.environ.get("ROUNDS", "6"))
M = int(os.environ.get("M", "4"))
N = len(LA.OPEN_BOARD)
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/false_belief")


def play(mA, tA, mB, tB, dev, words, A, B, cond, tf):
    """cond in {'control','false'}. Returns arrays [game][round] for recovery on the
    believed set and the actual set."""
    rec_bel, rec_act = [], []
    for gi in range(GAMES):
        rng = np.random.default_rng(1000 + gi)
        actual = sorted(rng.choice(N, M, replace=False).tolist())
        if cond == "false":
            pool = [i for i in range(N) if i not in actual]
            believed = sorted(rng.choice(pool, M, replace=False).tolist())
        else:
            believed = list(actual)
        S = LA.LLMSpeakerOpen(mA, tA, words, believed, dev, remember=True)   # A believes `believed`
        G = LA.LLMListenerOpen(mB, tB, words, dev)
        gb, ga = [], []
        for r in range(CAP):
            if not S.remaining:
                break
            real, swap, count = S.clue(G)
            G.update(real, count)
            belief = G.belief()
            rb = K.target_mass(belief, believed); rc = K.target_mass(belief, actual)
            gb.append(rb); ga.append(rc)
            guesses = G.pick_guesses(count); res = []
            for g in guesses:
                ok = (g in actual) and (g not in G.known)          # public reveal vs ACTUAL T
                res.append((words[g], bool(ok)))
                G.observe(g, ok); S.observe(g, ok)                 # disjoint -> S.remaining (T') never shrinks
            if tf is not None:
                tf.write(json.dumps({
                    "cond": cond, "pair": f"{A}->{B}", "spymaster": A, "guesser": B,
                    "game": gi, "round": r, "believed": [words[i] for i in believed],
                    "actual": [words[i] for i in actual], "clue": real,
                    "guesses": [w for w, _ in res], "correct": [o for _, o in res],
                    "recovery_believed": round(float(rb), 4), "recovery_actual": round(float(rc), 4),
                }) + "\n")
            S.note_clue(real)
        rec_bel.append(gb); rec_act.append(ga)
        print(f"    [{cond}] {A}->{B} g{gi}: rec_believed={gb[-1]:.2f} rec_actual={ga[-1]:.2f}", flush=True)
    return rec_bel, rec_act


def pad(rows, R):
    a = np.full((len(rows), R), np.nan)
    for i, x in enumerate(rows):
        a[i, :len(x)] = x
    return a


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    words = LA.OPEN_BOARD
    loaded = {m: LA.load(m, dev) for m in set(MODELS)}
    tf = open(os.path.join(RUN_DIR, f"false_belief_{'_vs_'.join(MODELS)}_transcript.jsonl"), "w")
    out = {"models": MODELS, "M": M, "games": GAMES, "rounds": CAP, "pairs": {}}
    for A, B in itertools.permutations(MODELS, 2):
        out["pairs"][f"{A}->{B}"] = {"spymaster": A, "guesser": B}
        for cond in ("control", "false"):
            rb, ra = play(*loaded[A], *loaded[B], dev, words, A, B, cond, tf)
            R = max(len(x) for x in rb)
            out["pairs"][f"{A}->{B}"][cond] = {
                "recovery_believed": np.nanmean(pad(rb, R), 0).tolist(),
                "recovery_actual": np.nanmean(pad(ra, R), 0).tolist()}
            print(f"[fb] {A}->{B} {cond}: final rec_believed={np.nanmean(pad(rb,R),0)[-1]:.2f} "
                  f"rec_actual={np.nanmean(pad(ra,R),0)[-1]:.2f}", flush=True)
    tf.close()
    tag = "_vs_".join(MODELS)
    json.dump(out, open(os.path.join(RUN_DIR, f"false_belief_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"false_belief_{tag}.pdf"))
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, f"false_belief_{tag}_transcript.jsonl"))
    except Exception:
        pass
    print(f"[fb] DONE -> {RUN_DIR}", flush=True)


def make_fig(out, path):
    pairs = list(out["pairs"])
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 4.8), squeeze=False)
        for ax, k in zip(axes[0], pairs):
            d = out["pairs"][k]
            for cond, style in (("control", "-"), ("false", "--")):
                if cond not in d:
                    continue
                rb = d[cond]["recovery_believed"]; ra = d[cond]["recovery_actual"]
                x = np.arange(1, len(rb) + 1)
                ax.plot(x, rb, style + "o", color="tab:green", label=f"{cond}: recovery(believed T')")
                ax.plot(x, ra, style + "s", color="tab:red", label=f"{cond}: recovery(actual T)")
            ax.set_title(f"{k}\nspy={d['spymaster']} believes T', guesser={d['guesser']}", fontsize=9)
            ax.set_xlabel("turn"); ax.set_ylabel("guesser belief mass"); ax.set_ylim(0, 1)
            ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.suptitle("False-belief: does the guesser recover what the spymaster BELIEVES (T') or the ACTUAL targets (T)?", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
