"""Two real models play OPEN-CLUE Codenames; MEASURE the mutual theory-of-mind.

Same measurement story as game_llm.py, but the REAL Codenames mechanic:
  * the spymaster free-generates a SINGLE one-word clue that is NOT on the board and
    that connects TWO of its targets (its clue "vocabulary" is the whole model vocab,
    restricted to the top-N valid off-board word tokens);
  * count = 2 -- the guesser guesses its TOP-2 board words each round;
  * the board is a FLAT, CROSS-CUTTING set (no clean category partition), so recovery
    is achievable but no longer trivial.

Per round we log:
  * recovery   : B's target-posterior mass (does B infer A's hidden set).
  * coupling   : KL(B_guess | swapped clue || B_guess | real clue), bounded over the
                 12 board words. The swap is the spymaster's 2nd-best word (the open
                 analog of the old c -> c' swap), so it stays in-distribution.
  * adaptivity : top-N full-vocab KL(A_clue | guesser-found-X || A_clue | naive). The
                 clue channel is now the whole vocab, so this is a top-N union KL
                 (same read-out as game1open_llm.py), not a KL over C=4 tokens.

Both role orderings, so we get the mutual picture. KLs are continuous ("how much
theory-of-mind did it show"), not an L1/L2 label.

Env: MODELS(LlamaInst,QwenInst) M(4) ROUNDS(4) GAMES(8) TOPN(50) DEVICE RUN_DIR
Out: <RUN_DIR>/game_llm_open_<A>_vs_<B>.json + .pdf + _transcript.jsonl (+ .json)
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
N = len(LA.OPEN_BOARD)                                  # flat cross-cutting board = 12 words
M = int(os.environ.get("M", "4"))                       # hidden targets per game
ROUNDS = int(os.environ.get("ROUNDS", "4"))
GAMES = int(os.environ.get("GAMES", "8"))
TOPN = int(os.environ.get("TOPN", "50"))
COUNT = 2                                               # words per clue / guesses per round
SPY_MEMORY = os.environ.get("SPY_MEMORY", "0") == "1"   # spymaster sees & never repeats past clues
SPY_SEES = os.environ.get("SPY_SEES", "remaining")      # "remaining" | "eliminated" (also sees wrong guesses)
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/llm_codenames_open")


def _clue_topk(tok, lg, k=10):
    """Top-k (token -> prob) of a full-vocab clue-position distribution, for the
    per-turn slideshow (the two distributions behind the adaptivity KL)."""
    import torch
    p = torch.softmax(lg, 0)
    v, idx = torch.topk(p, k)
    return {tok.decode([int(i)]).strip(): round(float(pr), 4) for pr, i in zip(v, idx)}


def topn_kl(lp, lq, n=TOPN):
    """KL(softmax(lp) || softmax(lq)) over the union of each side's top-N tokens plus a
    residual bucket -- tractable full-vocab KL (as in game1open_llm.py)."""
    import torch
    p = torch.softmax(lp, 0); q = torch.softmax(lq, 0)
    n = min(n, p.shape[-1])
    idx = torch.unique(torch.cat([torch.topk(p, n).indices, torch.topk(q, n).indices]))
    pp, qq = p[idx], q[idx]
    P = torch.cat([pp, (1 - pp.sum()).clamp(min=1e-9).view(1)]); P = P / P.sum()
    Q = torch.cat([qq, (1 - qq.sum()).clamp(min=1e-9).view(1)]); Q = Q / Q.sum()
    return float((P * (P.clamp(min=1e-12).log() - Q.clamp(min=1e-12).log())).sum())


def _pad(rows, R):
    a = np.full((len(rows), R), np.nan)
    for i, row in enumerate(rows):
        a[i, :len(row)] = row
    return a


def play_pair(mA, tA, mB, tB, dev, words, sname, gname, tf=None):
    """A = spymaster (its adaptivity = A's ToM of B); B = guesser (its coupling = B's ToM of A).
    A game ends the moment all targets are found (ROUNDS is only a safety cap), so game
    length is data-dependent -- the returned per-game lists are ragged."""
    mass, coup, adapt, coup2, finished = [], [], [], [], []
    for gi in range(GAMES):
        rng = np.random.default_rng(gi)
        targets = sorted(rng.choice(N, M, replace=False).tolist())
        S = LA.LLMSpeakerOpen(mA, tA, words, targets, dev, remember=SPY_MEMORY,
                              sees_eliminated={"eliminated": True, "inferred": "inferred"}.get(SPY_SEES, False))
        B = LA.LLMListenerOpen(mB, tB, words, dev)
        gm, gc, ga, gc2 = [], [], [], []
        for r in range(ROUNDS):
            if not S.remaining:                           # all targets already found -> stop
                break
            real, swap, count = S.clue(B)
            # coupling (1st pick): guesser belief under the real clue vs the swapped clue,
            # from an identical state (bounded KL over the 12 board words).
            clean = B.copy(); clean.update(real, count)
            swp = B.copy(); swp.update(swap, count)
            d1_clean, d1_swap = clean.guess_dist(), swp.guess_dist()
            kl_coup = K.kl(d1_swap, d1_clean)
            gc.append(kl_coup)
            # coupling (2nd pick): hold the first pick fixed (= clean's 1st pick), then measure
            # the clue-swap effect on the AUTOREGRESSIVE second-pick distribution.
            c1 = int(np.argmax(d1_clean))
            d2_clean, d2_swap = clean.second_dist(c1), swp.second_dist(c1)
            kl_coup2 = K.kl(d2_swap, d2_clean)
            gc2.append(kl_coup2)
            # adaptivity: spymaster clue distribution given B's revealed state vs a naive
            # (fresh) guesser -- top-N full-vocab KL.
            naive = LA.LLMListenerOpen(mB, tB, words, dev)
            lg_real, lg_naive = S.clue_logits(B), S.clue_logits(naive)
            kl_adapt = topn_kl(lg_real, lg_naive)
            ga.append(kl_adapt)
            belief = B.belief()
            found_before = [words[i] for i in sorted(B.known)]
            B.update(real, count)
            guesses = B.pick_guesses(count)               # sequential picks (2nd autoregressive)
            results = []
            for g in guesses:
                ok = g in S.remaining
                results.append((words[g], bool(ok)))
                B.observe(g, ok); S.observe(g, ok)
            if tf is not None:
                tf.write(json.dumps({
                    "pair": f"{sname}->{gname}", "spymaster": sname, "guesser": gname,
                    "game": gi, "round": r, "targets": [words[i] for i in targets],
                    "clue": real, "count": count, "guesses": [w for w, _ in results],
                    "correct": [ok for _, ok in results], "found_so_far": found_before,
                    "target_mass": round(float(K.target_mass(belief, targets)), 4),
                    "belief": {words[i]: round(float(belief[i]), 4) for i in range(N)},
                    "coupling": {"kl": round(float(kl_coup), 4), "clue_swapped_to": swap,
                                 "guess_dist_clean": {words[i]: round(float(d1_clean[i]), 4) for i in range(N)},
                                 "guess_dist_swap": {words[i]: round(float(d1_swap[i]), 4) for i in range(N)}},
                    "coupling2": {"kl": round(float(kl_coup2), 4), "cond_first": words[c1],
                                  "guess_dist2_clean": {words[i]: round(float(d2_clean[i]), 4) for i in range(N)},
                                  "guess_dist2_swap": {words[i]: round(float(d2_swap[i]), 4) for i in range(N)}},
                    "adaptivity": {"kl": round(float(kl_adapt), 4), "topn": TOPN,
                                   "clue_dist_real": _clue_topk(tA, lg_real),
                                   "clue_dist_naive": _clue_topk(tA, lg_naive)},
                }) + "\n")
            S.note_clue(real)                             # remember this round's clue (no-op if remember=False)
            gm.append(K.target_mass(B.belief(), targets))
        mass.append(gm); coup.append(gc); adapt.append(ga); coup2.append(gc2); finished.append(not S.remaining)
        print(f"    game {gi}: rounds={len(gm)} done={not S.remaining} recovery={gm[-1]:.2f} "
              f"coupling1={np.mean(gc):.2f} coupling2={np.mean(gc2):.2f} adaptivity={np.mean(ga):.2f}", flush=True)
    return mass, coup, adapt, coup2, finished


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    words = LA.OPEN_BOARD

    loaded = {}
    for m in set(MODELS):
        print(f"[open-cn] loading {m}", flush=True)
        loaded[m] = LA.load(m, dev)

    tag = "_vs_".join(MODELS)
    tf = open(os.path.join(RUN_DIR, f"game_llm_open_{tag}_transcript.jsonl"), "w")
    out = {"models": MODELS, "N": N, "M": M, "count": COUNT, "topn": TOPN,
           "rounds": ROUNDS, "games": GAMES, "board": words, "spy_memory": SPY_MEMORY,
           "spy_sees": SPY_SEES, "pairs": {}}
    for A, B in itertools.permutations(MODELS, 2):     # both role orderings
        print(f"[open-cn] {A} = spymaster,  {B} = guesser", flush=True)
        mass, coup, adapt, coup2, finished = play_pair(*loaded[A], *loaded[B], dev, words, A, B, tf)
        R = max(len(x) for x in mass)
        Mp, Cp, Ap, C2p = _pad(mass, R), _pad(coup, R), _pad(adapt, R), _pad(coup2, R)
        nper = np.sum(~np.isnan(Mp), 0)                    # games contributing at each round
        key = f"{A}->{B}"
        out["pairs"][key] = {
            "spymaster": A, "guesser": B, "max_rounds": R,
            "recovery_mean": np.nanmean(Mp, 0).tolist(),
            "recovery_se": (np.nanstd(Mp, 0) / np.sqrt(nper)).tolist(),
            "n_per_round": nper.tolist(),
            "coupling_mean": np.nanmean(Cp, 0).tolist(),   # B's ToM of A (1st pick)
            "coupling2_mean": np.nanmean(C2p, 0).tolist(), # 2nd-pick coupling
            "adaptivity_mean": np.nanmean(Ap, 0).tolist(), # A's ToM of B
            "coupling_overall": float(np.nanmean(Cp)), "coupling2_overall": float(np.nanmean(C2p)),
            "adaptivity_overall": float(np.nanmean(Ap)),
            "final_recovery": float(np.mean([m[-1] for m in mass])),
            "turns_mean": float(np.mean([len(m) for m in mass])),
            "completed_frac": float(np.mean(finished))}
        print(f"[open-cn] {key}: recovery={out['pairs'][key]['final_recovery']:.2f} "
              f"coupling1={out['pairs'][key]['coupling_overall']:.2f} "
              f"coupling2={out['pairs'][key]['coupling2_overall']:.2f} "
              f"adaptivity={out['pairs'][key]['adaptivity_overall']:.2f} "
              f"turns={out['pairs'][key]['turns_mean']:.1f} done={out['pairs'][key]['completed_frac']*100:.0f}%", flush=True)

    tf.close()
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, f"game_llm_open_{tag}_transcript.jsonl"))
    except Exception as e:
        print(f"[open-cn] json convert skipped: {e}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, f"game_llm_open_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"game_llm_open_{tag}.pdf"))
    print(f"[open-cn] DONE -> {RUN_DIR}/game_llm_open_{tag}.json  (+ transcript.jsonl)", flush=True)


def make_fig(out, path):
    pairs = list(out["pairs"])
    cols = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 2)))
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for i, k in enumerate(pairs):
            d = out["pairs"][k]
            rounds = np.arange(1, len(d["recovery_mean"]) + 1)
            lab = f"{k}  (turns {d['turns_mean']:.1f}, done {d['completed_frac']*100:.0f}%)"
            ax[0].errorbar(rounds, d["recovery_mean"], yerr=d["recovery_se"], color=cols[i], label=lab, capsize=2)
            ax[1].plot(rounds, d["coupling_mean"], "-o", color=cols[i], label=f"{k} 1st-pick")
            ax[1].plot(rounds, d["coupling2_mean"], "--s", color=cols[i], label=f"{k} 2nd-pick")
            ax[2].plot(rounds, d["adaptivity_mean"], "-o", color=cols[i], label=f"{k}  (spymaster={d['spymaster']})")
        ax[0].set_title("Recovery: guesser's target-posterior mass", fontsize=9); ax[0].set_ylim(0, 1)
        ax[1].set_title("Coupling = guesser's ToM of spymaster\nKL(guess|swap||guess|real): 1st pick (solid) vs 2nd pick (dashed)", fontsize=9)
        ax[2].set_title(f"Adaptivity = spymaster's ToM of guesser\ntop-{out['topn']} KL(clue|B-found-X||clue|B-naive)", fontsize=9)
        for a in ax:
            a.set_xlabel("round"); a.legend(fontsize=7); a.grid(alpha=.3)
        fig.suptitle(f"OPEN-CLUE Codenames between {' & '.join(out['models'])} "
                     f"(free one-word clue, count=2, cross-cutting board; neutral instructions).", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
