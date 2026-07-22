"""Can a spymaster that only sees the REMAINING set approximate the update it would make
if it also knew the guesser's ELIMINATED (wrong-guessed) words?

Idea (theory of mind): the guesser predictably picks the top board words for each clue, so
the spymaster can INFER the likely eliminated set by simulating a guesser on its OWN past
clues (any non-target it lands on is a likely wrong guess) -- no ground-truth needed.

The game is driven by a remaining-only spymaster. At each round, from the identical state,
we read the spymaster's clue distribution under three eliminated-sets:
    none      (remaining-only, the baseline)
    true      (ground-truth listener.dead)
    inferred  (self-simulated, infer_dead)
and measure top-N KL(none||true) vs KL(inferred||true): if inferred is closer to true, the
update is well approximated. We also score the inferred set vs the true set (precision/recall).

Env: MODELS(LlamaInst,QwenInst) GAMES(12) ROUNDS(8) M(4) TOPN(50) DEVICE RUN_DIR
Out: <RUN_DIR>/infer_dead_<A>_vs_<B>.{json,pdf} + _transcript.jsonl
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
GAMES = int(os.environ.get("GAMES", "12"))
CAP = int(os.environ.get("ROUNDS", "8"))
M = int(os.environ.get("M", "4"))
N = len(LA.OPEN_BOARD)
TOPN = int(os.environ.get("TOPN", "50"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/infer_dead")


def topn_kl(lp, lq, n=TOPN):
    import torch
    p = torch.softmax(lp, 0); q = torch.softmax(lq, 0)
    n = min(n, p.shape[-1])
    idx = torch.unique(torch.cat([torch.topk(p, n).indices, torch.topk(q, n).indices]))
    pp, qq = p[idx], q[idx]
    P = torch.cat([pp, (1 - pp.sum()).clamp(min=1e-9).view(1)]); P = P / P.sum()
    Q = torch.cat([qq, (1 - qq.sum()).clamp(min=1e-9).view(1)]); Q = Q / Q.sum()
    return float((P * (P.clamp(min=1e-12).log() - Q.clamp(min=1e-12).log())).sum())


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    words = LA.OPEN_BOARD
    loaded = {m: LA.load(m, dev) for m in set(MODELS)}

    tf = open(os.path.join(RUN_DIR, f"infer_dead_{'_vs_'.join(MODELS)}_transcript.jsonl"), "w")
    out = {"models": MODELS, "N": N, "M": M, "topn": TOPN, "games": GAMES, "pairs": {}}
    for A, B in itertools.permutations(MODELS, 2):                 # A=spymaster, B=guesser
        mS, tS = loaded[A]; mG, tG = loaded[B]
        rows = []                                                 # per (game,round)
        for gi in range(GAMES):
            rng = np.random.default_rng(gi)
            targets = sorted(rng.choice(N, M, replace=False).tolist())
            S = LA.LLMSpeakerOpen(mS, tS, words, targets, dev, remember=True)   # remaining-only driver
            G = LA.LLMListenerOpen(mG, tG, words, dev)
            for r in range(CAP):
                if not S.remaining:
                    break
                true_dead = set(G.dead)
                inferred = S.infer_dead()
                if true_dead or inferred:                          # only meaningful once misses exist
                    lg_none = S.clue_logits(G, dead_override=set())
                    lg_true = S.clue_logits(G, dead_override=true_dead)
                    lg_inf = S.clue_logits(G, dead_override=inferred)
                    kl_none = topn_kl(lg_none, lg_true)            # baseline gap to the informed clue
                    kl_inf = topn_kl(lg_inf, lg_true)              # residual gap after inference
                    inter = inferred & true_dead
                    prec = len(inter) / len(inferred) if inferred else float("nan")
                    rec = len(inter) / len(true_dead) if true_dead else float("nan")
                    rows.append({"game": gi, "round": r + 1, "kl_none_true": kl_none, "kl_inf_true": kl_inf,
                                 "n_true": len(true_dead), "n_inferred": len(inferred),
                                 "precision": prec, "recall": rec,
                                 "true_dead": [words[i] for i in sorted(true_dead)],
                                 "inferred_dead": [words[i] for i in sorted(inferred)]})
                    tf.write(json.dumps({"pair": f"{A}->{B}", **rows[-1]}) + "\n")
                # advance the game with the remaining-only clue
                real, swap, count = S.clue(G)
                G.update(real, count)
                for g in G.pick_guesses(count):
                    ok = g in S.remaining
                    G.observe(g, ok); S.observe(g, ok)
                S.note_clue(real)
            print(f"    [{A}->{B}] game {gi}: {len([x for x in rows if x['game']==gi])} scored rounds", flush=True)

        kln = np.array([x["kl_none_true"] for x in rows]); kli = np.array([x["kl_inf_true"] for x in rows])
        prec = np.array([x["precision"] for x in rows]); rec = np.array([x["recall"] for x in rows])
        key = f"{A}->{B}"
        out["pairs"][key] = {
            "spymaster": A, "guesser": B, "n_scored_rounds": len(rows),
            "kl_none_true_mean": float(np.nanmean(kln)), "kl_inf_true_mean": float(np.nanmean(kli)),
            "gap_closed_frac": float(1 - np.nanmean(kli) / np.nanmean(kln)) if np.nanmean(kln) > 0 else float("nan"),
            "precision_mean": float(np.nanmean(prec)), "recall_mean": float(np.nanmean(rec)),
            "rows": rows}
        p = out["pairs"][key]
        print(f"[infer] {key}: KL(none||true)={p['kl_none_true_mean']:.2f} -> KL(inferred||true)={p['kl_inf_true_mean']:.2f} "
              f"(gap closed {p['gap_closed_frac']*100:.0f}%)  inferred-set precision={p['precision_mean']:.2f} recall={p['recall_mean']:.2f}", flush=True)
    tf.close()
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, f"infer_dead_{'_vs_'.join(MODELS)}_transcript.jsonl"))
    except Exception:
        pass
    tag = "_vs_".join(MODELS)
    json.dump({k: (v if k != "pairs" else {kk: {x: y for x, y in vv.items() if x != "rows"}
                                           for kk, vv in v.items()}) for k, v in out.items()},
              open(os.path.join(RUN_DIR, f"infer_dead_{tag}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"infer_dead_{tag}.pdf"))
    print(f"[infer] DONE -> {RUN_DIR}", flush=True)


def make_fig(out, path):
    pairs = list(out["pairs"])
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for i, k in enumerate(pairs):
            rows = out["pairs"][k]["rows"]
            byr = {}
            for x in rows:
                byr.setdefault(x["round"], {"n": [], "i": [], "p": [], "r": []})
                byr[x["round"]]["n"].append(x["kl_none_true"]); byr[x["round"]]["i"].append(x["kl_inf_true"])
                byr[x["round"]]["p"].append(x["precision"]); byr[x["round"]]["r"].append(x["recall"])
            ts = sorted(byr)
            c = plt.cm.tab10(i)
            ax[0].plot(ts, [np.nanmean(byr[t]["n"]) for t in ts], "-o", color=c, label=f"{k}  KL(none‖true)")
            ax[0].plot(ts, [np.nanmean(byr[t]["i"]) for t in ts], "--s", color=c, label=f"{k}  KL(inferred‖true)")
            ax[1].plot(ts, [np.nanmean(byr[t]["p"]) for t in ts], "-o", color=c, label=f"{k}  precision")
            ax[1].plot(ts, [np.nanmean(byr[t]["r"]) for t in ts], "--s", color=c, label=f"{k}  recall")
        ax[0].set_title("Clue-distribution gap to the informed (true-dead) clue\nlower dashed = inference recovers the update", fontsize=9)
        ax[0].set_xlabel("round"); ax[0].set_ylabel("top-N KL to true-dead clue"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        ax[1].set_title("Inferred eliminated-set vs true eliminated-set", fontsize=9)
        ax[1].set_xlabel("round"); ax[1].set_ylabel("precision / recall"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.suptitle("Approximating the eliminated-words update from remaining-only, via guesser self-simulation", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
