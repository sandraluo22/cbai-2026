"""CROSS-MODEL KL slides for the yoked/restricted Game-1 runs, ONE SLIDESHOW PER GAME.

REPLAY each recorded game through the model, read each player's next-word distribution per
turn, compute the KL BETWEEN Qwen1's and Qwen2's distributions at that step (both directions;
NO swaps / no coupling). Then write ONE combined slideshow per condition:

  <SRC_DIR>/kl/<cond>_crossKL_curve.pdf     (all games' KL curves, one figure)
  <SRC_DIR>/kl/<cond>_crossKL_perturn.pdf   (every turn of every game, one page each:
                                             both players' guess distributions + picks + KL)
  <SRC_DIR>/kl/<cond>_crossKL.json          (raw per-turn capture, for re-plotting)

PLOT_ONLY=1 skips the model and re-renders the per-game folders from the saved *_crossKL.json
(no GPU -- runs anywhere).

Env: MODEL(QwenInst32) SRC_DIR CONDS(reactive,restrict-city,restrict-fruit) START_FILE
     TOPK(15) PLOT_ONLY(0) DEVICE
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/game-1/qwen32/qwen32_variations")
CONDS = os.environ.get("CONDS", "reactive,restrict-city,restrict-fruit").split(",")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
TOPK = int(os.environ.get("TOPK", "15"))
PLOT_ONLY = os.environ.get("PLOT_ONLY", "0") == "1"
N1, N2 = "Qwen1", "Qwen2"                                 # Qwen1 = free player A, Qwen2 = (maybe restricted) B


def _safe(t):
    s = "".join(c if (c.isascii() and c.isprintable()) else "·" for c in str(t))
    return s if s.strip() else "∅"


def plot_condition(cond, captured, out_dir):
    """ONE merged slideshow per condition (<cond>_crossKL.pdf): page 1 = all games'
    cross-KL curves, then one page per turn per game. Everything is color-coded by the
    GAME's final outcome: green = the game eventually converges, red = it never does."""
    C_MET, C_NO = "tab:green", "tab:red"
    items = sorted(captured.items(), key=lambda kv: int(kv[0]))
    with PdfPages(os.path.join(out_dir, f"{cond}_crossKL.pdf")) as pdf:
        # page 1: overview curves
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for roll, turns in items:
            met = bool(turns[-1]["agreed"])
            x = [t["turn"] for t in turns]
            ax.plot(x, [t["kl_ab"] for t in turns], "-", alpha=.6, color=C_MET if met else C_NO)
            if met:
                ax.plot([turns[-1]["turn"]], [turns[-1]["kl_ab"]], "*", color=C_MET, ms=10)
        ax.plot([], [], "-", color=C_MET, label="game converges (★ = meet turn)")
        ax.plot([], [], "-", color=C_NO, label="game never converges")
        ax.set_xlabel("turn"); ax.set_ylabel("KL(Qwen1‖Qwen2)")
        ax.set_title(f"{cond}: cross-model KL per turn, all games", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # per-turn pages, game outcome color on title + frame
        for roll, turns in items:
            met = bool(turns[-1]["agreed"])
            col = C_MET if met else C_NO
            for t in turns:
                da, db = t["topA"], t["topB"]
                toks = list(dict.fromkeys(list(da) + list(db)))[:20]
                x = np.arange(len(toks))
                fig, ax = plt.subplots(figsize=(12, 4.6))
                ax.bar(x - 0.2, [da.get(k, 0) for k in toks], 0.4, color="tab:blue", label=f"Qwen1 → {t['pickA']}")
                ax.bar(x + 0.2, [db.get(k, 0) for k in toks], 0.4, color="tab:orange", label=f"Qwen2 → {t['pickB']}")
                ax.set_xticks(x); ax.set_xticklabels([_safe(k) for k in toks], rotation=90, fontsize=7)
                ax.set_ylim(0, 1); ax.legend(fontsize=9)
                for sp in ax.spines.values():
                    sp.set_color(col); sp.set_linewidth(2.5)
                ax.set_title(f"{cond} game {roll} [{'CONVERGES' if met else 'NEVER CONVERGES'}], "
                             f"turn {t['turn']}: KL(Q1‖Q2)={t['kl_ab']:.2f}  "
                             f"KL(Q2‖Q1)={t['kl_ba']:.2f}   Q1={t['pickA']}  Q2={t['pickB']}"
                             + ("   ★AGREED" if t["agreed"] else ""), fontsize=10, color=col)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[crossKL] {cond}: wrote {cond}_crossKL.pdf "
          f"(1 curve page + {sum(len(t) for t in captured.values())} turn-pages)", flush=True)


def load_starts():
    out = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            out.append((p[-2], p[-1]))
    return out


def load_games(cond):
    rows = [json.loads(l) for l in open(os.path.join(SRC_DIR, f"game1_yoked_{cond}_transcript.jsonl"))]
    games = {}
    for r in rows:
        games.setdefault(r["rollout"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["turn"])
    return games


def capture(cond, model, tok, dev, starts):
    import torch
    import game1_yoked as Y

    @torch.no_grad()
    def readout(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        logits = None
        for _ in range(3):                                # Qwen3 bolds (`**word**`) -> skip formatting tokens
            logits = model(ids).logits[0, -1].float()
            s = tok.decode([int(logits.argmax())]).strip()
            if s and any(c.isalpha() for c in s):
                break
            ids = torch.cat([ids, logits.argmax()[None, None]], dim=1)
        p = torch.softmax(logits, -1)
        v, i = p.topk(TOPK)
        top = {(tok.decode([tid]).strip() or "·"): round(pv, 4) for tid, pv in zip(i.tolist(), v.tolist())}
        return top, p

    def kl(p, q):
        return float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum())

    concept = cond.split("-", 1)[1] if cond.startswith("restrict-") else None
    games = load_games(cond)
    captured = {}
    for roll, recs in games.items():
        sa, sb = starts[roll]
        histA = [(sb, sa)]; histB = [(sa, sb)]; used = {sa, sb}
        turns = []
        for r in recs:
            wA, wB = r["A"], r["B"]
            topA, pA = readout(Y.build_prompt(tok, histA, used))
            topB, pB = readout(Y.build_prompt(tok, histB, used, restrict=concept))
            turns.append({"turn": r["turn"], "agreed": bool(r["agreed"]), "pickA": wA, "pickB": wB,
                          "topA": topA, "topB": topB, "kl_ab": kl(pA, pB), "kl_ba": kl(pB, pA)})
            histA.append((wB, wA)); histB.append((wA, wB)); used |= {wA, wB}
        captured[roll] = turns
        print(f"[crossKL]   {cond} game {roll} ({sa}/{sb}): {len(turns)} turns", flush=True)
    return captured


def main():
    out_dir = os.path.join(SRC_DIR, "kl")
    os.makedirs(out_dir, exist_ok=True)
    if PLOT_ONLY:
        for cond in CONDS:
            captured = json.load(open(os.path.join(out_dir, f"{cond}_crossKL.json")))
            plot_condition(cond, captured, out_dir)
        print(f"[crossKL] PLOT_ONLY done -> {out_dir}", flush=True)
        return
    import torch
    import llm_agents as LA
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    starts = load_starts()
    for cond in CONDS:
        captured = capture(cond, model, tok, dev, starts)
        json.dump(captured, open(os.path.join(out_dir, f"{cond}_crossKL.json"), "w"))
        plot_condition(cond, captured, out_dir)
    print(f"[crossKL] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
