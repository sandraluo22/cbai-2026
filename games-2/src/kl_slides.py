"""Every-turn KL slideshows from ANY Game-1 transcript (bounded / semantic / topic /
unbounded) -- a page for EVERY turn of EVERY game, plus per-game KL curves. No GPU.

Handles both transcript formats automatically:
  * bounded/semantic: each model has a full 'dist' {word: prob} + 'coupling'.'swap_dist'
  * unbounded:        each model has 'top' {token: prob} + 'coupling'.'swap_top', and
                      exact per-turn 'coupling'.kl / 'step'.kl stored from full logits.

Coupling KL and (when present) step KL are taken EXACT from the transcript; otherwise
step KL is computed from consecutive distributions. Bars show the distributions behind
each number over the union of their top tokens.

Usage:  python src/kl_slides.py <transcript.jsonl>
Out:    <base>_coupling_allturns.pdf   <base>_stepkl_allturns.pdf
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core as K


def load(tpath):
    recs = [json.loads(l) for l in open(tpath)]
    names = list(recs[0]["picks"].keys())
    games = {}
    for r in recs:
        games.setdefault(r["game"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["turn"])
    return names, games


def clean_dist(rec, name):
    return rec.get(name, {}).get("dist") or rec.get(name, {}).get("top") or {}


def swap_dist(rec, name):
    cp = rec.get(name, {}).get("coupling")
    if not cp:
        return None, None
    return cp["kl"], (cp.get("swap_dist") or cp.get("swap_top") or {})


def dict_kl(p, q):
    keys = list(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys]); qv = np.array([q.get(k, 0.0) for k in keys])
    P = np.append(pv, max(0.0, 1 - pv.sum())); Q = np.append(qv, max(0.0, 1 - qv.sum()))
    return K.kl(P, Q)


def step_kl(rec, name, prev):
    st = rec.get(name, {}).get("step")
    if st is not None:
        return st["kl"]
    if prev is None:
        return None
    return dict_kl(clean_dist(rec, name), prev)


def _safe(t):
    """ASCII-safe label so non-Latin tokens (e.g. Qwen CJK subwords) don't render as
    tofu boxes / raise font warnings."""
    s = "".join(c if (c.isascii() and c.isprintable()) else "·" for c in str(t))
    return s if s.strip() else "∅"


def _bars(ax, da, db, la, lb, ca, title):
    toks = list(dict.fromkeys(list(da) + list(db)))[:18]
    x = np.arange(len(toks))
    ax.bar(x - 0.2, [da.get(t, 0) for t in toks], 0.4, color=ca, alpha=.85, label=la)
    ax.bar(x + 0.2, [db.get(t, 0) for t in toks], 0.4, color="0.55", alpha=.85, label=lb)
    ax.set_xticks(x); ax.set_xticklabels([_safe(t) for t in toks], rotation=90, fontsize=6); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7)


def build(names, games, which, out):
    nA, nB = names
    with PdfPages(out) as pdf:
        # per-game KL curves (every turn of every game visible)
        fig, ax = plt.subplots(figsize=(9, 5))
        for gi, recs in games.items():
            ta, va = [], []; tb, vb = [], []
            prev = {nA: None, nB: None}
            for r in recs:
                for nm, tt, vv in ((nA, ta, va), (nB, tb, vb)):
                    k = swap_dist(r, nm)[0] if which == "coupling" else step_kl(r, nm, prev[nm])
                    if k is not None:
                        tt.append(r["turn"] + 1); vv.append(k)
                prev[nA], prev[nB] = clean_dist(r, nA), clean_dist(r, nB)
            ax.plot(ta, va, "-o", ms=3, alpha=.7, color="tab:blue", label=(nA if gi == 0 else None))
            ax.plot(tb, vb, "-s", ms=3, alpha=.7, color="tab:orange", label=(nB if gi == 0 else None))
        ax.set_xlabel("turn"); ax.set_ylabel(f"{which} KL")
        ax.set_title(f"{which} KL every turn, every game ({len(games)} games) — {os.path.basename(out)}", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # one page per (game, turn)
        for gi, recs in games.items():
            prev = {nA: None, nB: None}
            for r in recs:
                kA = swap_dist(r, nA)[0] if which == "coupling" else step_kl(r, nA, prev[nA])
                kB = swap_dist(r, nB)[0] if which == "coupling" else step_kl(r, nB, prev[nB])
                if kA is not None and kB is not None:
                    fig = plt.figure(figsize=(13, 5)); gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
                    axk, axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
                    axk.bar([0, 1], [kA, kB], color=["tab:blue", "tab:orange"])
                    axk.set_xticks([0, 1]); axk.set_xticklabels([nA, nB], fontsize=8, rotation=20)
                    axk.set_ylabel(f"{which} KL"); axk.set_title(which, fontsize=9)
                    for axm, nm, kv, c in ((axA, nA, kA, "tab:blue"), (axB, nB, kB, "tab:orange")):
                        now = clean_dist(r, nm)
                        if which == "coupling":
                            other = swap_dist(r, nm)[1]; lb = "swap→" + r[nm]["coupling"].get("swap_other_to", "?")
                        else:
                            other = prev[nm] or {}; lb = "prev turn"
                        _bars(axm, now, other, "now", lb, c, f"{nm}: {r[nm]['pick']}  KL={kv:.2f}")
                    fig.suptitle(f"game {gi}, turn {r['turn']+1}: {nA}={r['picks'][nA]}, {nB}={r['picks'][nB]}"
                                 + ("  ★AGREED" if r["agreed"] else ""), fontsize=11)
                    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
                prev[nA], prev[nB] = clean_dist(r, nA), clean_dist(r, nB)
    print(f"  {out}")


def generate(tpath):
    """Emit both every-turn KL slideshows next to a transcript, plus a readable .json
    copy of the transcript. Callable from runners."""
    base = os.path.splitext(tpath)[0]
    names, games = load(tpath)
    total = sum(len(v) for v in games.values())
    print(f"[kl_slides] {os.path.basename(tpath)}: {len(games)} games, {total} turns total", flush=True)
    build(names, games, "coupling", base + "_coupling_allturns.pdf")
    build(names, games, "step", base + "_stepkl_allturns.pdf")
    try:
        import jsonl_to_json
        jsonl_to_json.convert(tpath)                    # also emit a browser-openable .json
    except Exception as e:
        print(f"[kl_slides] json convert skipped: {e}", flush=True)


def main():
    generate(sys.argv[1])


if __name__ == "__main__":
    main()
