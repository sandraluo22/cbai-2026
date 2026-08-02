"""Aggregate qsg_gossip v2 transcripts (no GPU): belief-update reputation readouts.

Per step, the listener's belief (softmax over the K labels) was read BEFORE and
AFTER the speaker's label entered its memory. We report the DIRECTED update
    dm = p_after[spoken] - p_before[spoken]
(mass moved toward the label just heard) and the total variation TV(p_after,
p_before), grouped by:
  1) the listener's conversation depth (its 1st-5th conversation vs later bins)
  2) the source identity (P1 = the potentially informed/misinformed agent vs rest)
  3) round.
Reputation = the source term: dm(P1) vs dm(others), and its trajectory over rounds
(rising in informed conditions / falling in misinformed = trust formation).
Also: per-round probe accuracy and echo-of-previous-reveal rate.

usage: python gossip_analyze.py [dir-with-run-subdirs]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
DEPTH_BINS = [(1, 5), (6, 15), (16, 10**9)]


def bin_depth(d):
    for lo, hi in DEPTH_BINS:
        if lo <= d <= hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"


def analyze(path):
    lines = [json.loads(l) for l in open(path)]
    meta = lines[0]
    starts = {l["round"]: l for l in lines if l["type"] == "round_start"}
    idx_by_round = {r: {l: i for i, l in enumerate(st.get("labels", meta["labels"]))}
                    for r, st in starts.items()}
    correct = {r: st["correct"] for r, st in starts.items()}
    by_src, by_depth, by_round_src = defaultdict(list), defaultdict(list), defaultdict(list)
    tv_all = []
    for l in lines:
        if l["type"] != "step":
            continue
        p0, p1 = np.array(l["p_before"]), np.array(l["p_after"])
        idx = idx_by_round[l["round"]]
        dm = p1[idx[l["s_label"]]] - p0[idx[l["s_label"]]]
        tv = 0.5 * np.abs(p1 - p0).sum()
        src = (f"P{l['S']}" if meta.get("var") == "duel" and l["S"] in (1, 2)
               else ("P1" if l["S"] == 1 else "rest"))
        by_src[src].append(dm)
        by_depth[bin_depth(l["conv_depth"])].append(dm)
        by_round_src[(l["round"], src)].append(dm)
        tv_all.append(tv)
    acc, echo = {}, {}
    for l in lines:
        if l["type"] == "probe":
            r = l["round"]
            acc[r] = sum(p["correct"] for p in l["probes"]) / len(l["probes"])
            if r - 1 in correct:
                echo[r] = sum(p["argmax"] == correct[r - 1] for p in l["probes"]) / len(l["probes"])
    return meta, acc, echo, by_src, by_depth, by_round_src, tv_all, correct


def fm(vals):
    return f"{np.mean(vals):+.3f}±{np.std(vals) / max(1, len(vals)) ** .5:.3f}(n={len(vals)})" \
        if vals else "   -   "


def main():
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if not f.endswith("_transcript.jsonl"):
                continue
            meta, acc, echo, by_src, by_depth, by_round_src, tv, correct = \
                analyze(os.path.join(sub, f))
            rs = sorted(acc)
            rep = {r for r in rs if r - 1 in correct and correct[r] == correct[r - 1]}
            print("=" * 78)
            print(f"{d}  (N={meta['n']} K={meta['k']} steps={meta['steps']}/round, "
                  f"chance={1 / meta['k']:.2f}, mean TV/step={np.mean(tv):.3f})")
            print("  probe_acc  :", " ".join(
                f"r{r}={acc[r]:.2f}{'*' if r in rep else ''}" for r in rs),
                " (*=truth repeats prev round)")
            print("  echo_prev  :", " ".join(f"r{r}={echo[r]:.2f}" for r in sorted(echo)))
            srcs = [k for k in ("P1", "P2", "rest") if by_src.get(k)]
            print("  dm by src  :", "  ".join(f"{s}: {fm(by_src[s])}" for s in srcs))
            print("  dm by depth:", "  ".join(
                f"{b}: {fm(by_depth[b])}" for b in (bin_depth(1), bin_depth(6), bin_depth(16))))
            print("  dm src x round:")
            for r in rs:
                print("    r%d:  " % r + "   ".join(
                    f"{s2} {fm(by_round_src[(r, s2)])}" for s2 in srcs))


if __name__ == "__main__":
    main()
