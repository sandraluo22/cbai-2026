"""Mechanistic analysis of the reliability cliff (Qwen3-32B, N=3 duel microscope).

Stage 0  behavioral check: logit margin for the reliable source's label in matched
         synthetic contexts (duel 10/10-vs-0, duel 8/10-vs-0, single source, tally
         true/false, explicit who-is-reliable query).
Stage 1  per-layer linear probes: decode WHICH source (P1/P2) is the reliable one
         from the residual stream at the answer position, per context type.
Stage 2  activation patching duel100 -> duel80 (token-aligned minimal pair):
         per-layer x position-group recovery of the reliable-label logit margin.
         Groups: current-round source-name tokens, reveal rows, last history
         token, answer (prefill) tokens.

Outputs mech_out/*.json + printed summaries. Stage 3 (head/MLP path patching)
is a follow-up script once this localizes.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
import qsg_gossip_fast as Q  # noqa: E402  (user_msg, make_labels-compatible format)
from run_games import load  # noqa: E402

OUT = os.path.join(_HERE, "mech_out")
os.makedirs(OUT, exist_ok=True)
R_HIST = 10                                        # history rounds
DEV = "cuda"


def same_len_labels(tok, rng, k=3):
    """Labels that all tokenize to the same number of tokens (patch alignment)."""
    while True:
        labs = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                for _ in range(k)]
        if len(set(labs)) < k:
            continue
        lens = [len(tok(l, add_special_tokens=False)["input_ids"]) for l in labs]
        firsts = [tok(l, add_special_tokens=False)["input_ids"][0] for l in labs]
        if len(set(lens)) == 1 and len(set(firsts)) == k:
            return labs


def build(mode, rel_src, rng, tok):
    """One matched context. rel_src in (1,2) = which source has the good record.
    Returns dict(text, labels(X=rel label,Y=other,Z), spans{group:[(c0,c1)..]})."""
    bad_src = 2 if rel_src == 1 else 1
    hist = [same_len_labels(tok, rng) for _ in range(R_HIST)]
    cur = same_len_labels(tok, rng)
    X, Y, Z = cur                                   # X = reliable source's current label
    mem, reveals = [], {}
    err_rounds = {2, 6} if mode == "duel80" else set()
    swap = os.environ.get("SWAPORDER", "") == "1"   # control: put the BAD source first
    for r0, (a, b, _) in enumerate(hist, start=1):
        if swap and mode != "single":
            mem.append((r0, bad_src, b))
            mem.append((r0, rel_src, a))
        else:
            mem.append((r0, rel_src, a))
            if mode != "single":
                mem.append((r0, bad_src, b))
        if mode not in ("tally_true", "tally_false"):  # tally: table is the ONLY signal
            reveals[r0] = b if r0 in err_rounds else a
    r = R_HIST + 1
    if swap and mode != "single":
        mem.append((r, bad_src, Y))
        mem.append((r, rel_src, X))
    else:
        mem.append((r, rel_src, X))
        if mode != "single":
            mem.append((r, bad_src, Y))
    tally = ""
    if mode in ("tally_true", "tally_false"):
        g, bcnt = (10, 0) if mode == "tally_true" else (0, 10)
        tally = (f"Track record so far: P{rel_src} {g}/10 correct; "
                 f"P{bad_src} {bcnt}/10 correct.\n")
    labels = sorted([X, Y, Z], key=lambda _: rng.random())
    msg = Q.user_msg(2, labels, mem, reveals, r, None, rng, False, False, "", tally)
    if mode == "query":
        msg = (msg.split("\nConstraints:")[0]
               + "\nQuestion: which player has been more reliable so far?"
               + '\n\nConstraints:\n- Output JSON only.\n\n'
               + 'Output JSON exactly: {"player": "<P1 or P2>"}')
        text = msg + '\n{"player": "P'
    else:
        text = msg + '{"label": "'
    spans = dict(cur_names=[], reveal_rows=[], last_hist=[], answer=[])
    tagline = f"Round {r} memories (current round)"
    ti = text.find(tagline)
    if ti >= 0:
        line_end = text.find("\n", ti)
        for pn in (f"P{rel_src}:", f"P{bad_src}:"):
            j = text.find(pn, ti, line_end)
            if j >= 0:
                spans["cur_names"].append((j, j + 2))
    k0 = 0
    for r0 in range(1, R_HIST + 1):
        pat = f'The correct answer for round {r0} was "'
        j = text.find(pat, k0)
        if j >= 0:
            spans["reveal_rows"].append((j, text.find("\n", j)))
            k0 = j + 1
    eh = text.find("Each memory entry is of the form")
    if eh > 0:
        spans["last_hist"].append((eh - 2, eh - 1))
    spans["answer"].append((len(text) - 12, len(text)))
    return dict(text=text, X=X, Y=Y, Z=Z, labels=labels, spans=spans, rel=rel_src,
                mode=mode)


def encode_with_spans(tok, ex):
    enc = tok(ex["text"], return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    idx = {}
    for g, ranges in ex["spans"].items():
        ids = [i for i, (a, b) in enumerate(offs)
               if any(a < c1 and b > c0 for (c0, c1) in ranges)]
        idx[g] = ids
    return {k: v.to(DEV) for k, v in enc.items()}, idx


@torch.no_grad()
def margins(model, tok, enc, ex):
    out = model(**enc, output_hidden_states=False)
    lg = out.logits[0, -1]
    if ex["mode"] == "query":
        i1 = tok("1", add_special_tokens=False)["input_ids"][0]
        i2 = tok("2", add_special_tokens=False)["input_ids"][0]
        rel = lg[i1] if ex["rel"] == 1 else lg[i2]
        oth = lg[i2] if ex["rel"] == 1 else lg[i1]
        return float(rel - oth)
    ix = tok(ex["X"], add_special_tokens=False)["input_ids"][0]
    iy = tok(ex["Y"], add_special_tokens=False)["input_ids"][0]
    return float(lg[ix] - lg[iy])


@torch.no_grad()
def hidden_last(model, enc):
    out = model(**enc, output_hidden_states=True)
    return [h[0, -1].float().cpu().numpy() for h in out.hidden_states]  # len L+1


def probe_stage(model, tok, n_per=24, seed=0):
    rng = random.Random(seed)
    modes = ["duel100", "duel80", "single", "tally_true", "tally_false", "query"]
    feats = {m: [] for m in modes}
    ys = {m: [] for m in modes}
    behav = {m: [] for m in modes}
    for m in modes:
        for i in range(n_per):
            rel = 1 if i % 2 == 0 else 2
            ex = build(m, rel, rng, tok)
            enc, _ = encode_with_spans(tok, ex)
            feats[m].append(hidden_last(model, enc))
            ys[m].append(rel)
            behav[m].append(margins(model, tok, enc, ex))
        print(f"[probe-data] {m}: margin(rel-label) mean {np.mean(behav[m]):+.3f} "
              f"sd {np.std(behav[m]):.3f} n={n_per}", flush=True)
    L = len(feats[modes[0]][0])
    acc = {m: [] for m in modes}
    for m in modes:
        Xs = np.stack([np.stack(f) for f in feats[m]])   # n x L x d
        y = np.array(ys[m])
        for l in range(L):
            A = Xs[:, l]
            A = (A - A.mean(0)) / (A.std(0) + 1e-6)
            tr = slice(0, len(y) // 2)
            te = slice(len(y) // 2, None)
            w = np.zeros(A.shape[1])
            lr = 0.1
            for _ in range(300):                          # tiny logistic reg
                p = 1 / (1 + np.exp(-(A[tr] @ w)))
                w += lr * A[tr].T @ ((y[tr] == 2) - p) / max(1, tr.stop)
                w *= (1 - 1e-3)
            pred = (A[te] @ w > 0) + 1
            acc[m].append(float((pred == y[te]).mean()))
        best = int(np.argmax(acc[m]))
        print(f"[probe] {m}: best layer {best} acc {acc[m][best]:.2f}; "
              f"final-layer acc {acc[m][-1]:.2f}", flush=True)
    json.dump(dict(acc=acc, behav={m: behav[m] for m in modes}),
              open(os.path.join(OUT, "probes.json"), "w"))
    return behav


def patch_stage(model, tok, n_ex=12, seed=100):
    rng = random.Random(seed)
    nL = model.config.num_hidden_layers
    groups = ["cur_names", "reveal_rows", "last_hist", "answer"]
    mCs, mXs = [], []
    mPs = np.zeros((nL, len(groups), n_ex)) * np.nan
    cnt = 0
    for i in range(n_ex):
        rel = 1 if i % 2 == 0 else 2
        rng2 = random.Random(seed + 1000 + i)
        clean = build("duel100", rel, rng2, tok)
        rng2 = random.Random(seed + 1000 + i)
        corr = build("duel80", rel, rng2, tok)
        encC, idxC = encode_with_spans(tok, clean)
        encX, idxX = encode_with_spans(tok, corr)
        if encC["input_ids"].shape[1] != encX["input_ids"].shape[1]:
            print(f"[patch] ex{i}: length mismatch, skipped", flush=True)
            continue
        mC = margins(model, tok, encC, clean)
        mX = margins(model, tok, encX, corr)
        if mC - mX < 0.5:
            print(f"[patch] ex{i}: weak contrast clean {mC:+.2f} corrupt {mX:+.2f}, "
                  "kept anyway", flush=True)
        stash = {}
        hooks = []
        def mk_cap(l):
            def f(mod, inp, out):
                stash[l] = (out[0] if isinstance(out, tuple) else out).detach()
            return f
        for l, blk in enumerate(model.model.layers):
            hooks.append(blk.register_forward_hook(mk_cap(l)))
        with torch.no_grad():
            model(**encC)
        for h in hooks:
            h.remove()
        mCs.append(mC); mXs.append(mX)
        for l in range(nL):
            for gi, g in enumerate(groups):
                pos = idxX[g]
                if not pos:
                    continue
                def mk_patch(l0, pos0):
                    def f(mod, inp, out):
                        tup = isinstance(out, tuple)
                        h0 = (out[0] if tup else out).clone()
                        h0[0, pos0] = stash[l0][0, pos0]
                        return (h0,) + tuple(out[1:]) if tup else h0
                    return f
                hk = model.model.layers[l].register_forward_hook(mk_patch(l, pos))
                mP = margins(model, tok, encX, corr)
                hk.remove()
                mPs[l, gi, cnt] = mP
        cnt += 1
        print(f"[patch] ex{i} done  clean {mC:+.2f} corrupt {mX:+.2f}", flush=True)
        del stash
        torch.cuda.empty_cache()
    mC0, mX0 = np.mean(mCs), np.mean(mXs)
    json.dump(dict(mC=mCs, mX=mXs, mP=mPs[:, :, :cnt].tolist(), groups=groups, n=cnt),
              open(os.path.join(OUT, "patching_raw.json"), "w"))
    print(f"[patch] pooled clean {mC0:+.3f} corrupt {mX0:+.3f} gap {mC0-mX0:+.3f}", flush=True)
    for gi, g in enumerate(groups):
        pooled = (np.nanmean(mPs[:, gi, :cnt], axis=1) - mX0) / max(1e-3, mC0 - mX0)
        top = np.argsort(pooled)[-3:][::-1]
        print(f"[patch] group {g}: pooled recovery top "
              + ", ".join(f"L{t}={pooled[t]:+.2f}" for t in top)
              + f" | L20-35 mean {np.nanmean(pooled[20:36]):+.2f}", flush=True)


def main():
    model, tok, _ = load(os.environ.get("MODEL", "Qwen32"))
    model.eval()
    if os.environ.get("SKIP_PROBE", "") != "1":
        print("== stage 0+1: behavior + probes ==", flush=True)
        probe_stage(model, tok, n_per=int(os.environ.get("NPROBE", "24")))
    print("== stage 2: activation patching duel100 -> duel80 ==", flush=True)
    patch_stage(model, tok, n_ex=int(os.environ.get("NPATCH", "12")))
    print("MECH_STAGES_012_DONE", flush=True)


if __name__ == "__main__":
    main()
