"""Invert the method: find the direction that PREDICTS the model's stated trust.

Every direction so far was built by contrasting prompts I wrote and assuming the
contrast was about trust. The behavioural check showed those directions do not track
the model's own stance: asked "do you trust X?" the model moves 11.6 logits across a
conversation while every mean-difference projection drifts upward regardless.

So stop assuming and start regressing. For a large set of contexts:
    y = the model's stated trust      logit(yes) - logit(no) to a direct question
    x = the activation at the appended name token in the SAME context
and fit y ~ w.x by ridge regression. If a linear direction predicting stated trust
exists, w is it; if held-out prediction fails everywhere, that is a much stronger
statement than any cosine.

Three things are then asked of w:
  1. does it generalise ACROSS FAMILIES? Cross-validation holds out whole families,
     so training never sees the way of describing trust it is tested on. Within-family
     CV is reported alongside: items inside a family are near-duplicates, so a high
     within / low across gap means the fit memorised phrasing, not trust.
  2. how does it compare to the hand-built directions, as a cosine and as a predictor?
     Each mean-difference direction is given its own fitted scale and intercept, so
     the comparison is about DIRECTION only, not units.
  3. does it track the conversation? Projected along the scam and helpful
     conversations, against the behavioural trajectory those conversations produce.

env: MODEL (Qwen32) LAYERS (27,35,45,52) NITEM (12) LAM (ridge, auto) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import elicit as E  # noqa: E402
import project as P  # noqa: E402
import stimuli2 as S2  # noqa: E402
from common import chat, load, resid, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
Q = "Do you trust {n}?\nAnswer with one word, yes or no."


@torch.no_grad()
def stated(model, tok, system, body, name):
    """logit(yes) - logit(no), case-pooled, for a direct trust question."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": body + "\n\n" + Q.format(n=name)}]
    text = E.render(tok, msgs)
    enc = tok(text, return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    f = lambda w: tok(w, add_special_tokens=False)["input_ids"][0]  # noqa: E731
    return float(torch.logsumexp(torch.stack([lg[f("yes")], lg[f("Yes")]]), 0) -
                 torch.logsumexp(torch.stack([lg[f("no")], lg[f("No")]]), 0))


def ridge(X, y, lam):
    """Dual form: n << d here, so invert the n x n gram matrix, not d x d."""
    n = X.shape[0]
    K = X @ X.T
    a = np.linalg.solve(K + lam * np.eye(n), y)
    return X.T @ a


def r_of(pred, y):
    if np.std(pred) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def main():
    model, tok, _ = load()
    model.eval()
    layers = [int(x) for x in os.environ.get("LAYERS", "27,35,45,52").split(",")]
    nitem = int(os.environ.get("NITEM", "12"))
    fams = list(S2.ALL) + (S2.STORY_FAMILIES
                           if os.path.exists(os.path.join(OUT, "stories.json")) else [])

    X = {l: [] for l in layers}
    y, grp, meta = [], [], []
    for fam in fams:
        for it in S2.items(fam, nitem):
            for c in S2.CONDS:
                body = it["texts"][c][: it["texts"][c].rstrip().rfind("\n")]
                r = resid(model, tok, chat(tok, it["system"], it["texts"][c], ""),
                          layers, None)
                for l in layers:
                    X[l].append(r[l])
                y.append(stated(model, tok, it["system"], body, it["name"]))
                grp.append(fam)
                meta.append((fam, c))
        print(f"[data] {fam}: {nitem*4} contexts", flush=True)
    y = np.array(y)
    grp = np.array(grp)
    print(f"[data] {len(y)} contexts; stated trust mean {y.mean():+.2f} "
          f"sd {y.std():.2f} range {y.min():+.1f}..{y.max():+.1f}", flush=True)

    z = np.load(os.path.join(OUT, "vectors2.npz"))
    zl = [int(v) for v in z["layers"]]
    hand = [k for k in z.files if k.endswith("--last--full")]

    res = {}
    for l in layers:
        A = np.stack(X[l]).astype(np.float64)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        A = (A - mu) / sd
        lam = float(os.environ.get("LAM", 0)) or 10.0 * A.shape[0]
        # --- leave-one-FAMILY-out: never train on the phrasing being tested
        pred_x = np.zeros_like(y)
        for f in np.unique(grp):
            tr, te = grp != f, grp == f
            w = ridge(A[tr], y[tr] - y[tr].mean(), lam)
            pred_x[te] = A[te] @ w + y[tr].mean()
        # --- 5-fold within-family (items shuffled across all families)
        idx = np.arange(len(y))
        rng = np.random.default_rng(0).permutation(idx)
        pred_w = np.zeros_like(y)
        for k in range(5):
            te = rng[k::5]
            tr = np.setdiff1d(idx, te)
            w = ridge(A[tr], y[tr] - y[tr].mean(), lam)
            pred_w[te] = A[te] @ w + y[tr].mean()
        w_full = unit(ridge(A, y - y.mean(), lam))
        rx, rw = r_of(pred_x, y), r_of(pred_w, y)
        # --- how well does each HAND-BUILT direction predict the same y?
        hb = {}
        for k in hand:
            if l not in zl:
                continue
            v = unit(z[k][zl.index(l)])
            hb[k.replace("--last--full", "")] = r_of(A @ ((v / sd) * sd), y)
        best = sorted(hb.items(), key=lambda kv: -abs(kv[1]))[:6]
        cosb = {k: float(unit(z[k + "--last--full"][zl.index(l)]) @ (w_full / np.linalg.norm(w_full)))
                for k, _ in best}
        res[f"L{l}"] = dict(r_across_family=rx, r_within=rw,
                            hand_best=[(k, v, cosb[k]) for k, v in best],
                            w=w_full.tolist())
        print(f"\n=== layer {l} ===", flush=True)
        print(f"  fitted direction, held-out r:  across-family {rx:+.3f} | "
              f"within-family {rw:+.3f}", flush=True)
        print("  best hand-built directions as predictors of the same stated trust:",
              flush=True)
        for k, v in best:
            print(f"    {k:<24} r {v:+.3f}   cos(with fitted w) {cosb[k]:+.3f}",
                  flush=True)

    # --- does the fitted direction track the conversation?
    print("\n=== fitted direction along the conversations ===", flush=True)
    names = S2.NAMES[:4]
    conv = {}
    for l in layers:
        w = unit(np.array(res[f"L{l}"]["w"]))
        for tag, turns in (("scam", P.SCAM), ("helpful", P.HELPFUL)):
            traj = []
            for upto in range(len(turns) + 1):
                traj.append(float(np.mean(
                    [resid(model, tok, P.convo_prefix(tok, turns, upto, nm), [l],
                           None)[l] @ w for nm in names])))
            traj = [(t - traj[0]) for t in traj]          # relative to the opening
            conv[f"L{l}_{tag}"] = traj
            print(f"  L{l} {tag:<8} " + " ".join(f"{x:+.2f}" for x in traj), flush=True)
    res["conversation"] = conv
    json.dump(res, open(os.path.join(OUT, "fit_direction.json"), "w"))
    print("\nFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
