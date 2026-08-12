"""Stage 3b — is there anything trust-specific LEFT once the controls are removed?

`compare.py` found that the five candidate trust directions are no more similar to
each other than they are to valence / competence / tall-short: separation ~0 at
every layer, at both read anchors. The natural reading is that diff-in-means over
these stimuli recovers a general "positive vs negative attribute of this person"
direction, and that the trust content is a small part of it or is not linearly
recoverable this way at all.

That reading makes a sharp prediction, which this script tests. Project each trust
vector off the subspace spanned by the three control vectors and ask two things:

  1. RELIABILITY  does the residual still replicate across split halves?
     cos(resid(h0), resid(h1)). If this collapses toward 0, everything repeatable
     about the direction WAS the control component — there is no trust-specific
     direction here. If it stays high, a trust-specific component exists and is
     simply much smaller than the evaluative one.

  2. AGREEMENT    do the residuals of DIFFERENT methods still agree with each other?
     This is the real test. A trust-specific component that is shared by "asserted
     disposition", "observed record", and "source credibility" is a trust direction.
     One that is reliable per-method but uncorrelated across methods is just each
     stimulus set's own idiosyncrasy.

Both are read against the same floor/ceiling logic as compare.py: reliability is the
ceiling on agreement, 1/sqrt(d) is the floor.

Caveat kept in view: the residual is orthogonal to the control vectors AS ESTIMATED
from this stimulus set. Controls estimated with error leave some of their own
variance behind, which inflates residual reliability. The split halves bound that
but do not eliminate it.

env: OUT (../out) ANCHOR (last) WRITE (1 -> append *R keys to vectors.npz)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli as S  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def project_off(v, basis):
    """Component of v orthogonal to span(basis), via Gram-Schmidt on the basis."""
    B = []
    for b in basis:
        u = b.astype(np.float64).copy()
        for e in B:
            u -= (u @ e) * e
        n = np.linalg.norm(u)
        if n > 1e-8:
            B.append(u / n)
    r = v.astype(np.float64).copy()
    for e in B:
        r -= (r @ e) * e
    return r


def main():
    anchor = os.environ.get("ANCHOR", "last")
    z = np.load(os.path.join(OUT, "vectors.npz"))
    layers = [int(x) for x in z["layers"]]
    methods = [m for m in S.METHODS if f"{m}--{anchor}--full" in z.files]
    ctrls = [c for c in S.CONTROLS if f"{c}--{anchor}--full" in z.files]
    d = z[f"{methods[0]}--{anchor}--full"].shape[1]
    print(f"[cfg] anchor={anchor} d={d} methods={methods} controls={ctrls}")
    print(f"[floor] random-pair cosine sd = {1/np.sqrt(d):.4f}\n")

    res, new = {}, {}
    for li, l in enumerate(layers):
        basis = [z[f"{c}--{anchor}--full"][li] for c in ctrls]
        rel_r, rel_o, keep = {}, {}, {}
        R = {}
        for m in methods:
            v = z[f"{m}--{anchor}--full"][li]
            h0 = z[f"{m}--{anchor}--h0"][li]
            h1 = z[f"{m}--{anchor}--h1"][li]
            r, r0, r1 = (project_off(x, basis) for x in (v, h0, h1))
            R[m] = r
            rel_o[m] = cos(h0, h1)
            rel_r[m] = cos(r0, r1)
            nv = np.linalg.norm(v)
            keep[m] = float(np.linalg.norm(r) / nv) if nv > 0 else float("nan")
            new.setdefault(f"{m}R--{anchor}--full", []).append(r)
            new.setdefault(f"{m}R--{anchor}--h0", []).append(r0)
            new.setdefault(f"{m}R--{anchor}--h1", []).append(r1)
        agree = {f"{a}|{b}": cos(R[a], R[b])
                 for a, b in itertools.combinations(methods, 2)}
        res[int(l)] = dict(rel_orig=rel_o, rel_resid=rel_r, norm_kept=keep,
                           agree_resid=agree)

    usable = [l for l in layers
              if np.all(np.isfinite(list(res[int(l)]["rel_orig"].values())))
              and min(res[int(l)]["rel_orig"].values()) >= 0.5]
    print("=== per layer: does anything survive removing valence+competence+arbitrary? ===")
    print("  L    ||resid||/||v||   split-half reliability      cross-method agreement")
    print("         (mean)          original -> residual        original -> residual")
    for l in usable:
        if l % 6 and l not in (26, 49):
            continue
        B = res[int(l)]
        ao = np.mean([cos(z[f'{a}--{anchor}--full'][layers.index(l)],
                          z[f'{b}--{anchor}--full'][layers.index(l)])
                      for a, b in itertools.combinations(methods, 2)])
        ar = np.mean(list(B["agree_resid"].values()))
        print(f"  {l:<4} {np.mean(list(B['norm_kept'].values())):.3f}"
              f"            {np.mean(list(B['rel_orig'].values())):.3f} -> "
              f"{np.mean(list(B['rel_resid'].values())):.3f}"
              f"             {ao:+.3f} -> {ar:+.3f}")

    best = max(usable, key=lambda l: np.mean(list(res[int(l)]["agree_resid"].values())))
    B = res[int(best)]
    print(f"\n=== residual detail at L{best} (best cross-method agreement) ===")
    for m in methods:
        print(f"  {m:<8} keeps {B['norm_kept'][m]*100:5.1f}% of its norm | "
              f"reliability {B['rel_orig'][m]:.3f} -> {B['rel_resid'][m]:.3f}")
    print("  pairwise agreement between residuals:")
    for k, v in sorted(B["agree_resid"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {v:+.3f}")

    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(OUT, f"residualize_{anchor}.json"), "w"), indent=1)
    if os.environ.get("WRITE", "1") == "1":
        keep = {k: z[k] for k in z.files}
        for k, v in new.items():
            keep[k] = np.stack(v)
        np.savez(os.path.join(OUT, "vectors.npz"), **keep)
        print(f"\n[write] appended {len(new)} residualised directions to vectors.npz")
    print("RESIDUALIZE_DONE")


if __name__ == "__main__":
    main()
