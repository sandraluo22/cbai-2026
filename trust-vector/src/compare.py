"""Stage 3 — compare the candidate directions. No GPU.

Cosine similarity between methods is uninterpretable on its own, so every number is
printed against both bounds:

  FLOOR    two independent random directions in d dims: E[cos]=0, sd=1/sqrt(d).
           At d=5120 that is 0.014 — so cos=0.05 is "significant" and still nothing.
  CEILING  the within-method split-half cosine cos(h0, h1). A method that only
           agrees with itself at 0.6 cannot agree with another method above ~0.6.
           `disatt` = cos(A,B) / sqrt(rel_A * rel_B) is the reliability-corrected
           estimate; it is meaningless (and suppressed) when either rel <= 0.

Two variants are reported:
  raw       cosine between the diff-in-means vectors as built.
  centered  after subtracting, at each layer, the mean of all method vectors. Every
            paired-prompt contrast in this stimulus set shares a "the two prompts
            differ in some evaluative way" component; centering removes what is
            common to trust AND valence AND competence AND tall/short, and asks
            whether anything method-specific is left.

The control block is the finding, not a footnote: if cos(trait, valence) is at the
level of cos(trait, record), the direction is valence, not trust.

env: OUT (../out) PLOT (1)
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


def main():
    z = np.load(os.path.join(OUT, "vectors.npz"))
    layers = list(z["layers"])
    anchor = os.environ.get("ANCHOR", "last")
    methods = [m for m in S.ALL if f"{m}--{anchor}--full" in z]
    d = z[f"{methods[0]}--{anchor}--full"].shape[1]
    floor = 1.0 / np.sqrt(d)
    print(f"[cfg] anchor={anchor} d={d} methods={methods}")
    print(f"[floor] random-pair cosine: mean 0.000, sd {floor:.4f} "
          f"(|cos| < {3*floor:.3f} is indistinguishable from unrelated)\n")

    res = {"floor_sd": floor, "layers": [int(l) for l in layers], "anchor": anchor,
           "by_layer": {}}
    for li, l in enumerate(layers):
        V = {m: z[f"{m}--{anchor}--full"][li] for m in methods}
        H = {m: (z[f"{m}--{anchor}--h0"][li], z[f"{m}--{anchor}--h1"][li])
             for m in methods}
        rel = {m: cos(*H[m]) for m in methods}
        mu = np.mean([V[m] for m in methods], 0)
        Vc = {m: V[m] - mu for m in methods}
        raw = {f"{a}|{b}": cos(V[a], V[b]) for a, b in itertools.combinations(methods, 2)}
        cen = {f"{a}|{b}": cos(Vc[a], Vc[b]) for a, b in itertools.combinations(methods, 2)}
        dis = {}
        for a, b in itertools.combinations(methods, 2):
            ra, rb = rel[a], rel[b]
            dis[f"{a}|{b}"] = (raw[f"{a}|{b}"] / np.sqrt(ra * rb)
                               if ra > 0.05 and rb > 0.05 else None)
        res["by_layer"][int(l)] = dict(reliability=rel, raw=raw, centered=cen,
                                       disattenuated=dis,
                                       norm={m: float(np.linalg.norm(V[m])) for m in methods})

    # ---- printed tables at the layer with the best mean reliability --------
    # L0 is the embedding output: at a read anchor inside the SHARED tail the two
    # branches are the same token there, so the diff is exactly 0 and every cosine
    # is nan. That is a sanity check passing, not a failure — but nan must not be
    # allowed to win an argmax (nan comparisons are always False).
    def ok(l):
        return np.all(np.isfinite(list(res["by_layer"][int(l)]["reliability"].values())))

    usable = [l for l in layers if ok(l)]
    if len(usable) < len(layers):
        print(f"[note] degenerate (zero-norm) layers excluded from selection: "
              f"{[int(l) for l in layers if not ok(l)]}\n")
    best = max(usable or layers, key=lambda l: np.mean(list(
        res["by_layer"][int(l)]["reliability"].values())))
    print(f"=== split-half reliability by layer (ceiling on any cross-method cosine) ===")
    hdr = "  layer " + " ".join(f"{m[:9]:>10}" for m in methods)
    print(hdr)
    for l in layers:
        r = res["by_layer"][int(l)]["reliability"]
        print(f"  L{int(l):<5} " + " ".join(f"{r[m]:>10.3f}" for m in methods))
    print(f"\n  -> most reliable layer overall: L{int(best)}\n")

    for tag in ("raw", "centered"):
        print(f"=== cross-method cosine ({tag}) at L{int(best)} ===")
        B = res["by_layer"][int(best)]
        print("            " + " ".join(f"{m[:9]:>10}" for m in methods))
        for a in methods:
            row = []
            for b in methods:
                if a == b:
                    row.append(f"{B['reliability'][a]:>10.3f}")
                else:
                    k = f"{a}|{b}" if f"{a}|{b}" in B[tag] else f"{b}|{a}"
                    row.append(f"{B[tag][k]:>10.3f}")
            mark = "*" if a in S.CONTROLS else " "
            print(f"  {mark}{a[:10]:<10}" + " ".join(row))
        print("   (diagonal = split-half reliability; * = control, not a trust method)\n")

    print(f"=== the discriminating comparison at L{int(best)} ===")
    B = res["by_layer"][int(best)]

    def g(a, b):
        return B["raw"].get(f"{a}|{b}", B["raw"].get(f"{b}|{a}"))

    within = [g(a, b) for a, b in itertools.combinations(S.METHODS, 2)
              if g(a, b) is not None and np.isfinite(g(a, b))]
    across = [g(a, b) for a in S.METHODS for b in S.CONTROLS
              if g(a, b) is not None and np.isfinite(g(a, b))]
    print(f"  trust-method <-> trust-method   mean cos {np.mean(within):+.3f} "
          f"(n={len(within)}, range {np.min(within):+.3f}..{np.max(within):+.3f})")
    print(f"  trust-method <-> control        mean cos {np.mean(across):+.3f} "
          f"(n={len(across)}, range {np.min(across):+.3f}..{np.max(across):+.3f})")
    print(f"  separation                      {np.mean(within)-np.mean(across):+.3f}")
    print("  -> a separation near zero means these are not trust directions; they are\n"
          "     whatever the controls share with them (valence / evaluative polarity).")
    res["summary"] = dict(best_layer=int(best), within_mean=float(np.mean(within)),
                          across_mean=float(np.mean(across)),
                          separation=float(np.mean(within) - np.mean(across)))
    json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1)

    if os.environ.get("PLOT", "1") == "1":
        try:
            plot(res, methods, layers)
        except Exception as e:
            print(f"[plot] skipped ({type(e).__name__}: {e})")
    print("COMPARE_DONE")


def plot(res, methods, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = res["summary"]["best_layer"]
    B = res["by_layer"][best]
    n = len(methods)
    M = np.eye(n)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            M[i, j] = (B["reliability"][a] if a == b
                       else B["raw"].get(f"{a}|{b}", B["raw"].get(f"{b}|{a}", np.nan)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(n)); ax[0].set_xticklabels(methods, rotation=45, ha="right")
    ax[0].set_yticks(range(n)); ax[0].set_yticklabels(methods)
    for i in range(n):
        for j in range(n):
            ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax[0].set_title(f"cosine between candidate directions, layer {best}\n"
                    "(diagonal = split-half reliability)")
    fig.colorbar(im, ax=ax[0], label="cosine similarity")
    for m in methods:
        ax[1].plot([int(l) for l in layers],
                   [res["by_layer"][int(l)]["reliability"][m] for l in layers],
                   marker="o", ms=3, label=m)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhline(3 * res["floor_sd"], color="grey", ls=":", lw=1,
                  label="3 sd of random-pair cosine")
    ax[1].set_xlabel("residual-stream layer")
    ax[1].set_ylabel("split-half cosine (reliability)")
    ax[1].set_title("how repeatable is each method's direction?")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(OUT, "compare.png")
    fig.savefig(p, dpi=150)
    print(f"[plot] -> {p}")


if __name__ == "__main__":
    main()
