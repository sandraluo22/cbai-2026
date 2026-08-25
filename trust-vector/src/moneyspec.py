"""Money-trust spectrum (Sandra 2026-08-14).

"How much would you trust {n} with?" operationalised as a dose grid:
  "Would you trust {n} with ${y} of your money? Answer with one word, yes or no."
  y in 1 .. 100,000 (log decades); margin = logit(yes)-logit(no) at each y;
  THRESHOLD = zero-crossing of the linear fit margin ~ log10(y), clipped to
  [-1, 7] -- the model's max trusted amount in log10 dollars.

Contexts, two kinds:
  in-context  the trust stories (pos/neg/mix/neu x 16) instantiated with the
              derivation names -- evidence controls the threshold
  famous      20 public figures spanning trusted -> distrusted -- the model's
              prior controls the threshold

Three questions:
  1. Does the spectrum behave (pos > mix/neu > neg; Mandela > Madoff)?
  2. PROBE: is the threshold linearly decodable from the name-token activation
     of the context alone (story + appended name, question never shown)?
     Ridge, leave-one-story-index-out (in-context) / LOO (famous), r held-out.
  3. STEERING: does +/-v at the name tokens move the threshold, per direction?
     Delta(thr) in log10-dollar decades, mix contexts + ambiguous famous.

env: MODEL ALPHA (0.35) LAYERS_PROBE (27,35,45,52) STEER_LAYER (45) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dirs as DIRS  # noqa: E402
from common import chat, load, resid_at_name  # noqa: E402
from sweep import read  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.35"))
L_STEER = int(os.environ.get("STEER_LAYER", "45"))
L_PROBE = [int(x) for x in os.environ.get("LAYERS_PROBE", "27,35,45,52").split(",")]

SPEC = os.environ.get("SPEC", "money")
if SPEC == "money":
    YLAB = ["$1", "$10", "$100", "$1,000", "$10,000", "$100,000"]
    Q = ("Would you trust {n} with {amt} of your own money?\n"
         "Answer with one word, yes or no.")
else:                       # SPEC=objects: severity ladder instead of dollars
    YLAB = ["a rock", "a garden rake", "a bicycle", "your house keys",
            "your car", "a kitchen knife", "a loaded gun", "a bomb"]
    Q = "Would you trust {n} with {amt}?\nAnswer with one word, yes or no."
LOGY = np.arange(float(len(YLAB)))         # grid coordinate (log10 $ / ladder rank)
STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
FAME_SYS = "Answer from your general knowledge of this person."

FAMOUS = {
    "trusted": ["Nelson Mandela", "Marie Curie", "Fred Rogers", "Tom Hanks",
                "Keanu Reeves", "Angela Merkel", "Warren Buffett"],
    "mid": ["Elon Musk", "Kanye West", "Boris Johnson", "Kim Kardashian",
            "Jeff Bezos", "Cristiano Ronaldo"],
    "distrusted": ["Richard Nixon", "Bernie Madoff", "Elizabeth Holmes",
                   "Sam Bankman-Fried", "Jordan Belfort", "Vladimir Putin",
                   "Lance Armstrong"],
}
STEER_DIRS = os.environ.get(
    "STEER_DIRS", "FITTED trust,optim,story_comb,story_trust,"
                  "story_trust@acctnb,nomfame,warmth_b,random").split(",")
STEER_ONLY = bool(os.environ.get("STEER_ONLY", ""))   # skip probe; steering only


def threshold(margins):
    """Zero of the linear fit margin ~ grid coordinate, clipped to the grid +-1."""
    hi = float(len(YLAB)) + 1.0
    m = np.asarray(margins, float)
    b, a = np.polyfit(LOGY, m, 1)
    if b >= 0:                       # non-decreasing in severity: no crossing
        return hi if m.mean() > 0 else -1.0
    return float(np.clip(-a / b, -1.0, hi))


def contexts_incontext():
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    out = []
    for cell in ("pos", "neg", "mix", "neu"):
        for i, story in enumerate(sb[cell]):
            nm = SU.NAMES_TRAIN[(len(out)) % len(SU.NAMES_TRAIN)]
            out.append(dict(kind="story", cell=cell, idx=i, name=nm,
                            body=story.replace("{n}", nm), system=STORY_SYS))
    return out


def contexts_famous():
    out = []
    for grp, names in FAMOUS.items():
        for nm in names:
            out.append(dict(kind="famous", cell=grp, idx=0, name=nm,
                            body=f"Consider {nm}.", system=FAME_SYS))
    return out


def curve(model, tok, ctx, inj=None, pos_fn=None):
    """Margins over the amount grid, optionally under an injection."""
    ms = []
    for amt in YLAB:
        q = Q.format(n=ctx["name"], amt=amt)
        body = (ctx["body"] + "\n\n" + q) if ctx["kind"] == "story" else q
        txt = chat(tok, ctx["system"], body, "")
        pos = DIRS.name_positions(tok, txt, ctx["name"]) if inj is not None else None
        m, _ = read(model, tok, txt, inj, pos)
        ms.append(m)
    return ms


def ridge_loo(X, y, groups, lam=1000.0):
    """Dual-form ridge with leave-one-group-out; returns held-out predictions."""
    yhat = np.zeros_like(y)
    for g in sorted(set(groups)):
        tr = np.array([i for i, gg in enumerate(groups) if gg != g])
        te = np.array([i for i, gg in enumerate(groups) if gg == g])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        A = Xtr @ Xtr.T + lam * np.eye(len(tr))
        alpha = np.linalg.solve(A, y[tr] - ym)
        yhat[te] = Xte @ (Xtr.T @ alpha) + ym
    return yhat


def main():
    model, tok, _ = load()
    model.eval()
    res = {"alpha": ALPHA, "steer_layer": L_STEER, "ylab": YLAB}

    ctxs = contexts_incontext() + contexts_famous()
    if STEER_ONLY:
        ctxs = [c for c in ctxs if (c["kind"], c["cell"]) in
                (("story", "mix"), ("famous", "mid"))]
    print(f"[cfg] {len(ctxs)} contexts", flush=True)

    # ---- stage 1: baseline curves + thresholds ----------------------------
    for c in ctxs:
        c["margins"] = curve(model, tok, c)
        c["thr"] = threshold(c["margins"])
    mono = np.mean([all(np.diff(c["margins"]) <= 1e-9) or
                    np.corrcoef(LOGY, c["margins"])[0, 1] < 0
                    for c in ctxs])
    print(f"[stage1] fraction with decreasing margin-vs-amount: {mono:.2f}", flush=True)
    for kind in ("story", "famous"):
        cells = sorted({c["cell"] for c in ctxs if c["kind"] == kind})
        for cell in cells:
            t = [c["thr"] for c in ctxs if c["kind"] == kind and c["cell"] == cell]
            print(f"  {kind}/{cell:<10} thr(log10$) {np.mean(t):+5.2f} "
                  f"+- {np.std(t)/np.sqrt(len(t)):.2f}  (n={len(t)})", flush=True)
    for nm in ("Nelson Mandela", "Bernie Madoff", "Elon Musk"):
        c = next((c for c in ctxs if c["name"] == nm), None)
        if c:
            print(f"  {nm:<18} margins " +
                  " ".join(f"{m:+5.1f}" for m in c["margins"]) +
                  f"  thr {c['thr']:+.2f}", flush=True)

    # ---- stage 2: activations + probe -------------------------------------
    if STEER_ONLY:
        return steer_stage(model, tok, ctxs, res)
    for c in ctxs:
        r = resid_at_name(model, tok, c["system"], c["body"], c["name"], L_PROBE)
        c["act"] = {l: r[l] for l in L_PROBE}
    res["probe"] = {}
    for l in L_PROBE:
        st = [c for c in ctxs if c["kind"] == "story"]
        X = np.stack([c["act"][l] for c in st]); y = np.array([c["thr"] for c in st])
        yh = ridge_loo(X, y, [c["idx"] for c in st])
        r_st = float(np.corrcoef(y, yh)[0, 1])
        fm = [c for c in ctxs if c["kind"] == "famous"]
        Xf = np.stack([c["act"][l] for c in fm]); yf = np.array([c["thr"] for c in fm])
        yhf = ridge_loo(Xf, yf, list(range(len(fm))))
        r_fm = float(np.corrcoef(yf, yhf)[0, 1])
        res["probe"][f"L{l}"] = dict(story=r_st, famous=r_fm)
        print(f"[probe L{l}] held-out r: story {r_st:+.3f} (n={len(st)}), "
              f"famous {r_fm:+.3f} (n={len(fm)})", flush=True)

    # ---- stage 3: steering the threshold ----------------------------------
    steer_stage(model, tok, ctxs, res)


def steer_stage(model, tok, ctxs, res):
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L_STEER)])
    D = DIRS.load_all(OUT, L_STEER)
    ns = int(os.environ.get("STEER_N", "999"))
    steer_ctx = ([c for c in ctxs if c["kind"] == "story" and c["cell"] == "mix"][:ns] +
                 [c for c in ctxs if c["kind"] == "famous" and c["cell"] == "mid"])
    res["steer"] = {}
    for dn in STEER_DIRS:
        if dn not in D:
            print(f"[steer] {dn} missing, skipped", flush=True)
            continue
        v = D[dn] * nrm * ALPHA
        for kind in ("story", "famous"):
            dts = []
            for c in [x for x in steer_ctx if x["kind"] == kind]:
                tp = threshold(curve(model, tok, c, inj=(L_STEER, v)))
                tm = threshold(curve(model, tok, c, inj=(L_STEER, -v)))
                dts.append(tp - tm)
            dts = np.array(dts)
            res["steer"][f"{dn}|{kind}"] = (float(dts.mean()),
                                            float(dts.std(ddof=1) / np.sqrt(len(dts))),
                                            len(dts))
            print(f"[steer] {dn:<20} {kind:<7} Δthr(decades) {dts.mean():+5.2f} "
                  f"+- {dts.std(ddof=1)/np.sqrt(len(dts)):.2f} (n={len(dts)})", flush=True)

    res["contexts"] = [{k: c[k] for k in ("kind", "cell", "name", "margins", "thr")}
                      for c in ctxs]
    stem = "moneyspec" if SPEC == "money" else f"moneyspec_{SPEC}"
    out_name = f"{stem}_steeronly.json" if STEER_ONLY else f"{stem}.json"
    json.dump(res, open(os.path.join(OUT, out_name), "w"), indent=1)

    # cross-spectrum: does the object ladder read the same per-person quantity
    # as the dollar grid? Align contexts by construction order, correlate
    # WITHIN cell (across-cell correlation is trivially driven by pos vs neg).
    mp = os.path.join(OUT, "moneyspec.json")
    if SPEC != "money" and os.path.exists(mp) and not STEER_ONLY:
        other = json.load(open(mp))["contexts"]
        key = lambda c: (c["kind"], c["cell"], c["name"])  # noqa: E731
        om = {}
        for i, c in enumerate(other):
            om.setdefault(key(c), []).append(c["thr"])
        pairs = {}
        seen = {}
        for c in ctxs:
            k = key(c)
            j = seen.get(k, 0)
            if k in om and j < len(om[k]):
                pairs.setdefault((c["kind"], c["cell"]), []).append(
                    (om[k][j], c["thr"]))
                seen[k] = j + 1
        for cell, pr in sorted(pairs.items()):
            a, b = np.array(pr).T
            if len(pr) > 3 and a.std() > 0 and b.std() > 0:
                print(f"[xspec] {cell[0]}/{cell[1]:<10} r(money, {SPEC}) = "
                      f"{np.corrcoef(a, b)[0, 1]:+.3f} (n={len(pr)})", flush=True)
    print("MONEYSPEC_DONE", flush=True)


if __name__ == "__main__":
    main()
