"""Follow-ups on the fitted direction.

1  COSINE with everything built so far, in full rather than a top-6 list.

2  WHAT "GENERALISES" MEANT, made concrete. The cross-validation held out whole
   prompt families, so the direction was scored on phrasings absent from training --
   that is all "generalises" claimed. It did NOT show the direction behaves like a
   mean-difference direction on the stimuli. So: project it on each family's positive
   and negative conditions separately and check it is high on one, low on the other.

3  IS THE NAME TOKEN THE WRONG PLACE? Everything so far reads one token. Directions
   are now built by mean-difference at several read positions (the appended name, the
   token before it, four positions spread through the body, and the mean over all
   tokens), and every one is tested on the conversations. If some position tracks the
   scam conversation and the name does not, the read position was the problem rather
   than the method.

4  TRUST vs DISTRUST as separate fits. Splitting contexts at the median stated trust
   and fitting within each half gives w_hi (what varies among trusted people) and
   w_lo (what varies among distrusted ones). One axis predicts cos(w_hi, w_lo) ~ 1.
   Rectified fits (predict max(y-med,0) and max(med-y,0)) are reported alongside; on
   a single axis those come out near -1.

5  Trajectories saved for plotting against the behavioural curve.

env: MODEL LAYERS (35,45,52) NITEM (10) OUT (../out)
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
import fit_direction as FD  # noqa: E402
import project as P  # noqa: E402
import stimuli2 as S2  # noqa: E402
from common import chat, load, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
POS = ["name", "pre", "p20", "p40", "p60", "p80", "meanall"]


def positions_for(n):
    """token indices for each named read position in a sequence of length n"""
    return {"name": [n - 1], "pre": [max(0, n - 2)],
            "p20": [int(n * .2)], "p40": [int(n * .4)],
            "p60": [int(n * .6)], "p80": [int(n * .8)],
            "meanall": list(range(n))}


@torch.no_grad()
def reads(model, tok, text, layers):
    """{position: {layer: vector}} in ONE forward pass."""
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}
    o = model(**enc, output_hidden_states=True)
    n = enc["input_ids"].shape[1]
    idx = positions_for(n)
    out = {}
    for p, ii in idx.items():
        t = torch.tensor(ii, device=o.hidden_states[0].device)
        out[p] = {l: o.hidden_states[l][0][t].mean(0).float().cpu().numpy()
                  for l in layers}
    return out


def r_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    model, tok, _ = load()
    model.eval()
    layers = [int(x) for x in os.environ.get("LAYERS", "35,45,52").split(",")]
    nitem = int(os.environ.get("NITEM", "10"))
    fams = list(S2.ALL) + (S2.STORY_FAMILIES
                           if os.path.exists(os.path.join(OUT, "stories.json")) else [])

    A = {p: {l: [] for l in layers} for p in POS}
    y, grp, cond = [], [], []
    for fam in fams:
        for it in S2.items(fam, nitem):
            for c in S2.CONDS:
                txt = chat(tok, it["system"], it["texts"][c], "")
                rr = reads(model, tok, txt, layers)
                for p in POS:
                    for l in layers:
                        A[p][l].append(rr[p][l])
                body = it["texts"][c][: it["texts"][c].rstrip().rfind("\n")]
                y.append(FD.stated(model, tok, it["system"], body, it["name"]))
                grp.append(fam); cond.append(c)
        print(f"[data] {fam}", flush=True)
    y = np.array(y); grp = np.array(grp); cond = np.array(cond)
    print(f"[data] {len(y)} contexts, stated trust sd {y.std():.2f}", flush=True)

    z = np.load(os.path.join(OUT, "vectors2.npz"))
    zl = [int(v) for v in z["layers"]]
    res = {}

    def fit(Araw, sel=None):
        X = np.stack(Araw).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X = (X - mu) / sd
        s = np.ones(len(y), bool) if sel is None else sel
        lam = 10.0 * s.sum()
        w = FD.ridge(X[s], y[s] - y[s].mean(), lam)
        return unit(w), X

    for l in layers:
        w_name, Xn = fit(A["name"][l])
        # --- 1. cosine with every hand-built direction
        cosall = {k.replace("--last--full", ""):
                  float(unit(z[k][zl.index(l)]) @ w_name)
                  for k in z.files if k.endswith("--last--full") and l in zl}
        top = sorted(cosall.items(), key=lambda kv: -kv[1])
        print(f"\n===== layer {l} =====")
        print("1. cosine of the fitted direction with each hand-built direction")
        print("   highest:  " + ", ".join(f"{k} {v:+.2f}" for k, v in top[:5]))
        print("   lowest :  " + ", ".join(f"{k} {v:+.2f}" for k, v in top[-5:]))
        print(f"   mean {np.mean(list(cosall.values())):+.3f}   "
              f"n={len(cosall)}")

        # --- 2. does it behave like a mean-difference direction on the stimuli?
        proj = Xn @ w_name
        print("2. projection of the fitted direction on each family "
              "(pos / mix / neg), z-scored")
        seps = []
        for f in fams:
            m = grp == f
            pp = [float(np.mean(proj[m & (cond == c)])) for c in ("pos", "mix", "neg")]
            seps.append(pp[0] - pp[2])
            print(f"     {f:<22} {pp[0]:+6.2f} {pp[1]:+6.2f} {pp[2]:+6.2f}   "
                  f"pos-neg {pp[0]-pp[2]:+6.2f}")
        print(f"     -> positive-minus-negative is >0 for "
              f"{sum(s > 0 for s in seps)}/{len(seps)} families, mean {np.mean(seps):+.2f}")

        # --- 4. trust vs distrust
        med = np.median(y)
        w_hi, _ = fit(A["name"][l], y > med)
        w_lo, _ = fit(A["name"][l], y < med)
        Xz = (np.stack(A["name"][l]) - np.stack(A["name"][l]).mean(0)) / (
            np.stack(A["name"][l]).std(0) + 1e-6)
        lam = 10.0 * len(y)
        w_up = unit(FD.ridge(Xz, np.maximum(y - med, 0) - np.maximum(y - med, 0).mean(), lam))
        w_dn = unit(FD.ridge(Xz, np.maximum(med - y, 0) - np.maximum(med - y, 0).mean(), lam))
        print("4. trust vs distrust")
        print(f"     cos(w_hi, w_lo)   {float(w_hi @ w_lo):+.3f}   "
              "(fit within the trusted half vs the distrusted half; 1 = one axis)")
        print(f"     cos(w_up, w_dn)   {float(w_up @ w_dn):+.3f}   "
              "(rectified targets; -1 = one axis)")
        res[f"L{l}"] = dict(cos_hand=cosall, sep_mean=float(np.mean(seps)),
                            cos_hi_lo=float(w_hi @ w_lo), cos_up_dn=float(w_up @ w_dn),
                            w=w_name.tolist(), w_hi=w_hi.tolist(), w_lo=w_lo.tolist())

    # --- 3. read-position sweep, evaluated on the conversations
    beh = json.load(open(os.path.join(OUT, "elicit.json")))
    names = S2.NAMES[:4]
    print("\n===== 3. which READ POSITION tracks the conversations? =====")
    print("   correlation of each trajectory with the model's stated trust")
    print(f"   {'layer/pos':<16}{'fitted:scam':>13}{'fitted:help':>13}"
          f"{'meandiff:scam':>15}{'meandiff:help':>15}")
    track = {}
    for l in layers:
        for p in POS:
            w, X = fit(A[p][l])
            md = unit(np.stack(A[p][l])[cond == "pos"].mean(0) -
                      np.stack(A[p][l])[cond == "neg"].mean(0))
            traj = {}
            for tag, turns in (("scam", P.SCAM), ("helpful", P.HELPFUL)):
                tw, tm = [], []
                for upto in range(len(turns) + 1):
                    vals_w, vals_m = [], []
                    for nm in names:
                        txt = P.convo_prefix(tok, turns, upto, nm)
                        rr = reads(model, tok, txt, [l])
                        vals_w.append(float(rr[p][l] @ w))
                        vals_m.append(float(rr[p][l] @ md))
                    tw.append(float(np.mean(vals_w))); tm.append(float(np.mean(vals_m)))
                traj[tag] = (tw, tm)
            rs = [r_of(traj["scam"][0], beh["scam_behav"]),
                  r_of(traj["helpful"][0], beh["helpful_behav"]),
                  r_of(traj["scam"][1], beh["scam_behav"]),
                  r_of(traj["helpful"][1], beh["helpful_behav"])]
            track[f"L{l}_{p}"] = dict(r=rs, fitted=traj["scam"][0],
                                      fitted_help=traj["helpful"][0],
                                      md=traj["scam"][1], md_help=traj["helpful"][1])
            print(f"   L{l}/{p:<11}" + "".join(f"{x:>+13.3f}" if i < 2 else
                                               f"{x:>+15.3f}" for i, x in enumerate(rs)),
                  flush=True)
    n_ok = sum(1 for v in track.values() if v["r"][0] > 0.5)
    n_md = sum(1 for v in track.values() if v["r"][2] > 0.5)
    print(f"\n   read positions where the SCAM trajectory tracks stated trust "
          f"(r > 0.5): fitted {n_ok}/{len(track)}, mean-difference {n_md}/{len(track)}")
    res["track"] = track
    json.dump(res, open(os.path.join(OUT, "fit2.json"), "w"))
    print("FIT2_DONE", flush=True)


if __name__ == "__main__":
    main()
