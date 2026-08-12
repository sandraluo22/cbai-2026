"""Trajectories across four domains, for the trust fit and the distrust fit, then
steering with both.

The fitted direction was tested on one bank phone call. Four domains now (bank, used
car, builder, colleague), each a matched good/bad pair, with the model's own stated
trust measured at every depth as the thing to be tracked.

  w      fitted to predict stated trust over the whole range
  w_lo   fitted only among contexts BELOW median stated trust -- "what varies among
         people who are not trusted". cos(w, w_lo) was -0.32 to -0.41, so this is not
         simply the negation of w, and it may track the falling half better.

Then the causal test the fitted direction has never had: inject +/- alpha*w and
+/- alpha*w_lo and see whether the quantity they PREDICT actually moves. Two beds:
  * stated trust on neutral-condition stimuli (headroom in both directions)
  * the iterated game's cooperation margin, so it is comparable with everything in v1
A matched-norm random direction is run at the same magnitude throughout.

env: MODEL LAYERS (45,52) NNAME (6) ALPHA (0.25,0.5,1.0) OUT (../out)
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
import qsg_games as G  # noqa: E402
import scenarios as SC  # noqa: E402
import stimuli2 as S2  # noqa: E402
from common import Inject, chat, first_id, load, rand_like, resid, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


@torch.no_grad()
def stated_at(model, tok, sc, which, upto, name, inject=None):
    """The model's stated trust at a conversation depth, optionally under injection."""
    s = SC.SCENARIOS[sc]
    msgs = [{"role": "system", "content": s["sys"]},
            {"role": "user", "content": s["open"].format(n=name)}]
    for role, txt in s[which][:upto]:
        msgs.append({"role": role, "content": txt.format(n=name)})
    msgs.append({"role": "user",
                 "content": f"Setting the conversation aside: do you trust {name}?\n"
                            "Answer with one word, yes or no."})
    text = E.render(tok, msgs)
    enc = tok(text, return_tensors="pt").to(model.device)
    if inject is None:
        lg = model(**enc).logits[0, -1]
    else:
        layer, vec, pos = inject
        with Inject(model, layer, torch.tensor(vec), pos):
            lg = model(**enc).logits[0, -1]
    f = lambda w: first_id(tok, w)  # noqa: E731
    return float(torch.logsumexp(torch.stack([lg[f("yes")], lg[f("Yes")]]), 0) -
                 torch.logsumexp(torch.stack([lg[f("no")], lg[f("No")]]), 0))


def main():
    model, tok, _ = load()
    model.eval()
    layers = [int(x) for x in os.environ.get("LAYERS", "45,52").split(",")]
    names = S2.NAMES[:int(os.environ.get("NNAME", "6"))]
    fit = json.load(open(os.path.join(OUT, "fit2.json")))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    W = {l: dict(trust=unit(np.array(fit[f"L{l}"]["w"])),
                 hi=unit(np.array(fit[f"L{l}"]["w_hi"])),
                 distrust=unit(np.array(fit[f"L{l}"]["w_lo"]))) for l in layers
         if f"L{l}" in fit}
    layers = [l for l in layers if l in W]
    print(f"[cfg] layers={layers} names={names} scenarios={list(SC.SCENARIOS)}",
          flush=True)

    # ---------- trajectories across the four domains ----------
    res = {"traj": {}}
    for sc in SC.SCENARIOS:
        D = SC.depth(sc)
        for which in ("good", "bad"):
            beh = []
            for upto in range(D):
                beh.append(float(np.mean([stated_at(model, tok, sc, which, upto, nm)
                                          for nm in names])))
            res["traj"][f"{sc}_{which}_stated"] = beh
            for l in layers:
                for wn, w in W[l].items():
                    tr = []
                    for upto in range(D):
                        tr.append(float(np.mean(
                            [resid(model, tok, SC.prefix(tok, sc, which, upto, nm),
                                   [l], None)[l] @ w for nm in names])))
                    tr = [x - tr[0] for x in tr]
                    res["traj"][f"{sc}_{which}_{wn}_L{l}"] = tr
        s = SC.SCENARIOS[sc]
        print(f"\n== {sc} ==  (turn {s['turn_of_interest']}: {s['note']})", flush=True)
        for which in ("good", "bad"):
            print(f"  {which:<5} stated " +
                  " ".join(f"{x:+6.1f}" for x in res["traj"][f"{sc}_{which}_stated"]),
                  flush=True)
            for l in layers:
                for wn in ("trust", "distrust"):
                    k = f"{sc}_{which}_{wn}_L{l}"
                    r = np.corrcoef(res["traj"][k],
                                    res["traj"][f"{sc}_{which}_stated"])[0, 1]
                    print(f"        {wn:<9}L{l} " +
                          " ".join(f"{x:+6.1f}" for x in res["traj"][k]) +
                          f"   r={r:+.2f}", flush=True)

    # ---------- steering ----------
    print("\n" + "=" * 78 + "\nSTEERING — does injecting the direction move what it "
          "predicts?\n" + "=" * 78, flush=True)
    alphas = [float(a) for a in os.environ.get("ALPHA", "0.25,0.5,1.0").split(",")]
    probes = []
    for fam in ("direct_b", "relational", "game_b"):
        for it in S2.items(fam, 6):
            probes.append((it, "mix"))
    steer = {}
    for l in layers:
        nrm = float(meta["resid_norm"][str(l)])
        base = float(np.mean([stated_at(model, tok, "bank", "good", 0, it["name"])
                              for it, _ in probes[:6]]))
        for wn in ("trust", "distrust", "rand"):
            for a in alphas:
                v = (rand_like(W[l]["trust"], seed=5) if wn == "rand"
                     else W[l][wn]) * nrm * a
                up, dn = [], []
                for it, c in probes:
                    body = it["texts"][c][: it["texts"][c].rstrip().rfind("\n")]
                    for sgn, acc in ((+1, up), (-1, dn)):
                        msgs_text = chat(tok, it["system"],
                                         body + f"\n\nDo you trust {it['name']}?\n"
                                                "Answer with one word, yes or no.", "")
                        enc = tok(msgs_text, return_tensors="pt")
                        enc = {k: t.to(model.device) for k, t in enc.items()}
                        with Inject(model, l, torch.tensor(sgn * v), None):
                            with torch.no_grad():
                                lg = model(**enc).logits[0, -1]
                        f = lambda w: first_id(tok, w)  # noqa: E731
                        acc.append(float(
                            torch.logsumexp(torch.stack([lg[f("yes")], lg[f("Yes")]]), 0) -
                            torch.logsumexp(torch.stack([lg[f("no")], lg[f("No")]]), 0)))
                d = np.array(up) - np.array(dn)
                steer[f"stated_{wn}_L{l}_a{a}"] = dict(
                    plus=float(np.mean(up)), minus=float(np.mean(dn)),
                    diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
                s_ = steer[f"stated_{wn}_L{l}_a{a}"]
                print(f"  stated trust  {wn:<9} L{l} a={a:<5} "
                      f"+v {s_['plus']:+7.2f}  -v {s_['minus']:+7.2f}  "
                      f"(+v)-(-v) {s_['diff']:+6.2f} +- {s_['se']:.2f}", flush=True)

    # ---------- the iterated game, for comparability with v1 ----------
    print("", flush=True)
    for l in layers:
        nrm = float(meta["resid_norm"][str(l)])
        for wn in ("trust", "distrust", "rand"):
            for a in alphas:
                v = (rand_like(W[l]["trust"], seed=5) if wn == "rand"
                     else W[l][wn]) * nrm * a
                d = []
                for g in G.GAMES:
                    for sch in ("one_lapse", "frequent"):
                        for flip in (False, True):
                            ex = G.build(g, sch, "unconditional", flip=flip)
                            txt = chat(tok, G.SYS, ex["user"], ex["prefill"])
                            pos = G.positions(tok, txt, ex)["partner_all"]
                            m = []
                            for sgn in (+1, -1):
                                with Inject(model, l, torch.tensor(sgn * v), pos):
                                    enc = tok(txt, return_tensors="pt")
                                    enc = {k: t.to(model.device) for k, t in enc.items()}
                                    with torch.no_grad():
                                        lg = model(**enc).logits[0, -1]
                                m.append(float(lg[first_id(tok, ex["coop"])] -
                                               lg[first_id(tok, ex["defect"])]))
                            d.append(m[0] - m[1])
                d = np.array(d)
                steer[f"game_{wn}_L{l}_a{a}"] = dict(
                    diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
                print(f"  game margin   {wn:<9} L{l} a={a:<5} "
                      f"(+v)-(-v) {d.mean():+6.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}",
                      flush=True)
    res["steer"] = steer
    json.dump(res, open(os.path.join(OUT, "fit3.json"), "w"))
    print("\nFIT3_DONE", flush=True)


if __name__ == "__main__":
    main()
