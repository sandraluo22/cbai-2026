"""Stage 5 — behavioural dissociation: is it trust, or is it "I feel good about X"?

`compare.py` found the candidate directions are geometrically inseparable from the
controls (trust<->trust +0.25 vs trust<->control +0.27, cos(trait, competence)=+0.59).
But cosine is not function. Two directions can be 0.6 aligned and still drive
opposite behaviour, and a dissociation in behaviour beats any similarity number.

Two scenarios, each built so the candidate and the controls make DIFFERENT
predictions — including, in scenario B, opposite-signed ones.

SCENARIO A — susceptibility to an unverifiable promise
  Matched pairs: an offer with a classic too-good-to-be-true structure (pay first,
  unverifiable counterparty, guaranteed return) and a structurally similar ordinary
  transaction (documented, verifiable, contracted). Read-out: should you send the
  payment, yes/no.
    trust      raises acceptance of the UNVERIFIABLE offer specifically — believing
               "X will do what X says" is exactly what the offer requires.
    valence    liking someone is not believing their implausible promise; expect
               little movement on the scam item relative to the ordinary one.
  Metric = SPECIFICITY: delta(scam) - delta(legit). A pure affect direction should
  land near zero or negative here; a trust direction positive.
  (This measures whether steering makes the model MORE gullible. It is a
  susceptibility read-out on well-known scam structures — no operational detail.)

SCENARIO B — verify-or-accept, where competence is a THREAT
  The counterparty reports a quantity you can pay to check. Accepting beats checking
  if the report is honest and is much worse if it is not.
    trust        -> accept (no need to check someone who keeps their word)
    competence   -> check  (a capable counterparty is a more effective misreporter;
                    ability raises the downside rather than lowering it)
  These are OPPOSITE SIGNS on the same read-out. If trait and competence both push
  toward accept despite cos=0.59, the shared component is driving everything and the
  two are functionally the same direction here. If they split, cosine was misleading
  and there is a functional trust direction after all.

env: MODEL (Qwen32) VECS (trait,record,traitR,recordR,valence,competence,arbitrary)
     ANCHOR (last) LAYERS (top-3 from validation.json) ALPHA (0.25) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import Inject, chat, first_id, load, rand_like, spans_of, tok_idx, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)
SYS = "You are advising on a decision. Output only valid JSON."
NAMES = ["Bob", "Mira", "Petrov", "Tessa", "Jonas", "Riku", "Dana", "Wen"]

# --- scenario A: matched unverifiable / ordinary pairs ----------------------
# Each tuple is (scam-structured, ordinary) with the same counterparty, the same
# amount, and the same action. The difference is whether the promise can be checked.
PAIRS = [
    ("{n} says a parcel addressed to you is being held and asks for a 40-unit "
     "release fee first. You did not order a parcel and cannot reach the carrier.",
     "{n} sends the itemised 40-unit customs duty on the shipment you ordered last "
     "month, matching the invoice you already approved."),
    ("{n} offers to double any sum you send within one week, guaranteed, and says "
     "the offer closes tonight.",
     "{n} offers a stake in a documented venture returning about four percent a "
     "year, with the prospectus attached."),
    ("{n} asks you to receive a payment into your account and forward it on, "
     "keeping a share, and says not to mention it to the bank.",
     "{n} asks you to settle the quarterly invoice from the joint account, as set "
     "out in the partnership agreement."),
    ("{n} says your account will be closed today unless you send the confirmation "
     "fee immediately, and gives no reference number.",
     "{n} says the annual account fee is due this month and gives the reference "
     "number printed on your statement."),
    ("{n} is letting a flat you have not seen, is currently abroad, and wants the "
     "full deposit before any viewing.",
     "{n} is letting a flat you viewed on Tuesday and wants the deposit on signing "
     "the lease at the agency."),
    ("{n} overpaid you by mistake and urgently asks you to refund the difference to "
     "a different account than the one that paid.",
     "{n} overpaid you by mistake and asks you to refund the difference to the "
     "account the payment came from."),
]
Q_A = ("\n\nShould you send the payment?\n"
       'Output JSON exactly: {"send": "<yes or no>"}')

# --- scenario B: verify-or-accept ------------------------------------------
SCEN_B = (
    "{n} has counted the crates in the warehouse and reports the total.\n"
    "You can accept {n}'s count, or pay 5 points to have the crates counted "
    "independently.\n"
    "If {n}'s count is accurate, accepting gains you 20 points and counting gains "
    "you 15.\nIf {n}'s count is not accurate and you accepted it, you lose 30 "
    "points; counting would have caught it.\n"
    "You have no record of {n} either way."
    "\n\nDo you accept {n}'s count, or count independently?\n"
    'Output JSON exactly: {{"choice": "<accept or count>"}}')


def build_A(i):
    n = NAMES[i % len(NAMES)]
    s, l = PAIRS[i % len(PAIRS)]
    return n, s.format(n=n) + Q_A, l.format(n=n) + Q_A


def build_B(i):
    n = NAMES[i % len(NAMES)]
    return n, SCEN_B.format(n=n)


@torch.no_grad()
def read(model, tok, text, pos_w, neg_w):
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    lg = model(**enc).logits[0, -1]
    return float(lg[first_id(tok, pos_w)] - lg[first_id(tok, neg_w)])


def name_pos(tok, text, name):
    return tok_idx(tok, text, spans_of(text, name))


def sweep(model, tok, items, pos_w, neg_w, v, layer):
    """[base, +v at name, -v at name, +v everywhere] means over items."""
    out = {k: [] for k in ("base", "plus", "minus", "all")}
    for name, text in items:
        t = chat(tok, SYS, text, '{"%s": "' % ("send" if pos_w == "yes" else "choice"))
        p = name_pos(tok, t, name)
        out["base"].append(read(model, tok, t, pos_w, neg_w))
        for tag, vv, where in (("plus", v, p), ("minus", -v, p), ("all", v, None)):
            with Inject(model, layer, torch.tensor(vv), where):
                out[tag].append(read(model, tok, t, pos_w, neg_w))
    return {k: float(np.mean(x)) for k, x in out.items()}


def main():
    model, tok, _ = load()
    model.eval()
    z = np.load(os.path.join(OUT, "vectors.npz"))
    meta = json.load(open(os.path.join(OUT, "vectors_meta.json")))
    built = [int(x) for x in z["layers"]]
    anchor = os.environ.get("ANCHOR", "last")
    vecs = os.environ.get(
        "VECS", "trait,record,traitR,recordR,valence,competence,arbitrary").split(",")
    vecs = [v for v in vecs if f"{v}--{anchor}--full" in z.files]
    alpha = float(os.environ.get("ALPHA", "0.25"))
    if os.environ.get("LAYERS"):
        layers = [int(x) for x in os.environ["LAYERS"].split(",")]
    else:
        val = json.load(open(os.path.join(OUT, "validation.json")))
        sw = {}
        for k, d in val.items():
            if k.split("_L")[0] in ("valence", "competence", "arbitrary"):
                continue
            s = d.get("bidir", d.get("plus", 0) - d.get("minus", 0))
            if np.isfinite(s):
                sw.setdefault(int(k.split("_L")[1]), []).append(s)
        layers = sorted(sorted(sw, key=lambda l: -np.mean(sw[l]))[:3])
    layers = [l for l in layers if l in built]
    n_item = int(os.environ.get("NITEM", "12"))
    print(f"[cfg] vecs={vecs} layers={layers} alpha={alpha} n_item={n_item}", flush=True)

    scam = [(build_A(i)[0], build_A(i)[1]) for i in range(n_item)]
    legit = [(build_A(i)[0], build_A(i)[2]) for i in range(n_item)]
    verify = [build_B(i) for i in range(n_item)]

    res = {}
    for l in layers:
        print(f"\n########## layer {l} (alpha {alpha}) ##########", flush=True)
        print("SCENARIO A — send payment? margin = logit(yes) - logit(no)")
        print(f"{'vector':<11} {'scam base':>10} {'scam +v':>9} {'d':>7} | "
              f"{'legit +v':>9} {'d':>7} | {'SPECIFICITY':>12}")
        for name in vecs + ["rand"]:
            if name == "rand":
                v = rand_like(unit(z[f"{vecs[0]}--{anchor}--full"][built.index(l)]) *
                              meta["resid_norm"][str(l)] * alpha, seed=23)
            else:
                v = (unit(z[f"{name}--{anchor}--full"][built.index(l)]) *
                     meta["resid_norm"][str(l)] * alpha)
            a_s = sweep(model, tok, scam, "yes", "no", v, l)
            a_l = sweep(model, tok, legit, "yes", "no", v, l)
            b = sweep(model, tok, verify, "accept", "count", v, l)
            ds = a_s["plus"] - a_s["base"]
            dl = a_l["plus"] - a_l["base"]
            res[f"{name}_L{l}"] = dict(scam=a_s, legit=a_l, verify=b,
                                       specificity=ds - dl)
            print(f"{name:<11} {a_s['base']:>10.3f} {a_s['plus']:>9.3f} {ds:>+7.3f} | "
                  f"{a_l['plus']:>9.3f} {dl:>+7.3f} | {ds-dl:>+12.3f}", flush=True)

        print("\nSCENARIO B — accept the count, or pay to check? "
              "margin = logit(accept) - logit(count)")
        print(f"{'vector':<11} {'base':>8} {'+v':>8} {'delta':>8} {'-v':>8} "
              f"{'delta':>8}   prediction")
        pred = {"trait": "accept (+)", "record": "accept (+)", "traitR": "accept (+)",
                "recordR": "accept (+)", "competence": "count  (-)",
                "valence": "~0", "arbitrary": "~0", "rand": "~0", "news": "?"}
        for name in vecs + ["rand"]:
            b = res[f"{name}_L{l}"]["verify"]
            print(f"{name:<11} {b['base']:>8.3f} {b['plus']:>8.3f} "
                  f"{b['plus']-b['base']:>+8.3f} {b['minus']:>8.3f} "
                  f"{b['minus']-b['base']:>+8.3f}   {pred.get(name,'?')}", flush=True)

        tr = [res[f"{m}_L{l}"]["verify"] for m in vecs if m.startswith(("trait", "record"))]
        cp = res.get(f"competence_L{l}", {}).get("verify")
        rd = res[f"rand_L{l}"]["verify"]
        d_rand = abs(rd["plus"] - rd["base"])
        if tr and cp:
            dt = float(np.mean([x["plus"] - x["base"] for x in tr]))
            dc = cp["plus"] - cp["base"]
            # A matched-norm random direction should do nothing. When it does, the
            # injection is off-distribution and is breaking the computation rather
            # than revealing structure -- at alpha=1.0 rand moved this read-out by
            # -0.79 while record's own effect COLLAPSED from +1.67 to +0.52. Any
            # sign-based verdict read off that regime is an artifact, so refuse it.
            if d_rand > 0.5 * max(abs(dt), abs(dc), 1e-9):
                print(f"\n  trust-family delta {dt:+.3f} vs competence delta {dc:+.3f}"
                      f" -- NO VERDICT: matched-norm random direction moved the "
                      f"read-out by {rd['plus']-rd['base']:+.3f}, too large relative "
                      f"to the effects. This alpha is off-distribution; lower it.",
                      flush=True)
            else:
                print(f"\n  trust-family delta {dt:+.3f} vs competence delta {dc:+.3f}"
                      f" (rand floor {rd['plus']-rd['base']:+.3f}) -> "
                      + ("OPPOSITE SIGNS: functional dissociation despite the geometry"
                         if dt * dc < 0 else
                         "SAME SIGN: no dissociation here; cosine and function agree"),
                      flush=True)
    json.dump(res, open(os.path.join(OUT, "dissociate.json"), "w"), indent=1)
    print("\nDISSOCIATE_DONE", flush=True)


if __name__ == "__main__":
    main()
