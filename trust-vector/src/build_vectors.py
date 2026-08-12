"""Stage 1 — derive one candidate trust direction per method, save them all.

For each method and each read-anchor, v = mean over matched pairs of
(resid[positive] - resid[negative]), taken at a position inside the SHARED tail so
the difference is not the contrast tokens. Written unnormalised; `compare.py` and
`steer_qsg.py` normalise and rescale.

Split halves (even / odd stimulus pairs) are saved alongside the full vector: the
within-method split-half cosine is the ceiling any cross-method cosine has to be
read against. Two methods cannot agree more than each agrees with itself.

Stage 2 (VALIDATE=1) is the home-domain causal check: inject +/-alpha*v into
evidence-free "will {name} keep their word?" prompts and read p(yes). A direction
that does not move its own domain has no business being pushed into the game.

env: MODEL (Qwen32) LAYERS (every 4th, or "all") NPAIR (12) ALPHA (0.5)
     FMT (chat|raw) VALIDATE (1) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli as S  # noqa: E402
from common import (Inject, fmt_fn, load, margin, n_tokens, p_first,  # noqa: E402
                    resid, tok_idx, unit)

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)


def pick_layers(model):
    nL = model.config.num_hidden_layers
    env = os.environ.get("LAYERS", "")
    if env == "all":
        return list(range(nL + 1))
    if env:
        return [int(x) for x in env.split(",")]
    return list(range(4, nL + 1, 4))


def main():
    model, tok, _ = load()
    model.eval()
    fmt = fmt_fn()
    layers = pick_layers(model)
    npair = int(os.environ.get("NPAIR", "12"))
    print(f"[cfg] layers={layers} npair={npair} fmt={os.environ.get('FMT','chat')}",
          flush=True)

    npz_p = os.path.join(OUT, "vectors.npz")
    meta_p = os.path.join(OUT, "vectors_meta.json")
    if os.environ.get("SKIP_VEC") == "1" and os.path.exists(npz_p):
        z = np.load(npz_p)
        layers = [int(x) for x in z["layers"]]
        store = {k: z[k] for k in z.files if k != "layers"}
        rn = json.load(open(meta_p))["resid_norm"]
        norms = {l: [float(rn[str(l)])] for l in layers}
        print(f"[vec] SKIP_VEC: reusing {len(store)} directions from {npz_p}", flush=True)
        return validate(model, tok, fmt, store, layers, norms)

    store, meta, norms = {}, {}, {l: [] for l in layers}
    for method in S.ALL:
        sysmsg, prefill, anchors = S.SPEC[method]
        acc = {a: [] for a in anchors}           # per-pair diffs, per anchor
        lenmatch = []
        for p_user, n_user, m in S.pairs(method, npair):
            # rationale methods carry the contrast in a per-pair prefill
            p_txt = fmt(tok, sysmsg, p_user, m.get("prefill_pos", prefill))
            n_txt = fmt(tok, sysmsg, n_user, m.get("prefill_neg", prefill))
            lenmatch.append(n_tokens(tok, p_txt) - n_tokens(tok, n_txt))
            for anchor in anchors:
                sp_p = S.anchor_spans(method, p_txt, m["name"]).get(anchor)
                sp_n = S.anchor_spans(method, n_txt, m["name"]).get(anchor)
                pos_p = None if anchor == "last" else tok_idx(tok, p_txt, sp_p)
                pos_n = None if anchor == "last" else tok_idx(tok, n_txt, sp_n)
                if anchor != "last" and (not pos_p or not pos_n):
                    continue
                rp = resid(model, tok, p_txt, layers, pos_p)
                rn = resid(model, tok, n_txt, layers, pos_n)
                acc[anchor].append({l: rp[l] - rn[l] for l in layers})
                if anchor == "last":
                    for l in layers:
                        norms[l] += [float(np.linalg.norm(rp[l])),
                                     float(np.linalg.norm(rn[l]))]
        for anchor in anchors:
            d = acc[anchor]
            if not d:
                print(f"[warn] {method}/{anchor}: no usable pairs", flush=True)
                continue
            for half, sel in (("full", range(len(d))),
                              ("h0", range(0, len(d), 2)),
                              ("h1", range(1, len(d), 2))):
                sel = list(sel)
                V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
                store[f"{method}--{anchor}--{half}"] = V.mean(0)      # (n_layers, d)
        meta[method] = dict(anchors=list(anchors), n_pairs=len(acc[anchors[0]]),
                            tok_len_delta=lenmatch)
        md = int(np.max(np.abs(lenmatch))) if lenmatch else 0
        print(f"[vec] {method}: {len(acc[anchors[0]])} pairs, anchors {list(anchors)}, "
              f"max |token-length mismatch| {md}", flush=True)

    np.savez(os.path.join(OUT, "vectors.npz"), layers=np.array(layers), **store)
    json.dump(dict(meta=meta, layers=layers,
                   resid_norm={str(l): float(np.mean(v)) for l, v in norms.items()},
                   model=os.environ.get("MODEL", "Qwen32"),
                   fmt=os.environ.get("FMT", "chat")),
              open(os.path.join(OUT, "vectors_meta.json"), "w"), indent=1)
    print(f"[vec] wrote {len(store)} directions -> {OUT}/vectors.npz", flush=True)

    if os.environ.get("VALIDATE", "1") != "1":
        print("BUILD_DONE", flush=True)
        return
    return validate(model, tok, fmt, store, layers, norms)


def validate(model, tok, fmt, store, layers, norms):
    # ---- stage 2: home-domain causal check --------------------------------
    # Primary read-out is the LOGIT MARGIN, not p(yes): with no evidence either way
    # the model answers "no" to these probes with p(yes) < 1e-3, so a probability is
    # pinned at the floor, -v has nowhere to go, and swings are incomparable across
    # methods. The margin is unbounded and moves in both directions. p(yes) is kept
    # alongside it only to show where on the curve the margin sits.
    print("\n=== validation: evidence-free trust question under +/-alpha*v ===\n"
          "    margin = logit(yes) - logit(no), unbounded; p(yes) shown for scale",
          flush=True)
    alpha = float(os.environ.get("ALPHA", "0.5"))
    fams = {}
    for kind in ("neutral", "trusting"):
        pr = [fmt(tok, S.SYS_JSON, q, '{"answer": "') for q, _ in S.holdout(8, kind=kind)]
        fams[kind] = (pr, float(np.mean([margin(model, tok, t, "yes", "no") for t in pr])),
                      float(np.mean([p_first(model, tok, t, ["yes", "no"]) for t in pr])))
        print(f"  baseline [{kind:<8}] margin {fams[kind][1]:+8.3f}  "
              f"(p(yes) {fams[kind][2]:.4f})", flush=True)
    val = {}
    for key in sorted(store):
        method, anchor, half = key.split("--")
        if half != "full" or anchor != "last":
            continue
        for l_i, l in enumerate(layers):
            nrm = float(np.mean(norms[l])) if len(norms[l]) else 0.0
            v = unit(store[key][l_i]) * nrm * alpha
            rec = {}
            for kind, (probes, bm, bp) in fams.items():
                up, dn = [], []
                for t in probes:
                    with Inject(model, l, torch.tensor(v), None):
                        up.append(margin(model, tok, t, "yes", "no"))
                    with Inject(model, l, torch.tensor(-v), None):
                        dn.append(margin(model, tok, t, "yes", "no"))
                rec[kind] = dict(base=bm, plus=float(np.mean(up)),
                                 minus=float(np.mean(dn)))
            # bidirectional score: +v must raise the neutral probe AND -v must lower
            # the trusting one. A direction that only does the first is pushing on the
            # answer, not on how the partner is represented.
            rec["bidir"] = ((rec["neutral"]["plus"] - rec["neutral"]["base"]) +
                            (rec["trusting"]["base"] - rec["trusting"]["minus"])) / 2
            rec["plus"] = rec["neutral"]["plus"]      # keys steer_qsg.py reads
            rec["minus"] = rec["neutral"]["minus"]
            val[f"{method}_L{l}"] = rec
        cand = [l for l in layers if np.isfinite(val[f"{method}_L{l}"]["bidir"])]
        best = max(cand, key=lambda l: val[f"{method}_L{l}"]["bidir"])
        b = val[f"{method}_L{best}"]
        print(f"  {method:<11} best L{best:<3} neutral +v {b['neutral']['plus']:+8.3f} "
              f"(base {b['neutral']['base']:+7.3f}) | trusting -v "
              f"{b['trusting']['minus']:+8.3f} (base {b['trusting']['base']:+7.3f}) | "
              f"bidir {b['bidir']:+.3f}", flush=True)
    json.dump(val, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
