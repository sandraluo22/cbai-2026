"""CAA-style sycophancy vector (Rimsky et al. 2024, 'Steering Llama 2 via
Contrastive Activation Addition'): Anthropic model-written sycophancy evals,
contrastive answer-letter completions, activation at the letter token, mean
difference. Derivation pool is DISJOINT from the 120-item syco test subset.

Stage 1: derive `syco_caa` (n=NDERIVE pairs), store in vectors2.npz (all
layers, h0/h1 split-half), print cos vs the existing landscape at L45.
Stage 2: home-bed validation on the held-out 120 test items:
  allpos arm  literature site -- +-v at EVERY position, small alphas, with
              answer-mass integrity check
  holder arm  our standard site -- +-v at the opinion holder's name (Ana)
-> out/syco_vec.json

env: MODEL NDERIVE (400) OUT
needs: out/syco_nlp.jsonl out/syco_phil.jsonl out/syco_subset.jsonl
"""
from __future__ import annotations
import json, os, random, sys
import numpy as np
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import Inject, chat, first_id, load, resid, unit
from newvec_build import save_family
from syco import margin_match, variants

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NDERIVE = int(os.environ.get("NDERIVE", "400"))
LAYER = 45


def pairs():
    test_qs = {json.loads(l)["question"] for l in open(os.path.join(OUT, "syco_subset.jsonl"))}
    pool = []
    for fn in ("syco_nlp.jsonl", "syco_phil.jsonl"):
        for line in open(os.path.join(OUT, fn)):
            try:
                ex = json.loads(line)
            except Exception:
                continue
            q = ex.get("question", "")
            mb = ex.get("answer_matching_behavior", "").strip()
            if q and mb in ("(A)", "(B)") and q not in test_qs and 200 < len(q) < 2500:
                pool.append((q, mb[1]))
    random.seed(1)
    random.shuffle(pool)
    print(f"[derive] pool {len(pool)}, using {NDERIVE}", flush=True)
    return pool[:NDERIVE]


def main():
    model, tok, _ = load(); model.eval()
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    items = []
    for i, (q, m) in enumerate(pairs()):
        other = "B" if m == "A" else "A"
        tp = chat(tok, "", q, f" ({m}")      # sycophantic completion
        tn = chat(tok, "", q, f" ({other}")  # non-sycophantic completion
        last_p = len(tok(tp)["input_ids"]) - 1
        last_n = len(tok(tn)["input_ids"]) - 1
        rp = resid(model, tok, tp, layers, [last_p])
        rn = resid(model, tok, tn, layers, [last_n])
        items.append({l: rp[l] - rn[l] for l in layers})
        if i and i % 100 == 0:
            print(f"[derive] {i}", flush=True)
    save_family(z, "syco_caa", items, layers)
    np.savez(npz, **z)
    D = DIRS.load_all(OUT, LAYER)
    v = D["syco_caa"]
    cs = sorted(((k, float(v @ w)) for k, w in D.items() if k != "syco_caa"),
                key=lambda kv: -abs(kv[1]))[:8]
    print("[cos L45] syco_caa: " + "  ".join(f"{k} {c:+.2f}" for k, c in cs), flush=True)

    # ---- stage 2: home bed, held-out test items ---------------------------
    exs = [json.loads(l) for l in open(os.path.join(OUT, "syco_subset.jsonl"))][:120]
    tok_a, tok_b = first_id(tok, "A"), first_id(tok, "B")
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    res = {"steer": {}}

    def mass(text, inj, pos):
        enc = tok(text, return_tensors="pt")
        enc = {k: t.to(model.device) for k, t in enc.items()}
        with torch.no_grad():
            l, vv = inj
            with Inject(model, l, torch.tensor(vv), pos):
                lg = model(**enc).logits[0, -1]
        p = torch.softmax(lg.float(), -1)
        return float(p[tok_a] + p[tok_b])

    for arm, alphas in (("allpos", (0.05, 0.1, 0.2)), ("holder", (0.35, 0.5))):
        for a in alphas:
            vv = v * nrm * a
            ds, ms = [], []
            for ex in exs:
                text, anchor = variants(ex)["holder"]
                full = chat(tok, "", text, " (")
                pos = None if arm == "allpos" else DIRS.name_positions(tok, full, anchor)
                mp = margin_match(model, tok, full, ex, tok_a, tok_b, (LAYER, vv), pos)
                mm = margin_match(model, tok, full, ex, tok_a, tok_b, (LAYER, -vv), pos)
                ds.append(mp - mm)
            ds = np.array(ds)
            intg = np.mean([mass(chat(tok, "", variants(ex)["holder"][0], " ("),
                                 (LAYER, vv), None if arm == "allpos" else None)
                            for ex in exs[:12]]) if arm == "allpos" else 1.0
            res["steer"][f"{arm}|a{a}"] = (float(ds.mean()),
                                           float(ds.std(ddof=1) / np.sqrt(len(ds))),
                                           float(intg))
            print(f"[home] {arm:<7} a={a:<5} Δ syco-margin {ds.mean():+5.2f} "
                  f"+- {ds.std(ddof=1)/np.sqrt(len(ds)):.2f}  mass {intg:.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "syco_vec.json"), "w"), indent=1)
    print("SYCOVEC_DONE", flush=True)


if __name__ == "__main__":
    main()
