"""Name-read sycophancy vectors (Sandra 2026-08-20): is syco_caa's inertness at
name tokens a READ-POSITION artifact? Two person-bound variants:

  syco_name2    CAA pairs unchanged; persona's first name APPENDED after the
                answer letter; read at that second mention (downstream of the
                agree/disagree divergence, so the diff is nonzero).
  syco_endname  contrast moved UPSTREAM (opinionated vs no-view persona,
                intro anonymized); name revealed in a final sentence before
                the choices; read there. (A pre-letter read of the CAA pairs
                would be identically zero by causal attention.)

Both stored in vectors2.npz; tested on the held-out syco bed at the HOLDER-NAME
site (the read-position-matched test), bystander, and all-positions arms.
-> out/syco_vec2.json
env: MODEL NDERIVE (400) OUT
"""
from __future__ import annotations
import json, os, random, re, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, first_id, load, resid
from newvec_build import save_family
from syco import margin_match, variants

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NDERIVE = int(os.environ.get("NDERIVE", "400"))
LAYER = 45
BELIEF = re.compile(r"I (believe|think|agree with the claim|disagree with the claim)[^.]*\.")
NAME_RE = re.compile(r"[Mm]y name is ([A-Z][a-z]+)")


def pool():
    test_qs = {json.loads(l)["question"] for l in open(os.path.join(OUT, "syco_subset.jsonl"))}
    out = []
    for fn in ("syco_nlp.jsonl", "syco_phil.jsonl"):
        for line in open(os.path.join(OUT, fn)):
            try:
                ex = json.loads(line)
            except Exception:
                continue
            q = ex.get("question", "")
            mb = ex.get("answer_matching_behavior", "").strip()
            nm = NAME_RE.search(q)
            if (q and mb in ("(A)", "(B)") and q not in test_qs
                    and 200 < len(q) < 2500 and nm and BELIEF.search(q)):
                out.append((q, mb[1], nm.group(1)))
    random.seed(2)
    random.shuffle(out)
    print(f"[pool] {len(out)} usable, taking {NDERIVE}", flush=True)
    return out[:NDERIVE]


def last_tok_read(model, tok, text, layers):
    n = len(tok(text)["input_ids"])
    return resid(model, tok, text, layers, [n - 1])


def main():
    model, tok, _ = load(); model.eval()
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    P = pool()

    items = []
    for q, m, first in P:
        other = "B" if m == "A" else "A"
        tp = chat(tok, "", q, f" ({m})\n\n{first}")
        tn = chat(tok, "", q, f" ({other})\n\n{first}")
        rp = last_tok_read(model, tok, tp, layers)
        rn = last_tok_read(model, tok, tn, layers)
        items.append({l: rp[l] - rn[l] for l in layers})
    save_family(z, "syco_name2", items, layers)
    np.savez(npz, **z)

    items = []
    kept = 0
    for q, m, first in P:
        anon = re.sub(r"Hello, [Mm]y name is [A-Z][a-zA-Z]+( [A-Z][a-zA-Z.]+)?\.?", "Hello.", q)
        neg_body = BELIEF.sub("I have not yet formed a view on this.", anon, count=1)
        if neg_body == anon:
            continue
        reveal = f"\n\nThe person speaking to you is named {first}.\n"
        ci = anon.find("\nChoices:")
        if ci < 0:
            continue
        pos_t = anon[:ci] + reveal
        neg_t = neg_body[:ci if len(neg_body) >= ci else 0]
        neg_ci = neg_body.find("\nChoices:")
        if neg_ci < 0:
            continue
        neg_t = neg_body[:neg_ci] + reveal
        rp = last_tok_read(model, tok, chat(tok, "", pos_t, ""), layers)
        rn = last_tok_read(model, tok, chat(tok, "", neg_t, ""), layers)
        items.append({l: rp[l] - rn[l] for l in layers})
        kept += 1
    print(f"[endname] {kept} pairs", flush=True)
    save_family(z, "syco_endname", items, layers)
    np.savez(npz, **z)

    D = DIRS.load_all(OUT, LAYER)
    for new in ("syco_name2", "syco_endname"):
        cs = sorted(((k, float(D[new] @ w)) for k, w in D.items() if k != new),
                    key=lambda kv: -abs(kv[1]))[:6]
        print(f"[cos L45] {new}: " + "  ".join(f"{k} {c:+.2f}" for k, c in cs), flush=True)

    # ---- held-out syco bed: holder-name (matched site), bystander, allpos --
    exs = [json.loads(l) for l in open(os.path.join(OUT, "syco_subset.jsonl"))][:120]
    tok_a, tok_b = first_id(tok, "A"), first_id(tok, "B")
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    res = {"steer": {}}
    for dn in ("syco_name2", "syco_endname", "syco_caa"):
        for arm, alphas in (("holder", (0.35, 0.5)), ("bystander", (0.5,)),
                            ("allpos", (0.1, 0.2))):
            for a in alphas:
                vv = D[dn] * nrm * a
                ds = []
                for ex in exs:
                    text, anchor = variants(ex)[arm if arm != "allpos" else "holder"]
                    full = chat(tok, "", text, " (")
                    pos = None if arm == "allpos" else DIRS.name_positions(tok, full, anchor)
                    mp = margin_match(model, tok, full, ex, tok_a, tok_b, (LAYER, vv), pos)
                    mm = margin_match(model, tok, full, ex, tok_a, tok_b, (LAYER, -vv), pos)
                    ds.append(mp - mm)
                ds = np.array(ds)
                res["steer"][f"{dn}|{arm}|a{a}"] = (
                    float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(len(ds))))
                print(f"[bed] {dn:<13} {arm:<9} a={a:<4} Δ {ds.mean():+5.2f} "
                      f"+- {ds.std(ddof=1)/np.sqrt(len(ds)):.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "syco_vec2.json"), "w"), indent=1)
    print("SYCOVEC2_DONE", flush=True)


if __name__ == "__main__":
    main()
