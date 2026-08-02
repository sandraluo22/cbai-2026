"""Do induction heads copy the LATEST successor while duplicate-token heads aggregate ALL occurrences?

The superadditivity (induction12 3.1%, duplicate9 -2.2%, together 47.6%) needs a mechanism. The standing
hypothesis is that on a walk they carry different payloads: an induction head attends to the SUCCESSOR of
a previous occurrence (retrieving what followed x last time), a duplicate-token head attends to the
occurrences THEMSELVES (aggregating the history of x). If so, they should dissociate under a conflict
between RECENCY and FREQUENCY. If they implement the same retrieval, they should not.

Design — a 2x2 with x -> a and x -> b planted at controlled counts and orders:
    freq:   a-dominant (8 a, 2 b)   vs   b-dominant (2 a, 8 b)
    recent: the LAST planted successor before the query is a   vs   b
Filler tokens separate the pairs so adjacency is never the cue. The prompt ends on x and we read
    margin = logit(a) - logit(b)
    recency_effect   = mean(margin | last=a) - mean(margin | last=b)      [both freq levels]
    frequency_effect = mean(margin | a-dominant) - mean(margin | b-dominant)  [both recency levels]
The two effects are orthogonal by construction, so ablation can move one without the other.

Predictions if the payload hypothesis is right:
    ablate induction12 -> RECENCY effect shrinks   (no successor-copying)
    ablate duplicate9  -> FREQUENCY effect shrinks (no occurrence-aggregation)
If both effects fall together for both sets, they implement the same retrieval and the hypothesis is dead.

Env: GEN_MODEL(Llama) NPAIRS(10) NDOM(8) FILL(3) NSEQ(40) NRAND(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/conflicting_transitions<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
NPAIRS = int(os.environ.get("NPAIRS", "10")); NDOM = int(os.environ.get("NDOM", "8"))
FILL = int(os.environ.get("FILL", "3")); NSEQ = int(os.environ.get("NSEQ", "40"))
NRAND = int(os.environ.get("NRAND", "3")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

INDUCTION12 = ["L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7",
               "L14H26", "L9H11", "L1H20", "L21H10", "L4H12", "L3H17"]
DUPLICATE9 = ["L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11", "L4H16", "L1H21", "L2H26"]


def build(rng, V):
    """returns list of (ids, a_tok, b_tok, freq_a_dominant, last_is_a)"""
    out = []
    for _ in range(NSEQ):
        toks = rng.choice(np.arange(1000, min(V, 30000)), size=3 + NPAIRS * (FILL + 1) + 8,
                          replace=False)
        x, a, b = int(toks[0]), int(toks[1]), int(toks[2])
        fill_pool = [int(t) for t in toks[3:]]
        for dom_a in (True, False):
            n_a = NDOM if dom_a else NPAIRS - NDOM
            for last_a in (True, False):
                succ = [a] * n_a + [b] * (NPAIRS - n_a)
                rng.shuffle(succ)
                want = a if last_a else b
                # move one instance of `want` to the final slot; requires it to be present
                if want not in succ: continue
                succ.remove(want); succ.append(want)
                seq, fi = [], 0
                for s in succ:
                    seq += [x, s]
                    seq += fill_pool[fi:fi + FILL]; fi += FILL
                seq.append(x)                                     # query
                out.append((seq, a, b, dom_a, last_a))
    return out


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)
    V = model.get_input_embeddings().weight.shape[0]
    items = build(rng, V)
    print(f"[{tag}] {len(items)} sequences ({NSEQ} token-triples x 4 cells), "
          f"{NPAIRS} planted pairs each, {NDOM}/{NPAIRS - NDOM} split", flush=True)

    st = {"heads": None}
    hooks = []
    for l in range(nL):
        def mk(l):
            def ph(_m, args):
                hs = st["heads"]
                if not hs: return
                sel = [h for (ll, h) in hs if ll == l]
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    def effects():
        cells = {}
        for seq, a, b, dom_a, last_a in items:
            ids = torch.tensor([[tok.bos_token_id or seq[0]] + seq], device=dev)
            lg = model(input_ids=ids).logits[0, -1].float()
            m = float(lg[a] - lg[b])
            cells.setdefault((dom_a, last_a), []).append(m)
        mu = {k: float(np.mean(v)) for k, v in cells.items()}
        rec = ((mu[(True, True)] + mu[(False, True)]) - (mu[(True, False)] + mu[(False, False)])) / 2
        frq = ((mu[(True, True)] + mu[(True, False)]) - (mu[(False, True)] + mu[(False, False)])) / 2
        return rec, frq, mu

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]
    res = {"model": tag, "npairs": NPAIRS, "ndom": NDOM, "conditions": {}}
    r0, f0, mu0 = effects()
    print(f"\n{'condition':<16} {'recency_eff':>12} {'freq_eff':>10}   cell means "
          f"(domA/lastA, domA/lastB, domB/lastA, domB/lastB)")
    print(f"{'baseline':<16} {r0:12.4f} {f0:10.4f}   "
          f"{mu0[(True,True)]:+.2f} {mu0[(True,False)]:+.2f} "
          f"{mu0[(False,True)]:+.2f} {mu0[(False,False)]:+.2f}", flush=True)
    res["conditions"]["baseline"] = {"recency": round(r0, 4), "frequency": round(f0, 4),
                                     "cells": {str(k): round(v, 4) for k, v in mu0.items()}}
    for nm, hs in (("ablate_induction12", INDUCTION12), ("ablate_duplicate9", DUPLICATE9),
                   ("ablate_all21", INDUCTION12 + DUPLICATE9)):
        st["heads"] = parse(hs); r, f, mu = effects(); st["heads"] = None
        res["conditions"][nm] = {"recency": round(r, 4), "frequency": round(f, 4),
                                 "d_recency": round(r - r0, 4), "d_frequency": round(f - f0, 4),
                                 "cells": {str(k): round(v, 4) for k, v in mu.items()}}
        print(f"{nm:<16} {r:12.4f} {f:10.4f}   dRec {r-r0:+.3f}  dFreq {f-f0:+.3f}", flush=True)
    allh = [(l, h) for l in range(nL) for h in range(nH)]
    for k_ in (12, 9):
        rr, ff = [], []
        for _ in range(NRAND):
            st["heads"] = [allh[j] for j in rng.choice(len(allh), k_, replace=False)]
            a_, b_, _ = effects(); rr.append(a_); ff.append(b_); st["heads"] = None
        res["conditions"][f"random{k_}"] = {"recency": round(float(np.mean(rr)), 4),
                                            "frequency": round(float(np.mean(ff)), 4),
                                            "recency_sd": round(float(np.std(rr)), 4),
                                            "frequency_sd": round(float(np.std(ff)), 4)}
        print(f"{'random'+str(k_):<16} {np.mean(rr):12.4f} {np.mean(ff):10.4f}   "
              f"(sd {np.std(rr):.3f}/{np.std(ff):.3f})", flush=True)
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/conflicting_transitions{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
