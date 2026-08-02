"""Teacher-forced block-dynamics probe (single-token prediction only, no generation).

Streams: prefix = 600-word walk on the context's own graph; continuation = 6 blocks of
100 GROUND-TRUTH steps alternating ring-block / grid-block (each thread continues its
own genuine walk across its blocks). Both primed contexts (grid-prefix, ring-prefix)
read the same continuation. At every continuation position we record the model's
one-step predictive mass on grid-neighbours and ring-neighbours of the previous token.

Out: out_blockprobe/blockprobe.json  (per-step mass series averaged over pairs, plus
the streams themselves for local model fitting)
"""
from __future__ import annotations
import os, sys, json
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import graph as G
import models as M

DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WORDS = "clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,chair,candle,bread,mirror,apple,river".split(",")
N, P, CTX = 16, 8, 600
BL = int(os.environ.get("BL", "100"))
NB = int(os.environ.get("NB", "6"))
OUT = os.environ.get("OUTDIR", "/root/test-1/out_blockprobe")


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=P, walk_length=CTX + BL * NB // 2, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    A_g, A_r = adjacency(grid), adjacency(ring)
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))

    T = BL * NB
    streams, srcs = [], []
    for p in range(P):
        ri, gi = CTX, CTX
        seq, src = [], []
        for b in range(NB):
            thread = "R" if b % 2 == 0 else "G"       # ring block first
            for _ in range(BL):
                if thread == "R":
                    seq.append(rw[p].nodes[ri]); ri += 1
                else:
                    seq.append(gw[p].nodes[gi]); gi += 1
                src.append(thread)
        streams.append(seq); srcs.append(src)

    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)

    series = {}
    for ctxname, walks in (("grid", gw), ("ring", rw)):
        mg = np.zeros(T); mr = np.zeros(T)
        for p in range(P):
            pref = walks[p].nodes[:CTX]
            full = pref + streams[p]
            ids = torch.tensor([[bos] + [cand[x] for x in full]], device=DEVICE)
            pr = torch.softmax(model(input_ids=ids).logits[0][:, cand_t].float(),
                               -1).cpu().numpy()
            for t in range(T):
                pv = full[CTX + t - 1]
                probs = pr[CTX + t]
                mg[t] += probs[A_g[pv]].sum() / P
                mr[t] += probs[A_r[pv]].sum() / P
        series[ctxname] = {"grid_mass": np.round(mg, 4).tolist(),
                           "ring_mass": np.round(mr, 4).tolist()}
        print(f"{ctxname} context done", flush=True)

    json.dump({"P": P, "ctx": CTX, "bl": BL, "nb": NB, "words": WORDS,
               "streams": [list(map(int, s)) for s in streams],
               "sources": srcs,
               "prefixes": {"grid": [w.nodes[:CTX] for w in gw],
                            "ring": [w.nodes[:CTX] for w in rw]},
               "series": series},
              open(os.path.join(OUT, "blockprobe.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
