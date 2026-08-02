"""A-under-responsive-B vs A-under-broadcasting-B: behavioral policy comparison.

For conditions bidir (B responds to A), owBA (B deaf, broadcasts own stream; A live),
and indep (reference), reconstruct the grid-primed A context on each condition's
stream and collect its one-step predictive at every late position. Build the
state-conditional policy matrix M[prev, :] = mean predictive given previous node,
late window (t >= 300).

Comparisons:
  behavioral: count-weighted mean JS between policy rows across conditions,
              against a split-half WITHIN-condition noise floor (pairs 0-3 vs 4-7)
  behavioral profile: late mass on grid-/ring-neighbours per condition

Out: out_recip/a_policy.json
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
N, P, CTX, T = 16, 8, 600, 600
RECIP = os.environ.get("RECIP", "/root/test-1/out_recip")
CONDS = ["bidir", "owBA", "indep"]


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def js(a, b):
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    def kl(x, y):
        mk = x > 0
        return float((x[mk] * np.log(x[mk] / np.maximum(y[mk], 1e-12))).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


@torch.no_grad()
def main():
    rec = json.load(open(os.path.join(RECIP, "recip.json")))
    streams = rec["streams"]
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=P, walk_length=CTX, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    A_g, A_r = adjacency(grid), adjacency(ring)
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)

    pol = {}     # cond -> half -> (M[N,N], cnt[N])
    masses = {}
    for cond in CONDS:
        Mh = {0: (np.zeros((N, N)), np.zeros(N)), 1: (np.zeros((N, N)), np.zeros(N))}
        mg = mr = 0.0; nn = 0
        for p in range(P):
            pre = gw[p].nodes
            seq = streams[cond][p]
            full = pre + seq
            ids = torch.tensor([[bos] + [cand[x] for x in full]], device=DEVICE)
            pr = torch.softmax(model(input_ids=ids).logits[0][:, cand_t].float(),
                               -1).cpu().numpy()
            half = 0 if p < P // 2 else 1
            for t in range(300, T):
                pv = full[CTX + t - 1]
                probs = pr[CTX + t]
                Mh[half][0][pv] += probs
                Mh[half][1][pv] += 1
                mg += probs[A_g[pv]].sum(); mr += probs[A_r[pv]].sum(); nn += 1
        pol[cond] = Mh
        masses[cond] = {"grid_nbrs": float(mg / nn), "ring_nbrs": float(mr / nn)}
        print(cond, "policy collected", flush=True)

    def full_policy(cond):
        M0 = pol[cond][0][0] + pol[cond][1][0]
        c0 = pol[cond][0][1] + pol[cond][1][1]
        return M0, c0

    def pol_js(Ma, ca, Mb, cb):
        w = np.minimum(ca, cb)
        tot = w.sum()
        s = 0.0
        for i in range(N):
            if ca[i] > 2 and cb[i] > 2:
                s += w[i] * js(Ma[i] / ca[i], Mb[i] / cb[i])
        return float(s / max(tot, 1e-9))

    out = {"masses": masses, "policy_js": {}, "noise_floor": {}}
    for i in range(len(CONDS)):
        for j2 in range(i + 1, len(CONDS)):
            a, b = CONDS[i], CONDS[j2]
            Ma, ca = full_policy(a); Mb, cb = full_policy(b)
            out["policy_js"][f"{a}_vs_{b}"] = pol_js(Ma, ca, Mb, cb)
    for cond in CONDS:
        (M0, c0), (M1, c1) = pol[cond][0], pol[cond][1]
        out["noise_floor"][cond] = pol_js(M0, c0, M1, c1)
    json.dump(out, open(os.path.join(RECIP, "a_policy.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
