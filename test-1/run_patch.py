"""Activation patching between trap basins.

From the basin experiment: pick rollouts locked in basin X and steer their continued
generation toward basin Y by adding alpha * v_L to the residual stream at deep layers
(24-31) at each newly generated position, where
    v_L = mean(NM[Y, ctx, L, {Y-trap pair}]) - mean(NM[X, ctx, L, {X-trap pair}])
computed from the stored per-rollout node-mean captures. Arms:
    steer alpha in {1, 2, 4}  |  random vector matched norm (alpha=2)
    same-basin vector (alpha=2, Y' in X's own basin)
Outcome: last-50 dominant pair == target trap (switched) / == own trap (stayed) / other.

Env: DEVICE. In: out_basin/{basin.json,nodemeans_basin.npz}. Out: out_patch/patch.json
"""
from __future__ import annotations
import os, sys, json, time
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
BASIN = os.environ.get("BASIN", "/root/test-1/out_basin")
OUT = os.environ.get("OUTDIR", "/root/test-1/out_patch")
N, CTX, T, TP, K = 16, 1000, 600, 100, 2
DEEP = list(range(24, 32))


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    b = json.load(open(os.path.join(BASIN, "basin.json")))
    NM = np.load(os.path.join(BASIN, "nodemeans_basin.npz"))["nm"].astype(np.float32)
    WORDS = b["words"]
    traps = [tuple(t[0]) for t in b["main_traps"]]
    groups = {}
    for r, t in enumerate(traps):
        groups.setdefault(frozenset(t), []).append(r)
    big = sorted([g for g in groups.items() if len(g[1]) >= 4],
                 key=lambda kv: -len(kv[1]))
    bA, bB = big[0], big[1]                       # two largest basins
    X = bA[1][:6] + bB[1][:4]                     # rollouts to steer
    tgt = {r: (bB if r in bA[1] else bA) for r in X}
    same = {r: (bA if r in bA[1] else bB) for r in X}
    print("basin A:", sorted(bA[0]), len(bA[1]), " basin B:", sorted(bB[0]), len(bB[1]),
          flush=True)

    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=1, walk_length=CTX, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    gwalk = G.generate_walks(grid, replace(cfg, graph_type="grid"))[0].nodes
    rwalk = G.generate_walks(ring, replace(cfg, graph_type="ring"))[0].nodes
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    blocks = M._decoder_blocks(model)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)
    grow = [bos] + [cand[x] for x in gwalk]
    rrow = [bos] + [cand[x] for x in rwalk]
    streams = b["main_streams"]
    NX = len(X)

    def vecs(kind, rng):
        """per (row, layer) steering vectors; rows = NX grid ctx then NX ring ctx."""
        V = np.zeros((2 * NX, len(DEEP), NM.shape[-1]), np.float32)
        for i, r in enumerate(X):
            tset = tgt[r] if kind in ("target",) else same[r]
            Yr = [y for y in tset[1] if y != r][0]
            ta = sorted(tset[0]); xa = sorted(traps[r])
            for ci in (0, 1):
                for li in range(len(DEEP)):
                    vy = NM[Yr, ci, li, ta].mean(0)
                    vx = NM[r, ci, li, xa].mean(0)
                    v = vy - vx
                    if kind == "random":
                        rv = rng.standard_normal(v.shape).astype(np.float32)
                        v = rv * (np.linalg.norm(v) / max(np.linalg.norm(rv), 1e-9))
                    V[i + ci * NX, li] = v
        return V

    def run_arm(name, V, alpha):
        rows = [grow + [cand[x] for x in streams[r]] for r in X] + \
               [rrow + [cand[x] for x in streams[r]] for r in X]
        ids = torch.tensor(rows, device=DEVICE)
        try:
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=ids, use_cache=True)
        past, logits = o.past_key_values, o.logits[:, -1, :]
        Vt = {L: torch.tensor(alpha * V[:, li], device=DEVICE, dtype=torch.bfloat16)
              for li, L in enumerate(DEEP)}
        hooks = []
        def mk(L):
            def hh(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h[:, -1, :] += Vt[L]
                return out
            return hh
        for L in DEEP:
            hooks.append(blocks[L].register_forward_hook(mk(L)))
        rng = np.random.default_rng(55)
        bstreams = [[] for _ in range(NX)]
        try:
            for i in range(TP):
                t = T + i
                gen_rows = list(range(NX, 2 * NX)) if t % 2 == 0 else list(range(NX))
                pr = torch.softmax(logits[gen_rows][:, cand_t].float(), -1).cpu().numpy()
                toks = np.zeros(NX, np.int64)
                for j in range(NX):
                    pp = pr[j].copy()
                    pp[np.argsort(pp)[:-K]] = 0.0
                    node = int(rng.choice(N, p=pp / pp.sum()))
                    bstreams[j].append(node)
                    toks[j] = cand[node]
                inp = torch.tensor(np.concatenate([toks, toks]), device=DEVICE)[:, None]
                o = model(input_ids=inp, past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[:, -1, :]
        finally:
            for h in hooks:
                h.remove()
        del past
        torch.cuda.empty_cache()
        res = []
        for j, r in enumerate(X):
            s = bstreams[j][-50:]
            cnt = {}
            for a, b2 in zip(s, s[1:]):
                kk = frozenset((a, b2)) if a != b2 else frozenset((a,))
                cnt[kk] = cnt.get(kk, 0) + 1
            top = max(cnt.items(), key=lambda kv: kv[1])
            lab = ("switched" if top[0] == tgt[r][0] else
                   "stayed" if top[0] == frozenset(traps[r]) else "other")
            res.append({"rollout": r, "top": sorted(top[0]), "frac": top[1] / 49,
                        "label": lab})
        print(name, {l: sum(1 for x in res if x["label"] == l)
                     for l in ("switched", "stayed", "other")}, flush=True)
        return res

    rngv = np.random.default_rng(7)
    arms = {}
    for name, kind, alpha in (("steer_a1", "target", 1.0), ("steer_a2", "target", 2.0),
                              ("steer_a4", "target", 4.0),
                              ("random_a2", "random", 2.0),
                              ("samebasin_a2", "same", 2.0)):
        V = vecs("random" if kind == "random" else
                 ("target" if kind == "target" else "same"), rngv)
        arms[name] = run_arm(name, V, alpha)
    json.dump({"X": X, "traps": {r: sorted(traps[r]) for r in X},
               "targets": {r: sorted(tgt[r][0]) for r in X}, "arms": arms},
              open(os.path.join(OUT, "patch.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
