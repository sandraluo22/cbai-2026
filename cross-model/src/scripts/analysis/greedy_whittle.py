"""Whittle the 21-head set down: is there a SMALLER sufficient subset, and is its identity special?

Context: the 21 heads recover 47.6% of the parity margin and ~90% neighbour validity with the other 1003
mean-ablated, but NO 12/9 partition works — and random 12/9 splits behave the same as the motif split, so
the superadditivity is a property of the set, not of motif structure. That leaves open whether some
smaller subset suffices, and whether greedy selection finds anything a random subset of the same size
would not.

Two searches, both keep-only (everything outside the current set mean-ablated):
  forward   start empty, repeatedly ADD the head that most improves the objective
  backward  start from all 21, repeatedly REMOVE the head whose loss costs least

**The control that makes the curve mean something**: at every size k we also evaluate NRAND RANDOM
k-subsets OF THE 21. If greedy tracks random, selection order is irrelevant and only the count matters —
which is what the 12/9 partition result would predict. If greedy separates from random, some heads carry
disproportionate weight and there is real internal structure.

Objective is neighbour validity (the task), with the parity margin reported alongside; both are read off
the same forward passes. Greedy is run on the objective only, so parity is an out-of-objective check.

Env: GEN_MODEL(Llama) K(4) OBJ(nbr|parity) NWALKS(3) WLEN(1200) CTXLO(800) NRAND(4)
     BACKWARD(1) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/greedy_whittle<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4")); OBJ = os.environ.get("OBJ", "nbr")
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "4"))
BACKWARD = os.environ.get("BACKWARD", "1") == "1"; SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

ALL21 = ["L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7", "L14H26", "L9H11", "L1H20",
         "L21H10", "L4H12", "L3H17", "L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11",
         "L4H16", "L1H21", "L2H26"]
DUP9 = {"L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11", "L4H16", "L1H21", "L2H26"}


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev)
    neg_i = torch.tensor(np.where(col < 0)[0], device=dev)
    data = []
    for w in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(w.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in w.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [w.nodes[s] for s in steps]))

    st = {"keep": None}
    hooks = []
    for l in range(nL):
        def mk(l):
            def ph(_m, args):
                kp = st["keep"]
                if kp is None: return
                sel = [h for h in range(nH) if (l, h) not in kp]
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    def parse(x): return {(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x}

    def evaluate(keep):
        st["keep"] = None if keep is None else parse(keep)
        ok = tot = 0; pm = 0.0
        for ids, rp, nds in data:
            lsm = torch.log_softmax(model(input_ids=ids).logits[0][rp][:, cand_t].float(), 1)
            top = lsm.argmax(1).tolist()
            cur = torch.tensor([col[u] for u in nds], device=dev)
            opp = torch.where((cur > 0)[:, None], lsm[:, neg_i], lsm[:, pos_i])
            sam = torch.where((cur > 0)[:, None], lsm[:, pos_i], lsm[:, neg_i])
            pm += float((torch.logsumexp(opp, 1) - torch.logsumexp(sam, 1)).sum())
            for t_, u in zip(top, nds):
                ok += int(t_ in graph.adjacency[u]); tot += 1
        st["keep"] = None
        return ok / tot, pm / tot

    full = evaluate(None); floor = evaluate([])
    all21 = evaluate(ALL21)
    obj_i = 0 if OBJ == "nbr" else 1
    print(f"[{tag}] full {full[0]:.4f}/{full[1]:+.3f}   all21 {all21[0]:.4f}/{all21[1]:+.3f}   "
          f"floor {floor[0]:.4f}/{floor[1]:+.3f}   (objective = {OBJ})", flush=True)
    res = {"model": tag, "obj": OBJ, "full": full, "floor": floor, "all21": all21,
           "forward": [], "backward": [], "random_by_k": {}}

    # ---- greedy FORWARD ----
    print(f"\n{'k':>3} {'added':<9} {'nbr':>8} {'parity':>9} {'rand_nbr':>9} {'rand_sd':>8}  set")
    cur, rest = [], list(ALL21)
    for k in range(1, len(ALL21) + 1):
        best, bestv = None, None
        for h in rest:
            v = evaluate(cur + [h])
            if bestv is None or v[obj_i] > bestv[obj_i]: best, bestv = h, v
        cur = cur + [best]; rest.remove(best)
        rv = []
        for _ in range(NRAND):
            sub = [ALL21[j] for j in rng.choice(len(ALL21), k, replace=False)]
            rv.append(evaluate(sub))
        rn = float(np.mean([x[obj_i] for x in rv])); rs = float(np.std([x[obj_i] for x in rv]))
        res["forward"].append({"k": k, "added": best, "nbr": round(bestv[0], 4),
                               "parity": round(bestv[1], 4), "rand_mean": round(rn, 4),
                               "rand_sd": round(rs, 4), "set": list(cur)})
        res["random_by_k"][str(k)] = {"mean": round(rn, 4), "sd": round(rs, 4)}
        mark = "*" if best in DUP9 else "+"
        print(f"{k:3} {best+mark:<9} {bestv[0]:8.4f} {bestv[1]:+9.3f} {rn:9.4f} {rs:8.4f}  "
              f"{'/'.join(cur[-3:])}", flush=True)
        if k >= 12 and bestv[obj_i] >= all21[obj_i] - 1e-6: break

    # ---- greedy BACKWARD ----
    if BACKWARD:
        print(f"\n{'k':>3} {'removed':<9} {'nbr':>8} {'parity':>9}   (backward: drop the cheapest head)")
        cur2 = list(ALL21)
        while len(cur2) > 1:
            best, bestv = None, None
            for h in cur2:
                v = evaluate([x for x in cur2 if x != h])
                if bestv is None or v[obj_i] > bestv[obj_i]: best, bestv = h, v
            cur2 = [x for x in cur2 if x != best]
            res["backward"].append({"k": len(cur2), "removed": best, "nbr": round(bestv[0], 4),
                                    "parity": round(bestv[1], 4), "set": list(cur2)})
            mark = "*" if best in DUP9 else "+"
            print(f"{len(cur2):3} {best+mark:<9} {bestv[0]:8.4f} {bestv[1]:+9.3f}", flush=True)
    for h in hooks: h.remove()
    print("\n(* = duplicate-token head, + = induction/prev-token head)")
    p_ = f"{OUTDIR}/greedy_whittle{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"DONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
