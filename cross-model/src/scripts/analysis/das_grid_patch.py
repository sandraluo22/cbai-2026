"""DAS on grid parity in L14H26 with TWO interchange counterfactuals defined by a node permutation pi
(true source-based interchange, vs das_parity's colour-prototype delta):

  PATCH A 'global'  pi = 90-degree grid rotation = an AUTOMORPHISM of the 4x4 grid that maps every node to an
                    OPPOSITE-parity node (parity (r+c) -> (c+3-r) = (r+c+1) mod 2). A clean global odd<->even swap.
  PATCH B 'swap2'   pi = one transposition of two opposite-parity nodes; delta is zero except at those two
                    positions -> a minimal, LOCAL parity flip.

For each patch we inject, into the aligned rank-r subspace at every node token t (current node X),
    delta_t = znode[pi(X)] - znode[X]        (source = counterfactual node's mean head output)
train an orthogonal rotation R (frozen model) to make the model predict pi(X)'s neighbour-parity at the
FLIPPED positions (where pi(X)!=X). Sweep r, compare to full head + random subspace. Then compare the two
patches' learned subspaces (cosine / principal angle) and map each to the parity axis + Laplacian eigenmodes.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NWALKS(24) WLEN(260) CTXLO(100)
     RDIMS(1,2,4,8,128) STEPS(80) BATCH(3) LR(0.02) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/das_grid_patch_<model>_L<layer>H<head>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
NWALKS = int(os.environ.get("NWALKS", "24")); WLEN = int(os.environ.get("WLEN", "260")); CTXLO = int(os.environ.get("CTXLO", "100"))
RDIMS = [int(x) for x in os.environ.get("RDIMS", "1,2,4,8,128").split(",")]
STEPS = int(os.environ.get("STEPS", "80")); BATCH = int(os.environ.get("BATCH", "3"))
LR = float(os.environ.get("LR", "0.02")); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/das")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    n = graph.n_nodes; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(float)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def rot90_perm(coords, R, Cn):
    """node permutation pi for a 90-deg rotation of an RxCn grid: (r,c) -> (c, R-1-r). Requires R==Cn."""
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    perm = np.zeros(len(coords), int)
    for i, (r, c) in enumerate(coords):
        perm[i] = idx[(int(c), int(R - 1 - r))]
    return perm


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12); return float(abs(a @ b))


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4, n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    col = two_colour(graph)

    # ---- permutations ----
    pi_global = rot90_perm(coords, 4, 4)
    assert all(col[pi_global[i]] == -col[i] for i in range(n)), "rotation must invert parity of every node"
    pos_nodes = [i for i in range(n) if col[i] > 0]; neg_nodes = [i for i in range(n) if col[i] < 0]
    a = pos_nodes[0]; b = next(j for j in neg_nodes if j not in graph.neighbors(a))     # 2 opposite-parity, non-adjacent
    pi_swap2 = np.arange(n); pi_swap2[a] = b; pi_swap2[b] = a
    PIS = {"global": pi_global, "swap2": pi_swap2}
    print(f"[{tag}] global pi inverts all parities; swap2 swaps nodes {a}<->{b} (col {col[a]:+.0f}/{col[b]:+.0f})", flush=True)

    model, tok = load_with_fallback(hf, mirror, cfg)
    for p in model.parameters(): p.requires_grad_(False)
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    walks = G.generate_walks(graph, cfg)
    pos_idx = torch.tensor(pos_nodes, device=dev); neg_idx = torch.tensor(neg_nodes, device=dev)

    # ---- clean pass: per-walk ids/node-tokens/readout, per-node mean head output znode ----
    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hcap = proj.register_forward_pre_hook(cap)
    wdata = []; znode_sum = np.zeros((n, hd)); znode_cnt = np.zeros(n)
    with torch.no_grad():
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
            model(input_ids=ids); seqlen = ids.shape[1]
            ntok = []; readpos = []; readnode = []
            for s in range(len(nodes)):
                t = spans[s][-1]; nd = nodes[s]; ntok.append((t, nd))
                if cl[s] >= CTXLO and s < len(nodes) - 1:
                    readpos.append(t); readnode.append(nd)
                    znode_sum[nd] += zc["z"][0, t, csl].float().cpu().numpy(); znode_cnt[nd] += 1
            wdata.append({"ids": ids, "ntok": ntok, "readpos": readpos, "readnode": readnode, "seqlen": seqlen})
    hcap.remove()
    znode = znode_sum / np.maximum(znode_cnt, 1)[:, None]                                 # [n, hd]
    znode_t = torch.tensor(znode, dtype=torch.float32, device=dev)
    print(f"[{tag}] collected {len(walks)} walks, znode ready (hd={hd})", flush=True)

    # normalized-Laplacian eigenmodes for mode correlation
    A = np.zeros((n, n))
    for u in range(n):
        for v in graph.adjacency[u]: A[u, v] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(np.maximum(dg, 1e-12)); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    eigw, eigU = np.linalg.eigh(Lap); parity_mode = int(np.argmax(eigw))
    proto_delta = (znode[pos_nodes].mean(0) - znode[neg_nodes].mean(0))                   # raw parity axis (head space)

    state = {"delta": None, "Rr": None}
    def patch_pre(_m, args):
        if state["delta"] is not None and state["Rr"] is not None:
            x = args[0].clone(); Rr = state["Rr"]
            patch = (state["delta"] @ Rr.t()) @ Rr
            x[0, :, csl] = x[0, :, csl] + patch.to(x.dtype)
            return (x,) + tuple(args[1:])
    ph = proj.register_forward_pre_hook(patch_pre)

    def run_patch(name, pi):
        pi_t = pi
        tgt_sign = np.array([-col[pi[i]] for i in range(n)])                              # predict pi(X)'s neighbour parity
        flipped = np.array([pi[i] != i for i in range(n)])
        for w in wdata:
            D = torch.zeros(w["seqlen"], hd, device=dev)
            for t, nd in w["ntok"]:
                if flipped[nd]: D[t] = znode_t[pi[nd]] - znode_t[nd]
            w["delta"] = D
            rp = [(p, nd) for p, nd in zip(w["readpos"], w["readnode"]) if flipped[nd]]   # only flipped positions
            w["rp_t"] = torch.tensor([p for p, _ in rp], device=dev, dtype=torch.long)
            w["tgt_t"] = torch.tensor([tgt_sign[nd] for _, nd in rp], device=dev)

        def eval_walk(w, Rr):
            if len(w["rp_t"]) == 0: return torch.tensor(0.0, device=dev), None, None
            state["delta"] = w["delta"] if Rr is not None else None; state["Rr"] = Rr
            logits = model(input_ids=w["ids"]).logits[0][w["rp_t"]][:, cand_t].float()
            state["delta"] = None
            lsm = torch.log_softmax(logits, 1); tgt_pos = w["tgt_t"] > 0
            same = torch.where(tgt_pos[:, None], lsm[:, pos_idx], lsm[:, neg_idx])
            opp = torch.where(tgt_pos[:, None], lsm[:, neg_idx], lsm[:, pos_idx])
            sm = torch.logsumexp(same, 1); om = torch.logsumexp(opp, 1); loss = -sm.mean()
            with torch.no_grad():
                am = logits.argmax(1)
                in_same = torch.where(tgt_pos, (am[:, None] == pos_idx).any(1), (am[:, None] == neg_idx).any(1))
                return loss, in_same.float().mean().item(), (sm - om).mean().item()

        def evaluate(Rr):
            fs, ms = [], []
            for w in wdata:
                with torch.no_grad():
                    _, f, m = eval_walk(w, Rr)
                    if f is not None: fs.append(f); ms.append(m)
            return float(np.mean(fs)), float(np.mean(ms))

        res = {}
        wf = [w for w in wdata if len(w["rp_t"]) > 0]
        for r in RDIMS:
            lin = nn.Linear(hd, hd, bias=False).to(dev); nn.utils.parametrizations.orthogonal(lin)
            opt = torch.optim.Adam(lin.parameters(), lr=LR)
            if 0 < r < hd:
                for step in range(STEPS):
                    opt.zero_grad(); bs = [wf[i] for i in rng.choice(len(wf), min(BATCH, len(wf)), replace=False)]
                    loss = sum(eval_walk(w, lin.weight[:r])[0] for w in bs) / len(bs)
                    loss.backward(); opt.step()
            Rr = None if r == 0 else (torch.eye(hd, device=dev) if r >= hd else lin.weight[:r])
            f, m = evaluate(Rr.detach() if Rr is not None else None)
            if 0 < r < hd:
                linr = nn.Linear(hd, hd, bias=False).to(dev); nn.utils.parametrizations.orthogonal(linr)
                fr, mr = evaluate(linr.weight[:r].detach())
                res[r] = {"flip_acc": f, "margin": m, "flip_rand": fr, "margin_rand": mr,
                          "subspace": lin.weight[:r].detach().cpu().numpy()}
            else:
                res[r] = {"flip_acc": f, "margin": m, "flip_rand": f, "margin_rand": m}
            print(f"[{tag}] {name:7} r={r}: flip={f:.3f} margin={m:+.3f}"
                  + (f"  (rand flip={res[r].get('flip_rand',f):.3f})" if 0 < r < hd else ""), flush=True)
        return res

    RES = {name: run_patch(name, pi) for name, pi in PIS.items()}
    ph.remove()

    # ---- compare subspaces + map to parity axis / eigenmodes ----
    def node_scores(direction):
        s = znode @ (direction / (np.linalg.norm(direction) + 1e-12)); return s - s.mean()
    def best_mode(direction):
        sc = node_scores(direction); cs = [abs(np.corrcoef(sc, eigU[:, k])[0, 1]) for k in range(n)]
        return int(np.argmax(cs)), round(float(np.max(cs)), 3), round(float(abs(np.corrcoef(sc, eigU[:, parity_mode])[0, 1])), 3)

    cmp = {}
    for name in PIS:
        if 1 not in RES[name] or "subspace" not in RES[name][1]: continue
        d1 = RES[name][1]["subspace"][0]
        bm, bc, pc = best_mode(d1)
        cmp[name] = {"cos_with_parity_axis": round(cos(d1, proto_delta), 3),
                     "best_eigmode": bm, "best_eigmode_corr": bc, "parity_mode_corr": pc,
                     "parity_mode_idx": parity_mode}
    if all(1 in RES[nm] and "subspace" in RES[nm][1] for nm in PIS):
        gd = RES["global"][1]["subspace"][0]; sd = RES["swap2"][1]["subspace"][0]
        cmp["global_vs_swap2_rank1_cos"] = round(cos(gd, sd), 3)
    if all(2 in RES[nm] and "subspace" in RES[nm][2] for nm in PIS):
        g2 = RES["global"][2]["subspace"]; s2 = RES["swap2"][2]["subspace"]
        sv = np.linalg.svd(g2 @ s2.T, compute_uv=False); cmp["global_vs_swap2_rank2_meancos2"] = round(float((sv ** 2).mean()), 3)

    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "hd": hd, "swap_nodes": [int(a), int(b)],
           "rdims": RDIMS, "parity_mode": parity_mode,
           "results": {nm: {str(r): {k: (v if k != "subspace" else None) for k, v in rv.items()} for r, rv in RES[nm].items()} for nm in PIS},
           "compare": cmp, "eigw": eigw.tolist(), "two_colour": col.tolist()}
    p = f"{OUTDIR}/das_grid_patch_{tag}_L{HEAD_LAYER}H{HEAD_IDX}.json"
    json.dump(out, open(p, "w"), indent=2)
    # subspaces npz
    npz = {}
    for nm in PIS:
        for r in RDIMS:
            if "subspace" in RES[nm][r]: npz[f"{nm}_R{r}"] = RES[nm][r]["subspace"].astype("float32")
    npz["proto_delta"] = proto_delta.astype("float32"); npz["znode"] = znode.astype("float32")
    npz["eigU"] = eigU.astype("float32"); npz["eigw"] = eigw.astype("float32"); npz["two_colour"] = col.astype("float32")
    np.savez_compressed(p.replace(".json", ".npz"), **npz)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    print(f"DONE -> {p}\ncompare = {json.dumps(cmp, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
