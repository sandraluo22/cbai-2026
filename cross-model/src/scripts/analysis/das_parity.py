"""Experiment 5 -- Distributed Alignment Search (DAS) inside a single head, MULTI-POSITION interchange.

Exp 4 localised parity to a few heads (L14H26 top parity-specific). A single-position head swap does NOT
move the readout parity (parity is accumulated over many occurrences, cf. exp 6), so DAS here patches the
head's subspace at EVERY node position of the walk: at a position whose current node has colour c, inject
the OPPOSITE-colour prototype into the aligned subspace only,
    z'_t = z_t + P_r (z̄(-c_t) - z̄(c_t)),   P_r = R[:r]^T R[:r]  (rank-r projector, differentiable in R)
where z̄(±) are the per-colour mean head outputs. If the head's parity lives in an r-dim subspace, this
flips the model's next-node parity (it should now predict the SAME colour as the current node). Train the
orthogonal rotation R (frozen model), sweep r, and compare to r=full-head and an untrained RANDOM subspace.
Metric: parity margin = logsumexp(same-colour logits) - logsumexp(opposite-colour logits), and flip acc.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NWALKS(24) WLEN(260) CTXLO(100)
     RDIMS(1,2,4,8,16,128) STEPS(80) BATCH(3) LR(0.02) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/das_parity_<model>_L<layer>H<head>.json
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
VARIABLE = os.environ.get("VARIABLE", "parity")   # parity (checkerboard) | row (top/bottom) | col (left/right)
NWALKS = int(os.environ.get("NWALKS", "24")); WLEN = int(os.environ.get("WLEN", "260"))
CTXLO = int(os.environ.get("CTXLO", "100"))
RDIMS = [int(x) for x in os.environ.get("RDIMS", "1,2,4,8,16,128").split(",")]
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


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    base = f"{OUTDIR}/das_{VARIABLE}_{tag}_L{HEAD_LAYER}H{HEAD_IDX}"
    ckpt_path = base + ".npz"; collect_path = base + "_collect.pt"; json_path = base + ".json"
    # Persist so re-running doesn't repeat the GPU job: skip entirely if a checkpoint already exists.
    if os.path.exists(ckpt_path) and os.environ.get("FORCE", "0") != "1":
        print(f"[{tag}] DAS checkpoint exists -> {ckpt_path}\n"
              f"        Reuse offline (rotations/znode/eigenmodes/metrics all saved); set FORCE=1 to retrain. Skipping.", flush=True)
        return
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4, n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    if VARIABLE == "parity":
        col = two_colour(graph)                                       # checkerboard (disassortative)
    elif VARIABLE == "row":
        col = np.sign(coords[:, 0] - coords[:, 0].mean())             # top/bottom (assortative coordinate)
    elif VARIABLE == "col":
        col = np.sign(coords[:, 1] - coords[:, 1].mean())             # left/right
    else:
        raise ValueError(f"unknown VARIABLE {VARIABLE}")
    col = col.astype(float)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    # flipped-target orientation: after patching a node toward the opposite prototype, which colour should it
    # predict? = the neighbour-class of the flipped value. Disassortative (parity) -> same as current;
    # assortative (coord half-cut) -> opposite. Captured by one bool.
    def mean_nbr_class(mask):
        vals = [np.mean([col[j] for j in graph.neighbors(i)]) for i in range(n) if mask[i]]
        return float(np.mean(vals)) if vals else 0.0
    SAME_CLASS_TARGET = bool(np.sign(mean_nbr_class(col < 0)) > 0)     # current(+) flips to (-); target sign=sign(m[-])
    print(f"[{tag}] VARIABLE={VARIABLE} same_class_target={SAME_CLASS_TARGET} "
          f"({int((col>0).sum())}+/{int((col<0).sum())}-)", flush=True)

    model, tok = load_with_fallback(hf, mirror, cfg)
    for p in model.parameters(): p.requires_grad_(False)
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    walks = G.generate_walks(graph, cfg)

    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(np.maximum(dg, 1e-12)); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    eigw, eigU = np.linalg.eigh(Lap)                                  # normalized-Laplacian eigenmodes (match gma/index)

    # ---- clean pass (cached): per-walk ids + node/readout positions, per-colour prototypes, per-node head means.
    #      Reused across retrains via the collect cache so only the trained rotation must be recomputed. ----
    if os.path.exists(collect_path) and os.environ.get("RECOLLECT", "0") != "1":
        C = torch.load(collect_path, weights_only=False)
        wdata = [{"ids": ids.to(dev), "ntok": nt, "readpos": rp, "readcol": rc, "seqlen": int(ids.shape[1])}
                 for ids, nt, rp, rc in zip(C["ids"], C["ntok"], C["readpos"], C["readcol"])]
        proto = {c: torch.tensor(C["proto"][c], dtype=torch.float32, device=dev) for c in (1.0, -1.0)}
        znode = C["znode"]
        print(f"[{tag}] loaded collection cache {collect_path} ({len(wdata)} walks)", flush=True)
    else:
        zc = {}
        def cap(_m, args): zc["z"] = args[0].detach()
        hcap = proj.register_forward_pre_hook(cap)
        wdata = []; zsum = {1.0: np.zeros(hd), -1.0: np.zeros(hd)}; zcnt = {1.0: 0, -1.0: 0}
        znode_sum = np.zeros((n, hd)); znode_cnt = np.zeros(n)         # per-node mean head output (for eigenmode corr)
        with torch.no_grad():
            for wk in walks:
                ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
                spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
                model(input_ids=ids); seqlen = ids.shape[1]
                ntok = []; readpos = []; readcol = []
                for s in range(len(nodes)):
                    c = float(col[nodes[s]]); t = spans[s][-1]
                    ntok.append((t, c))                                # patch every node's readout token
                    if cl[s] >= CTXLO and s < len(nodes) - 1:
                        readpos.append(t); readcol.append(c)
                        zrow = zc["z"][0, t, csl].float().cpu().numpy()
                        zsum[c] += zrow; zcnt[c] += 1
                        znode_sum[nodes[s]] += zrow; znode_cnt[nodes[s]] += 1
                wdata.append({"ids": ids, "ntok": ntok, "readpos": readpos, "readcol": readcol, "seqlen": seqlen})
        hcap.remove()
        proto = {c: torch.tensor(zsum[c] / max(zcnt[c], 1), dtype=torch.float32, device=dev) for c in (1.0, -1.0)}
        znode = znode_sum / np.maximum(znode_cnt, 1)[:, None]          # [n, hd] per-node head output
        torch.save({"ids": [w["ids"].cpu() for w in wdata], "ntok": [w["ntok"] for w in wdata],
                    "readpos": [w["readpos"] for w in wdata], "readcol": [w["readcol"] for w in wdata],
                    "proto": {c: proto[c].cpu().numpy() for c in (1.0, -1.0)}, "znode": znode}, collect_path)
        print(f"[{tag}] L{HEAD_LAYER}H{HEAD_IDX} hd={hd} | {len(walks)} walks, prototypes from {zcnt[1.0]}+/{zcnt[-1.0]}- occ "
              f"(collection cached -> {collect_path})", flush=True)

    # per-walk Delta [seqlen, hd]: opposite-minus-same colour prototype at each node token
    for w in wdata:
        D = torch.zeros(w["seqlen"], hd, device=dev)
        for t, c in w["ntok"]:
            D[t] = proto[-c] - proto[c]
        w["delta"] = D
        w["readpos_t"] = torch.tensor(w["readpos"], device=dev, dtype=torch.long)
        w["readcol_t"] = torch.tensor(w["readcol"], device=dev)

    state = {"delta": None, "Rr": None}
    def patch_pre(_m, args):
        if state["delta"] is not None and state["Rr"] is not None:
            x = args[0].clone()
            Rr = state["Rr"]
            patch = (state["delta"] @ Rr.t()) @ Rr                     # Delta @ P_r  (P_r = R[:r]^T R[:r])
            x[0, :, csl] = x[0, :, csl] + patch.to(x.dtype)
            return (x,) + tuple(args[1:])
    ph = proj.register_forward_pre_hook(patch_pre)

    def eval_walk(w, Rr):
        state["delta"] = w["delta"] if Rr is not None else None; state["Rr"] = Rr   # Rr=None -> no patch (r=0)
        logits = model(input_ids=w["ids"]).logits[0][w["readpos_t"]][:, cand_t].float()   # [P, n]
        state["delta"] = None
        lsm = torch.log_softmax(logits, 1)
        rc = w["readcol_t"]                                                               # current node value (+/-)
        cur_pos = rc > 0                                                                  # [P]
        tgt_pos = cur_pos if SAME_CLASS_TARGET else ~cur_pos                              # flipped-target class per sample
        same = torch.where(tgt_pos[:, None], lsm[:, pos_idx], lsm[:, neg_idx])            # "same" = flipped target class
        opp = torch.where(tgt_pos[:, None], lsm[:, neg_idx], lsm[:, pos_idx])
        same_m = torch.logsumexp(same, 1); opp_m = torch.logsumexp(opp, 1)
        loss = -same_m.mean()
        with torch.no_grad():
            am = logits.argmax(1)
            in_same = torch.where(tgt_pos, (am[:, None] == pos_idx).any(1), (am[:, None] == neg_idx).any(1))
            flip = in_same.float().mean().item(); margin = (same_m - opp_m).mean().item()
        return loss, flip, margin

    def evaluate(Rr):
        fs, ms = [], []
        for w in wdata:
            with torch.no_grad():
                _, f, m = eval_walk(w, Rr); fs.append(f); ms.append(m)
        return float(np.mean(fs)), float(np.mean(ms))

    def Rr_of(lin, r): return None if r == 0 else (torch.eye(hd, device=dev) if r >= hd else lin.weight[:r])

    results = {}
    for r in RDIMS:
        lin = nn.Linear(hd, hd, bias=False).to(dev); nn.utils.parametrizations.orthogonal(lin)
        opt = torch.optim.Adam(lin.parameters(), lr=LR)
        if 0 < r < hd:
            for step in range(STEPS):
                opt.zero_grad(); batch = [wdata[i] for i in rng.choice(len(wdata), min(BATCH, len(wdata)), replace=False)]
                loss = sum(eval_walk(w, lin.weight[:r])[0] for w in batch) / len(batch)
                loss.backward(); opt.step()
                if step % 20 == 0:
                    f, m = evaluate(lin.weight[:r].detach())
                    print(f"  r={r:<3} step {step:3d} loss={loss.item():.3f} flip={f:.3f} margin={m:+.3f}", flush=True)
        f, m = evaluate(Rr_of(lin, r).detach() if Rr_of(lin, r) is not None else None)
        if 0 < r < hd:
            linr = nn.Linear(hd, hd, bias=False).to(dev); nn.utils.parametrizations.orthogonal(linr)
            f_rand, m_rand = evaluate(linr.weight[:r].detach())
        else:
            f_rand, m_rand = f, m
        results[r] = {"flip_acc": f, "margin": m, "flip_acc_random_subspace": f_rand, "margin_random_subspace": m_rand}
        if 0 < r < hd:
            results[r]["subspace"] = lin.weight[:r].detach().cpu().numpy().tolist()   # learned aligned basis [r, hd]
        print(f"[{tag}] r={r}: flip={f:.3f} margin={m:+.3f} (random r-subspace flip={f_rand:.3f} margin={m_rand:+.3f})", flush=True)
    ph.remove()

    proto_delta = (proto[1.0] - proto[-1.0]).cpu().numpy()            # raw parity axis in head-output space
    out = {"model": tag, "variable": VARIABLE, "head": [HEAD_LAYER, HEAD_IDX], "hd": hd, "n_walks": len(walks),
           "steps": STEPS, "batch": BATCH, "results": {str(k): v for k, v in results.items()},
           "znode": znode.tolist(), "eigU": eigU.tolist(), "eigw": eigw.tolist(),
           "two_colour": col.tolist(), "proto_delta": proto_delta.tolist()}
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open(json_path, "w"), indent=2); print(f"DONE -> {json_path}", flush=True)
    # compact reusable checkpoint: rotations + arrays + metrics as float32 arrays (analysis needs no GPU/model)
    ck = {"head": np.array([HEAD_LAYER, HEAD_IDX]), "hd": hd, "znode": znode.astype("float32"),
          "eigU": eigU.astype("float32"), "eigw": eigw.astype("float32"), "two_colour": col.astype("float32"),
          "proto_delta": proto_delta.astype("float32"),
          "metrics": json.dumps({str(k): {kk: vv for kk, vv in v.items() if kk != "subspace"} for k, v in results.items()})}
    for k, v in results.items():
        if "subspace" in v: ck[f"R_{k}"] = np.array(v["subspace"], dtype="float32")   # learned rotation basis [r, hd]
    np.savez_compressed(ckpt_path, **ck); print(f"CHECKPOINT -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
