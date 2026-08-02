"""PERSISTENCE-LAW test via lazy walks. On a lazy random walk (stay put with probability p, else step to
a uniform neighbour) the transition eigenvalue of Laplacian mode λ becomes mu' = (1-p)(1-λ) + p, so the
parity mode's one-step autocorrelation is 1-2p+pλ.. -> for λ=2: mu' = 2p-1 — laziness destroys the
alternation (persistence 0 at p=0.5) WITHOUT changing the graph. If the model allocates representation by
predictive persistence, parity structure should collapse with p while the low-λ coordinate modes survive.
Laziness also DECOUPLES node parity from token-position parity (identical at p=0), so we measure both:
if the "parity feature" is really a temporal even/odd counter, the position-parity separation survives
laziness while node-parity separation dies with graph-expected persistence.

Per p: (a) node-parity separation of L14H26 output (mean-diff axis, per-occurrence AUC), (b) POSITION-
parity separation the same way, (c) parity-mode write power (chance-normalized) and coordinate decode R^2
from node means, (d) behavioural parity hedge: log-odds of predicting opposite- vs same-colour class at
readouts (should track the true transition probability 1-p .. shrink to 0 at p=0.5).

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) K(4) PS(0,0.1,0.25,0.4,0.5) NWALKS(12)
     CTXLO(1000) WLEN(1300) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/lazy_walk_persistence<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
from graph import Walk
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
K = int(os.environ.get("K", "4")); PS = [float(x) for x in os.environ.get("PS", "0,0.1,0.25,0.4,0.5").split(",")]
NWALKS = int(os.environ.get("NWALKS", "12")); CTXLO = int(os.environ.get("CTXLO", "1000"))
WLEN = int(os.environ.get("WLEN", "1300")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def lazy_walks(graph, p, rng):
    walks = []
    for w in range(NWALKS):
        cur = w % graph.n_nodes; nodes = [cur]
        for _ in range(WLEN - 1):
            if rng.random() >= p: cur = int(rng.choice(graph.neighbors(cur)))
            nodes.append(cur)
        walks.append(Walk(walk_id=w, nodes=nodes, words=[graph.words[n] for n in nodes]))
    return walks


def sep_auc(Z, y):
    """mean-diff separation + AUC of the mean-diff axis for binary labels y (+1/-1)."""
    a = Z[y > 0].mean(0) - Z[y < 0].mean(0); a /= (np.linalg.norm(a) + 1e-12)
    s = Z @ a
    sep = float(s[y > 0].mean() - s[y < 0].mean())
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = int((y > 0).sum()); n0 = len(y) - n1
    auc = float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0 + 1e-12))
    return sep, auc, a


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, float)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    A = np.zeros((n, n))
    for u in range(n):
        for v in graph.adjacency[u]: A[u, v] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(dg); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    eigw, eigU = np.linalg.eigh(Lap); parity_mode = int(np.argmax(eigw))

    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hk = proj.register_forward_pre_hook(cap)
    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "k": K, "ctxlo": CTXLO, "wlen": WLEN, "ps": PS, "results": []}
    for p in PS:
        rng = np.random.default_rng(SEED)
        walks = lazy_walks(graph, p, rng)
        Zocc = []; ynode = []; ypos = []; occ_nd = []; zsum = np.zeros((n, hd)); zcnt = np.zeros(n)
        hedge = []
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; zc.clear()
            lg = model(input_ids=ids).logits
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                t = spans[s][-1]; nd = nodes[s]
                z = zc["z"][0, t, csl].float().cpu().numpy()
                Zocc.append(z); ynode.append(col[nd]); ypos.append(1.0 if s % 2 == 0 else -1.0); occ_nd.append(nd)
                zsum[nd] += z; zcnt[nd] += 1
                lsm = torch.log_softmax(lg[0, t, cand_t].float(), 0)
                same = torch.logsumexp(lsm[pos_idx if col[nd] > 0 else neg_idx], 0)
                opp = torch.logsumexp(lsm[neg_idx if col[nd] > 0 else pos_idx], 0)
                hedge.append(float(opp - same))
        Zocc = np.array(Zocc); ynode = np.array(ynode); ypos = np.array(ypos)
        Zc = Zocc - Zocc.mean(0)
        sep_n, auc_n, ax_n = sep_auc(Zc, ynode)
        sep_p, auc_p, ax_p = sep_auc(Zc, ypos)
        znode = zsum / np.maximum(zcnt, 1)[:, None]; Hc = znode - znode.mean(0)
        pw = np.array([np.linalg.norm(Hc.T @ eigU[:, m]) ** 2 for m in range(n)]); pw /= (pw.sum() + 1e-12)
        # held-out coordinate decode from per-occurrence activations (ridge; odd occurrences train, even test)
        Yco = coords[np.array(occ_nd)]
        tr = np.arange(len(Zc)) % 2 == 1; te = ~tr
        lam_r = 10.0
        Wr = np.linalg.solve(Zc[tr].T @ Zc[tr] + lam_r * np.eye(hd), Zc[tr].T @ (Yco[tr] - Yco[tr].mean(0)))
        pred = Zc[te] @ Wr + Yco[tr].mean(0)
        ssr = ((Yco[te] - pred) ** 2).sum(0); sst = ((Yco[te] - Yco[te].mean(0)) ** 2).sum(0)
        r2 = [round(float(1 - ssr[i] / (sst[i] + 1e-12)), 3) for i in range(2)]
        row = {"p": p, "persistence_parity": round(1 - 2 * p, 3),
               "node_parity_sep": round(sep_n, 4), "node_parity_auc": round(auc_n, 4),
               "pos_parity_sep": round(sep_p, 4), "pos_parity_auc": round(auc_p, 4),
               "cos_nodeaxis_posaxis": round(float(abs(ax_n @ ax_p)), 3),
               "parity_mode_power_xn": round(float(pw[parity_mode] * n), 3),
               "coord_decode_r2": r2, "behav_opp_logodds": round(float(np.mean(hedge)), 3),
               "n_occ": int(len(ynode))}
        out["results"].append(row)
        print(f"[p={p}] node-par AUC={auc_n:.3f} pos-par AUC={auc_p:.3f} cos(axes)={row['cos_nodeaxis_posaxis']:.2f} "
              f"par_pow*n={row['parity_mode_power_xn']:.2f} coordR2={r2} hedge={row['behav_opp_logodds']:+.2f}", flush=True)
    hk.remove()
    pth = f"{OUTDIR}/lazy_walk_persistence{OUTTAG}_{tag}.json"
    json.dump(out, open(pth, "w"), indent=2); print(f"DONE -> {pth}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
