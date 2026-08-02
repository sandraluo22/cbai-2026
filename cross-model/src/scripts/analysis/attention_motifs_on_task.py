"""Are the 'neither9' heads really NOT induction heads?

The only evidence so far is that they rank outside the top-100 on the Olsson score — mean attention to
the SUCCESSOR of a prefix match, measured on REPEATED RANDOM TOKENS. That is one operationalisation on an
off-task input. A head doing induction-like retrieval on structured input would score low there and still
be induction. So measure the motifs directly, ON THE WALK TASK, where the distinction is sharp:

at a readout position t sitting on node u, let S = {s < t : node_s = u} (previous visits to u)
    same_token   attention from t to S            — aggregate the history of BEING at u
    induction    attention from t to {s+1 : s in S} — the SUCCESSOR of each previous visit, which on a
                 graph walk is always a NEIGHBOUR of u, so this motif directly predicts a valid next node
    prev_token   attention from t to t-1
    far          attention to everything else (control; sums to 1 with the above)

This is the discriminating measurement: induction and same_token both key on previous occurrences of the
current token, but attend one position apart. Olsson's score can only see the first.

Reports per head, then group means for neither9 / induction12 / all others, with a permutation test.

Env: GEN_MODEL(Llama) GRAPH(ring|grid) K(16) NWALKS(2) WLEN(400) CTXLO(250) NPERM(20000)
     SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/attention_motifs_on_task<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GRAPH = os.environ.get("GRAPH", "ring"); K = int(os.environ.get("K", "16"))
NWALKS = int(os.environ.get("NWALKS", "2")); WLEN = int(os.environ.get("WLEN", "400"))
CTXLO = int(os.environ.get("CTXLO", "250")); NPERM = int(os.environ.get("NPERM", "20000"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

INDUCTION12 = ["L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7",
               "L14H26", "L9H11", "L1H20", "L21H10", "L4H12", "L3H17"]
NEITHER9 = ["L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11", "L4H16", "L1H21", "L2H26"]


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    rng = np.random.default_rng(SEED)

    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

    acc = np.zeros((nL, nH, 4)); nq = 0
    for wk in G.generate_walks(graph, cfg):
        nodes = wk.nodes
        ids = torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev)
        o = model(input_ids=ids, output_attentions=True)
        T = ids.shape[1]
        qs = [t for t in range(CTXLO, len(nodes)) if t >= CTXLO]
        # build index sets per query position (token position = node index + 1 because of BOS)
        same_idx, ind_idx, prev_idx = [], [], []
        for t in qs:
            u = nodes[t]
            S = [s for s in range(t) if nodes[s] == u]
            same_idx.append([s + 1 for s in S])
            ind_idx.append([s + 2 for s in S if s + 1 < t])       # successor of a previous visit
            prev_idx.append([t])                                   # t-1 in node space = token t
        A = torch.stack([o.attentions[l][0] for l in range(nL)])   # [nL, nH, T, T]
        for j, t in enumerate(qs):
            qt = t + 1
            col = A[:, :, qt, :]                                   # [nL, nH, T]
            for c, idxs in enumerate((same_idx[j], ind_idx[j], prev_idx[j])):
                if idxs:
                    acc[:, :, c] += col[:, :, torch.tensor(idxs, device=dev)].sum(-1).float().cpu().numpy()
            nq += 0
        nq += len(qs)
        del o, A
    acc /= max(nq, 1)
    acc[:, :, 3] = 1.0 - acc[:, :, :3].sum(-1)

    names = [f"L{l}H{h}" for l in range(nL) for h in range(nH)]
    flat = acc.reshape(-1, 4)
    lab = ["same_token", "induction", "prev_token", "other"]
    print(f"[{tag}] {GRAPH}{n} walk, {nq} readout positions. Motif attention mass per head.\n")
    print(f"{'head':<9} {'same_tok':>9} {'induction':>10} {'prev_tok':>9} {'other':>8}   group")
    grp = {}
    for g, hs in (("neither9", NEITHER9), ("induction12", INDUCTION12)):
        for h in hs:
            i = names.index(h)
            print(f"{h:<9} {flat[i,0]:9.4f} {flat[i,1]:10.4f} {flat[i,2]:9.4f} {flat[i,3]:8.4f}   {g}")
        grp[g] = [names.index(h) for h in hs]
    rest = [i for i in range(len(names)) if i not in set(grp["neither9"]) | set(grp["induction12"])]
    print(f"\n{'group':<14} {'same_tok':>9} {'induction':>10} {'prev_tok':>9}   perm p (same_token high)")
    out = {"model": tag, "graph": GRAPH, "n": n, "n_readouts": nq,
           "per_head": {names[i]: {lab[c]: round(float(flat[i, c]), 5) for c in range(4)}
                        for i in range(len(names))}, "groups": {}}
    pool = flat[:, 0]
    for g in ("neither9", "induction12"):
        idx = grp[g]; m = flat[idx].mean(0)
        perm = np.array([pool[rng.permutation(len(pool))[:len(idx)]].mean() for _ in range(NPERM)])
        p = float((perm >= m[0]).mean())
        print(f"{g:<14} {m[0]:9.4f} {m[1]:10.4f} {m[2]:9.4f}   p={p:.4f}")
        out["groups"][g] = {lab[c]: round(float(m[c]), 5) for c in range(4)}
        out["groups"][g]["perm_p_same_token"] = round(p, 4)
    m = flat[rest].mean(0)
    print(f"{'all others':<14} {m[0]:9.4f} {m[1]:10.4f} {m[2]:9.4f}")
    out["groups"]["all_others"] = {lab[c]: round(float(m[c]), 5) for c in range(4)}
    # who tops each motif overall?
    for c, nm in ((0, "same_token"), (1, "induction"), (2, "prev_token")):
        top = np.argsort(-flat[:, c])[:8]
        tags = [f"{names[i]}({flat[i,c]:.3f})" + ("*" if names[i] in NEITHER9 else
                ("+" if names[i] in INDUCTION12 else "")) for i in top]
        print(f"\ntop-8 {nm:<11}: " + ", ".join(tags))
        out.setdefault("top8", {})[nm] = [names[i] for i in top]
    print("\n(* = in neither9, + = in induction12)")
    p_ = f"{OUTDIR}/attention_motifs_on_task{OUTTAG}_{tag}.json"
    json.dump(out, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
