"""If you project the GEOMETRY out of the residual, does the task break?

Everything so far tested geometry->task INDIRECTLY: rank-r DAS patching (which asks whether a learned
subspace can IMPLEMENT a remap) and projecting out a ring-derived DAS subspace. Neither is the direct
question. The direct one: the geometry IS the top PCs of the per-node mean representation, so take those
PCs at layer L, project them out of the residual at every position, and measure task accuracy.

  geom_k     project out the top-k PCs of the node-mean cloud (k = 1,2,4,8)
  coord2     project out the 2 directions best aligned with (row, col) specifically — the coordinate
             plane rather than the top-variance plane; these differ (coord_r peaks at L10 while the
             top-2 PCs carry only ~26% of variance)
  random_k   random rank-k subspace = the control that makes any drop meaningful

CRITICAL CONFOUND this is designed to expose: the top PCs of a node-mean cloud carry NODE IDENTITY, not
only its geometric arrangement. Projecting them out can destroy the task simply by deleting which node
we are on. So we also report `concept_decode` — leave-one-out nearest-centroid accuracy for node identity
under the same projection. A drop in task accuracy is only evidence about GEOMETRY if identity survives.

Env: GEN_MODEL(Llama) K(4) GRAPH(grid) LAYERS("10,14,18,24") KS("1,2,4,8") NWALKS(3) WLEN(1200)
     CTXLO(800) NRAND(5) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/project_out_geometry<OUTTAG>_<model>.json
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
K = int(os.environ.get("K", "4")); GRAPH = os.environ.get("GRAPH", "grid")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "10,14,18,24").split(",")]
KS = [int(x) for x in os.environ.get("KS", "1,2,4,8").split(",")]
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K if GRAPH == "grid" else K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "grid", "grid_rows": K, "grid_cols": K} if GRAPH == "grid"
                     else {"graph_type": "ring", "ring_size": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    rng = np.random.default_rng(SEED)

    data = []
    for wk in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [wk.nodes[s] for s in steps]))

    st = {"proj": None, "layer": None}
    hooks = []
    for l in range(cm.num_hidden_layers):
        def mk(l):
            def rh(_m, _i, out):
                if st["proj"] is None or st["layer"] != l: return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); P = st["proj"]; f = h[0].float()
                h[0] = (f - (f @ P.t()) @ P).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return rh
        hooks.append(blocks[l].register_forward_hook(mk(l)))

    def run(L):
        """returns (nbr_validity, concept_decode_acc, node_mean_matrix)"""
        S = torch.zeros(n, cm.hidden_size, device=dev); C = torch.zeros(n, device=dev)
        X, Y, ok, tot = [], [], 0, 0
        for ids, rp, nds in data:
            o = model(input_ids=ids, output_hidden_states=True)
            H = o.hidden_states[L + 1][0, rp].float()
            top = o.logits[0][rp][:, cand_t].float().argmax(1).tolist()
            for i, u in enumerate(nds):
                S[u] += H[i]; C[u] += 1; ok += int(top[i] in graph.adjacency[u]); tot += 1
                X.append(H[i].cpu().numpy()); Y.append(u)
        Mn = (S / C.clamp(min=1)[:, None]); Mn = Mn - Mn.mean(0, keepdim=True)
        X = np.stack(X); Y = np.array(Y)
        sel = rng.permutation(len(X))[:400]                      # subsample for the decode probe
        Xs, Ys = X[sel], Y[sel]
        dok = 0
        for i in range(len(Xs)):
            m = np.ones(len(Xs), bool); m[i] = False
            cen = np.stack([Xs[m & (Ys == c)].mean(0) if (m & (Ys == c)).any()
                            else np.zeros(Xs.shape[1]) for c in range(n)])
            dok += int(int(np.argmin(np.linalg.norm(cen - Xs[i], axis=1))) == Ys[i])
        return ok / tot, dok / len(Xs), Mn.cpu().numpy()

    res = {"model": tag, "graph": GRAPH, "n": n, "layers": {}}
    print(f"{'L':>3} {'proj':<12} {'k':>3} {'nbr_valid':>10} {'concept_dec':>12} {'note'}")
    for L in LAYERS:
        base_nbr, base_dec, Mn = run(L)
        U, S_, Vt = np.linalg.svd(Mn, full_matrices=False)
        # coordinate plane: the 2 right-singular directions whose scores best track (row, col)
        sc = U * S_
        r0 = [abs(np.corrcoef(sc[:, j], coords[:, 0])[0, 1]) for j in range(min(8, sc.shape[1]))]
        r1 = [abs(np.corrcoef(sc[:, j], coords[:, 1])[0, 1]) for j in range(min(8, sc.shape[1]))]
        j0, j1 = int(np.argmax(r0)), int(np.argmax(r1))
        if j1 == j0: j1 = int(np.argsort(r1)[-2])
        row = {"baseline_nbr": round(base_nbr, 4), "baseline_decode": round(base_dec, 4),
               "coord_dirs": [j0, j1], "coord_r": [round(r0[j0], 3), round(r1[j1], 3)], "proj": {}}
        print(f"{L:3} {'(none)':<12} {0:3} {base_nbr:10.4f} {base_dec:12.4f}   "
              f"coord dirs = PC{j0},PC{j1} (r={r0[j0]:.2f},{r1[j1]:.2f})", flush=True)
        for k_ in KS:
            P = torch.tensor(Vt[:k_], dtype=torch.float32, device=dev)
            st["layer"] = L; st["proj"] = P
            nb, dc, _ = run(L); st["proj"] = None; st["layer"] = None
            rs = []
            for _ in range(NRAND):
                Rr = np.linalg.qr(rng.standard_normal((cm.hidden_size, k_)))[0].T.astype(np.float32)
                st["layer"] = L; st["proj"] = torch.tensor(Rr, device=dev)
                a, b, _ = run(L); rs.append((a, b)); st["proj"] = None; st["layer"] = None
            rn, rd = float(np.mean([x[0] for x in rs])), float(np.mean([x[1] for x in rs]))
            row["proj"][f"top{k_}"] = {"nbr": round(nb, 4), "decode": round(dc, 4),
                                       "rand_nbr": round(rn, 4), "rand_decode": round(rd, 4),
                                       "excess_nbr": round(nb - rn, 4)}
            print(f"{L:3} {'top-PCs':<12} {k_:3} {nb:10.4f} {dc:12.4f}   "
                  f"random {rn:.4f}/{rd:.4f}  excess {nb-rn:+.4f}", flush=True)
        Pc = torch.tensor(Vt[[j0, j1]], dtype=torch.float32, device=dev)
        st["layer"] = L; st["proj"] = Pc
        nb, dc, _ = run(L); st["proj"] = None; st["layer"] = None
        row["proj"]["coord2"] = {"nbr": round(nb, 4), "decode": round(dc, 4)}
        print(f"{L:3} {'coord-plane':<12} {2:3} {nb:10.4f} {dc:12.4f}   "
              f"(the 2 coordinate-aligned dirs)", flush=True)
        res["layers"][str(L)] = row
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/project_out_geometry{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
