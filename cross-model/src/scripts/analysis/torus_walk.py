"""Random walk on a TORUS C_w x C_L (both axes periodic), ctx~1000. For each (w,L) we capture per-node
residual means and ask what structure the model uses:
  - RSA of rep distances vs TORUS (both wrap) vs CYLINDER (width wraps, length open) vs GRID (no wrap):
    the wrap-index = RSA(torus) - RSA(grid) tells us whether the model closes the loops.
  - PCA: top principal components + variance explained + 2D embedding (does it show a torus/two circles?).
  - EIGENMODE decomposition: normalized-Laplacian (2D Fourier) power spectrum; is power on the periodic
    fundamentals of BOTH cycles? best-4D fit to the torus's natural 4D embedding (cosθ,sinθ,cosφ,sinφ).
Sweep: width w in {4,5,6} (minor circle) x length L in {6,10,16} (major circle).

Env: GEN_MODEL(Llama) WIDTHS(4,5,6) LENGTHS(6,10,16) WLEN(1000) CTXLO(300) NWALKS(40) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/torus_walk_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import dataclass, replace
from typing import List
import numpy as np
import torch

from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
# two families as explicit (w,L) pairs: A = vary L at fixed w; B = vary w at fixed L. Default = their union.
FIX_W = int(os.environ.get("FIX_W", "5")); FIX_L = int(os.environ.get("FIX_L", "8"))
VARY_L = [int(x) for x in os.environ.get("VARY_L", "4,5,6,7,8,10,12,14,16").split(",")]
VARY_W = [int(x) for x in os.environ.get("VARY_W", "3,4,5,6,7,8").split(",")]
if os.environ.get("COMBOS"):
    COMBOS = [tuple(int(y) for y in c.lower().split("x")) for c in os.environ["COMBOS"].split(",")]  # "wxL"
else:
    COMBOS = sorted(set([(FIX_W, L) for L in VARY_L] + [(w, FIX_L) for w in VARY_W]))
NTOP = int(os.environ.get("NTOP", "6"))                     # number of top-firing eigenmodes to store per combo
WLEN = int(os.environ.get("WLEN", "1000")); CTXLO = int(os.environ.get("CTXLO", "300"))
NWALKS = int(os.environ.get("NWALKS", "40")); SEED = int(os.environ.get("SEED", "0"))
ALLMODES = os.environ.get("ALLMODES", "0") == "1"   # also store int8/base64 node-mean projections for ALL n modes
CLOUD_N = int(os.environ.get("CLOUD_N", "140"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/7_torus")

WORDS = ("apple bird sand math chair river music glass cloud knife table house tree stone water fire book "
         "phone door window clock plant coffee bread cheese wine garden mountain ocean forest desert island "
         "bridge tower castle engine wheel rope hammer nail brush paint paper pencil camera mirror candle lamp "
         "pillow blanket carpet basket bottle spoon fork plate bowl kettle oven fridge shelf drawer ladder "
         "fence gate roof wall floor tunnel cave valley hill meadow pond stream glacier volcano canyon cliff "
         "beach harbor anchor saddle feather marble copper velvet ribbon button pocket collar helmet shield "
         "arrow torch barrel wagon").split()


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def build_torus(w, L):
    n = w * L
    def nid(i, j): return i * w + j
    adj = [[] for _ in range(n)]
    for i in range(L):
        for j in range(w):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                adj[nid(i, j)].append(nid((i + di) % L, (j + dj) % w))
    adjacency = [sorted(set(a)) for a in adj]
    coords = [(i, j) for i in range(L) for j in range(w)]                 # (length, width)
    return Graph(n_nodes=n, words=WORDS[:n], adjacency=adjacency, coords=coords)


def cyc(a, b, n): d = abs(a - b) % n; return min(d, n - d)


def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def rdm(H): iu = np.triu_indices(H.shape[0], 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def best_fit_rsa(H, C, target, k=8):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    kk = min(k, Vt.shape[0]); Z = U[:, :kk] * S[:kk]
    W = np.linalg.lstsq(Z, C - C.mean(0), rcond=None)[0]
    return sp(rdm(Z @ W), target)


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, n, dev, cloud_n=140, rng=None):
    """per-layer node means + a reservoir-sampled cloud of individual occurrences (full per-layer residuals,
    so each occurrence can be projected onto every layer's node-mean PCs, like the grid viewer)."""
    nL = cm.num_hidden_layers; grabbed = {}; rng = rng or np.random.default_rng(0)
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    res_rows = []; res_node = []; seen = 0                                     # reservoir of occurrences
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]
            allrows = {L: grabbed[L][0][single].float().cpu().numpy() for L in range(nL)}   # [S,d] per layer
            for s in range(len(nodes)):
                if cl[s] < CTXLO: continue
                for L in range(nL): nsum[L][nodes[s]] += allrows[L][s]
                ncnt[nodes[s]] += 1; seen += 1
                stacked = np.stack([allrows[L][s] for L in range(nL)]).astype(np.float32)   # [nL,d]
                if len(res_rows) < cloud_n: res_rows.append(stacked); res_node.append(nodes[s])
                else:
                    j = int(rng.integers(seen))
                    if j < cloud_n: res_rows[j] = stacked; res_node[j] = nodes[s]
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    cloud = np.array(res_rows) if res_rows else np.zeros((0, nL, cm.hidden_size), np.float32)   # [cloud_n,nL,d]
    return means, ncnt, cloud, np.array(res_node)


def freq_label(u, coords, L, w):
    """dominant torus Fourier frequency (a around length, b around width) of a graph eigenvector."""
    i, j = coords[:, 0], coords[:, 1]; best = (0, 0, -1.0)
    for a in range(L // 2 + 1):
        for b in range(w // 2 + 1):
            if a == 0 and b == 0: continue
            for pa in (np.cos(2 * np.pi * a * i / L), np.sin(2 * np.pi * a * i / L)):
                for pb in (np.cos(2 * np.pi * b * j / w), np.sin(2 * np.pi * b * j / w)):
                    v = pa * pb
                    if v.std() < 1e-9: continue
                    c = abs(np.corrcoef(u, v)[0, 1])
                    if c > best[2]: best = (a, b, c)
    return best[0], best[1]


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = load_with_fallback(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    out = {"model": tag, "wlen": WLEN, "ctxlo": CTXLO, "fix_w": FIX_W, "fix_l": FIX_L, "ntop": NTOP, "combos": {}}

    for (w, L) in COMBOS:
        if True:
            n = w * L
            if n > len(WORDS):
                print(f"[skip] w{w}xL{L} needs {n} words > {len(WORDS)}", flush=True); continue
            graph = build_torus(w, L)
            cfg = replace(get_config("gemma_qwen"), n_walks=NWALKS, walk_length=WLEN, seed=SEED, device=dev)
            walks = G.generate_walks(graph, cfg)
            coords = np.array(graph.coords)                                  # (i=length, j=width)
            iu = np.triu_indices(n, 1)
            D_torus = np.array([[cyc(a[0], b[0], L) + cyc(a[1], b[1], w) for b in coords] for a in coords])[iu]
            D_cyl = np.array([[abs(a[0] - b[0]) + cyc(a[1], b[1], w) for b in coords] for a in coords])[iu]
            D_grid = np.array([[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in coords] for a in coords])[iu]
            C4 = np.column_stack([np.cos(2 * np.pi * coords[:, 0] / L), np.sin(2 * np.pi * coords[:, 0] / L),
                                  np.cos(2 * np.pi * coords[:, 1] / w), np.sin(2 * np.pi * coords[:, 1] / w)])
            A = np.zeros((n, n))
            for a in range(n):
                for b in graph.adjacency[a]: A[a, b] = 1.0
            dg = A.sum(1); di = 1 / np.sqrt(dg); Lap = np.eye(n) - di[:, None] * A * di[None, :]
            eigw, eigU = np.linalg.eigh(Lap)

            means, ncnt, cloud, cloud_node = node_means(model, tok, blocks, cm, walks, n, dev, CLOUD_N, np.random.default_rng(SEED))
            per = []; layer_embed = []; layer_cloud = []
            for Lyr in range(nL):
                H = means[Lyr]; mu = H.mean(0); Hc = H - mu; R = rdm(Hc)
                U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
                emb = U[:, :3] * S[:3]
                clp = (cloud[:, Lyr, :] - mu) @ Vt[:3].T if len(cloud) else np.zeros((0, 3))
                per.append({"layer": Lyr, "rsa_torus": round(sp(R, D_torus), 3), "rsa_cyl": round(sp(R, D_cyl), 3),
                            "rsa_grid": round(sp(R, D_grid), 3), "best4d": round(best_fit_rsa(H, C4, D_torus), 3),
                            "pc2var": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3)})
                layer_embed.append([[round(v, 3) for v in r] for r in emb])
                layer_cloud.append([[round(v, 2) for v in r] for r in clp])
            Lstar = max(range(nL), key=lambda l: per[l]["rsa_torus"]); b = per[Lstar]
            Hs = means[Lstar] - means[Lstar].mean(0); pS = (eigU.T @ Hs) ** 2; pS = pS.sum(1); pS[0] = 0; pS = pS / (pS.sum() + 1e-12)
            idx = [int(k) for k in np.argsort(pS[1:])[::-1][:NTOP] + 1]           # top-NTOP firing modes at best layer
            freqs = [freq_label(eigU[:, k], coords, L, w) for k in idx]
            # MODEL eigenmode projection per layer: node's score along the model's mode-k representational
            # direction (chat_k = Hc^T u_k), sign-aligned to the graph eigenmode u_k. Not the ground truth.
            eig_proj = []; cloud_eig = []                              # node-mean AND per-occurrence mode projections
            for Lyr in range(nL):
                mu = means[Lyr].mean(0); Hc = means[Lyr] - mu; row = []; crow = []
                clr = (cloud[:, Lyr, :] - mu) if len(cloud) else np.zeros((0, cm.hidden_size))
                for k in idx:
                    u = eigU[:, k]; chat = Hc.T @ u; dirn = chat / (np.linalg.norm(chat) + 1e-9)
                    mc = Hc @ dirn
                    sign = -1.0 if np.corrcoef(mc, u)[0, 1] < 0 else 1.0
                    row.append([round(float(x), 3) for x in mc * sign])
                    crow.append([round(float(x), 2) for x in (clr @ dirn) * sign])
                eig_proj.append(row); cloud_eig.append(crow)
            key = f"w{w}_L{L}"
            out["combos"][key] = {"w": w, "L": L, "n": n, "best_layer": Lstar, "n_layers": nL,
                                  "rsa_torus": b["rsa_torus"], "rsa_cyl": b["rsa_cyl"], "rsa_grid": b["rsa_grid"],
                                  "best4d": b["best4d"], "wrap_index": round(b["rsa_torus"] - b["rsa_grid"], 3),
                                  "coords": coords.tolist(), "cloud_node": cloud_node.tolist(),
                                  "layer_embed": layer_embed, "layer_cloud": layer_cloud, "per_layer": per,
                                  "eig_idx": idx, "eig_freq": [[int(a), int(bb)] for a, bb in freqs],
                                  "eig_lambda": [round(float(eigw[k]), 3) for k in idx],
                                  "eig_power": [round(float(pS[k]), 4) for k in idx],
                                  "eig_gt": [[round(float(eigU[m, k]), 4) for m in range(n)] for k in idx],
                                  "eig_proj": eig_proj, "cloud_eig": cloud_eig,
                                  "eig_power_by_mode": [round(float(x), 4) for x in pS[:min(n, 24)]],
                                  "eig_lambdas": [round(float(x), 3) for x in eigw[:min(n, 24)]]}
            if ALLMODES:
                import base64
                P = np.zeros((nL, n, n), np.float32)
                for Lyr in range(nL):
                    Hc = means[Lyr] - means[Lyr].mean(0)
                    chat = Hc.T @ eigU; Dirs = chat / (np.linalg.norm(chat, axis=0) + 1e-9)
                    MC = Hc @ Dirs
                    signs = np.sign(np.einsum("nm,nm->m", MC, eigU)); signs[signs == 0] = 1
                    P[Lyr] = (MC * signs).T
                scale = np.abs(P).max(axis=2) + 1e-9
                Q = np.clip(np.round(P / scale[:, :, None] * 127), -127, 127).astype(np.int8)
                gd = out["combos"][key]
                gd["eig_all_b64"] = base64.b64encode(Q.tobytes()).decode()
                gd["eig_all_scale"] = [[round(float(s), 4) for s in row] for row in scale]
                gd["eig_freq_all"] = [[0, 0]] + [[int(a), int(bb)] for a, bb in
                                                 (freq_label(eigU[:, m], coords, L, w) for m in range(1, n))]
                gd["eig_lambda_all"] = [round(float(x), 3) for x in eigw]
                gd["eig_power_all"] = [round(float(x), 4) for x in pS]
                gd["eig_gt_all"] = [[round(float(eigU[m, k2]), 3) for m in range(n)] for k2 in range(n)]
            print(f"[{tag}] {key} n={n}: L*={Lstar} torus={b['rsa_torus']:.2f} grid={b['rsa_grid']:.2f} "
                  f"wrap={out['combos'][key]['wrap_index']:+.2f} freqs={freqs} min_occ={ncnt.min():.0f}", flush=True)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/torus_walk{os.environ.get('OUTTAG', '')}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
