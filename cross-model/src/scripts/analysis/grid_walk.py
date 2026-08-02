"""Random walk on k x k GRIDS of varying size, capturing the SAME data as torus_walk.py so the interactive
viewer can show a PCA view and a model-eigenmode-projection view with a single grid-size lever. Per grid size
k we store: per-layer PCA embedding of node-means (+ a reservoir cloud of individual occurrences), per-layer
model eigenmode projections (node score along Hc^T u_k, sign-aligned) for the top-firing modes, coords and
parity for colour-coding, and the eigenmode power spectrum.

Env: GEN_MODEL(Llama) SIZES(3,4,..,16) NWALKS(30) SAMPLES_PER_NODE(60) WLEN_CAP(900) CTXLO(200) NTOP(6)
     CLOUD_N(120) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/grid_walk_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans
from grid_parity_compare import build_word_pool

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
SIZES = [int(x) for x in os.environ.get("SIZES", "3,4,5,6,7,8,9,10,11,12,13,14,15,16").split(",")]
NWALKS = int(os.environ.get("NWALKS", "30")); SPN = int(os.environ.get("SAMPLES_PER_NODE", "60"))
WLEN_CAP = int(os.environ.get("WLEN_CAP", "900")); CTXLO = int(os.environ.get("CTXLO", "200"))
NTOP = int(os.environ.get("NTOP", "6")); CLOUD_N = int(os.environ.get("CLOUD_N", "120")); SEED = int(os.environ.get("SEED", "0"))
ALLMODES = os.environ.get("ALLMODES", "0") == "1"   # also store int8/base64 node-mean projections for ALL n modes
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


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


def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def rdm(H): iu = np.triu_indices(H.shape[0], 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def freq_label(u, coords, R, C):
    """dominant 2D-DCT spatial frequency (p down rows, q across cols) of a grid eigenvector."""
    r, c = coords[:, 0], coords[:, 1]; best = (0, 0, -1.0)
    for p in range(R):
        for q in range(C):
            if p == 0 and q == 0: continue
            v = np.cos(np.pi * p * (r + 0.5) / R) * np.cos(np.pi * q * (c + 0.5) / C)
            if v.std() < 1e-9: continue
            cc = abs(np.corrcoef(u, v)[0, 1])
            if cc > best[2]: best = (p, q, cc)
    return best[0], best[1]


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, n, dev, cloud_n, rng):
    nL = cm.num_hidden_layers; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    res_rows = []; res_node = []; seen = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]
            allrows = {L: grabbed[L][0][single].float().cpu().numpy() for L in range(nL)}
            for s in range(len(nodes)):
                if cl[s] < CTXLO: continue
                for L in range(nL): nsum[L][nodes[s]] += allrows[L][s]
                ncnt[nodes[s]] += 1; seen += 1
                stacked = np.stack([allrows[L][s] for L in range(nL)]).astype(np.float32)
                if len(res_rows) < cloud_n: res_rows.append(stacked); res_node.append(nodes[s])
                else:
                    j = int(rng.integers(seen))
                    if j < cloud_n: res_rows[j] = stacked; res_node[j] = nodes[s]
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1); means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    cloud = np.array(res_rows) if res_rows else np.zeros((0, nL, cm.hidden_size), np.float32)
    return means, ncnt, cloud, np.array(res_node)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = load_with_fallback(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    import config as _config
    need = max(k * k for k in SIZES)
    if need > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, need)
    out = {"model": tag, "ntop": NTOP, "sizes": SIZES, "grids": {}}
    for k in SIZES:
        n = k * k
        wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
        cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=k, grid_cols=k, n_walks=NWALKS, walk_length=wl, seed=SEED, device=dev)
        graph = G.build_graph(cfg); coords = np.array(graph.coords); col = two_colour(graph)
        walks = G.generate_walks(graph, cfg)
        iu = np.triu_indices(n, 1)
        D_grid = np.array([[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in coords] for a in coords])[iu]
        A = np.zeros((n, n))
        for a in range(n):
            for b in graph.adjacency[a]: A[a, b] = 1.0
        dg = A.sum(1); di = 1 / np.sqrt(dg); Lap = np.eye(n) - di[:, None] * A * di[None, :]
        eigw, eigU = np.linalg.eigh(Lap)
        means, ncnt, cloud, cloud_node = node_means(model, tok, blocks, cm, walks, n, dev, CLOUD_N, np.random.default_rng(SEED))
        per = []; layer_embed = []; layer_cloud = []
        for Lyr in range(nL):
            H = means[Lyr]; mu = H.mean(0); Hc = H - mu; Rr = rdm(Hc)
            U, S, Vt = np.linalg.svd(Hc, full_matrices=False); emb = U[:, :3] * S[:3]
            clp = (cloud[:, Lyr, :] - mu) @ Vt[:3].T if len(cloud) else np.zeros((0, 3))
            per.append({"layer": Lyr, "rsa_grid": round(sp(Rr, D_grid), 3),
                        "pc2var": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3)})
            layer_embed.append([[round(v, 3) for v in r] for r in emb])
            layer_cloud.append([[round(v, 2) for v in r] for r in clp])
        Lstar = max(range(nL), key=lambda l: per[l]["rsa_grid"])
        Hs = means[Lstar] - means[Lstar].mean(0); pS = (eigU.T @ Hs) ** 2; pS = pS.sum(1); pS[0] = 0; pS = pS / (pS.sum() + 1e-12)
        idx = [int(kk) for kk in np.argsort(pS[1:])[::-1][:NTOP] + 1]
        freqs = [freq_label(eigU[:, kk], coords, k, k) for kk in idx]
        parity_mode = int(np.argmax(eigw))
        eig_proj = []; cloud_eig = []
        for Lyr in range(nL):
            mu = means[Lyr].mean(0); Hc = means[Lyr] - mu; row = []; crow = []
            clr = (cloud[:, Lyr, :] - mu) if len(cloud) else np.zeros((0, cm.hidden_size))
            for kk in idx:
                u = eigU[:, kk]; chat = Hc.T @ u; dirn = chat / (np.linalg.norm(chat) + 1e-9); mc = Hc @ dirn
                sign = -1.0 if np.corrcoef(mc, u)[0, 1] < 0 else 1.0
                row.append([round(float(x), 3) for x in mc * sign]); crow.append([round(float(x), 2) for x in (clr @ dirn) * sign])
            eig_proj.append(row); cloud_eig.append(crow)
        out["grids"][f"k{k}"] = {"k": k, "n": n, "best_layer": Lstar, "n_layers": nL, "rsa_grid": per[Lstar]["rsa_grid"],
                                 "coords": coords.tolist(), "parity": [int(x) for x in col], "cloud_node": cloud_node.tolist(),
                                 "layer_embed": layer_embed, "layer_cloud": layer_cloud, "per_layer": per,
                                 "eig_idx": idx, "eig_freq": [[int(a), int(bb)] for a, bb in freqs],
                                 "eig_lambda": [round(float(eigw[kk]), 3) for kk in idx],
                                 "eig_power": [round(float(pS[kk]), 4) for kk in idx],
                                 "eig_is_parity": [int(kk == parity_mode) for kk in idx],
                                 "eig_proj": eig_proj, "cloud_eig": cloud_eig,
                                 "eig_power_by_mode": [round(float(x), 4) for x in pS[:min(n, 30)]],
                                 "eig_lambdas": [round(float(x), 3) for x in eigw[:min(n, 30)]]}
        if ALLMODES:
            # per-layer projections onto EVERY Laplacian mode, int8-quantized (layer-major [L][mode][node])
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
            gd = out["grids"][f"k{k}"]
            gd["eig_all_b64"] = base64.b64encode(Q.tobytes()).decode()
            gd["eig_all_scale"] = [[round(float(s), 4) for s in row] for row in scale]
            gd["eig_freq_all"] = [[0, 0]] + [[int(a), int(bb)] for a, bb in
                                             (freq_label(eigU[:, m], coords, k, k) for m in range(1, n))]
            gd["eig_lambda_all"] = [round(float(x), 3) for x in eigw]
            gd["eig_power_all"] = [round(float(x), 4) for x in pS]
        print(f"[{tag}] k{k} n={n}: L*={Lstar} rsa_grid={per[Lstar]['rsa_grid']:.2f} wl={wl} top_freqs={freqs} min_occ={ncnt.min():.0f}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/grid_walk{os.environ.get('OUTTAG', '')}_{tag}.json"
    json.dump(out, open(p, "w")); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
