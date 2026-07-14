"""Per-EIGENVECTOR single-head ablation: is each Laplacian eigenvector controlled by a different set
of heads? For a graph family, ablate every head one at a time, recompute teacher-forced node-means,
and measure the damage to EACH non-trivial Laplacian eigenmode's power (at a common readout layer).
Then the correlation between eigenmodes' damage maps says whether they share circuitry or not.

Env: PRESET GEN_MODEL(Llama) FAM(grid) N(16) SEED(0) NWALKS(16) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/head_eig_sweep_<model>_<fam>.json + .pdf
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "distilgpt2": ("distilgpt2", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
FAM = os.environ.get("FAM", "grid"); N = int(os.environ.get("N", "16")); SEED = int(os.environ.get("SEED", "0"))
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/head_eig_sweep")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def _finish(edges, n):
    A = np.zeros((n, n))
    for a, b in edges: A[a, b] = A[b, a] = 1
    return [sorted(np.where(A[i] > 0)[0].tolist()) for i in range(n)], A


def _tree(rng, n):
    perm = rng.permutation(n); e = set()
    for i in range(1, n):
        j = perm[rng.integers(0, i)]; a, b = int(perm[i]), int(j); e.add((min(a, b), max(a, b)))
    return e


def build_family(name, n, seed):
    rng = np.random.default_rng(seed)
    if name == "grid":
        r = c = int(round(n ** 0.5)); e = set()
        for i in range(r):
            for j in range(c):
                u = i * c + j
                if j + 1 < c: e.add((u, u + 1))
                if i + 1 < r: e.add((u, u + c))
        return _finish(e, n)
    if name == "ring":
        return _finish({(i, (i + 1) % n) if i + 1 < n else (0, n - 1) for i in range(n)}, n)
    if name == "tree":
        return _finish(_tree(rng, n), n)
    if name == "er_random":
        e = _tree(rng, n); tgt = n * 4 // 2
        while len(e) < tgt:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b: e.add((min(a, b), max(a, b)))
        return _finish(e, n)
    if name in ("sbm2", "sbm4"):
        k = 2 if name == "sbm2" else 4; block = np.repeat(np.arange(k), n // k)
        e = _tree(rng, n)
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < (0.75 if block[a] == block[b] else 0.04): e.add((a, b))
        return _finish(e, n)
    raise ValueError(name)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def ablate_head_hook(block, cm, dev, h):
    proj, hd = attn_proj(block, cm)
    ct = torch.arange(h * hd, (h + 1) * hd, device=dev, dtype=torch.long)
    def pre(_m, args, ct=ct):
        x = args[0].clone(); x[..., ct] = 0
        return (x,) + tuple(args[1:])
    return proj.register_forward_pre_hook(pre)


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, dev, n, layers):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in layers}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1); first = layers[0]
            for L in layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == first: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in layers}


def eig_power(H, V):
    Hc = H - H.mean(0); c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0.0
    return p / (p.sum() + 1e-12)                        # power per eigenmode


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    adj, A = build_family(FAM, N, SEED)
    Lap = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(Lap)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=N, n_walks=NWALKS, walk_length=WLEN, device=dev)
    words = cfg.words()[:N]
    graph = Graph(n_nodes=N, words=words, adjacency=adj, coords=[tuple(c) for c in V[:, 1:3]])
    print(f"[{tag}/{FAM}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    walks = G.generate_walks(graph, cfg)

    meansL = node_means(model, tok, blocks, cm, walks, dev, N, list(range(nL)))
    powL = np.stack([eig_power(meansL[L], V) for L in range(nL)])     # (nL, N)
    Lstar = int(powL[:, 1:].max(1).argmax())                          # layer with the single most-dominant eigenmode (concentration; sum-based pick was degenerate)
    clean = powL[Lstar]                                              # (N,)
    print(f"[{tag}/{FAM}] readout L*={Lstar}  clean top modes: " +
          ", ".join(f"m{k}(λ{w[k]:.1f}):{clean[k]:.2f}" for k in np.argsort(clean)[::-1][:4]), flush=True)

    nE = N - 1                                                        # non-trivial modes 1..N-1
    damage = np.zeros((nE, nL, nH))
    for L in range(nL):
        for h in range(nH):
            hd = ablate_head_hook(blocks[L], cm, dev, h)
            try:
                mm = node_means(model, tok, blocks, cm, walks, dev, N, [Lstar])
                p = eig_power(mm[Lstar], V)
                damage[:, L, h] = clean[1:] - p[1:]
            finally:
                hd.remove()
        if L % 8 == 0: print(f"[{tag}/{FAM}] swept layer {L}/{nL}", flush=True)

    # correlation between eigenmodes' damage maps (do different modes use different heads?)
    D = damage.reshape(nE, -1); C = np.corrcoef(D)
    out = {"model": tag, "fam": FAM, "N": N, "nL": nL, "nH": nH, "Lstar": Lstar,
           "eigenvalues": w[1:].tolist(), "clean_power": clean[1:].tolist(),
           "damage": damage.tolist(), "corr": C.tolist(),
           "top_heads": {int(k): [[int(l), int(hh), round(float(damage[k, l, hh]), 4)]
                          for l, hh in np.dstack(np.unravel_index(np.argsort(damage[k], axis=None)[::-1], damage[k].shape))[0][:6]]
                          for k in range(nE)}}
    json.dump(out, open(f"{OUTDIR}/head_eig_sweep_{tag}_{FAM}.json", "w"), indent=2)
    make_fig(out, damage, w, C, f"{OUTDIR}/head_eig_sweep_{tag}_{FAM}.pdf")
    print(f"DONE -> {OUTDIR}/head_eig_sweep_{tag}_{FAM}.json", flush=True)


def make_fig(out, damage, w, C, path):
    nE = damage.shape[0]
    with PdfPages(path) as pdf:
        # eigenmode damage-map correlation matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{out['model']} {out['fam']} — do eigenmodes share head circuits?\n"
                     "corr of per-mode single-head damage maps (1=same heads, 0=different)", fontsize=9)
        ax.set_xlabel("eigenmode (freq →)"); ax.set_ylabel("eigenmode (freq →)")
        fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # damage maps for the top-power modes
        order = np.argsort(out["clean_power"])[::-1][:6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, k in zip(axes.flat, order):
            Dk = damage[k]; lim = np.abs(Dk).max() + 1e-9
            im = ax.imshow(Dk, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"mode {k+1} (λ{out['eigenvalues'][k]:.1f}, pow {out['clean_power'][k]:.2f})", fontsize=8)
            ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.suptitle(f"{out['model']} {out['fam']} — single-head damage per eigenmode (red=head builds it)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
