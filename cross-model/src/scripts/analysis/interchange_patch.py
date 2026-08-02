"""Experiment 4 -- causal head interchange across STRUCTURE. Base run = 4x4 GRID walk. For each
attention head, inject that head's per-node output from a SOURCE-graph run (node-aligned by shared word)
into the grid run, and measure how far the grid's next-node behaviour swaps toward the source's structure.

Two arms (SOURCE env):
  antiprism (A_8, 16 nodes, non-bipartite): a DIFFERENT parity -- inner/outer shell instead of even/odd
      checkerboard. Head that carries grid-parity, once overwritten, should make the grid predict the
      antiprism's opposite-SHELL nodes.  -> the PARITY arm.
  ring (16-cycle): a DIFFERENT coordinate -- ring position instead of grid (row,col).  -> the COORD arm.

Per head we log (nL x nH maps):
  d_src_nbr : source-neighbour prob mass at grid readouts, patched - base (behaviour toward source graph)
  d_src_par : source opposite-2-colour(shell) mass, patched - base (parity/shell swap specifically)
  d_grid_nbr: grid-neighbour mass, patched - base (how much the ORIGINAL grid behaviour is destroyed)
The source==grid node identity is shared (same words), so this is a clean interchange on the structural
variable. Cross-check vs the known parity/coord circuit is done offline (viz/interchange_patch_plot.py)
against the head->mode contribution map.

Env: GEN_MODEL(Llama) SOURCE(antiprism|ring) NWALKS(16) WLEN(300) CTXLO(100) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/interchange_<model>_<source>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
SOURCE = os.environ.get("SOURCE", "antiprism")                 # antiprism (parity arm) | ring (coord arm)
SRC_KW = {"antiprism": dict(graph_type="antiprism", prism_k=8), "ring": dict(graph_type="ring", ring_size=16),
          "prism": dict(graph_type="prism", prism_k=8)}
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/interchange")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    """bipartite 2-colouring if it exists, else the inner/outer shell split (first-half vs second-half
    node ids -- exactly the antiprism/prism inner vs outer rings)."""
    n = graph.n_nodes; col = np.zeros(n); bip = True
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
                elif col[v] == col[u]: bip = False
    if not bip:                                                      # non-bipartite -> inner/outer shell
        col = np.array([1.0 if i < n // 2 else -1.0 for i in range(n)])
    return col.astype(float)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def tok2node_map(spans, nodes, seqlen):
    t2n = np.full(seqlen, -1, int)
    for s in range(len(nodes)):
        for t in range(spans[s][0], spans[s][-1] + 1):
            if t < seqlen: t2n[t] = nodes[s]
    return t2n


@torch.no_grad()
def src_node_means(model, tok, blocks, cm, walks, dev, n, D):
    """per-layer per-node mean of the o_proj INPUT (concatenated head outputs) on SOURCE walks."""
    nL = cm.num_hidden_layers; zc = {}
    def mkz(L):
        def pre(_m, args): zc[L] = args[0].detach()
        return pre
    hs = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkz(L)) for L in range(nL)]
    zsum = {L: np.zeros((n, D)) for L in range(nL)}; zcnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
            model(input_ids=ids)
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    for L in range(nL): zsum[L][nodes[s]] += zc[L][0, spans[s][-1]].float().cpu().numpy()
                    zcnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(zcnt, 1)
    return {L: zsum[L] / cn[:, None] for L in range(nL)}


@torch.no_grad()
def grid_pass(model, tok, blocks, cm, walks, grid, src, srccol, gridcol, dev, cand_t, srcz, patchL, patchcols):
    """grid run, optionally patching head at patchL/patchcols with source node-means. Returns behaviour
    masses: source-neighbour, source-opposite-colour(shell), grid-neighbour -- averaged over readouts."""
    n = grid.n_nodes; state = {"mask": None, "replz": None}; handles = []
    if patchL is not None:
        proj, _hd = attn_proj(blocks[patchL], cm)
        def pre(_m, args):
            if state["mask"] is not None:
                x = args[0].clone()
                x[0][state["mask"].unsqueeze(1), patchcols.unsqueeze(0)] = \
                    state["replz"][state["mask"]][:, patchcols].to(x.dtype)
                return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    mass = np.zeros(3); mc = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); seqlen = ids.shape[1]
            if patchL is not None:
                t2n = tok2node_map(spans, nodes, seqlen)
                state["mask"] = torch.tensor(np.where(t2n >= 0)[0], device=dev, dtype=torch.long)
                state["replz"] = torch.tensor(srcz[patchL][np.where(t2n >= 0, t2n, 0)], device=dev)
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if cl[s] < CTXLO: continue
                cur = nodes[s]
                p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                src_opp = np.where(srccol == -srccol[cur])[0]
                mass[0] += float(p[src.neighbors(cur)].sum())               # source-neighbour mass
                mass[1] += float(p[src_opp].sum())                          # source opposite-colour/shell mass
                mass[2] += float(p[grid.neighbors(cur)].sum())              # grid-neighbour mass
                mc += 1
    finally:
        for h in handles: h.remove()
    return mass / max(mc, 1)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    gcfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4, n_walks=NWALKS, walk_length=WLEN, device=dev)
    scfg = replace(get_config("gemma_qwen"), **SRC_KW[SOURCE], n_walks=NWALKS, walk_length=WLEN, device=dev)
    grid = G.build_graph(gcfg); src = G.build_graph(scfg); n = grid.n_nodes
    assert src.n_nodes == n, f"source {SOURCE} has {src.n_nodes} nodes != grid {n} (need node alignment)"
    gridcol = two_colour(grid); srccol = two_colour(src)
    print(f"[{tag}] source={SOURCE} n={n} | grid∩src same-neighbour overlap "
          f"{np.mean([len(set(grid.neighbors(i)) & set(src.neighbors(i))) / max(len(grid.neighbors(i)),1) for i in range(n)]):.2f}", flush=True)

    model, tok = load_with_fallback(hf, mirror, gcfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH); D = nH * hd
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in grid.words], device=dev)
    gwalks = G.generate_walks(grid, gcfg); swalks = G.generate_walks(src, scfg)

    srcz = src_node_means(model, tok, blocks, cm, swalks, dev, n, D)
    base = grid_pass(model, tok, blocks, cm, gwalks, grid, src, srccol, gridcol, dev, cand_t, srcz, None, None)
    print(f"[{tag}] base: src_nbr={base[0]:.3f} src_par={base[1]:.3f} grid_nbr={base[2]:.3f}", flush=True)

    d_src_nbr = np.zeros((nL, nH)); d_src_par = np.zeros((nL, nH)); d_grid_nbr = np.zeros((nL, nH))
    for L in range(nL):
        _, hdd = attn_proj(blocks[L], cm)
        for h in range(nH):
            cols = torch.arange(h * hdd, (h + 1) * hdd, device=dev)
            m = grid_pass(model, tok, blocks, cm, gwalks, grid, src, srccol, gridcol, dev, cand_t, srcz, L, cols)
            d_src_nbr[L, h] = m[0] - base[0]; d_src_par[L, h] = m[1] - base[1]; d_grid_nbr[L, h] = m[2] - base[2]
        print(f"[{tag}] layer {L} done (max Δsrc_par so far {d_src_par.max():.3f})", flush=True)

    def tops(mat, k=10):
        idx = np.argsort(mat, axis=None)[::-1][:k]
        return [{"layer": int(i // nH), "head": int(i % nH), "val": round(float(mat.flatten()[i]), 4)} for i in idx]
    out = {"model": tag, "source": SOURCE, "n_layers": nL, "n_heads": nH, "n_nodes": n,
           "base": {"src_nbr": base[0], "src_par": base[1], "grid_nbr": base[2]},
           "d_src_nbr": d_src_nbr.tolist(), "d_src_par": d_src_par.tolist(), "d_grid_nbr": d_grid_nbr.tolist(),
           "top_src_nbr": tops(d_src_nbr), "top_src_par": tops(d_src_par)}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/interchange_{tag}_{SOURCE}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"[{tag}] top Δsrc_par: " + ", ".join(f"L{t['layer']}H{t['head']}={t['val']}" for t in out["top_src_par"][:5]), flush=True)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
