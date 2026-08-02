"""Compare the PARITY eigenmode representation between grid sizes (4x4 balanced 8/8 vs 5x5 unbalanced 13/12).
For each grid capture L14H26's per-node output, derive the parity axis (mean output of parity-+ minus parity--
nodes), the normalized-Laplacian eigenmodes, and the head's eigenmode write-spectrum. Then compare across
sizes: (a) cosine of the two parity axes, (b) cross-transfer (does the 4x4 axis separate 5x5's two colours,
and vice versa), (c) whether L14H26 writes the parity (top-eig) mode in both. Tells us if the model uses ONE
size-invariant parity direction (consistent with the natural-text boundary-feature finding) or a per-size one.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) GRIDS("4x4,5x5") NWALKS(24) WLEN(260) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/grid_parity_compare_<model>.json (+ .npz)
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from dataclasses import replace as dreplace
import config as _config
from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw

def build_word_pool(tok, need):
    """distinct clean single-token lowercase words from the tokenizer vocab, so grids up to 16x16 (256
    nodes) get a unique word per node. Deterministic (sorted by token id)."""
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    pool, seen = [], set()
    for token, _idx in vocab:
        if not (token.startswith("Ġ") or token.startswith("▁")): continue     # word-initial only
        s = tok.convert_tokens_to_string([token]).strip()
        if s.isascii() and s.isalpha() and s.islower() and 4 <= len(s) <= 10 and s not in seen:
            seen.add(s); pool.append(s)
            if len(pool) >= need + 30: break
    return pool

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
GRIDS = [tuple(int(x) for x in g.split("x")) for g in os.environ.get("GRIDS", "4x4,5x5").split(",")]
NWALKS = int(os.environ.get("NWALKS", "24")); WLEN = int(os.environ.get("WLEN", "260")); CTXLO = int(os.environ.get("CTXLO", "100"))
NPERM = int(os.environ.get("NPERM", "6"))     # random word->node assignments to average out word content
SPN = int(os.environ.get("SAMPLES_PER_NODE", "0"))  # if >0, scale walk_length so each node gets ~SPN readout occurrences
WLEN_CAP = int(os.environ.get("WLEN_CAP", "900"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


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


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12); return float(abs(a @ b))


@torch.no_grad()
def znode_for_words(model, tok, blocks, cm, graph, cfg, dev, csl, hd, n):
    """per-node mean of L14H26 output for the graph's CURRENT word assignment."""
    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    proj = attn_proj(blocks[HEAD_LAYER], cm)[0]; hk = proj.register_forward_pre_hook(cap)
    walks = G.generate_walks(graph, cfg); zsum = np.zeros((n, hd)); zcnt = np.zeros(n)  # zcnt = per-node readout visit counts
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
        model(input_ids=ids)
        for s in range(len(nodes) - 1):
            if cl[s] < CTXLO: continue
            zsum[nodes[s]] += zc["z"][0, spans[s][-1], csl].float().cpu().numpy(); zcnt[nodes[s]] += 1
    hk.remove()
    return zsum / np.maximum(zcnt, 1)[:, None], zcnt


@torch.no_grad()
def capture(model, tok, blocks, cm, R, C, dev):
    n = R * C
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS))) if SPN else WLEN   # equalize samples/node across sizes
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=R, grid_cols=C, n_walks=NWALKS, walk_length=wl, device=dev)
    graph0 = G.build_graph(cfg); n = graph0.n_nodes; col = two_colour(graph0); coords = np.array(graph0.coords, int)
    print(f"  [{R}x{C}] n={n} walk_length={wl} (~{NWALKS*(wl-CTXLO)//n} readouts/node)", flush=True)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    # average node reps over NPERM random word->node assignments (from the shared 36-word pool) -> word-agnostic
    # PERMSEED_PER_SIZE=1: seed word permutations per grid size. With the legacy shared seed(0), perm i is
    # byte-identical across sizes (same pool), so node j gets the same word at every size; for ODD k node j's
    # colour is exactly j%2 at every size, so word-specific components add coherently across odd sizes and
    # inflate odd-odd cross-size axis cosines (a word confound, not shared geometry).
    seed = (97 * R + 31 * C + 7) if os.environ.get("PERMSEED_PER_SIZE", "0") == "1" else 0
    rng = np.random.default_rng(seed); zperm = []; axperm = []; cnt = np.zeros(n)
    for p in range(NPERM):
        w = list(np.array(_config.WORDS)[rng.permutation(len(_config.WORDS))[:n]])
        gp = dreplace(graph0, words=w)
        zn, zc_p = znode_for_words(model, tok, blocks, cm, gp, cfg, dev, csl, hd, n)
        cnt += zc_p
        zperm.append(zn); a = zn[col > 0].mean(0) - zn[col < 0].mean(0); axperm.append(a / (np.linalg.norm(a) + 1e-12))
    znode = np.mean(zperm, 0); cnt /= NPERM                                # word-agnostic per-node rep
    axis = znode[col > 0].mean(0) - znode[col < 0].mean(0); axis = axis / (np.linalg.norm(axis) + 1e-12)
    # visit-frequency / degree controls: odd grids put all four corners in one colour class, so the raw parity
    # axis may carry a shared visit-frequency (degree) component that inflates odd-odd cross-size cosines.
    # Remove the activation direction predicted by node visit counts (resp. node degree) and re-derive the axis.
    Zc = znode - znode.mean(0)
    def _ctrl_axis(feature):
        f = feature - feature.mean(); f = f / (np.linalg.norm(f) + 1e-12)
        b = Zc.T @ f; b = b / (np.linalg.norm(b) + 1e-12)
        ax = axis - (axis @ b) * b
        return ax / (np.linalg.norm(ax) + 1e-12), float(abs(axis @ b))
    deg = np.array([len(graph0.adjacency[u]) for u in range(n)], float)
    axis_fc, freq_axis_cos = _ctrl_axis(cnt)
    axis_dc, deg_axis_cos = _ctrl_axis(deg)
    # within-grid consistency of the per-perm axes = 1 - word confound (high => geometric, low => word-driven)
    aps = np.array(axperm); pc = [abs(aps[i] @ aps[j]) for i in range(NPERM) for j in range(i + 1, NPERM)]
    within_axis_cos = float(np.mean(pc)) if pc else 1.0
    # split-half ceiling: axis from perms[:half] vs perms[half:] — the max cross-size cosine one could expect
    h1, h2 = np.mean(zperm[:NPERM // 2], 0), np.mean(zperm[NPERM // 2:], 0)
    sh = [h[col > 0].mean(0) - h[col < 0].mean(0) for h in (h1, h2)]
    splithalf_cos = cos(sh[0], sh[1]) if NPERM >= 2 else 1.0
    # normalized-Laplacian eigenmodes + head write spectrum
    A = np.zeros((n, n))
    for u in range(n):
        for v in graph0.adjacency[u]: A[u, v] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(np.maximum(dg, 1e-12)); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    eigw, eigU = np.linalg.eigh(Lap); parity_mode = int(np.argmax(eigw))
    Hc = znode - znode.mean(0)
    pw = np.array([np.linalg.norm(Hc.T @ eigU[:, k]) ** 2 for k in range(n)]); pw = pw / (pw.sum() + 1e-12)
    gp = znode @ axis; gpf = znode @ axis_fc
    return {"R": R, "C": C, "n": n, "col": col, "coords": coords, "znode": znode, "axis": axis,
            "axis_fc": axis_fc, "axis_dc": axis_dc,
            "freq_axis_cos": round(freq_axis_cos, 3), "deg_axis_cos": round(deg_axis_cos, 3),
            "sep_own_fc": float(gpf[col > 0].mean() - gpf[col < 0].mean()),
            "eigw": eigw, "eigU": eigU, "parity_mode": parity_mode, "spectrum": pw,
            "within_axis_cos": round(within_axis_cos, 3), "splithalf_cos": round(splithalf_cos, 3),
            "n_pos": int((col > 0).sum()), "n_neg": int((col < 0).sum()),
            "sep_own": float(gp[col > 0].mean() - gp[col < 0].mean()),
            "parity_mode_power": float(pw[parity_mode]),
            "parity_axis_eig_corr": round(float(abs(np.corrcoef(znode @ axis - (znode @ axis).mean(), eigU[:, parity_mode])[0, 1])), 3)}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    need = max(R * C for R, C in GRIDS)
    if need > len(_config.WORDS):
        _config.WORDS[:] = build_word_pool(tok, need)
        print(f"[{tag}] word pool -> {len(_config.WORDS)} distinct tokens (need {need})", flush=True)
    D = {f"{R}x{C}": capture(model, tok, blocks, cm, R, C, dev) for R, C in GRIDS}
    keys = list(D)
    # cross comparisons
    cross = {}
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j: continue
            a, b = keys[i], keys[j]
            gp = D[b]["znode"] @ D[a]["axis"]                       # project grid-b nodes on grid-a axis
            colb = D[b]["col"]
            cross[f"{a}_axis_on_{b}"] = {"sep": round(float(gp[colb > 0].mean() - gp[colb < 0].mean()), 3),
                                        "sep_vs_own": round(float((gp[colb > 0].mean() - gp[colb < 0].mean()) / (D[b]["sep_own"] + 1e-9)), 3)}
    axis_cos = {}; axis_cos_fc = {}; axis_cos_dc = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            kk = f"{keys[i]}_vs_{keys[j]}"
            axis_cos[kk] = round(cos(D[keys[i]]["axis"], D[keys[j]]["axis"]), 3)
            axis_cos_fc[kk] = round(cos(D[keys[i]]["axis_fc"], D[keys[j]]["axis_fc"]), 3)
            axis_cos_dc[kk] = round(cos(D[keys[i]]["axis_dc"], D[keys[j]]["axis_dc"]), 3)

    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "grids": keys,
           "per_grid": {k: {"n": D[k]["n"], "balance": f"{D[k]['n_pos']}/{D[k]['n_neg']}",
                            "within_grid_axis_consistency": D[k]["within_axis_cos"],
                            "axis_splithalf_cos": D[k]["splithalf_cos"],
                            "freq_axis_cos": D[k]["freq_axis_cos"], "deg_axis_cos": D[k]["deg_axis_cos"],
                            "sep_own_freqctrl": round(D[k]["sep_own_fc"], 3),
                            "sep_own": round(D[k]["sep_own"], 3), "parity_mode_idx": D[k]["parity_mode"],
                            "parity_mode_power": round(D[k]["parity_mode_power"], 3),
                            "parity_axis_eig_corr": D[k]["parity_axis_eig_corr"],
                            "top3_write_modes": [[int(m), round(float(D[k]["spectrum"][m]), 3), round(float(D[k]["eigw"][m]), 3)]
                                                 for m in np.argsort(D[k]["spectrum"])[::-1][:3]]} for k in keys},
           "parity_axis_cosine_across_sizes": axis_cos,
           "parity_axis_cosine_across_sizes_freqctrl": axis_cos_fc,
           "parity_axis_cosine_across_sizes_degctrl": axis_cos_dc, "cross_transfer": cross}
    p = f"{OUTDIR}/grid_parity_compare{os.environ.get('OUTTAG', '')}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    npz = {}
    for k in keys:
        for f in ("col", "coords", "axis", "spectrum", "eigw", "znode"):
            npz[f"{k}_{f}"] = np.asarray(D[k][f]).astype("float32")
        npz[f"{k}_eigU"] = D[k]["eigU"].astype("float32")
        npz[f"{k}_parity_eigvec"] = D[k]["eigU"][:, D[k]["parity_mode"]].astype("float32")
    np.savez_compressed(p.replace(".json", ".npz"), **npz)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(out, indent=2), flush=True); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
