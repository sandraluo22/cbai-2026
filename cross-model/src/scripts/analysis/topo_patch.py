"""Head patching across TOPOLOGIES: grid (original) vs ring, same 16 words. Grid and ring
walks are different node sequences (no position alignment), so this uses NODE-ALIGNED mean
patching: capture each head's per-node output in the RING run, inject it (by current node)
into the GRID run, and measure whether the geometry / behaviour moves toward the ring.

Per head h:
  restore_rsa : (RSA_ring(grid patched h) - RSA_ring(grid_base)) / (RSA_ring(ring) - RSA_ring(grid_base))
                best-2D RSA of the grid-run geometry vs the RING layout. 1 = fully ring-like.
  d_ring_mass : ring-neighbour mass at grid readouts, patched minus base (behaviour toward ring).
Also RSA_grid(patch) to flag heads that just DISRUPT (both low).

Env: PRESET MODELS_FILTER NWALKS(16) WLEN(300) CTXLO(100) INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/topo_patch.json + .pdf
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
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
if PRESET == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None)]
else:
    MODELS = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
              ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
              ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
NWALKS = int(os.environ.get("NWALKS", "16"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/topo_patch")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GDu, iu):
    if np.isnan(H).any():
        return float("nan")
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GDu)


def tok2node_map(spans, nodes, seqlen):
    t2n = np.full(seqlen, -1, int)
    for s in range(len(nodes)):
        for t in range(spans[s][0], spans[s][-1] + 1):
            if t < seqlen: t2n[t] = nodes[s]
    return t2n


@torch.no_grad()
def ring_node_means(model, tok, blocks, cm, walks, dev, nH, hd):
    """per-layer per-node mean of the o_proj INPUT (concatenated head outputs) on ring walks."""
    nL = cm.num_hidden_layers; D = nH * hd
    zc = {}
    def mkz(L):
        def pre(_m, args): zc[L] = args[0].detach()
        return pre
    hs = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkz(L)) for L in range(nL)]
    zsum = {L: np.zeros((16, D)) for L in range(nL)}; zcnt = np.zeros(16)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
            model(input_ids=ids)
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    for L in range(nL):
                        zsum[L][nodes[s]] += zc[L][0, spans[s][-1]].float().cpu().numpy()
                    zcnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(zcnt, 1)
    return {L: zsum[L] / cn[:, None] for L in range(nL)}


@torch.no_grad()
def grid_pass(model, tok, blocks, cm, walks, grid, ring, dev, Lstar, cand_t, ringz, patchL, patchcols):
    """grid run (optionally patch head at patchL/patchcols with ring node-means); return
    node-mean geometry at Lstar and ring/grid neighbour mass at readouts."""
    n = 16; state = {"mask": None, "replz": None}
    handles = []
    if patchL is not None:
        proj, _hd = attn_proj(blocks[patchL], cm)
        def pre(_m, args):
            if state["mask"] is not None:
                x = args[0].clone()
                x[0][state["mask"].unsqueeze(1), patchcols.unsqueeze(0)] = \
                    state["replz"][state["mask"]][:, patchcols].to(x.dtype)
                return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    grab = {}
    def mk(L):
        def hh(_m, _i, out): grab[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    handles.append(blocks[Lstar].register_forward_hook(mk(Lstar)))
    nsum = np.zeros((n, cm.hidden_size)); ncnt = np.zeros(n); mass = np.zeros(2); mc = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
            seqlen = ids.shape[1]
            if patchL is not None:
                t2n = tok2node_map(spans, nodes, seqlen)
                mask = torch.tensor(np.where(t2n >= 0)[0], device=dev, dtype=torch.long)
                replz = torch.tensor(ringz[patchL][np.where(t2n >= 0, t2n, 0)], device=dev)
                state["repl"] = True; state["mask"] = mask; state["replz"] = replz
            grab.clear(); logits = model(input_ids=ids).logits[0]
            hs = grab[Lstar][0]
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    nsum[nodes[s]] += hs[spans[s][-1]].float().cpu().numpy(); ncnt[nodes[s]] += 1
                    if s < len(nodes) - 1:
                        p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                        rn = [(nodes[s] - 1) % 16, (nodes[s] + 1) % 16]
                        mass[0] += float(p[rn].sum()); mass[1] += float(p[grid.neighbors(nodes[s])].sum()); mc += 1
    finally:
        for h in handles: h.remove()
    cn = np.maximum(ncnt, 1)
    return nsum / cn[:, None], (mass[0] / max(mc, 1), mass[1] / max(mc, 1))


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    gcfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4)
    rcfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=16)
    out = {"models": {}}
    for tag, hf, mirror in MODELS:
        gc_cfg = replace(gcfg, n_walks=NWALKS, walk_length=WLEN, device=dev)
        rc_cfg = replace(rcfg, n_walks=NWALKS, walk_length=WLEN, device=dev)
        grid = G.build_graph(gc_cfg); ring = G.build_graph(rc_cfg)
        iu = np.triu_indices(16, 1)
        GDg = grid.distance_matrix()[iu]; GDr = ring.distance_matrix()[iu]
        Gc_g = np.array(grid.coords, float); Gc_r = np.array(ring.coords, float)
        print(f"[{tag}] RSA(grid_dist, ring_dist)={sp(GDg, GDr):.3f}", flush=True)
        gwalks = G.generate_walks(grid, gc_cfg); rwalks = G.generate_walks(ring, rc_cfg)
        model, tok = load_with_fallback(tag, hf, mirror, gc_cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH)
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in grid.words], device=dev)

        ringz = ring_node_means(model, tok, blocks, cm, rwalks, dev, nH, hd)
        # grid base geometry: pick L* = grid best-2D peak (capture all layers once)
        grab = {}
        def mk(L):
            def hh(_m, _i, out): grab[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
        nsum = {L: np.zeros((16, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(16)
        for wk in gwalks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grab.clear()
            model(input_ids=ids)
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    for L in range(nL): nsum[L][nodes[s]] += grab[L][0, spans[s][-1]].float().cpu().numpy()
                    ncnt[nodes[s]] += 1
        for h in hs: h.remove()
        cn = np.maximum(ncnt, 1); gmeans = {L: nsum[L] / cn[:, None] for L in range(nL)}
        Lstar = int(np.nanargmax([best2d_rsa(gmeans[L], Gc_g, GDg, iu) for L in range(nL)]))
        RSAg_grid = best2d_rsa(gmeans[Lstar], Gc_g, GDg, iu); RSAg_ring = best2d_rsa(gmeans[Lstar], Gc_r, GDr, iu)
        # ring full geometry (for the ring reference)
        rsum = {Lstar: np.zeros((16, cm.hidden_size))}; rcnt = np.zeros(16); h2 = blocks[Lstar].register_forward_hook(mk(Lstar))
        for wk in rwalks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grab.clear()
            model(input_ids=ids)
            for s in range(len(nodes)):
                if cl[s] >= CTXLO: rsum[Lstar][nodes[s]] += grab[Lstar][0, spans[s][-1]].float().cpu().numpy(); rcnt[nodes[s]] += 1
        h2.remove(); rmean = rsum[Lstar] / np.maximum(rcnt, 1)[:, None]
        RSAr_ring = best2d_rsa(rmean, Gc_r, GDr, iu)
        base_geo, base_mass = grid_pass(model, tok, blocks, cm, gwalks, grid, ring, dev, Lstar, cand_t, ringz, None, None)
        print(f"[{tag}] L*={Lstar} | grid geo: vs grid={RSAg_grid:.2f} vs ring={RSAg_ring:.2f} | ring geo vs ring={RSAr_ring:.2f} "
              f"| base ring_mass={base_mass[0]:.2f} grid_mass={base_mass[1]:.2f}", flush=True)
        dR = (RSAr_ring - RSAg_ring) if abs(RSAr_ring - RSAg_ring) > 1e-4 else 1.0

        r_rsa = np.zeros((nL, nH)); r_rsa_grid = np.zeros((nL, nH)); d_ring = np.zeros((nL, nH))
        for L in range(nL):
            _, hdd = attn_proj(blocks[L], cm)
            for h in range(nH):
                cols = torch.arange(h * hdd, (h + 1) * hdd, device=dev)
                geo, mass = grid_pass(model, tok, blocks, cm, gwalks, grid, ring, dev, Lstar, cand_t, ringz, L, cols)
                r_rsa[L, h] = (best2d_rsa(geo, Gc_r, GDr, iu) - RSAg_ring) / dR
                r_rsa_grid[L, h] = best2d_rsa(geo, Gc_g, GDg, iu)
                d_ring[L, h] = mass[0] - base_mass[0]
            print(f"[{tag}] layer {L} done", flush=True)
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
        def tops(mat):
            t = np.argsort(mat, axis=None)[::-1][:8]
            return [{"layer": int(i // nH), "head": int(i % nH), "val": round(float(mat.flatten()[i]), 3),
                     "qk": round(float(gen.flatten()[i]), 2), "dla": round(float(att.flatten()[i]), 2)} for i in t]
        rec = {"n_layers": nL, "n_heads": nH, "Lstar": Lstar, "rsa_grid_vs_ring_dist": sp(GDg, GDr),
               "RSAg_grid": RSAg_grid, "RSAg_ring": RSAg_ring, "RSAr_ring": RSAr_ring,
               "base_ring_mass": base_mass[0], "base_grid_mass": base_mass[1],
               "restore_rsa": r_rsa.tolist(), "rsa_grid": r_rsa_grid.tolist(), "d_ring_mass": d_ring.tolist(),
               "top_rsa": tops(r_rsa), "top_ring_mass": tops(d_ring),
               "corr_rsa_ringmass": float(np.corrcoef(r_rsa.flatten(), d_ring.flatten())[0, 1])}
        out["models"][tag] = rec
        print(f"[{tag}] top ΔRSA→ring L{rec['top_rsa'][0]['layer']}H{rec['top_rsa'][0]['head']}={rec['top_rsa'][0]['val']} "
              f"| top Δring-mass L{rec['top_ring_mass'][0]['layer']}H{rec['top_ring_mass'][0]['head']}={rec['top_ring_mass'][0]['val']} "
              f"| corr={rec['corr_rsa_ringmass']:.2f}", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/topo_patch.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/topo_patch.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            for j, (key, lab) in enumerate([("restore_rsa", "ΔRSA→ring (geometry)"), ("d_ring_mass", "Δ ring-neighbour mass (behaviour)")]):
                Rm = np.array(r[key]); v = max(0.1, float(np.nanpercentile(np.abs(Rm), 99)))
                im = ax[j].imshow(Rm, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                ax[j].set_xlabel("head"); ax[j].set_ylabel("layer"); ax[j].set_title(f"{m}: {lab}", fontsize=9)
                fig.colorbar(im, ax=ax[j], fraction=.046)
            fig.suptitle(f"{m}: inject RING head-output into GRID run — which head makes geometry/behaviour ring-like? "
                         f"(GT sim={r['rsa_grid_vs_ring_dist']:.2f}, corr={r['corr_rsa_ringmass']:.2f})", fontsize=9)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
