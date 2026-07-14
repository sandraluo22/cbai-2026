"""Mediation test: do specific heads READ the geometry representation?

Steer node X's residual to node Y's location at an early-ish layer L_steer (where the map
is already formed), THEN ablate a downstream head-group G, and measure whether the steering
redirect (mass moving to Y's neighbours) survives. If the redirect COLLAPSES when G is
ablated, G reads the steered representation to produce the prediction.

Per pair (X,Y) and group G in {writers(top DLA), qk(top prefix-match), random} (all restricted
to layers > L_steer), measure mass on Y's neighbours in the 2x2 {steer?}x{ablate G?}:
  redirect_noabl = nbrY(steer) - nbrY(nosteer)          # baseline steering effect
  redirect_ablG  = nbrY(steer,ablG) - nbrY(nosteer,ablG)
  mediation(G)   = redirect_noabl - redirect_ablG        # how much G carries the read

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(12) WLEN(300) CTXLO(100) KGROUP(15)
     STEER(full|plane) INDJSON DLAJSON PAIRS OUTDIR DEVICE
Out: <OUTDIR>/steer_x_ablate_<graph>.json + .pdf
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
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "12"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
KGROUP = int(os.environ.get("KGROUP", "15"))
STEER = os.environ.get("STEER", "full")
PAIRS = [tuple(int(x) for x in p.split("-")) for p in os.environ.get("PAIRS", "0-12,3-15,0-15,5-10").split(",")]
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/steer_x_ablate")
RNG = np.random.default_rng(0)


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


def best2d_plane(H, Gc):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    Q, _ = np.linalg.qr(Vt[:k].T @ W)
    return Q


@torch.no_grad()
def node_means_all(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(cm.num_hidden_layers)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(cm.num_hidden_layers)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(cm.num_hidden_layers):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(cm.num_hidden_layers)}


@torch.no_grad()
def measure(model, tok, blocks, cm, graph, cand_t, dev, L_steer, walks, X, steervec, ablate_by_layer):
    n = graph.n_nodes; handles = []
    # downstream head ablation (all positions)
    for L, heads in ablate_by_layer.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]), device=dev, dtype=torch.long)
        def pre(_m, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0; return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    # steer at L_steer, X readout positions
    st = {"pos": None}; v = None if steervec is None else torch.tensor(steervec, device=dev)
    def shook(_m, _i, out):
        if st["pos"]:
            hsd = (out[0] if isinstance(out, tuple) else out).clone()
            hsd[0, st["pos"], :] += v.to(hsd.dtype)
            return (hsd,) + tuple(out[1:]) if isinstance(out, tuple) else hsd
    if steervec is not None:
        handles.append(blocks[L_steer].register_forward_hook(shook))
    Psum = np.zeros(n); cnt = 0
    try:
        for wk in walks:
            nodes = wk.nodes; spans = resolve_token_spans(tok, wk); cl = np.arange(1, len(nodes) + 1)
            pos = [spans[s + 1][0] - 1 for s in range(len(nodes) - 1) if nodes[s] == X and cl[s] >= CTXLO]
            if not pos: continue
            st["pos"] = pos
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            for p in pos:
                Psum += torch.softmax(logits[p][cand_t].float(), 0).cpu().numpy(); cnt += 1
    finally:
        for h in handles: h.remove()
    return Psum / max(cnt, 1)


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "steer": STEER, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; Gc = np.array(graph.coords, float)
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
        means = node_means_all(model, tok, blocks, cm, walks, dev, n)
        rsaL = {L: sp(np.linalg.norm((means[L]-means[L].mean(0))[:,None]-(means[L]-means[L].mean(0))[None],axis=2)[iu], GD) for L in range(nL)}
        peak = max(rsaL.values())
        # steer layer: strongest-geometry layer in the first 60% of depth (leaves writers downstream)
        cand = [L for L in range(nL) if rsaL[L] >= 0.85 * peak and L <= 0.6 * nL] or [int(0.4 * nL)]
        L_steer = max(cand, key=lambda L: rsaL[L]); H = means[L_steer]; B = best2d_plane(H, Gc)
        print(f"[{tag}] L_steer={L_steer} (rsa {rsaL[L_steer]:.2f}, peak {peak:.2f})", flush=True)
        # head groups, restricted to layers strictly downstream of L_steer
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
        def topk(mat):
            order = np.argsort(mat, axis=None)[::-1]
            hs = [(int(i // nH), int(i % nH)) for i in order]
            return [x for x in hs if x[0] > L_steer][:KGROUP]
        writers = topk(att); qk = topk(gen)
        used = set(writers) | set(qk)
        pool = [(l, h) for l in range(L_steer + 1, nL) for h in range(nH) if (l, h) not in used]
        rand = [pool[i] for i in RNG.choice(len(pool), min(KGROUP, len(pool)), replace=False)]
        groups = {"writers": writers, "qk": qk, "random": rand}
        def by_layer(heads):
            d = {}
            for l, h in heads: d.setdefault(l, []).append(h)
            return d

        rec = {"L_steer": L_steer, "groups": {k: v for k, v in groups.items()}, "pairs": {}}
        for (X, Y) in PAIRS:
            nbrY = graph.neighbors(Y); nbrX = graph.neighbors(X)
            sv = (H[Y] - H[X]) if STEER == "full" else B @ (B.T @ (H[Y] - H[X]))
            pr = {}
            for gname, heads in [("none", []), ("writers", writers), ("qk", qk), ("random", rand)]:
                abl = by_layer(heads)
                Pns = measure(model, tok, blocks, cm, graph, cand_t, dev, L_steer, walks, X, None, abl)
                Pst = measure(model, tok, blocks, cm, graph, cand_t, dev, L_steer, walks, X, sv, abl)
                pr[gname] = {"nosteer_nbrY": float(Pns[nbrY].sum()), "steer_nbrY": float(Pst[nbrY].sum()),
                             "redirect": float(Pst[nbrY].sum() - Pns[nbrY].sum())}
            base = pr["none"]["redirect"]
            rec["pairs"][f"{X}->{Y}"] = pr
            print(f"[{tag}] {X}->{Y}: redirect none={base:+.2f} | writers={pr['writers']['redirect']:+.2f} "
                  f"qk={pr['qk']['redirect']:+.2f} random={pr['random']['redirect']:+.2f}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/steer_x_ablate_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/steer_x_ablate_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; pairs = list(r["pairs"]); x = np.arange(len(pairs)); w = 0.2
            for i, (g, c) in enumerate([("none", "k"), ("writers", "tab:red"), ("qk", "tab:green"), ("random", "tab:blue")]):
                ax[0, j].bar(x + (i - 1.5) * w, [r["pairs"][p][g]["redirect"] for p in pairs], w, label=g, color=c)
            ax[0, j].axhline(0, color=".7", lw=.6); ax[0, j].set_xticks(x); ax[0, j].set_xticklabels(pairs, fontsize=7)
            ax[0, j].set_title(f"{m} (L_steer={r['L_steer']})", fontsize=9); ax[0, j].set_ylabel("steering redirect (Δ mass on Y-nbrs)")
            ax[0, j].legend(fontsize=7)
        fig.suptitle(f"[{out['graph']}] steer geometry + ablate downstream group: does the redirect survive? "
                     "(lower bar under a group = that group READS the representation)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
