"""LONG-TERM GENERATION under head-group ablation: does removing (1) induction/QK heads or
(2) DLA writer heads break downstream BEHAVIOUR and/or the geometry REPRESENTATION over an
autoregressive rollout?

For each condition we ablate a head group everywhere (zero its per-head slice into o_proj at
ALL positions), seed the model with XCTX context steps, then sample GSTEPS steps freely. Per
generation window we track, from the REAL output:
  - neighbour mass : softmax mass the model puts on the true graph-neighbours of the last node
  - validity       : fraction of sampled steps that are a true neighbour (behaviour)
and, from one final ablated forward over the generated tokens, the geometry REPRESENTATION:
  - geom           : leave-one-node-out coord-probe R² of the generated tokens' node-means,
                     per (layer, window)  -> does the grid still live in the residual stream?

Conditions: clean | ablate_induction | ablate_dla | ablate_random (rank-matched control).
Head groups: top-KGROUP by induction "generic" score (INDJSON) and by DLA "head_attr" (DLAJSON);
random = KGROUP heads drawn from outside those two sets. This is the generate-mode analogue of
the teacher-forced 3_ablations knockouts, and the head-group analogue of removal_followup's
subspace removal -- so the two dissociation axes (which circuit / next-token vs long-term) are
directly comparable.

Env: PRESET GEN_MODEL(Llama) MODELS_FILTER GRAPH(square_grid) XCTX(150) GSTEPS(150) NSEED(4)
     NWIN(6) TEMP(1.0) KGROUP(15) INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/gen_head_ablation_<graph>.json + .pdf
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
from graph import Walk
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
           ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
if PRESET == "smoke":
    ALLSPEC = [("distilgpt2", "distilgpt2", None)]
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
XCTX = int(os.environ.get("XCTX", "150"))
GSTEPS = int(os.environ.get("GSTEPS", "150"))
NSEED = int(os.environ.get("NSEED", "4"))
NWIN = int(os.environ.get("NWIN", "6"))
TEMP = float(os.environ.get("TEMP", "1.0"))
KGROUP = int(os.environ.get("KGROUP", "15"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/gen_head_ablation")
RNG = np.random.default_rng(0)


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def mkwalk(nodes, graph):
    return Walk(walk_id=0, nodes=list(nodes), words=[graph.words[j] for j in nodes])


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def coord_loo_r2(H, coords):
    """Leave-one-node-out ridge coord-probe R² on node-means H (rows may be NaN if a node was
    never generated in a window). Alpha chosen by best LOO R²; matches removal_followup."""
    ok = np.isfinite(H).all(1)
    H = H[ok]; coords = coords[ok]
    n = H.shape[0]
    if n < 6:
        return float("nan")
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; Yc = coords - coords.mean(0)
    folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[k] @ Vt.T), U.T.copy(), S))
    best = -9.0
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        best = max(best, 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1])))
    return float(best)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def ablation_hooks(blocks, cm, dev, by_layer):
    """Zero the per-head slice into o_proj for every (layer -> [heads]) at ALL positions."""
    handles = []
    for L, heads in by_layer.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]),
                          device=dev, dtype=torch.long)
        def pre(_m, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    return handles


def by_layer(heads):
    d = {}
    for l, h in heads:
        d.setdefault(l, []).append(h)
    return d


def select_groups(ind, dla, tag, nL, nH):
    """top-KGROUP induction heads (by generic score) and DLA writer heads (by head_attr),
    plus a rank-matched random control drawn from outside both sets."""
    gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
    att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
    def topk(mat):
        order = np.argsort(mat, axis=None)[::-1]
        return [(int(i // nH), int(i % nH)) for i in order][:KGROUP]
    induction = topk(gen); writers = topk(att)
    used = set(induction) | set(writers)
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in used]
    rand = [pool[i] for i in RNG.choice(len(pool), min(KGROUP, len(pool)), replace=False)]
    return {"induction": induction, "dla": writers, "random": rand}


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
def generate_track(model, tok, blocks, cm, graph, cand_t, dev, seed_nodes, abl, coords, rng, acc, GBIN):
    """Ablate head-group `abl` (layer->heads, or None) at ALL positions, seed with seed_nodes,
    sample GSTEPS steps. Record real-output neighbour mass + validity per window, then one final
    ablated forward to grab generated-token node-means per (layer, window). Mutates `acc`."""
    nL = cm.num_hidden_layers
    NWIN = acc["nbr"].shape[0]; nodes = list(seed_nodes)
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    try:
        for t in range(GSTEPS):
            wk = mkwalk(nodes, graph); spans = resolve_token_spans(tok, wk)
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            last = model(input_ids=ids).logits[0, -1]
            p = torch.softmax(last[cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
            prev = nodes[-1]; nb = graph.neighbors(prev); j = int(rng.choice(len(p), p=p))
            b = min(t // GBIN, NWIN - 1)
            acc["nbr"][b] += float(p[nb].sum()); acc["val"][b] += int(j in nb); acc["cnt"][b] += 1
            nodes.append(j)
    finally:
        for h in handles: h.remove()
    # geometry of the generated tokens, ablation still applied
    gen = nodes[len(seed_nodes):]; gstart = len(seed_nodes); ng = len(gen)
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    wk = mkwalk(nodes, graph); spans = resolve_token_spans(tok, wk)
    ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev); grabbed.clear()
    model(input_ids=ids); single = [t[-1] for t in spans]
    for h in caps: h.remove()
    for h in handles: h.remove()
    bins = [min(i // GBIN, NWIN - 1) for i in range(ng)]
    for L in range(nL):
        rows = grabbed[L][0][[single[gstart + i] for i in range(ng)]].float().cpu().numpy()
        for i in range(ng):
            acc["gsum"][L, bins[i], gen[i]] += rows[i]
    for i in range(ng):
        acc["gcnt"][bins[i], gen[i]] += 1


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    tag = GEN_MODEL; hf, mirror = {t: (h, m) for t, h, m in ALLSPEC}[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NSEED, 8), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    seeds = G.generate_walks(graph, cfg)[:NSEED]
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)

    groups = select_groups(ind, dla, tag, nL, nH)
    print(f"[{tag}] KGROUP={KGROUP}  induction⊂{sorted({l for l,_ in groups['induction']})}  "
          f"dla⊂{sorted({l for l,_ in groups['dla']})}", flush=True)
    GBIN = max(1, GSTEPS // NWIN); win_mid = [b * GBIN + GBIN // 2 for b in range(NWIN)]
    conds = {"clean": None, "ablate_induction": by_layer(groups["induction"]),
             "ablate_dla": by_layer(groups["dla"]), "ablate_random": by_layer(groups["random"])}
    out = {"graph": GRAPH, "model": tag, "kgroup": KGROUP, "xctx": XCTX, "gsteps": GSTEPS,
           "nseed": NSEED, "nwin": NWIN, "win_mid": win_mid, "nL": nL,
           "groups": {k: [list(t) for t in v] for k, v in groups.items()}, "conds": {}}
    for cname, abl in conds.items():
        acc = {"nbr": np.zeros(NWIN), "val": np.zeros(NWIN), "cnt": np.zeros(NWIN),
               "gsum": np.zeros((nL, NWIN, n, cm.hidden_size)), "gcnt": np.zeros((NWIN, n))}
        for si, seed in enumerate(seeds):
            generate_track(model, tok, blocks, cm, graph, cand_t, dev, seed.nodes, abl, coords,
                           np.random.default_rng(1000 + si), acc, GBIN)
        cnt = np.maximum(acc["cnt"], 1)
        val = (acc["val"] / cnt); nbr = (acc["nbr"] / cnt)
        geom = np.full((nL, NWIN), np.nan)
        for L in range(nL):
            for w in range(NWIN):
                H = np.where(acc["gcnt"][w][:, None] > 0, acc["gsum"][L, w] / np.maximum(acc["gcnt"][w][:, None], 1), np.nan)
                geom[L, w] = coord_loo_r2(H, coords)
        out["conds"][cname] = {"val": val.tolist(), "nbr": nbr.tolist(), "geom": geom.tolist(),
                               "geom_peak_by_win": np.nanmax(geom, axis=0).tolist()}
        print(f"[{tag}/{GRAPH}/{cname}] nbr {nbr[0]:.2f}->{nbr[-1]:.2f}  val {val[0]:.2f}->{val[-1]:.2f}  "
              f"geom_peak {np.nanmax(geom):.2f} (final win {np.nanmax(geom[:, -1]):.2f})", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/gen_head_ablation_{GRAPH}.json"
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/gen_head_ablation_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    colors = {"clean": "k", "ablate_induction": "tab:green", "ablate_dla": "tab:red", "ablate_random": "tab:blue"}
    wm = out["win_mid"]
    with PdfPages(path) as pdf:
        # summary page: behaviour + peak-geometry vs generation step
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        for cn, c in colors.items():
            cd = out["conds"].get(cn)
            if not cd: continue
            ax[0].plot(wm, cd["nbr"], "-o", ms=3, color=c, label=cn)
            ax[1].plot(wm, cd["val"], "-o", ms=3, color=c, label=cn)
            ax[2].plot(wm, cd["geom_peak_by_win"], "-o", ms=3, color=c, label=cn)
        ax[0].set_title("downstream neighbour mass (behaviour)", fontsize=9); ax[0].set_ylim(0, 1.05)
        ax[1].set_title("generated-step validity (behaviour)", fontsize=9); ax[1].set_ylim(0, 1.05)
        ax[2].set_title("coord-probe R² over layers (representation)", fontsize=9)
        ax[2].set_ylim(-0.6, 1.0); ax[2].axhline(0, color=".7", lw=.6)
        for a in ax: a.set_xlabel("generation step"); a.legend(fontsize=7)
        fig.suptitle(f"[{out['graph']}] {out['model']} — LONG-TERM GENERATION under head-group ablation "
                     f"(K={out['kgroup']}/group).\nblack=clean, green=ablate induction/QK, red=ablate DLA writers, "
                     "blue=ablate random. Behaviour (mass/validity) vs representation (probe R²).", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # per-layer geometry pages
        for L in range(out["nL"]):
            fig, a = plt.subplots(figsize=(6.4, 4.4))
            for cn, c in colors.items():
                cd = out["conds"].get(cn)
                if not cd: continue
                a.plot(wm, np.array(cd["geom"])[L], "-o", ms=3, color=c, label=cn)
            a.set_title(f"{out['model']} coord-probe R² @ LAYER {L}", fontsize=9)
            a.set_ylim(-0.6, 1.0); a.axhline(0, color=".7", lw=.6)
            a.set_xlabel("generation step"); a.legend(fontsize=7)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
