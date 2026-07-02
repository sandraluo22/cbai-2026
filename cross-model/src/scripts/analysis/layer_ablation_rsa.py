"""Ablate the QK (prefix-matching) heads at ONE layer, then measure best-2D RSA over ALL
(subsequent) layers. Sweeps the ablation layer over the layers that contain QK>thresh heads,
so we see which layer's prefix-matchers (if any) the downstream geometry depends on.

For each model & graph: clean best-2D RSA per layer; then for each ablation layer L (all QK
heads at L zeroed), best-2D RSA per layer. Also a random-head control at the layer with the
strongest effect. Node means over ctx>=CTXLO.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(16) WLEN(250) CTXLO(100) QK_THRESH(0.2)
     MAXLAYERS(12) INDJSON OUTDIR DEVICE
Out: <OUTDIR>/layer_ablation_rsa_<graph>.json + .pdf
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
GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "16"))
WLEN = int(os.environ.get("WLEN", "250"))
CTXLO = int(os.environ.get("CTXLO", "100"))
QK_THRESH = float(os.environ.get("QK_THRESH", "0.2"))
MAXLAYERS = int(os.environ.get("MAXLAYERS", "12"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/layer_ablation")
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


def best2d_rsa(H, Gc, GD, iu):
    if np.isnan(H).any():
        return float("nan")
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GD)


@torch.no_grad()
def rsa_per_layer(model, tok, blocks, cm, walks, graph, dev, Gc, GD, iu, ablate_heads):
    n = graph.n_nodes; nL = cm.num_hidden_layers
    handles = []
    by_layer = {}
    for (l, h) in ablate_heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        proj, hdim = attn_proj(blocks[l], cm)
        cols = torch.tensor(np.concatenate([np.arange(h * hdim, (h + 1) * hdim) for h in hs]), device=dev)
        def pre(_m, args, cols=cols):
            x = args[0].clone(); x[..., cols] = 0; return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    for L in range(nL):
        handles.append(blocks[L].register_forward_hook(mk(L)))
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}
    ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0:
                            ncnt[nodes[s]] += 1
    finally:
        for hnd in handles:
            hnd.remove()
    cn = np.maximum(ncnt, 1)
    return [best2d_rsa(nsum[L] / cn[:, None], Gc, GD, iu) for L in range(nL)]


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        # QK heads per layer; ablation layers = those with the most QK>thresh heads
        qk_by_layer = {l: [h for h in range(nH) if gen[l, h] > QK_THRESH] for l in range(nL)}
        qk_by_layer = {l: hs for l, hs in qk_by_layer.items() if hs}
        abl_layers = sorted(qk_by_layer, key=lambda l: -len(qk_by_layer[l]))[:MAXLAYERS]
        abl_layers = sorted(abl_layers)
        print(f"[{tag}] ablation layers (QK>{QK_THRESH}): {abl_layers}", flush=True)

        clean = rsa_per_layer(model, tok, blocks, cm, walks, graph, dev, Gc, GD, iu, [])
        Lpeak = int(np.nanargmax(clean))
        rec = {"nL": nL, "clean": clean, "Lpeak": Lpeak, "qk_by_layer": qk_by_layer, "per_ablation_layer": {}}
        for L in abl_layers:
            heads = [(L, h) for h in qk_by_layer[L]]
            rec["per_ablation_layer"][str(L)] = rsa_per_layer(model, tok, blocks, cm, walks, graph, dev, Gc, GD, iu, heads)
            dpk = clean[Lpeak] - rec["per_ablation_layer"][str(L)][Lpeak]
            print(f"[{tag}] ablate QK@L{L} ({len(heads)} heads): ΔRSA@peak(L{Lpeak})={dpk:+.3f}", flush=True)
        # random control at the most-damaging ablation layer
        dmg = {L: clean[Lpeak] - rec["per_ablation_layer"][str(L)][Lpeak] for L in abl_layers}
        Lworst = max(dmg, key=dmg.get)
        allh = [(Lworst, h) for h in range(nH) if (Lworst, h) not in {(Lworst, x) for x in qk_by_layer[Lworst]}]
        rand = [allh[i] for i in RNG.choice(len(allh), min(len(qk_by_layer[Lworst]), len(allh)), replace=False)]
        rec["random_control"] = {"layer": Lworst, "rsa": rsa_per_layer(model, tok, blocks, cm, walks, graph, dev, Gc, GD, iu, rand)}
        rec["damage_at_peak"] = {str(L): dmg[L] for L in abl_layers}
        out["models"][tag] = rec
        print(f"[{tag}] worst ablation layer = L{Lworst} (ΔRSA@peak {dmg[Lworst]:+.3f}); "
              f"random@L{Lworst} ΔRSA@peak {clean[Lpeak]-rec['random_control']['rsa'][Lpeak]:+.3f}", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/layer_ablation_rsa_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/layer_ablation_rsa_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; nL = r["nL"]; L = list(range(nL))
            abl = sorted(int(k) for k in r["per_ablation_layer"])
            fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
            ax[0].plot(L, r["clean"], "-o", ms=3, color="k", lw=2, label="clean", zorder=5)
            cmap = plt.cm.viridis(np.linspace(0, 1, len(abl)))
            for c, La in zip(cmap, abl):
                curve = r["per_ablation_layer"][str(La)]
                ax[0].plot(L, curve, "-", lw=1, color=c, alpha=.8)
                ax[0].axvline(La, color=c, ls=":", lw=.6)
            ax[0].axvline(r["Lpeak"], color="red", ls="--", lw=1, label=f"peak L{r['Lpeak']}")
            ax[0].set_xlabel("layer (RSA read-out)"); ax[0].set_ylabel("best-2D RSA")
            ax[0].set_title(f"{m}: RSA over layers — clean (black) vs ablate QK @ each layer (viridis by abl layer)", fontsize=9)
            ax[0].legend(fontsize=7)
            # damage-at-peak vs ablation layer
            dmg = r["damage_at_peak"]; xs = sorted(int(k) for k in dmg)
            ax[1].bar(xs, [dmg[str(x)] for x in xs], width=0.8, color="tab:red", alpha=.8)
            rc = r.get("random_control")
            if rc:
                ax[1].plot([rc["layer"]], [r["clean"][r["Lpeak"]] - rc["rsa"][r["Lpeak"]]], "b*", ms=13, label="random ctrl")
            ax[1].axhline(0, color=".8", lw=.6); ax[1].set_xlabel("ablation layer"); ax[1].set_ylabel("ΔRSA at peak layer")
            ax[1].set_title(f"{m}: geometry damage at peak vs which layer's QK heads ablated", fontsize=9); ax[1].legend(fontsize=7)
            fig.suptitle(f"{m} [{out['graph']}]: ablate QK heads at ONE layer -> RSA over subsequent layers", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
