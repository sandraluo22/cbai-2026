"""Single-head ablation SWEEP: which head, ablated, damages the GEOMETRY most?

For each model & graph: a clean run gives best-2D RSA per layer -> pick the peak layer L*.
Then ablate EACH head (in layers 0..L*, since later heads can't affect L*) one at a time,
re-run, and re-measure best-2D RSA at L* AND next-step neighbour mass. Rank heads by
geometry damage (clean - ablated best-2D RSA); the behaviour delta is captured for free to
contrast geometry-critical vs behaviour-critical heads.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(8) WLEN(250) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/head_sweep_<graph>.json  and  head_sweep_<graph>.pdf
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
NWALKS = int(os.environ.get("NWALKS", "8"))
WLEN = int(os.environ.get("WLEN", "250"))
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/head_sweep")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        nh = cm.num_attention_heads
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // nh)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GD, iu):
    if np.isnan(H).any():
        return float("nan")
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GD)


@torch.no_grad()
def run(model, tok, blocks, cm, walks, graph, cand_t, dev, cap_layers, ablate=None):
    n = graph.n_nodes
    handles = []
    if ablate is not None:
        l, h = ablate; proj, hdim = attn_proj(blocks[l], cm)
        is_conv = not (hasattr(blocks[l], "self_attn"))
        ct = torch.arange(h * hdim, (h + 1) * hdim, device=dev)
        def pre(_m, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0; return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    for L in cap_layers:
        handles.append(blocks[L].register_forward_hook(mk(L)))
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in cap_layers}
    ncnt = {L: np.zeros(n) for L in cap_layers}
    mass = [0.0, 0]
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in cap_layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]; ncnt[L][nodes[s]] += 1
            for s in range(len(nodes) - 1):
                if cl[s] >= CTXLO:
                    p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                    mass[0] += float(p[graph.neighbors(nodes[s])].sum()); mass[1] += 1
    finally:
        for hnd in handles:
            hnd.remove()
    means = {L: np.where(ncnt[L][:, None] > 0, nsum[L] / np.maximum(ncnt[L][:, None], 1), np.nan) for L in cap_layers}
    return means, (mass[0] / mass[1] if mass[1] else float("nan"))


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
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
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)

        # clean: best-2D RSA per layer -> peak layer L*
        means, clean_mass = run(model, tok, blocks, cm, walks, graph, cand_t, dev, list(range(nL)))
        rsa_layer = {L: best2d_rsa(means[L], Gc, GD, iu) for L in range(nL)}
        Lstar = max(rsa_layer, key=lambda L: (rsa_layer[L] if rsa_layer[L] == rsa_layer[L] else -9))
        clean_rsa = rsa_layer[Lstar]
        print(f"[{tag}] L*={Lstar} clean best2dRSA={clean_rsa:+.2f} clean nbr_mass={clean_mass:.2f}; "
              f"sweeping {(Lstar+1)*nH} heads (layers 0..{Lstar})", flush=True)

        dmg = np.full((nL, nH), np.nan); dbeh = np.full((nL, nH), np.nan)
        for l in range(Lstar + 1):
            for h in range(nH):
                m2, mass2 = run(model, tok, blocks, cm, walks, graph, cand_t, dev, [Lstar], ablate=(l, h))
                dmg[l, h] = clean_rsa - best2d_rsa(m2[Lstar], Gc, GD, iu)   # geometry damage
                dbeh[l, h] = clean_mass - mass2                              # behaviour damage
            print(f"[{tag}] layer {l}/{Lstar} done; running max geom-damage={np.nanmax(dmg):+.3f}", flush=True)

        flat = np.argsort(np.nan_to_num(dmg, nan=-9), axis=None)[::-1][:10]
        top = [{"layer": int(i // nH), "head": int(i % nH), "geom_damage": round(float(dmg.flatten()[i]), 3),
                "behav_damage": round(float(dbeh.flatten()[i]), 3)} for i in flat]
        out["models"][tag] = {"Lstar": Lstar, "clean_rsa": clean_rsa, "clean_mass": clean_mass,
                              "geom_damage": np.where(np.isnan(dmg), None, dmg).tolist(),
                              "behav_damage": np.where(np.isnan(dbeh), None, dbeh).tolist(), "top_geometry_heads": top}
        print(f"[{tag}] TOP geometry-damaging head: L{top[0]['layer']}H{top[0]['head']} "
              f"drops RSA by {top[0]['geom_damage']:+.2f} (behav {top[0]['behav_damage']:+.2f})", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/head_sweep_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/head_sweep_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]
            dmg = np.array([[np.nan if v is None else v for v in row] for row in r["geom_damage"]])
            beh = np.array([[np.nan if v is None else v for v in row] for row in r["behav_damage"]])
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            vlim = max(0.05, float(np.nanmax(np.abs(dmg))))
            im = ax[0].imshow(dmg, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            t = r["top_geometry_heads"][0]; ax[0].plot(t["head"], t["layer"], "k*", ms=12)
            ax[0].set_title(f"{m}: best-2D RSA damage per head (L*={r['Lstar']}, clean {r['clean_rsa']:.2f})\n"
                            f"top: L{t['layer']}H{t['head']} ΔRSA={t['geom_damage']:+.2f}", fontsize=9)
            ax[0].set_xlabel("head"); ax[0].set_ylabel("layer"); fig.colorbar(im, ax=ax[0], fraction=.046, label="geometry damage")
            ax[1].scatter(beh.flatten(), dmg.flatten(), s=10, alpha=.4)
            for h in r["top_geometry_heads"][:5]:
                ax[1].annotate(f"L{h['layer']}H{h['head']}", (h["behav_damage"], h["geom_damage"]), fontsize=6, color="red")
            ax[1].axhline(0, color=".8", lw=.6); ax[1].axvline(0, color=".8", lw=.6)
            ax[1].set_xlabel("behaviour damage (Δ nbr mass)"); ax[1].set_ylabel("geometry damage (Δ best-2D RSA)")
            ax[1].set_title(f"{m}: geometry vs behaviour damage per head", fontsize=9)
            fig.suptitle(f"{m} [{out['graph']}]: single-head ablation — which head builds the geometry?", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
