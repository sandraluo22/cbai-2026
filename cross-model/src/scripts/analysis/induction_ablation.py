"""(1) Induction-head ABLATION: is induction the whole story, or just the local
front-end while a separate circuit builds the global geometry?

Ablate each model's top-K GENERIC induction heads (zero their attention-output
projection contribution) and re-measure, vs a clean run and a RANDOM-head ablation
control of the same size:
  (a) next-step neighbour accuracy   (behaviour)
  (b) node-mean RSA per layer        (the global geometry)

If geometry SURVIVES induction-head ablation -> there's a separate geometry circuit.
If it COLLAPSES -> the geometry is a readout of induction. Either is a real result.

Reads the top induction heads from induction-head/induction.json (generic probe).
Runs on the pod; PRESET=smoke uses distilgpt2 (CPU, c_proj head ablation).

Env: PRESET MODELS_FILTER GRAPHS(square_grid,days) ABLATE_K(8) NWALKS(20) WLEN(300)
     CTXLO(100) OUTDIR DEVICE  INDJSON(path to induction.json)
Out: <OUTDIR>/ablation.json , <OUTDIR>/induction_ablation.pdf
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
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPHS = os.environ.get("GRAPHS", "square_grid,days").split(",")
ABLATE_K = int(os.environ.get("ABLATE_K", "15"))
NWALKS = int(os.environ.get("NWALKS", "20"))
WLEN   = int(os.environ.get("WLEN", "300"))
CTXLO  = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/ablation" if PRESET != "smoke" else "runs/smoke_ablation")
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json"
                         if PRESET != "smoke" else "runs/smoke_induction/induction.json")
CKPTS = [20, 60, 150, 250]      # all < WLEN-1 so each has a valid transition
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception as e:
        if mirror:
            print(f"[{tag}] {hf} unavailable ({type(e).__name__}); mirror {mirror}", flush=True)
            return M.load_model(mirror, cfg)
        raise


def attn_proj(block, cfg_model):
    """Return (out_proj_module, head_dim, n_heads) for the attention output projection,
    handling Llama/Gemma/Qwen (self_attn.o_proj) and GPT-2 (attn.c_proj)."""
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        nh = cfg_model.num_attention_heads
        hd = getattr(cfg_model, "head_dim", None) or (cfg_model.hidden_size // nh)
        return block.self_attn.o_proj, hd, nh
    if hasattr(block, "attn") and hasattr(block.attn, "c_proj"):                 # GPT-2
        nh = cfg_model.n_head; hd = cfg_model.hidden_size // nh
        return block.attn.c_proj, hd, nh
    raise AttributeError("no known attention output projection")


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GD, iu):
    """Supervised best-2D RSA: top-6 node-mean PCs regressed onto the graph layout coords
    (as in best_2d.py), RDM of that 2-D embedding vs graph distance. This recovers the
    ring even when raw Euclidean RSA is swamped (the 'probed geometry')."""
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)                                   # n x 2 supervised embedding
    R = np.linalg.norm(P[:, None] - P[None], axis=2)[iu]
    return sp(R, GD)


@torch.no_grad()
def run_condition(model, tok, blocks, cfg_model, walks, graph, words, cand_t, dev,
                  ablate_by_layer, cap_layers):
    """One pass: zero ablated heads (o_proj input mask), capture post-block residuals
    (accumulate per-node sums for ctx>=CTXLO), tally next-step neighbour accuracy."""
    n = graph.n_nodes
    handles = []
    # ablation: zero head slices on the attention output projection input
    for L, heads in ablate_by_layer.items():
        proj, hd, nh = attn_proj(blocks[L], cfg_model)
        cols = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]) if heads else np.array([], int)
        ct = torch.tensor(cols, device=dev, dtype=torch.long)
        def pre(mod, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    # residual capture
    grabbed = {}
    def mk(l):
        def h(_m, _i, out): grabbed[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return h
    for L in cap_layers:
        handles.append(blocks[L].register_forward_hook(mk(L)))

    nsum = {L: np.zeros((n, cfg_model.hidden_size)) for L in cap_layers}
    ncnt = {L: np.zeros(n) for L in cap_layers}
    acc = {C: {"correct": 0, "total": 0, "mass": 0.0, "exact": 0.0} for C in CKPTS}
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]               # last-subword token per step
            cl = np.arange(1, len(nodes) + 1)
            for L in cap_layers:
                hs = grabbed[L][0]
                rows = hs[single].float().cpu().numpy()    # [n_occ, d]
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]; ncnt[L][nodes[s]] += 1
            # next-step prediction at the checkpoints (argmax acc + graded neighbour mass + exact)
            for C in CKPTS:
                s = C - 1
                if 0 <= s <= len(nodes) - 2:
                    pos = spans[s + 1][0] - 1
                    p = torch.softmax(logits[pos][cand_t].float(), 0).cpu().numpy()
                    nb = graph.neighbors(nodes[s])
                    acc[C]["correct"] += int(int(p.argmax()) in nb); acc[C]["total"] += 1
                    acc[C]["mass"] += float(p[nb].sum()); acc[C]["exact"] += float(p[nodes[s + 1]])
    finally:
        for h in handles:
            h.remove()

    iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]
    Gc = np.array(graph.coords, float)
    rsa, rsa_b2d = {}, {}
    for L in cap_layers:
        H = np.where(ncnt[L][:, None] > 0, nsum[L] / np.maximum(ncnt[L][:, None], 1), np.nan)
        if np.isnan(H).any():
            rsa[L] = float("nan"); rsa_b2d[L] = float("nan"); continue
        R = np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
        rsa[L] = sp(R, GD)                                    # raw Euclidean node-mean RSA
        rsa_b2d[L] = best2d_rsa(H, Gc, GD, iu)                # probed (supervised best-2D) RSA
    acc_by_ctx = {C: {"acc": (acc[C]["correct"] / acc[C]["total"] if acc[C]["total"] else float("nan")),
                      "neighbor_mass": (acc[C]["mass"] / acc[C]["total"] if acc[C]["total"] else float("nan")),
                      "exact": (acc[C]["exact"] / acc[C]["total"] if acc[C]["total"] else float("nan"))}
                  for C in CKPTS}
    return rsa, rsa_b2d, acc_by_ctx


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"]
    out = {"ablate_k": ABLATE_K, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), device=dev)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        cap_layers = sorted(set(int(round(r * (nL - 1))) for r in np.linspace(0.1, 0.95, 10)))
        # induction heads to ablate: top-K from the FULL generic-score matrix
        gen = np.array(ind[tag]["generic"])                      # [L, H]
        flat = np.argsort(gen, axis=None)[::-1][:ABLATE_K]
        ind_heads = [(int(i // gen.shape[1]), int(i % gen.shape[1])) for i in flat]
        allh = [(l, h) for l in range(nL) for h in range(nH)]
        pool = [x for x in allh if x not in set(ind_heads)]
        rand_heads = [pool[i] for i in RNG.choice(len(pool), ABLATE_K, replace=False)]
        def by_layer(heads):
            d = {}
            for l, h in heads:
                d.setdefault(l, []).append(h)
            return d
        conds = {"clean": {}, "ablate_induction": by_layer(ind_heads), "ablate_random": by_layer(rand_heads)}

        rec = {"n_layers": nL, "n_heads": nH, "cap_layers": cap_layers,
               "ablated_induction": ind_heads, "ablated_random": rand_heads, "graphs": {}}
        for gname in GRAPHS:
            gcfg = replace(cfg, **GKW[gname], n_walks=NWALKS, walk_length=WLEN)
            graph = G.build_graph(gcfg); words = graph.words
            walks = G.generate_walks(graph, gcfg)
            cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
            gres = {}
            for cname, abl in conds.items():
                rsa, rsa_b2d, accc = run_condition(model, tok, blocks, cm, walks, graph, words, cand_t, dev, abl, cap_layers)
                gres[cname] = {"rsa_by_layer": rsa, "best2d_by_layer": rsa_b2d, "acc_by_ctx": accc}
                pk = max((v for v in rsa.values() if np.isfinite(v)), default=float("nan"))
                pkb = max((v for v in rsa_b2d.values() if np.isfinite(v)), default=float("nan"))
                a250 = accc[250]
                print(f"[{tag}/{gname}/{cname}] rawRSA={pk:+.2f} best2dRSA={pkb:+.2f}  "
                      f"nbr_mass@250={a250['neighbor_mass']:.2f}", flush=True)
            rec["graphs"][gname] = gres
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(OUTDIR, exist_ok=True)
    prev = f"{OUTDIR}/ablation.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/induction_ablation.pdf")
    print(f"DONE -> {prev} + induction_ablation.pdf", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    graphs = sorted({g for m in models for g in out["models"][m]["graphs"]})
    colors = {"clean": "k", "ablate_induction": "tab:red", "ablate_random": "tab:blue"}
    with PdfPages(path) as pdf:
        for gname in graphs:
            fig, ax = plt.subplots(len(models), 3, figsize=(16, 3.4 * len(models)), squeeze=False)
            for row, m in enumerate(models):
                gr = out["models"][m]["graphs"].get(gname)
                if not gr:
                    continue
                for cname, c in colors.items():
                    def curve(key):
                        D = gr[cname][key]; Ls = sorted(int(k) for k in D)
                        return Ls, [D[str(L)] if str(L) in D else D[L] for L in Ls]
                    Ls, yr = curve("rsa_by_layer"); ax[row, 0].plot(Ls, yr, "-o", ms=3, color=c, label=cname)
                    Ls, yb = curve("best2d_by_layer"); ax[row, 1].plot(Ls, yb, "-o", ms=3, color=c, label=cname)
                    acc = gr[cname]["acc_by_ctx"]; Cs = sorted(int(k) for k in acc)
                    ya = [(acc[str(C)] if str(C) in acc else acc[C])["neighbor_mass"] for C in Cs]
                    ax[row, 2].plot(Cs, ya, "-o", ms=3, color=c, label=cname)
                ax[row, 0].set_title(f"{m}  raw node-mean RSA", fontsize=9)
                ax[row, 0].set_xlabel("layer"); ax[row, 0].set_ylabel("RSA"); ax[row, 0].legend(fontsize=6)
                ax[row, 1].set_title(f"{m}  PROBED best-2D RSA (supervised)", fontsize=9)
                ax[row, 1].set_xlabel("layer"); ax[row, 1].set_ylabel("best-2D RSA")
                ax[row, 2].set_title(f"{m}  next-step neighbour MASS (graded)", fontsize=9)
                ax[row, 2].set_xlabel("context length"); ax[row, 2].set_ylabel("neighbour mass"); ax[row, 2].set_ylim(0, 1.05)
            fig.suptitle(f"Induction-head ablation [{gname}]: raw geometry | PROBED best-2D geometry | behaviour\n"
                         "black = clean, red = ablate induction heads, blue = ablate random heads", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
