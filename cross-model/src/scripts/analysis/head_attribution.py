"""Direct logit attribution (DLA) for the next-step prediction: how much does each
attention head (and each MLP) DIRECTLY contribute to the logit of the correct next node?

At each readout token (node X at ctx>=CTXLO, predicting node_{s+1}=T), decompose the
direct path to logits: per head h, contribution = z_h . W_O_h (its o_proj-input slice
through W_O), LN-folded, unembedded, restricted to the node-words, centred; the attribution
is that head's push on T. Same for each MLP output. Aggregated over occurrences ->
per-(layer,head) attention DLA, per-layer MLP DLA, and the attention-vs-MLP split.

Expectation from the OV finding: attention heads write ~0 to the correct-node logit
(they move a feature); the MLPs complete the prediction.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(12) WLEN(300) CTXLO(100) INDJSON OUTDIR
Out: <OUTDIR>/head_attribution_<graph>.json + .pdf
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
       "days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "12"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/attribution")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def blocks_of(model):
    return model.model.layers if (hasattr(model, "model") and hasattr(model.model, "layers")) else model.transformer.h


def oproj_of(b, gpt2):
    return b.attn.c_proj if gpt2 else b.self_attn.o_proj


def mlp_of(b):
    return b.mlp


def final_norm(model):
    return model.model.norm if (hasattr(model, "model") and hasattr(model.model, "norm")) else model.transformer.ln_f


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; words = graph.words
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = blocks_of(model); gpt2 = not hasattr(blocks[0], "self_attn")
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH)
        WU = model.get_output_embeddings().weight.float()
        fn = final_norm(model); is_rms = "rms" in type(fn).__name__.lower(); gamma = fn.weight.detach().float()
        if "gemma" in (getattr(cm, "model_type", "") or "").lower():
            gamma = 1.0 + gamma
        cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
        WUn = WU[cand]                                              # [n, d] node-word unembeddings

        def dla(vec_d):                                             # residual delta [d] -> centred node logits
            c = vec_d.float()
            if not is_rms:
                c = c - c.mean()
            c = c * gamma
            g = (WUn @ c)                                           # [n]
            return (g - g.mean()).cpu().numpy()

        # capture o_proj inputs (per-head) and mlp outputs per layer
        zc, mc = {}, {}
        def mkz(L):
            def pre(_m, args): zc[L] = args[0].detach()
            return pre
        def mkm(L):
            def h(_m, _i, o): mc[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        handles = []
        for L in range(nL):
            handles.append(oproj_of(blocks[L], gpt2).register_forward_pre_hook(mkz(L)))
            handles.append(mlp_of(blocks[L]).register_forward_hook(mkm(L)))

        head_attr = np.zeros((nL, nH)); mlp_attr = np.zeros(nL); cnt = 0
        try:
            for wk in walks:
                ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
                spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
                zc.clear(); mc.clear(); model(input_ids=ids)
                for s in range(len(nodes) - 1):
                    if cl[s] < CTXLO:
                        continue
                    p = spans[s + 1][0] - 1; T = nodes[s + 1]
                    for L in range(nL):
                        z = zc[L][0, p].float().view(nH, hd)                  # [nH, hd]
                        Wo = oproj_of(blocks[L], gpt2).weight.float()
                        if gpt2:                                             # Conv1D [nH*hd, d]
                            c_all = torch.einsum('nh,nhd->nd', z, Wo.view(nH, hd, -1))
                        else:                                                # Linear [d, nH*hd]
                            c_all = torch.einsum('nh,dnh->nd', z, Wo.view(-1, nH, hd))
                        if not is_rms:
                            c_all = c_all - c_all.mean(1, keepdim=True)
                        c_all = c_all * gamma                                 # [nH, d]
                        g = c_all @ WUn.T                                     # [nH, n]
                        g = g - g.mean(1, keepdim=True)
                        head_attr[L] += g[:, T].cpu().numpy()
                        mlp_attr[L] += dla(mc[L][0, p])[T]
                    cnt += 1
        finally:
            for hnd in handles:
                hnd.remove()
        head_attr /= max(cnt, 1); mlp_attr /= max(cnt, 1)
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        top = np.argsort(head_attr, axis=None)[::-1][:8]
        rec = {"n_layers": nL, "n_heads": nH, "head_attr": head_attr.tolist(), "mlp_attr": mlp_attr.tolist(),
               "attn_total": float(head_attr.sum()), "mlp_total": float(mlp_attr.sum()),
               "top_heads": [{"layer": int(i // nH), "head": int(i % nH), "attr": round(float(head_attr.flatten()[i]), 3),
                              "qk": round(float(gen.flatten()[i]), 3)} for i in top]}
        out["models"][tag] = rec
        print(f"[{tag}] direct-logit attribution to correct next node: attn total={rec['attn_total']:+.2f} "
              f"mlp total={rec['mlp_total']:+.2f} | top head L{rec['top_heads'][0]['layer']}H{rec['top_heads'][0]['head']} "
              f"attr={rec['top_heads'][0]['attr']:+.2f} (qk {rec['top_heads'][0]['qk']:.2f})", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/head_attribution_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/head_attribution_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; ha = np.array(r["head_attr"]); ma = np.array(r["mlp_attr"])
            fig, ax = plt.subplots(1, 3, figsize=(17, 5))
            v = max(0.02, float(np.nanmax(np.abs(ha))))
            im = ax[0].imshow(ha, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ax[0].set_xlabel("head"); ax[0].set_ylabel("layer"); ax[0].set_title(f"{m}: per-head DLA to correct next node", fontsize=9)
            fig.colorbar(im, ax=ax[0], fraction=.046)
            ax[1].plot(ma, range(len(ma)), "-o", ms=3, color="tab:purple", label="MLP DLA")
            ax[1].plot(ha.sum(1), range(len(ha)), "-o", ms=3, color="tab:blue", label="attn DLA (layer sum)")
            ax[1].axvline(0, color=".8", lw=.6); ax[1].set_ylabel("layer"); ax[1].set_xlabel("DLA to correct node"); ax[1].legend(fontsize=8)
            ax[1].set_title(f"{m}: per-layer attention vs MLP DLA", fontsize=9)
            ax[2].bar(["attn total", "mlp total"], [r["attn_total"], r["mlp_total"]], color=["tab:blue", "tab:purple"])
            ax[2].axhline(0, color=".8", lw=.6); ax[2].set_ylabel("summed DLA to correct node")
            ax[2].set_title(f"{m}: who writes the answer? (attn {r['attn_total']:+.1f} vs mlp {r['mlp_total']:+.1f})", fontsize=9)
            fig.suptitle(f"{m} [{out['graph']}]: direct logit attribution of the next-step prediction", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
