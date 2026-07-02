"""What does a QK (prefix-matching) head OUTPUT when fed a node?

These heads don't copy the attended token (OV self-boost is negative), so we characterise
the full output over node-words. During the task we capture each head's per-node VALUE
vector (v_proj output, real LayerNorm applied), average per node, push through the head's
W_O and the (LN-folded) unembedding, and read the distribution over the n node-words:

    M[X, Y] = P(head writes node Y  |  fed node X)          (softmax over node-words)

Then: self mass (copying), neighbour mass (does it write the graph neighbours?), and the
argmax structure. Top-K QK heads per model (from induction.json).

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(10) WLEN(300) CTXLO(100) TOPK(4)
     INDJSON OUTDIR DEVICE
Out: <OUTDIR>/node_output_<graph>.json , node_output_<graph>.pdf
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
NWALKS = int(os.environ.get("NWALKS", "10"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
TOPK = int(os.environ.get("TOPK", "4"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/node_output")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def vproj_of(block, is_gpt2):
    return block.attn.c_attn if is_gpt2 else block.self_attn.v_proj


def wo_of(block, is_gpt2):
    return block.attn.c_proj if is_gpt2 else block.self_attn.o_proj


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; words = graph.words
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); is_gpt2 = not hasattr(blocks[0], "self_attn")
        nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        nkv = getattr(cm, "num_key_value_heads", None) or nH
        hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH); group = nH // nkv
        WU = model.get_output_embeddings().weight.float()
        fn = model.model.norm if hasattr(model, "model") else model.transformer.ln_f
        is_rms = "rms" in type(fn).__name__.lower(); gamma = fn.weight.detach().float()
        if "gemma" in (getattr(cm, "model_type", "") or "").lower():
            gamma = 1.0 + gamma
        cand = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)

        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((cm.num_hidden_layers, nH))))
        topheads = [(int(i // nH), int(i % nH)) for i in np.argsort(gen, axis=None)[::-1][:TOPK]]
        need_layers = sorted({l for l, _ in topheads})

        # capture per-head node-mean VALUE vectors during the task
        vcap = {}
        def mkv(L):
            def h(_m, _i, o): vcap[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        handles = [vproj_of(blocks[L], is_gpt2).register_forward_hook(mkv(L)) for L in need_layers]
        vsum = {L: None for L in need_layers}; vcnt = np.zeros(n)
        walks = G.generate_walks(graph, cfg)
        try:
            for wk in walks:
                ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
                spans = resolve_token_spans(tok, wk); nodes = wk.nodes; vcap.clear()
                model(input_ids=ids)
                single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
                for L in need_layers:
                    Vall = vcap[L][0]
                    if is_gpt2:
                        Vall = Vall[:, 2 * cm.hidden_size:]
                    rows = Vall[single].float().cpu().numpy()          # [n_occ, nkv*hd]
                    if vsum[L] is None:
                        vsum[L] = np.zeros((n, rows.shape[1]))
                    for s in range(len(nodes)):
                        if cl[s] >= CTXLO:
                            vsum[L][nodes[s]] += rows[s]
                            if L == need_layers[0]:
                                vcnt[nodes[s]] += 1
        finally:
            for hnd in handles:
                hnd.remove()
        # recompute counts per layer robustly (same mask -> same counts)
        cnts = np.maximum(vcnt, 1)
        vmean = {L: vsum[L] / cnts[:, None] for L in need_layers}

        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()
        deg = np.mean([len(graph.neighbors(i)) for i in range(n)])
        recs = []
        for (l, h) in topheads:
            kv = h // group
            vh = torch.tensor(vmean[l][:, kv * hd:(kv + 1) * hd], device=dev, dtype=WU.dtype)  # [n, hd]
            Wo = wo_of(blocks[l], is_gpt2).weight
            ov = (vh @ Wo[h * hd:(h + 1) * hd, :].float()) if is_gpt2 else (vh @ Wo[:, h * hd:(h + 1) * hd].T.float())
            if not is_rms:
                ov = ov - ov.mean(-1, keepdim=True)
            g = (ov * gamma) @ WU.T                                     # [n, V]
            Mnode = torch.softmax(g[:, cand].float(), dim=1).cpu().numpy()   # [n, n] output dist over nodes
            self_p = float(np.mean(np.diag(Mnode)))
            nbr_p = float(np.mean([Mnode[X, graph.neighbors(X)].sum() for X in range(n)]))
            argmax_is_nbr = float(np.mean([int(Mnode[X].argmax() in graph.neighbors(X)) for X in range(n)]))
            argmax_is_self = float(np.mean([int(Mnode[X].argmax() == X) for X in range(n)]))
            # does the output-distance structure match the graph? RSA of M-rows' distances
            recs.append({"layer": l, "head": h, "qk": round(float(gen[l, h]), 3),
                         "self_mass": round(self_p, 3), "neighbor_mass": round(nbr_p, 3),
                         "chance_neighbor": round(float(deg / n), 3),
                         "argmax_is_neighbor": round(argmax_is_nbr, 2), "argmax_is_self": round(argmax_is_self, 2),
                         "M": Mnode.tolist()})
            print(f"[{tag}] L{l}H{h} (qk{gen[l,h]:.2f}): self={self_p:.2f} nbr={nbr_p:.2f} "
                  f"(chance {deg/n:.2f}) argmax->nbr {argmax_is_nbr:.0%} self {argmax_is_self:.0%}", flush=True)
        out["models"][tag] = {"n": n, "words": words, "heads": recs}
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/node_output_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/node_output_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; heads = r["heads"]; words = r["words"]
            fig, ax = plt.subplots(1, len(heads), figsize=(4.2 * len(heads), 4.4), squeeze=False)
            for j, hd in enumerate(heads):
                Mx = np.array(hd["M"]); a = ax[0, j]
                im = a.imshow(Mx, cmap="magma", aspect="auto")
                a.set_xticks(range(len(words))); a.set_xticklabels(words, rotation=90, fontsize=5)
                a.set_yticks(range(len(words))); a.set_yticklabels(words, fontsize=5)
                a.set_xlabel("output node"); a.set_ylabel("input node")
                a.set_title(f"L{hd['layer']}H{hd['head']} qk{hd['qk']:.2f}\nself{hd['self_mass']:.2f} "
                            f"nbr{hd['neighbor_mass']:.2f} (ch{hd['chance_neighbor']:.2f})", fontsize=8)
                fig.colorbar(im, ax=a, fraction=.046)
            fig.suptitle(f"{m} [{out['graph']}]: top-QK heads — output distribution over nodes M[in,out]\n"
                         "(diagonal=copy self; bright off-diagonal=writes other nodes)", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
