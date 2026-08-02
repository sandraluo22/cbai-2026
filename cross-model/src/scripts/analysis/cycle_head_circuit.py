"""Circuit HEADS for in-context cycles of different node lengths. For each ring size, ablate every attention
head (zero its o_proj-input slice) one at a time and measure the drop in next-node neighbour validity ->
a per-(layer,head) causal-importance map. Then compare maps across sizes: do different-node-length cycles
recruit the SAME heads (one circuit) or different heads? Also a cheap per-layer MLP-ablation sweep.

Batched: all walks of a structure run in ONE padded forward per head, so 1024 heads is affordable.

Env: GEN_MODEL(Llama) SIZES(4,6,8,12,16) NWALKS(6) WLEN(200) CTXLO(100) TOPK(20) OUTDIR DEVICE
Out: <OUTDIR>/cycle_head_circuit_<model>.json  (head_map[size][L][H], mlp_map, base, overlap-ready top-K)
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
SIZES = [int(x) for x in os.environ.get("SIZES", "4,6,8,12,16").split(",")]
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "200"))
CTXLO = int(os.environ.get("CTXLO", "100")); TOPK = int(os.environ.get("TOPK", "20"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")
HEADS_LIMIT = os.environ.get("HEADS_LIMIT")  # e.g. "0,1" to test only layers 0-1 (smoke)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def prep_batch(tok, graph, walks, dev):
    """Precompute padded ids/mask + per-walk readout (span_end, node) list restricted to ctx>=CTXLO."""
    seqs, readouts = [], []
    for wk in walks:
        ids = tok(wk.text, add_special_tokens=True)["input_ids"]
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes
        ro = [(spans[s][-1], nodes[s]) for s in range(len(nodes) - 1) if (s + 1) >= CTXLO]
        seqs.append(ids); readouts.append(ro)
    ml = max(len(s) for s in seqs); B = len(seqs)
    pad = tok.pad_token_id or 0
    batch = torch.full((B, ml), pad, device=dev, dtype=torch.long)
    mask = torch.zeros((B, ml), device=dev, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s, device=dev); mask[i, :len(s)] = 1
    cand = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(graph.n_nodes)], device=dev)
    nbrs = [set(graph.neighbors(i)) for i in range(graph.n_nodes)]
    return batch, mask, readouts, cand, nbrs


@torch.no_grad()
def validity(model, batch, mask, readouts, cand, nbrs):
    """Continuous next-node score: softmax over the candidate node-words, prob mass on the true neighbours,
    averaged over readout positions. Sensitive to single-head ablation (unlike the argmax hit-rate)."""
    logits = model(input_ids=batch, attention_mask=mask).logits.float()  # [B,T,V]
    tot = 0.0; s = 0.0
    for b, ro in enumerate(readouts):
        for pos, node in ro:
            p = torch.softmax(logits[b, pos][cand], 0)
            s += float(p[list(nbrs[node])].sum()); tot += 1
    return s / max(tot, 1)


def head_hook(proj, hd, h, dev):
    ct = torch.arange(h * hd, (h + 1) * hd, device=dev, dtype=torch.long)
    def pre(_m, args):
        x = args[0].clone(); x[..., ct] = 0
        return (x,) + tuple(args[1:])
    return proj.register_forward_pre_hook(pre)


def mlp_hook(block):
    def h(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        return (torch.zeros_like(o),) + tuple(out[1:]) if isinstance(out, tuple) else torch.zeros_like(o)
    return block.mlp.register_forward_hook(h)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    layers = list(range(nL)) if not HEADS_LIMIT else [int(x) for x in HEADS_LIMIT.split(",")]

    out = {"model": tag, "sizes": SIZES, "nL": nL, "nH": nH, "base": {}, "head_map": {}, "mlp_map": {}, "topk": {}}
    for nn in SIZES:
        cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=nn, n_walks=NWALKS, walk_length=WLEN, device=dev)
        ring = G.build_graph(cfg); walks = G.generate_walks(ring, cfg)
        batch, mask, readouts, cand, nbrs = prep_batch(tok, ring, walks, dev)
        base = validity(model, batch, mask, readouts, cand, nbrs); out["base"][nn] = round(base, 4)
        hm = np.zeros((nL, nH), np.float32)
        for L in layers:
            proj, hd = attn_proj(blocks[L], cm)
            for h in range(nH):
                hh = head_hook(proj, hd, h, dev)
                hm[L, h] = base - validity(model, batch, mask, readouts, cand, nbrs)
                hh.remove()
        mm = np.zeros(nL, np.float32)
        for L in layers:
            hh = mlp_hook(blocks[L]); mm[L] = base - validity(model, batch, mask, readouts, cand, nbrs); hh.remove()
        out["head_map"][nn] = hm.tolist(); out["mlp_map"][nn] = mm.tolist()
        flat = [(float(hm[L, h]), L, h) for L in layers for h in range(nH)]
        flat.sort(reverse=True)
        out["topk"][nn] = [[L, h, round(v, 4)] for v, L, h in flat[:TOPK]]
        print(f"[{tag}] ring-{nn:2d}: base={base:.3f}  top head L{flat[0][1]}H{flat[0][2]} drop={flat[0][0]:.3f}  "
              f"top5={[(L,h) for _,L,h in flat[:5]]}", flush=True)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/cycle_head_circuit_{tag}.json"
    json.dump(out, open(p, "w")); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
