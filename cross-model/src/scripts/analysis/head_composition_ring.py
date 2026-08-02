"""Does one head control WHERE other heads look? (the indirect / routing role)

A head with no axis content and no direct logit path can still be the largest behavioural contributor,
which leaves exactly one mechanism: it changes what other heads attend to. This measures that directly.
For each SOURCE head: mean-ablate it, then recompute every TARGET head's on-task attention motif profile
(same_token / induction / prev_token, defined at readout positions against previous visits to the current
node) and report the change. Controls ablate a random head AT THE SOURCE'S OWN LAYER, so "ablating
something at L14 perturbs L16" is subtracted off and only the source's specific effect remains.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAZY(0) SOURCES("L14H26,...") TARGETS("L16H20,...")
     NWALKS(4) WLEN(1000) CTXLO(600) NRAND(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/head_composition_ring<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama"); K = int(os.environ.get("K", "16"))
GRAPH = os.environ.get("GRAPH", "ring"); LAZY = float(os.environ.get("LAZY", "0"))
SOURCES = [h for h in os.environ.get("SOURCES", "").split(",") if h]
TARGETS = [h for h in os.environ.get("TARGETS", "").split(",") if h]
NWALKS = int(os.environ.get("NWALKS", "4")); WLEN = int(os.environ.get("WLEN", "1000"))
CTXLO = int(os.environ.get("CTXLO", "600")); NRAND = int(os.environ.get("NRAND", "3"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev),)
    # SDPA silently returns no attention weights, so output_attentions=True yields all-zero motifs.
    # Eager attention is required for this measurement; assert rather than trust the flag.
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)
    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    tgt = [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in TARGETS]
    tl = sorted({l for l, _ in tgt})

    st = {"abl": None, "means": None}
    attn = {}
    hooks = []
    for L in range(nL):
        proj, _ = attn_proj(blocks[L], cm)
        def pre(_m, a, L=L):
            x = a[0]
            if st["abl"] is not None:
                hs = [h for (l, h) in st["abl"] if l == L]
                if hs:
                    x = x.clone()
                    for h in hs:
                        sl = slice(h * hd, (h + 1) * hd)
                        x[0, :, sl] = st["means"][L][sl].to(x.dtype)
            if st["means"] is None: st.setdefault("cap", {})[L] = x.detach()
            return (x,) + tuple(a[1:])
        hooks.append(proj.register_forward_pre_hook(pre))
    for L in tl:
        def ah(_m, _i, out, L=L):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is not None: attn[L] = w.detach()
        hooks.append(blocks[L].self_attn.register_forward_hook(ah))

    lr_ = np.random.default_rng(SEED); seqs = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(WLEN - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        seqs.append((torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev), nodes))

    def motifs():
        acc = {t: np.zeros(3) for t in tgt}; m = 0
        for ids, nodes in seqs:
            attn.clear()
            model(input_ids=ids, output_attentions=True)
            for si in range(len(nodes)):
                if si + 1 < CTXLO or si >= len(nodes) - 1: continue
                u = nodes[si]; t = si + 1
                prev = [s + 1 for s in range(si) if nodes[s] == u]
                if not prev: continue
                succ = [p + 1 for p in prev if p + 1 <= t - 1]
                for (L, h) in tgt:
                    if L not in attn: continue
                    a = attn[L][0, h, t].float()
                    acc[(L, h)] += np.array([float(a[prev].sum()), float(a[succ].sum()) if succ else 0.0,
                                             float(a[t - 1])])
                m += 1
        return {t: acc[t] / max(m, 1) for t in tgt}

    st["abl"] = None; st["means"] = None
    base = motifs()
    assert max(float(v.sum()) for v in base.values()) > 1e-6, \
        "all target motifs are zero — attention weights were not returned (SDPA?); need eager attention"
    st["means"] = {L: st["cap"][L][0].float().mean(0) for L in st["cap"]}
    res = {"model": tag, "sources": SOURCES, "targets": TARGETS,
           "baseline": {f"L{l}H{h}": [round(x, 4) for x in base[(l, h)]] for l, h in tgt}, "rows": {}}
    print(f"[{tag}] baseline motifs (same_token, induction, prev_token):")
    for l, h in tgt: print(f"   L{l}H{h:<3} {base[(l,h)].round(4)}", flush=True)
    for s in SOURCES:
        sl_, sh = int(s.split("H")[0][1:]), int(s.split("H")[1])
        st["abl"] = [(sl_, sh)]; ab = motifs()
        rnd = []
        for _ in range(NRAND):
            st["abl"] = [(sl_, int(rng.choice([x for x in range(nH) if x != sh])))]
            rnd.append(motifs())
        st["abl"] = None
        row = {}
        print(f"\nablate {s} -> change in target motifs (delta vs random head at L{sl_}):")
        print(f"{'target':<9}{'d same_tok':>12}{'d induction':>13}{'d prev_tok':>12}   (random-source deltas)")
        for l, h in tgt:
            if (l, h) == (sl_, sh): continue
            d = ab[(l, h)] - base[(l, h)]
            dr = np.mean([r[(l, h)] - base[(l, h)] for r in rnd], axis=0)
            row[f"L{l}H{h}"] = {"delta": [round(x, 4) for x in d], "rand_delta": [round(x, 4) for x in dr]}
            print(f"L{l}H{h:<7}{d[0]:12.4f}{d[1]:13.4f}{d[2]:12.4f}   "
                  f"({dr[0]:+.4f}, {dr[1]:+.4f}, {dr[2]:+.4f})", flush=True)
        res["rows"][s] = row
        json.dump(res, open(f"{OUTDIR}/head_composition_ring{OUTTAG}_{tag}.json", "w"), indent=2)
    for hk in hooks: hk.remove()
    print(f"\nDONE -> {OUTDIR}/head_composition_ring{OUTTAG}_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
