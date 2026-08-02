"""HOW does a parity head compute parity? On a bipartite grid walk, node parity alternates every step, so a
head tracking it should show attention structured by LAG PARITY: from a node-token query, attention mass to
node tokens an EVEN number of steps back lands on same-colour nodes, ODD lag on opposite-colour. We measure,
for the parity-writer heads vs coord/QK-induction/DLA heads vs random heads, the attention mass split by
(a) lag parity (even vs odd step distance), (b) source node colour (same vs opposite as the query node), and
(c) same-word attention (induction-style token matching). An induction-like mod-2 motif = strong same-word +
lag-even preference; a genuine colour-class reader = same/opposite-colour asymmetry beyond word matching.

Env: GEN_MODEL(Llama) K(4) NWALKS(6) WLEN(600) CTXLO(150)
     HEADS("2:26,14:26,14:19,21:10,16:20,15:30,25:7") NRANDHEAD(6) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/parity_attention<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "6"))
WLEN = int(os.environ.get("WLEN", "600")); CTXLO = int(os.environ.get("CTXLO", "150"))
HEADS = [tuple(int(x) for x in h.split(":")) for h in
         os.environ.get("HEADS", "2:26,14:26,14:19,21:10,16:20,15:30,25:7").split(",")]
NRANDHEAD = int(os.environ.get("NRANDHEAD", "6")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

LABELS = {(2, 26): "parity-writer", (14, 26): "parity-writer", (14, 19): "parity-writer",
          (21, 10): "coord-writer", (16, 20): "QK-induction", (15, 30): "QK-induction", (25, 7): "DLA-reader"}


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config
    # sdpa/flash return no attention weights — force the eager path so output_attentions works
    try: model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    rng = np.random.default_rng(SEED)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    rand_heads = []
    while len(rand_heads) < NRANDHEAD:
        h = (int(rng.integers(nL)), int(rng.integers(nH)))
        if h not in HEADS and h not in rand_heads: rand_heads.append(h)
    all_heads = HEADS + rand_heads
    layers = sorted({L for L, _ in all_heads})

    acc = {h: {"lag_even": 0.0, "lag_odd": 0.0, "same_col": 0.0, "opp_col": 0.0,
               "same_word": 0.0, "bos_other": 0.0, "nq": 0} for h in all_heads}
    walks = G.generate_walks(graph, cfg)
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes
        outp = model(input_ids=ids, output_attentions=True)
        attn = {L: outp.attentions[L][0].float().cpu().numpy() for L in layers}   # [heads, T, T]
        tok_of_step = [sp[-1] for sp in spans]
        step_of_tok = {t: s for s, t in enumerate(tok_of_step)}
        for qs in range(CTXLO, len(nodes) - 1):
            qt = tok_of_step[qs]; qnd = nodes[qs]
            for (L, H) in all_heads:
                a = attn[L][H, qt]                                    # [T] over keys <= qt
                e = acc[(L, H)]; e["nq"] += 1
                for t in range(qt + 1):
                    m = float(a[t])
                    if m < 1e-6: continue
                    s = step_of_tok.get(t)
                    if s is None or s == qs:
                        e["bos_other"] += m; continue
                    lag = qs - s
                    if lag % 2 == 0: e["lag_even"] += m
                    else: e["lag_odd"] += m
                    if col[nodes[s]] == col[qnd]: e["same_col"] += m
                    else: e["opp_col"] += m
                    if nodes[s] == qnd: e["same_word"] += m
    rows = []
    for h in all_heads:
        e = acc[h]; nq = max(e["nq"], 1)
        tot = e["lag_even"] + e["lag_odd"] + 1e-9
        rows.append({"head": f"L{h[0]}H{h[1]}", "role": LABELS.get(h, "random"),
                     "lag_even_frac": round(e["lag_even"] / tot, 4),
                     "same_col_frac": round(e["same_col"] / tot, 4),
                     "same_word_mass": round(e["same_word"] / nq, 4),
                     "node_token_mass": round(tot / nq, 4),
                     "bos_other_mass": round(e["bos_other"] / nq, 4)})
        print(f"  {rows[-1]['head']:8} {rows[-1]['role']:14} lag-even {rows[-1]['lag_even_frac']:.3f}  "
              f"same-colour {rows[-1]['same_col_frac']:.3f}  same-word/q {rows[-1]['same_word_mass']:.3f}  "
              f"node-mass/q {rows[-1]['node_token_mass']:.3f}", flush=True)
    out = {"model": tag, "k": K, "wlen": WLEN, "ctxlo": CTXLO, "heads": rows,
           "note": "lag_even_frac & same_col_frac are fractions of attention mass on PREVIOUS node tokens; "
                   "0.5 = no preference. On a bipartite walk lag-even == same-colour for p=0 walks; same_word "
                   "isolates the induction (token-matching) component."}
    p = f"{OUTDIR}/parity_attention{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
