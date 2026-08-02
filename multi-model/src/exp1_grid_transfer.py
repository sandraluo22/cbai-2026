"""Exp1 -- Grid-geometry TRANSFER: does a walk that Llama GENERATES induce the
square-grid representation inside Qwen?

Pipeline (all base models, 4x4 grid over 16 concept words):
  1. Seed Llama with a real random-walk prefix, then let Llama free-generate the
     rest of the walk, its next-token distribution CONSTRAINED to the 16 node
     words (cross-model gen_head_ablation style). -> `llama_gen` walks.
  2. Feed those generated word sequences to Qwen (teacher-forced) and fit the
     leave-one-node-out coordinate probe (row, col) at EVERY Qwen layer.
     -> does Qwen reconstruct the grid from Llama's generated tokens?

Conditions probed in Qwen (per layer, LOO R^2 with permutation null):
  - llama_gen : Qwen reads Llama's generated walk               (the transfer test)
  - real_walk : Qwen reads a genuine random walk               (upper bound)
  - shuffled  : Qwen reads a shuffled word sequence            (specificity floor)
Plus a sanity self-probe: Llama's OWN grid R^2 on its generated walk (it should
carry the grid), and the behavioural fidelity of the generation (neighbour mass,
validity) so we know the generated walk is grid-like in the first place.

Env: PRESET GRAPH(square_grid) NSEED(6) XCTX(80) GSTEPS(220) NWALKS_REAL(12)
     WLEN_REAL(300) CTXLO(100) TEMP(1.0) NPERM(200) RUN_DIR DEVICE
Out: <RUN_DIR>/exp1_grid_transfer.json + .pdf
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C  # noqa: E402
import graph as G   # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

GRAPH = os.environ.get("GRAPH", "square_grid")
NSEED = int(os.environ.get("NSEED", "6" if C.PRESET != "smoke" else "3"))
XCTX = int(os.environ.get("XCTX", "80" if C.PRESET != "smoke" else "15"))
GSTEPS = int(os.environ.get("GSTEPS", "220" if C.PRESET != "smoke" else "40"))
NWALKS_REAL = int(os.environ.get("NWALKS_REAL", "12" if C.PRESET != "smoke" else "4"))
WLEN_REAL = int(os.environ.get("WLEN_REAL", "300" if C.PRESET != "smoke" else "40"))
CTXLO = int(os.environ.get("CTXLO", "100" if C.PRESET != "smoke" else "5"))
TEMP = float(os.environ.get("TEMP", "1.0"))
NPERM = int(os.environ.get("NPERM", "200" if C.PRESET != "smoke" else "20"))
READER = os.environ.get("READER", "Qwen")     # model that READS the generated walk (Qwen or Llama)
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
SUF = "" if READER == "Qwen" else f"_{READER}"


def per_layer_probe(node_means, coords, nperm):
    nL = len(node_means)
    rec = {"r2_row": [], "r2_col": [], "null_row": [], "null_col": [],
           "p_row": [], "p_col": []}
    for L in range(nL):
        rr, rc, nr, nc, pr, pc = C.coord_loo_r2_with_null(node_means[L], coords, nperm=nperm, seed=L)
        rec["r2_row"].append(rr); rec["r2_col"].append(rc)
        rec["null_row"].append(nr); rec["null_col"].append(nc)
        rec["p_row"].append(pr); rec["p_col"].append(pc)
    mean_r2 = [(a + b) / 2 for a, b in zip(rec["r2_row"], rec["r2_col"])]
    rec["mean_r2"] = mean_r2
    rec["peak_layer"] = int(np.nanargmax(mean_r2))
    rec["peak_mean_r2"] = float(np.nanmax(mean_r2))
    return rec


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg(GRAPH, n_walks=max(NSEED, NWALKS_REAL, 8),
                     walk_length=max(XCTX, WLEN_REAL), device=dev)
    graph, n, coords = C.build_grid(cfg)

    # real walks (Qwen upper-bound control) + seeds for Llama generation
    real_cfg = C.make_cfg(GRAPH, n_walks=NWALKS_REAL, walk_length=WLEN_REAL, device=dev)
    real_walks = G.generate_walks(graph, real_cfg)
    seeds = G.generate_walks(graph, cfg)[:NSEED]

    out = {"graph": GRAPH, "n_nodes": n, "ctxlo": CTXLO, "xctx": XCTX, "gsteps": GSTEPS,
           "nseed": NSEED, "temp": TEMP, "reader": READER,
           "conds": {}, "behaviour": {}, "self_probe": {}}

    # ---- 1. Llama generates walks (constrained to node words) ----
    print("[exp1] loading Llama for generation", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(ltok, graph, dev)
    gen_walks = []
    beh = []
    for si, seed in enumerate(seeds):
        nodes, b = C.generate_walk(llama, ltok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                   temp=TEMP, rng=np.random.default_rng(1000 + si))
        gen_walks.append(C.mkwalk(nodes, graph))
        beh.append(b)
    out["behaviour"] = {"nbr_mass": float(np.nanmean([x["nbr_mass"] for x in beh])),
                        "validity": float(np.nanmean([x["validity"] for x in beh]))}
    print(f"[exp1] Llama gen: nbr_mass={out['behaviour']['nbr_mass']:.3f} "
          f"validity={out['behaviour']['validity']:.3f}", flush=True)

    # Llama self-probe on its own generated walk (should carry the grid)
    lm_gen_means, _ = C.node_means_all_layers(llama, ltok, graph, gen_walks, dev, n, ctxlo=CTXLO)
    out["self_probe"]["Llama_on_gen"] = per_layer_probe(lm_gen_means, coords, NPERM)
    C.free(llama, ltok)

    # shuffled control: same generated tokens, node identities permuted per walk
    rng = np.random.default_rng(7)
    shuf_walks = []
    for wk in gen_walks:
        perm = rng.permutation(n)
        shuf_walks.append(C.mkwalk([perm[x] for x in wk.nodes], graph))

    # ---- 2. READER reads each condition; probe all layers ----
    print(f"[exp1] loading {READER} for read-out", flush=True)
    reader, rtok = C.load_model(READER, cfg)
    for cname, walks in (("llama_gen", gen_walks), ("real_walk", real_walks),
                         ("shuffled", shuf_walks)):
        nm, ncnt = C.node_means_all_layers(reader, rtok, graph, walks, dev, n, ctxlo=CTXLO)
        rec = per_layer_probe(nm, coords, NPERM)
        rec["nodes_seen"] = int((ncnt > 0).sum())
        out["conds"][cname] = rec
        print(f"[exp1] {READER}/{cname}: peak L{rec['peak_layer']} mean R²={rec['peak_mean_r2']:.3f} "
              f"(row={rec['r2_row'][rec['peak_layer']]:.3f} col={rec['r2_col'][rec['peak_layer']]:.3f})", flush=True)
    C.free(reader, rtok)

    json.dump(out, open(os.path.join(RUN_DIR, f"exp1_grid_transfer{SUF}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"exp1_grid_transfer{SUF}.pdf"))
    print(f"[exp1] DONE ({READER} reader) -> {RUN_DIR}/exp1_grid_transfer{SUF}.json", flush=True)


def make_fig(out, path):
    rd = out.get("reader", "Qwen")
    colors = {"llama_gen": "tab:red", "real_walk": "k", "shuffled": "tab:gray"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        # left: mean R^2 per Qwen layer, all conditions
        for cn, c in colors.items():
            r = out["conds"].get(cn)
            if not r:
                continue
            L = range(len(r["mean_r2"]))
            ax[0].plot(L, r["mean_r2"], "-o", ms=3, color=c,
                       label=f"{cn} (peak L{r['peak_layer']}={r['peak_mean_r2']:.2f})")
        ax[0].axhline(0, color=".7", lw=.6); ax[0].set_ylim(-0.6, 1.0)
        ax[0].set_xlabel(f"{rd} layer"); ax[0].set_ylabel("LOO coord-probe mean R²")
        ax[0].set_title(f"{rd} grid recovery by source of the walk", fontsize=10)
        ax[0].legend(fontsize=8)
        # right: llama_gen row/col + Llama self-probe for reference
        r = out["conds"]["llama_gen"]; sp = out["self_probe"]["Llama_on_gen"]
        L = range(len(r["r2_row"]))
        ax[1].plot(L, r["r2_row"], "-o", ms=3, color="tab:blue", label=f"{rd} row")
        ax[1].plot(L, r["r2_col"], "-o", ms=3, color="tab:red", label=f"{rd} col")
        Ls = range(len(sp["mean_r2"]))
        ax[1].plot(Ls, sp["mean_r2"], "--", color="tab:green", label="Llama self mean R² (gen walk)")
        ax[1].axhline(0, color=".7", lw=.6); ax[1].set_ylim(-0.6, 1.0)
        ax[1].set_xlabel("layer"); ax[1].set_ylabel("LOO R²")
        ax[1].set_title("llama_gen: Qwen per-axis vs Llama self-probe", fontsize=10)
        ax[1].legend(fontsize=8)
        fig.suptitle(f"[{out['graph']}] Exp1 grid transfer — Llama-generated walk "
                     f"(nbr_mass={out['behaviour']['nbr_mass']:.2f}, "
                     f"val={out['behaviour']['validity']:.2f}) fed to {rd}", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
