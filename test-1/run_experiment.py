"""Dueling-context experiment: two mirrored Llama-3.1-8B instances, one primed with a
4x4-grid random walk, one with a 16-ring random walk (same 16 concept words), then coupled.

Two "instances" = the same bf16 weights with two INDEPENDENT contexts / KV caches, which is
mathematically identical to loading two mirrored copies.

Phase 1 (prefill): context CTX=1000 node-words per instance -- a plain random walk on its own
graph, no instructions (Park et al. plain-walk condition; words = cross-model WORDS[:16]).

Phase 2 (joint generation, TGEN steps): starting from context-1000 the instances alternate.
Step t even -> the RING context predicts the next node (sampled over the 16 node-word tokens);
that word is appended to BOTH contexts. Step t odd -> the GRID context predicts; fed to both.
So after step 0 both contexts extend with the SAME jointly-authored node stream.

Phase 3 (capture): one full forward per final context with hooks on every decoder block;
per-node mean residuals accumulated in word-step windows:
  base        prefix steps [CTX-300, CTX)   -- pre-interaction geometry
  joint_early joint steps  [0, 100)
  joint_mid   joint steps  [100, 300)
  joint_late  joint steps  [300, TGEN)
pooled across NPAIRS independent walk pairs. Geometry analysis (PCA, coordinate-regression,
Laplacian eigenmode spectra -- NOT RSA) happens offline in analyze.py.

Env: NPAIRS(8) CTX(1000) TGEN(600) TEMP(1.0) SEED(0) OUTDIR(out) CM_SRC DEVICE(cuda)
Out: <OUTDIR>/nodemeans_dueling.npz  <OUTDIR>/gen_log.json
"""
from __future__ import annotations
import os, sys, json, time
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import graph as G
import models as M

NPAIRS = int(os.environ.get("NPAIRS", "8"))
CTX = int(os.environ.get("CTX", "1000"))
TGEN = int(os.environ.get("TGEN", "600"))
TEMP = float(os.environ.get("TEMP", "1.0"))
TOPK = int(os.environ.get("TOPK", "0"))      # >0: renormalized top-k sampling over the 16 nodes
TOPK_RING = int(os.environ.get("TOPK_RING", str(TOPK)))   # per-generator override
TOPK_GRID = int(os.environ.get("TOPK_GRID", str(TOPK)))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out"))
DEVICE = os.environ.get("DEVICE", "cuda")
BASE_WIN = 300                               # prefix window width for the baseline geometry
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]

WINDOWS = {  # name -> (segment, lo, hi) in word steps; segment "prefix" counts from walk start,
             # "joint" from the first jointly-generated word
    "base":        ("prefix", CTX - BASE_WIN, CTX),
    "joint_early": ("joint", 0, 100),
    "joint_mid":   ("joint", 100, 300),
    "joint_late":  ("joint", 300, TGEN),
}


def adjacency_matrix(graph):
    n = graph.n_nodes
    A = np.zeros((n, n), np.int8)
    for i in range(n):
        for j in graph.adjacency[i]:
            A[i, j] = 1
    return A


def load_model(cfg):
    last = None
    for name in MODEL_CANDS:
        try:
            print(f"loading {name}", flush=True)
            return M.load_model(name, cfg), name
        except Exception as e:                      # gated repo without access, etc.
            print(f"  failed: {e}", flush=True)
            last = e
    raise last


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=NPAIRS, walk_length=CTX, seed=SEED)
    grid_cfg = replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4)
    ring_cfg = replace(cfg, graph_type="ring", ring_size=16)
    grid = G.build_graph(grid_cfg)
    ring = G.build_graph(ring_cfg)
    assert grid.words == ring.words, "both graphs must share the node vocabulary"
    if os.environ.get("WORDS16"):            # low-bigram-prior assignment (pick_words.py)
        w16 = os.environ["WORDS16"].split(",")
        assert len(w16) == grid.n_nodes and len(set(w16)) == grid.n_nodes
        grid.words = list(w16); ring.words = list(w16)
    words = grid.words
    n = grid.n_nodes

    grid_walks = G.generate_walks(grid, grid_cfg)   # pair p uses grid_walks[p], ring_walks[p]
    ring_walks = G.generate_walks(ring, ring_cfg)

    (model, tok), model_name = load_model(cfg)
    cm = model.config
    blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers
    bos = tok.bos_token_id
    assert bos is not None

    # one token per node word (" word" form); required for uniform-length batching
    cand = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {w}' is not a single token: {ids}"
        cand.append(ids[0])
    cand_t = torch.tensor(cand, device=DEVICE)

    # ---- phase 1: batched prefill ---------------------------------------
    # batch rows 0..NPAIRS-1 = grid contexts, NPAIRS..2*NPAIRS-1 = ring contexts
    prefill = [[bos] + [cand[nd] for nd in wk.nodes] for wk in grid_walks] + \
              [[bos] + [cand[nd] for nd in wk.nodes] for wk in ring_walks]
    ids = torch.tensor(prefill, device=DEVICE)              # [2P, CTX+1]
    t0 = time.time()
    try:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[:, -1, :]                      # [2P, vocab]
    print(f"prefill done in {time.time()-t0:.1f}s", flush=True)

    # ---- phase 2: alternating joint generation --------------------------
    rng = np.random.default_rng(SEED + 1)
    A_ring = adjacency_matrix(ring).astype(bool)
    A_grid = adjacency_matrix(grid).astype(bool)
    joint = np.zeros((NPAIRS, TGEN), np.int32)              # fed node ids (shared per pair)
    genlog = {p: [] for p in range(NPAIRS)}
    prev = {("ring", p): ring_walks[p].nodes[-1] for p in range(NPAIRS)}
    prev.update({("grid", p): grid_walks[p].nodes[-1] for p in range(NPAIRS)})

    t0 = time.time()
    for t in range(TGEN):
        who = "ring" if t % 2 == 0 else "grid"
        rows = range(NPAIRS, 2 * NPAIRS) if who == "ring" else range(NPAIRS)
        lg = last_logits[list(rows)][:, cand_t].float()     # [P, n]
        probs = torch.softmax(lg / TEMP, dim=-1).cpu().numpy()
        step_tok = np.zeros(NPAIRS, np.int64)
        for p in range(NPAIRS):
            pv = prev[(who, p)] if t == 0 else int(joint[p, t - 1])
            pp = probs[p].copy()
            k = TOPK_RING if who == "ring" else TOPK_GRID
            if k > 0:                        # keep only this generator's top-k nodes
                pp[np.argsort(pp)[:-k]] = 0.0
            node = int(rng.choice(n, p=pp / pp.sum()))
            joint[p, t] = node
            step_tok[p] = cand[node]
            genlog[p].append({
                "t": t, "gen": who, "node": node, "word": words[node],
                "prev": pv,
                "ring_valid": bool(A_ring[pv, node]), "grid_valid": bool(A_grid[pv, node]),
                "p_ring_nbrs": float(probs[p][A_ring[pv]].sum()),
                "p_grid_nbrs": float(probs[p][A_grid[pv]].sum()),
            })
        inp = torch.tensor(np.concatenate([step_tok, step_tok]), device=DEVICE)[:, None]
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        if (t + 1) % 100 == 0:
            print(f"joint step {t+1}/{TGEN} ({time.time()-t0:.1f}s)", flush=True)
    del past, out, last_logits
    torch.cuda.empty_cache()

    # ---- phase 3: full-context capture with hooks on every block --------
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]

    nsum = {(c, wname): {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}
            for c in ("grid", "ring") for wname in WINDOWS}
    ncnt = {(c, wname): np.zeros(n) for c in ("grid", "ring") for wname in WINDOWS}

    t0 = time.time()
    try:
        for p in range(NPAIRS):
            for c, walks in (("grid", grid_walks), ("ring", ring_walks)):
                row = p if c == "grid" else NPAIRS + p
                full = prefill[row] + [cand[nd] for nd in joint[p]]
                fids = torch.tensor([full], device=DEVICE)
                grabbed.clear()
                try:
                    model(input_ids=fids, logits_to_keep=1)
                except TypeError:
                    model(input_ids=fids)
                # word step s (0-based over prefix+joint) sits at token position 1+s
                nodes_all = (grid_walks[p].nodes if c == "grid" else ring_walks[p].nodes) \
                            + list(joint[p])
                for wname, (seg, lo, hi) in WINDOWS.items():
                    off = 0 if seg == "prefix" else CTX
                    s0, s1 = off + lo, off + hi
                    pos = list(range(1 + s0, 1 + s1))
                    nds = nodes_all[s0:s1]
                    for L in range(nL):
                        rows_h = grabbed[L][0][pos].float().cpu().numpy()
                        np.add.at(nsum[(c, wname)][L], nds, rows_h)
                    np.add.at(ncnt[(c, wname)], nds, 1.0)
                print(f"capture pair {p} {c} ({time.time()-t0:.1f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    # ---- save ------------------------------------------------------------
    save = {"adjacency_grid": adjacency_matrix(grid), "adjacency_ring": adjacency_matrix(ring),
            "coords_grid": np.array(grid.coords, float),
            "coords_ring": np.array(ring.coords, float),
            "words": np.array(words), "n_layers": np.array([nL])}
    for (c, wname), per_layer in nsum.items():
        cnt = np.maximum(ncnt[(c, wname)], 1.0)
        for L in range(nL):
            save[f"{c}_{wname}_layer_{L}"] = (per_layer[L] / cnt[:, None]).astype(np.float16)
        save[f"{c}_{wname}_ncnt"] = ncnt[(c, wname)]
    npz_path = os.path.join(OUTDIR, "nodemeans_dueling.npz")
    np.savez_compressed(npz_path, **save)

    log = {"model": model_name, "npairs": NPAIRS, "ctx": CTX, "tgen": TGEN, "temp": TEMP,
           "topk": TOPK, "topk_ring": TOPK_RING, "topk_grid": TOPK_GRID,
           "seed": SEED, "windows": {k: list(v) for k, v in WINDOWS.items()},
           "words": words,
           "prefix_last": {f"pair{p}": {"grid": int(grid_walks[p].nodes[-1]),
                                        "ring": int(ring_walks[p].nodes[-1])}
                           for p in range(NPAIRS)},
           "steps": {f"pair{p}": genlog[p] for p in range(NPAIRS)}}
    json_path = os.path.join(OUTDIR, "gen_log.json")
    json.dump(log, open(json_path, "w"))
    print(f"DONE -> {npz_path}  {json_path}", flush=True)


if __name__ == "__main__":
    main()
