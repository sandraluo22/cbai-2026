"""Generalized dueling-context experiment for ANY 16-node graph pair (grid|ring|hex).

Same protocol as run_experiment.py (which is the grid+ring original): two mirrored
Llama-3.1-8B contexts, each primed with a CTX-word random walk on its own graph over the
same 16 words; from context-1000 they alternate generating (graph B speaks first),
each sampled word appended to BOTH contexts; per-node mean residuals captured at every
decoder block in word-step windows, pooled over NPAIRS pairs.

Env: GRAPH_A(hex) GRAPH_B(grid) TOPK_A(0) TOPK_B(0) WORDS16 NPAIRS(8) CTX(1000)
     TGEN(600) TEMP(1.0) SEED(0) OUTDIR CM_SRC DEVICE(cuda)
Out: <OUTDIR>/nodemeans_dueling.npz  <OUTDIR>/gen_log.json  (keys use the graph names)
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

GKW = {"grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "prism": dict(graph_type="prism", prism_k=8),
       "antiprism": dict(graph_type="antiprism", prism_k=8),
       "ring3": dict(graph_type="ring", ring_size=16)}   # rewired below


def build_named(name, cfg):
    g = G.build_graph(replace(cfg, **GKW[name]))
    if name == "ring3":
        # stride-3 relabeled 16-cycle: node i ~ i+-3 (mod 16); C16(3), isomorphic to C16(1).
        # cycle position of node j is j * 11 mod 16 (11 = 3^-1 mod 16) -> circle coords.
        g.adjacency = [sorted([(i - 3) % 16, (i + 3) % 16]) for i in range(16)]
        g.coords = [(float(np.cos(2 * np.pi * (11 * i % 16) / 16)),
                     float(np.sin(2 * np.pi * (11 * i % 16) / 16))) for i in range(16)]
    return g


GA = os.environ.get("GRAPH_A", "hex")
GB = os.environ.get("GRAPH_B", "grid")
assert GA in GKW and GB in GKW and GA != GB
TOPK_A = int(os.environ.get("TOPK_A", "0"))
TOPK_B = int(os.environ.get("TOPK_B", "0"))
NPAIRS = int(os.environ.get("NPAIRS", "8"))
CTX = int(os.environ.get("CTX", "1000"))
TGEN = int(os.environ.get("TGEN", "600"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out_pair"))
DEVICE = os.environ.get("DEVICE", "cuda")
BASE_WIN = 300
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WINDOWS = {
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
        except Exception as e:
            print(f"  failed: {e}", flush=True)
            last = e
    raise last


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=NPAIRS, walk_length=CTX, seed=SEED)
    gA = build_named(GA, cfg)
    gB = build_named(GB, cfg)
    if os.environ.get("WORDS16"):
        w16 = os.environ["WORDS16"].split(",")
        assert len(w16) == gA.n_nodes and len(set(w16)) == gA.n_nodes
        gA.words = list(w16); gB.words = list(w16)
    assert gA.words == gB.words
    words = gA.words
    n = gA.n_nodes

    walks = {GA: G.generate_walks(gA, replace(cfg, **GKW[GA])),
             GB: G.generate_walks(gB, replace(cfg, **GKW[GB]))}
    graphs = {GA: gA, GB: gB}

    (model, tok), model_name = load_model(cfg)
    cm = model.config
    blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers
    bos = tok.bos_token_id
    cand = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {w}' is not a single token: {ids}"
        cand.append(ids[0])
    cand_t = torch.tensor(cand, device=DEVICE)

    # batch rows 0..P-1 = A contexts, P..2P-1 = B contexts
    prefill = [[bos] + [cand[nd] for nd in wk.nodes] for wk in walks[GA]] + \
              [[bos] + [cand[nd] for nd in wk.nodes] for wk in walks[GB]]
    ids = torch.tensor(prefill, device=DEVICE)
    t0 = time.time()
    try:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[:, -1, :]
    print(f"prefill done in {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(SEED + 1)
    A_a = adjacency_matrix(gA).astype(bool)
    A_b = adjacency_matrix(gB).astype(bool)
    joint = np.zeros((NPAIRS, TGEN), np.int32)
    genlog = {p: [] for p in range(NPAIRS)}
    prev = {(GA, p): walks[GA][p].nodes[-1] for p in range(NPAIRS)}
    prev.update({(GB, p): walks[GB][p].nodes[-1] for p in range(NPAIRS)})
    TOPK = {GA: TOPK_A, GB: TOPK_B}

    t0 = time.time()
    for t in range(TGEN):
        who = GB if t % 2 == 0 else GA                # B speaks first (as ring did)
        rows = range(NPAIRS, 2 * NPAIRS) if who == GB else range(NPAIRS)
        lg = last_logits[list(rows)][:, cand_t].float()
        probs = torch.softmax(lg / TEMP, dim=-1).cpu().numpy()
        step_tok = np.zeros(NPAIRS, np.int64)
        for p in range(NPAIRS):
            pv = prev[(who, p)] if t == 0 else int(joint[p, t - 1])
            pp = probs[p].copy()
            k = TOPK[who]
            if k > 0:
                pp[np.argsort(pp)[:-k]] = 0.0
            node = int(rng.choice(n, p=pp / pp.sum()))
            joint[p, t] = node
            step_tok[p] = cand[node]
            genlog[p].append({
                "t": t, "gen": who, "node": node, "word": words[node], "prev": pv,
                f"{GA}_valid": bool(A_a[pv, node]), f"{GB}_valid": bool(A_b[pv, node]),
                f"p_{GA}_nbrs": float(probs[p][A_a[pv]].sum()),
                f"p_{GB}_nbrs": float(probs[p][A_b[pv]].sum()),
            })
        inp = torch.tensor(np.concatenate([step_tok, step_tok]), device=DEVICE)[:, None]
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        if (t + 1) % 200 == 0:
            print(f"joint step {t+1}/{TGEN} ({time.time()-t0:.1f}s)", flush=True)
    del past, out, last_logits
    torch.cuda.empty_cache()

    grabbed = {}
    def mk(L):
        def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {(c, wname): {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}
            for c in (GA, GB) for wname in WINDOWS}
    ncnt = {(c, wname): np.zeros(n) for c in (GA, GB) for wname in WINDOWS}
    t0 = time.time()
    try:
        for p in range(NPAIRS):
            for c in (GA, GB):
                row = p if c == GA else NPAIRS + p
                full = prefill[row] + [cand[nd] for nd in joint[p]]
                fids = torch.tensor([full], device=DEVICE)
                grabbed.clear()
                try:
                    model(input_ids=fids, logits_to_keep=1)
                except TypeError:
                    model(input_ids=fids)
                nodes_all = walks[c][p].nodes + list(joint[p])
                for wname, (seg, lo, hi) in WINDOWS.items():
                    off = 0 if seg == "prefix" else CTX
                    s0, s1 = off + lo, off + hi
                    pos = list(range(1 + s0, 1 + s1))
                    nds = nodes_all[s0:s1]
                    for L in range(nL):
                        rows_h = grabbed[L][0][pos].float().cpu().numpy()
                        np.add.at(nsum[(c, wname)][L], nds, rows_h)
                    np.add.at(ncnt[(c, wname)], nds, 1.0)
            print(f"capture pair {p} ({time.time()-t0:.1f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    save = {f"adjacency_{GA}": adjacency_matrix(gA), f"adjacency_{GB}": adjacency_matrix(gB),
            f"coords_{GA}": np.array(gA.coords, float), f"coords_{GB}": np.array(gB.coords, float),
            "ctx_names": np.array([GA, GB]), "words": np.array(words),
            "n_layers": np.array([nL])}
    for (c, wname), per_layer in nsum.items():
        cnt = np.maximum(ncnt[(c, wname)], 1.0)
        for L in range(nL):
            save[f"{c}_{wname}_layer_{L}"] = (per_layer[L] / cnt[:, None]).astype(np.float16)
        save[f"{c}_{wname}_ncnt"] = ncnt[(c, wname)]
    np.savez_compressed(os.path.join(OUTDIR, "nodemeans_dueling.npz"), **save)

    log = {"model": model_name, "graph_a": GA, "graph_b": GB, "npairs": NPAIRS,
           "ctx": CTX, "tgen": TGEN, "temp": TEMP, "topk_a": TOPK_A, "topk_b": TOPK_B,
           "seed": SEED, "windows": {k: list(v) for k, v in WINDOWS.items()},
           "words": words,
           "steps": {f"pair{p}": genlog[p] for p in range(NPAIRS)}}
    json.dump(log, open(os.path.join(OUTDIR, "gen_log.json"), "w"))
    print(f"DONE -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
