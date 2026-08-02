"""N-way dueling contexts: K mirrored Llama contexts, each primed with a CTX-word random
walk on its own 16-node graph (same words), then cycling generation: context k speaks at
steps t with t mod K == k, sampling the next node (top-k restricted); the word is appended
to ALL contexts. Node-mean capture identical to run_pair.py.

Env: GRAPHS("ring,grid,ring3") TOPKS("2,4,2") WORDS16 NPAIRS(8) CTX(1000) TGEN(600)
     TEMP(1.0) SEED(0) OUTDIR CM_SRC DEVICE
Out: <OUTDIR>/nodemeans_dueling.npz  <OUTDIR>/gen_log.json (same key layout as run_pair,
     with ctx_names = the K graph names; compatible with capture_fresh.py)
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
from run_pair import GKW, build_named, adjacency_matrix, load_model

NAMES = os.environ.get("GRAPHS", "ring,grid,ring3").split(",")
TOPKS = [int(x) for x in os.environ.get("TOPKS", "2,4,2").split(",")]
assert len(NAMES) == len(TOPKS) and len(set(NAMES)) == len(NAMES)
K = len(NAMES)
NPAIRS = int(os.environ.get("NPAIRS", "8"))
CTX = int(os.environ.get("CTX", "1000"))
TGEN = int(os.environ.get("TGEN", "600"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out_multi"))
DEVICE = os.environ.get("DEVICE", "cuda")
BASE_WIN = 300
WINDOWS = {
    "base":        ("prefix", CTX - BASE_WIN, CTX),
    "joint_early": ("joint", 0, 100),
    "joint_mid":   ("joint", 100, 300),
    "joint_late":  ("joint", 300, TGEN),
}


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=NPAIRS, walk_length=CTX, seed=SEED)
    graphs = {nm: build_named(nm, cfg) for nm in NAMES}
    if os.environ.get("WORDS16"):
        w16 = os.environ["WORDS16"].split(",")
        assert len(w16) == 16 and len(set(w16)) == 16
        for g in graphs.values():
            g.words = list(w16)
    words = graphs[NAMES[0]].words
    n = 16
    walks = {nm: G.generate_walks(graphs[nm], replace(cfg, **GKW[nm])) for nm in NAMES}
    adjs = {nm: adjacency_matrix(graphs[nm]).astype(bool) for nm in NAMES}

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

    # batch rows: graph k occupies rows [k*NPAIRS, (k+1)*NPAIRS)
    prefill = [[bos] + [cand[nd] for nd in wk.nodes]
               for nm in NAMES for wk in walks[nm]]
    ids = torch.tensor(prefill, device=DEVICE)
    t0 = time.time()
    try:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[:, -1, :]
    print(f"prefill [{ids.shape[0]}x{ids.shape[1]}] done in {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(SEED + 1)
    joint = np.zeros((NPAIRS, TGEN), np.int32)
    genlog = {p: [] for p in range(NPAIRS)}
    prev0 = {(nm, p): walks[nm][p].nodes[-1] for nm in NAMES for p in range(NPAIRS)}

    t0 = time.time()
    for t in range(TGEN):
        ki = t % K
        who = NAMES[ki]
        rows = list(range(ki * NPAIRS, (ki + 1) * NPAIRS))
        lg = last_logits[rows][:, cand_t].float()
        probs = torch.softmax(lg / TEMP, dim=-1).cpu().numpy()
        step_tok = np.zeros(NPAIRS, np.int64)
        for p in range(NPAIRS):
            pv = prev0[(who, p)] if t == 0 else int(joint[p, t - 1])
            pp = probs[p].copy()
            k = TOPKS[ki]
            if k > 0:
                pp[np.argsort(pp)[:-k]] = 0.0
            node = int(rng.choice(n, p=pp / pp.sum()))
            joint[p, t] = node
            step_tok[p] = cand[node]
            rec = {"t": t, "gen": who, "node": node, "word": words[node], "prev": pv}
            for nm in NAMES:
                rec[f"{nm}_valid"] = bool(adjs[nm][pv, node])
                rec[f"p_{nm}_nbrs"] = float(probs[p][adjs[nm][pv]].sum())
            genlog[p].append(rec)
        inp = torch.tensor(np.tile(step_tok, K), device=DEVICE)[:, None]
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
    nsum = {(c, w): {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}
            for c in NAMES for w in WINDOWS}
    ncnt = {(c, w): np.zeros(n) for c in NAMES for w in WINDOWS}
    t0 = time.time()
    try:
        for p in range(NPAIRS):
            for ki, c in enumerate(NAMES):
                row = ki * NPAIRS + p
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

    save = {"ctx_names": np.array(NAMES), "words": np.array(words),
            "n_layers": np.array([nL])}
    for nm in NAMES:
        save[f"adjacency_{nm}"] = adjacency_matrix(graphs[nm])
        save[f"coords_{nm}"] = np.array(graphs[nm].coords, float)
    for (c, wname), per_layer in nsum.items():
        cnt = np.maximum(ncnt[(c, wname)], 1.0)
        for L in range(nL):
            save[f"{c}_{wname}_layer_{L}"] = (per_layer[L] / cnt[:, None]).astype(np.float16)
        save[f"{c}_{wname}_ncnt"] = ncnt[(c, wname)]
    np.savez_compressed(os.path.join(OUTDIR, "nodemeans_dueling.npz"), **save)

    log = {"model": model_name, "graphs": NAMES, "topks": TOPKS, "npairs": NPAIRS,
           "ctx": CTX, "tgen": TGEN, "temp": TEMP, "seed": SEED,
           "windows": {k: list(v) for k, v in WINDOWS.items()}, "words": words,
           "steps": {f"pair{p}": genlog[p] for p in range(NPAIRS)}}
    json.dump(log, open(os.path.join(OUTDIR, "gen_log.json"), "w"))
    print(f"DONE -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
