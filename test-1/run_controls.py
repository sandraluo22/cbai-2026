"""Control conditions for the grid+ring dueling-context experiment (Llama, fixed vocab).

1) free_ring : ring-primed context self-generates TGEN tokens (top-k), sees ONLY itself.
2) free_grid : same for grid.
3) oneway    : driver self-generates (never sees receiver); receiver's context receives
               [driver-token, own-prediction] alternating -- influence flows one way.
               Run both directions (ring->grid and grid->ring).
4) realsteps : no generation. Both contexts receive the SAME interleaved stream of
               GROUND-TRUTH steps: odd turns advance the pair's real ring walk, even
               turns its real grid walk (each token graph-valid by construction).

Same capture as run_experiment.py: per-node mean residuals at every block, windows
base/joint_early/joint_mid/joint_late (driver windows scaled to its 300 tokens).

Env: NPAIRS(8) CTX(1000) TGEN(600) KRING(2) KGRID(4) TEMP(1.0) SEED(0) WORDS16
Out: out_ctrl_<cond>/nodemeans_dueling.npz + gen_log.json  (ctx_names key layout)
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
ROUNDS = TGEN // 2
KRING = int(os.environ.get("KRING", "2"))
KGRID = int(os.environ.get("KGRID", "4"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]


def windows_for(n_new):
    """base window on the prefix + three joint windows scaled to n_new appended words."""
    e, m = max(1, n_new // 6), n_new // 2
    return {"base": ("prefix", CTX - 300, CTX), "joint_early": ("joint", 0, e),
            "joint_mid": ("joint", e, m), "joint_late": ("joint", m, n_new)}


def adjacency_matrix(g):
    A = np.zeros((16, 16), np.int8)
    for i in range(16):
        for j in g.adjacency[i]:
            A[i, j] = 1
    return A


@torch.no_grad()
def capture_ctx(model, blocks, nL, hidden, row_ids, nodes_all, wins, nsum, ncnt):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    try:
        fids = torch.tensor([row_ids], device=DEVICE)
        try:
            model(input_ids=fids, logits_to_keep=1)
        except TypeError:
            model(input_ids=fids)
        for wname, (seg, lo, hi) in wins.items():
            off = 0 if seg == "prefix" else CTX
            pos = list(range(1 + off + lo, 1 + off + hi))
            nds = nodes_all[off + lo:off + hi]
            for L in range(nL):
                rows_h = grabbed[L][0][pos].float().cpu().numpy()
                np.add.at(nsum[wname][L], nds, rows_h)
            np.add.at(ncnt[wname], nds, 1.0)
    finally:
        for h in hs:
            h.remove()


def save_condition(outdir, ctxs, grid, ring, words, gen_log):
    """ctxs: name -> (nsum, ncnt, wins, nL, hidden)"""
    os.makedirs(outdir, exist_ok=True)
    save = {"ctx_names": np.array(list(ctxs)), "words": np.array(words),
            "n_layers": np.array([next(iter(ctxs.values()))[3]]),
            "adjacency_grid": adjacency_matrix(grid), "adjacency_ring": adjacency_matrix(ring),
            "coords_grid": np.array(grid.coords, float),
            "coords_ring": np.array(ring.coords, float)}
    for nm, (nsum, ncnt, wins, nL, hidden) in ctxs.items():
        save[f"nlayers_{nm}"] = np.array([nL])
        save[f"adjacency_{nm}"] = adjacency_matrix(ring if "ring" in nm else grid)
        save[f"coords_{nm}"] = np.array((ring if "ring" in nm else grid).coords, float)
        for wname in wins:
            cnt = np.maximum(ncnt[wname], 1.0)
            for L in range(nL):
                save[f"{nm}_{wname}_layer_{L}"] = (nsum[wname][L] / cnt[:, None]).astype(np.float16)
        save[f"{nm}_windows"] = np.array([f"{w}:{s}:{lo}:{hi}"
                                          for w, (s, lo, hi) in wins.items()])
    np.savez_compressed(os.path.join(outdir, "nodemeans_dueling.npz"), **save)
    json.dump(gen_log, open(os.path.join(outdir, "gen_log.json"), "w"))
    print(f"SAVED -> {outdir}", flush=True)


@torch.no_grad()
def main():
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=NPAIRS, walk_length=CTX + ROUNDS, seed=SEED)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    if os.environ.get("WORDS16"):
        w16 = os.environ["WORDS16"].split(",")
        grid.words = list(w16); ring.words = list(w16)
    words = grid.words
    gwalks = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rwalks = G.generate_walks(ring, replace(cfg, graph_type="ring"))
    A = {"grid": adjacency_matrix(grid).astype(bool),
         "ring": adjacency_matrix(ring).astype(bool)}
    K = {"grid": KGRID, "ring": KRING}
    WK = {"grid": gwalks, "ring": rwalks}

    model = tok = None
    for name in MODEL_CANDS:
        try:
            model, tok = M.load_model(name, cfg); break
        except Exception as e:
            print(f"failed {name}: {e}", flush=True)
    cm = model.config
    blocks = M._decoder_blocks(model)
    nL, hidden = cm.num_hidden_layers, cm.hidden_size
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    cand_t = torch.tensor(cand, device=DEVICE)

    def prefill(rows):
        ids = torch.tensor(rows, device=DEVICE)
        try:
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(input_ids=ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    def step(past, toks):
        inp = torch.tensor(toks, device=DEVICE)[:, None]
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    def sample(logits, prev, gname, rng, log_rows):
        lg = logits[:, cand_t].float()
        probs = torch.softmax(lg / TEMP, -1).cpu().numpy()
        out = np.zeros(NPAIRS, np.int64)
        for p in range(NPAIRS):
            pp = probs[p].copy()
            k = K[gname]
            pp[np.argsort(pp)[:-k]] = 0.0
            node = int(rng.choice(16, p=pp / pp.sum()))
            out[p] = node
            log_rows[p].append({"gen": gname, "node": node, "prev": int(prev[p]),
                                "ring_valid": bool(A["ring"][prev[p], node]),
                                "grid_valid": bool(A["grid"][prev[p], node]),
                                "p_ring_nbrs": float(probs[p][A["ring"][prev[p]]].sum()),
                                "p_grid_nbrs": float(probs[p][A["grid"][prev[p]]].sum())})
        return out

    def new_acc(wins):
        return ({w: {L: np.zeros((16, hidden)) for L in range(nL)} for w in wins},
                {w: np.zeros(16) for w in wins})

    prefix_nodes = {g: [WK[g][p].nodes[:CTX] for p in range(NPAIRS)] for g in ("grid", "ring")}
    cont_nodes = {g: [WK[g][p].nodes[CTX:CTX + ROUNDS] for p in range(NPAIRS)]
                  for g in ("grid", "ring")}
    prefix_rows = {g: [[bos] + [cand[n] for n in prefix_nodes[g][p]] for p in range(NPAIRS)]
                   for g in ("grid", "ring")}

    # ---- (1)+(2) free generation ------------------------------------------
    for gname in ("ring", "grid"):
        rng = np.random.default_rng(SEED + 1)
        past, logits = prefill(prefix_rows[gname])
        logr = {p: [] for p in range(NPAIRS)}
        gen = np.zeros((NPAIRS, TGEN), np.int32)
        prev = np.array([prefix_nodes[gname][p][-1] for p in range(NPAIRS)])
        t0 = time.time()
        for t in range(TGEN):
            nodes = sample(logits, prev, gname, rng, logr)
            gen[:, t] = nodes
            prev = nodes
            past, logits = step(past, [cand[x] for x in nodes])
        del past
        torch.cuda.empty_cache()
        wins = windows_for(TGEN)
        nsum, ncnt = new_acc(wins)
        for p in range(NPAIRS):
            row = prefix_rows[gname][p] + [cand[x] for x in gen[p]]
            capture_ctx(model, blocks, nL, hidden, row,
                        prefix_nodes[gname][p] + list(gen[p]), wins, nsum, ncnt)
        save_condition(os.path.join(_here, f"out_ctrl_free{gname}"),
                       {gname: (nsum, ncnt, wins, nL, hidden)}, grid, ring, words,
                       {"cond": f"free_{gname}", "npairs": NPAIRS, "ctx": CTX,
                        "tgen": TGEN, "k": K[gname],
                        "steps": {f"pair{p}": logr[p] for p in range(NPAIRS)}})
        print(f"free_{gname} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- (3) one-way influence, both directions ----------------------------
    for drv, rcv in (("ring", "grid"), ("grid", "ring")):
        rng = np.random.default_rng(SEED + 1)
        past_d, log_d = prefill(prefix_rows[drv])
        past_r, log_r = prefill(prefix_rows[rcv])
        logr = {p: [] for p in range(NPAIRS)}
        d_gen = np.zeros((NPAIRS, ROUNDS), np.int32)
        r_seq = np.zeros((NPAIRS, TGEN), np.int32)     # receiver's appended stream
        prev_d = np.array([prefix_nodes[drv][p][-1] for p in range(NPAIRS)])
        prev_r = np.array([prefix_nodes[rcv][p][-1] for p in range(NPAIRS)])
        t0 = time.time()
        for t in range(ROUNDS):
            d = sample(log_d, prev_d, drv, rng, logr)          # driver: own ctx only
            d_gen[:, t] = d
            prev_d = d
            past_d, log_d = step(past_d, [cand[x] for x in d])
            past_r, log_r = step(past_r, [cand[x] for x in d])  # receiver sees driver tok
            r = sample(log_r, d, rcv, rng, logr)                # receiver predicts next
            r_seq[:, 2 * t] = d
            r_seq[:, 2 * t + 1] = r
            past_r, log_r = step(past_r, [cand[x] for x in r])  # own pred: receiver only
        del past_d, past_r
        torch.cuda.empty_cache()
        wins_d, wins_r = windows_for(ROUNDS), windows_for(TGEN)
        nd, cd = new_acc(wins_d)
        nr, cr = new_acc(wins_r)
        for p in range(NPAIRS):
            capture_ctx(model, blocks, nL, hidden,
                        prefix_rows[drv][p] + [cand[x] for x in d_gen[p]],
                        prefix_nodes[drv][p] + list(d_gen[p]), wins_d, nd, cd)
            capture_ctx(model, blocks, nL, hidden,
                        prefix_rows[rcv][p] + [cand[x] for x in r_seq[p]],
                        prefix_nodes[rcv][p] + list(r_seq[p]), wins_r, nr, cr)
        save_condition(os.path.join(_here, f"out_ctrl_ow_{drv}2{rcv}"),
                       {f"{drv}-driver": (nd, cd, wins_d, nL, hidden),
                        f"{rcv}-receiver": (nr, cr, wins_r, nL, hidden)},
                       grid, ring, words,
                       {"cond": f"oneway_{drv}2{rcv}", "npairs": NPAIRS, "ctx": CTX,
                        "tgen": TGEN, "rounds": ROUNDS,
                        "steps": {f"pair{p}": logr[p] for p in range(NPAIRS)}})
        print(f"oneway_{drv}2{rcv} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- (4) real interleaved ground-truth steps ---------------------------
    wins = windows_for(TGEN)
    accs = {g: new_acc(wins) for g in ("ring", "grid")}
    inter = np.zeros((NPAIRS, TGEN), np.int32)
    for p in range(NPAIRS):
        for t in range(ROUNDS):
            inter[p, 2 * t] = cont_nodes["ring"][p][t]
            inter[p, 2 * t + 1] = cont_nodes["grid"][p][t]
    t0 = time.time()
    for p in range(NPAIRS):
        for g in ("ring", "grid"):
            nsum, ncnt = accs[g]
            capture_ctx(model, blocks, nL, hidden,
                        prefix_rows[g][p] + [cand[x] for x in inter[p]],
                        prefix_nodes[g][p] + list(inter[p]), wins, nsum, ncnt)
    save_condition(os.path.join(_here, "out_ctrl_real"),
                   {"ring": (*accs["ring"], wins, nL, hidden),
                    "grid": (*accs["grid"], wins, nL, hidden)},
                   grid, ring, words,
                   {"cond": "realsteps", "npairs": NPAIRS, "ctx": CTX, "tgen": TGEN,
                    "steps": {f"pair{p}": [{"gen": "real", "node": int(x)}
                                           for x in inter[p]] for p in range(NPAIRS)}})
    print(f"realsteps done ({time.time()-t0:.0f}s)", flush=True)
    print("ALL CONTROLS DONE", flush=True)


if __name__ == "__main__":
    main()
