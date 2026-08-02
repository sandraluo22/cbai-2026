"""Does the model MERGE the two graph priors into one posterior, or learn a
CONDITIONAL/latent-source structure ("even turns are ring-process, odd turns are
grid-process") and keep both?  Phase-averaged node-means cannot tell these apart;
this probe can.

Contexts are reconstructed exactly from saved gen_logs + reproducible walks:
  coupled : runs/out_k2_fix   (grid+ring, alternating generators, ring on even t)
  real    : runs/out_ctrl_real (perfect interleave: even t = true ring-walk step,
                                odd t = true grid-walk step)

Tests, on late joint steps t in [300, 600):
 (a) PHASE-CONDITIONAL PREDICTION: predictive mass on ring-nbrs(prev) and
     grid-nbrs(prev), split by parity of the position being predicted. A merged
     posterior => no parity contrast; a source-inferring learner => ring-leaning
     before even turns, grid-leaning before odd turns.
 (b) LAG-2 DE-INTERLEAVING (real condition): the true next R-step is a ring-neighbour
     of the token TWO steps back. Compare predictive mass on nbrs_ring(x_{t-2}) vs
     nbrs_ring(x_{t-1}) / nbrs_grid(x_{t-1}) at even t (mirror for odd/grid).
 (c) PHASE-SPLIT GEOMETRY: node-means from even-turn vs odd-turn occurrences
     separately (late window); per-phase R^2 (ring vs grid coords) at deep layers +
     Procrustes between the two phase constellations.

Env: DEVICE. Out: runs/out_probe/phase_probe.json
"""
from __future__ import annotations
import os, sys, json
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

DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
N = 16
LATE = (300, 600)
DEEP = range(24, 32)


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def r2(Hc, F):
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def psim(A, B):
    A = A - A.mean(0); A /= max(np.linalg.norm(A), 1e-12)
    B = B - B.mean(0); B /= max(np.linalg.norm(B), 1e-12)
    Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(B, full_matrices=False)
    return float(np.linalg.svd((Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :],
                               compute_uv=False).sum())


@torch.no_grad()
def main():
    cfg0 = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg0); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    blocks = M._decoder_blocks(model)
    nL = model.config.num_hidden_layers
    hid = model.config.hidden_size
    bos = tok.bos_token_id

    out_all = {}
    for tag, rundir, ctxlen, wl in (("coupled", "runs/out_k2_fix", 1000, 1000),
                                    ("real", "runs/out_ctrl_real", 1000, 1300)):
        log = json.load(open(os.path.join(_here, rundir, "gen_log.json")))
        znpz = np.load(os.path.join(_here, rundir, "nodemeans_dueling.npz"),
                       allow_pickle=False)
        words = [str(w) for w in znpz["words"]]
        P = log["npairs"]
        T = log["tgen"]
        joint = np.array([[s["node"] for s in log["steps"][f"pair{p}"]][:T]
                          for p in range(P)])
        cfg = replace(cfg0, n_walks=P, walk_length=wl, seed=log.get("seed", 0))
        grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
        ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
        grid.words = list(words)
        ring.words = list(words)
        A_g, A_r = adjacency(grid), adjacency(ring)
        cg = np.array(grid.coords, float)
        cr = np.array(ring.coords, float)
        gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
        rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))
        cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
        cand_t = torch.tensor(cand, device=DEVICE)

        # sanity: reconstructed prefixes end where the log says they should
        if "prefix_last" in log:
            for p in range(P):
                assert gw[p].nodes[ctxlen - 1] == log["prefix_last"][f"pair{p}"]["grid"]
                assert rw[p].nodes[ctxlen - 1] == log["prefix_last"][f"pair{p}"]["ring"]

        res = {"phase_pred": {}, "lag": {}, "geom": {}}
        grabbed = {}
        def mk(L):
            def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return hh
        for ctxname, walks in (("grid", gw), ("ring", rw)):
            acc = {ph: {L: np.zeros((N, hid)) for L in DEEP} for ph in (0, 1)}
            cnt = {ph: np.zeros(N) for ph in (0, 1)}
            pm = {(par, gname): [] for par in (0, 1) for gname in ("ring", "grid")}
            lag = {k: [] for k in ("even_ring_lag2", "even_ring_lag1", "even_grid_lag1",
                                   "odd_grid_lag2", "odd_grid_lag1", "odd_ring_lag1")}
            handles = [blocks[L].register_forward_hook(mk(L)) for L in DEEP]
            try:
                for p in range(P):
                    prefix = walks[p].nodes[:ctxlen]
                    full_nodes = prefix + list(joint[p])
                    ids = torch.tensor([[bos] + [cand[x] for x in full_nodes]],
                                       device=DEVICE)
                    grabbed.clear()
                    lg = model(input_ids=ids).logits[0].float()
                    probs = torch.softmax(lg[:, cand_t], -1).cpu().numpy()
                    for t in range(LATE[0], LATE[1]):
                        par = t % 2                      # 0 = ring turn, 1 = grid turn
                        pv = full_nodes[ctxlen + t - 1]
                        pos_prev = ctxlen + t            # BOS offset: token t-1 is here
                        pr = probs[pos_prev]
                        pm[(par, "ring")].append(pr[A_r[pv]].sum())
                        pm[(par, "grid")].append(pr[A_g[pv]].sum())
                        if tag == "real" and t >= 302:
                            x2 = full_nodes[ctxlen + t - 2]
                            x1 = pv
                            if par == 0:
                                lag["even_ring_lag2"].append(pr[A_r[x2]].sum())
                                lag["even_ring_lag1"].append(pr[A_r[x1]].sum())
                                lag["even_grid_lag1"].append(pr[A_g[x1]].sum())
                            else:
                                lag["odd_grid_lag2"].append(pr[A_g[x2]].sum())
                                lag["odd_grid_lag1"].append(pr[A_g[x1]].sum())
                                lag["odd_ring_lag1"].append(pr[A_r[x1]].sum())
                        # phase-split capture of the token AT step t
                        node_t = full_nodes[ctxlen + t]
                        pos_t = 1 + ctxlen + t
                        for L in DEEP:
                            acc[par][L][node_t] += grabbed[L][0][pos_t].float().cpu().numpy()
                        cnt[par][node_t] += 1
            finally:
                for h in handles:
                    h.remove()
            res["phase_pred"][ctxname] = {
                f"par{par}_{g}nbrs": float(np.mean(pm[(par, g)]))
                for par in (0, 1) for g in ("ring", "grid")}
            if tag == "real":
                res["lag"][ctxname] = {k: float(np.mean(v)) for k, v in lag.items() if v}
            geom = {}
            Hs = {}
            for ph in (0, 1):
                c = np.maximum(cnt[ph], 1.0)
                Hs[ph] = {L: acc[ph][L] / c[:, None] for L in DEEP}
            geom["r2"] = {f"par{ph}_{nmf}": float(np.mean(
                [r2(Hs[ph][L] - Hs[ph][L].mean(0), F) for L in DEEP]))
                for ph in (0, 1) for nmf, F in (("ring", cr), ("grid", cg))}
            geom["procrustes_even_vs_odd"] = float(np.mean(
                [psim(Hs[0][L], Hs[1][L]) for L in DEEP]))
            geom["min_occ"] = [int(cnt[0].min()), int(cnt[1].min())]
            res["geom"][ctxname] = geom
            print(f"[{tag}/{ctxname}] done", flush=True)
        out_all[tag] = res

    os.makedirs(os.path.join(_here, "runs", "out_probe"), exist_ok=True)
    path = os.path.join(_here, "runs", "out_probe", "phase_probe.json")
    json.dump(out_all, open(path, "w"), indent=1)
    print(json.dumps(out_all, indent=1))
    print("DONE ->", path, flush=True)


if __name__ == "__main__":
    main()
