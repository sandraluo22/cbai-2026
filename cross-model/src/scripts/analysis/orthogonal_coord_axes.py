"""How many ORTHOGONAL directions carry coordinates, and which of them are CAUSAL?

Two objections this addresses. (1) A linear probe finds a DECODABLE direction, not necessarily one the
model uses — so probe rank says nothing about causal rank. (2) A single rank-2 probe plane cannot tell
whether the code is genuinely 2-dimensional or spread redundantly across many directions.

Ladder construction: fit a ridge probe to (row, col), orthonormalise its 2 weight vectors, record them,
project them OUT of the representation, refit on the residual, repeat. Every rung is orthogonal to all
previous rungs BY CONSTRUCTION, so "coordinates still decodable at rung 3" means genuinely independent
coordinate information, not a rotation of rung 1.

Then each rung gets a CAUSAL test, which is the part a probe alone cannot give:
    steer   add +/- scale * direction to the residual at LAYER, and measure the change in the model's
            EXPECTED OUTPUT COORDINATE, E[coord] = sum_w p(w) * coord(w) over the node words.
            dE_row/dE_col are the same quantity the earlier steer_probe run reported (row axis moved the
            output row by ~1.85 with 0.15 col cross-talk; random directions gave ~0.01).
    A rung that decodes but does NOT steer is representational only. A rung that steers is used.

Also evaluated for comparison, on the same footing:
    das      directions from the saved residual-DAS npz (causal BY CONSTRUCTION — the subspace had to
             support an interchange intervention), so we can ask whether DAS and probe axes coincide
    random   matched random directions = the steering null

Env: GEN_MODEL(Llama) K(4) LAYER(14) RUNGS(5) SCALES("2,4") ALPHA(1e3) NWALKS(3) WLEN(1200)
     CTXLO(800) DAS_NPZ DAS_KEY SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/orthogonal_coord_axes<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4")); LAYER = int(os.environ.get("LAYER", "14"))
RUNGS = int(os.environ.get("RUNGS", "5"))
SCALES = [float(x) for x in os.environ.get("SCALES", "2,4").split(",")]
ALPHA = float(os.environ.get("ALPHA", "1e3"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); SEED = int(os.environ.get("SEED", "0"))
DAS_NPZ = os.environ.get("DAS_NPZ", ""); DAS_KEY = os.environ.get("DAS_KEY", "4x4_r4")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def ridge_lono(X, Y, nodes, n, alpha):
    d = X.shape[1]; preds = np.zeros_like(Y)
    for c in range(n):
        te = nodes == c
        if not te.any() or te.all(): continue
        tr = ~te; Xt, Yt = X[tr], Y[tr]; mu = Xt.mean(0); Xc = Xt - mu
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ (Yt - Yt.mean(0)))
        preds[te] = (X[te] - mu) @ W + Yt.mean(0)
    ss = ((Y - preds) ** 2).sum(0); sv = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = float(np.mean(1 - ss / np.maximum(sv, 1e-12)))
    mu = X.mean(0); Xc = X - mu
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ (Y - Y.mean(0)))
    return r2, W


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(SEED)

    data = []
    for w in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(w.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in w.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [w.nodes[s] for s in steps]))

    st = {"proj": None, "add": None}
    hooks = []
    for l in range(cm.num_hidden_layers):
        def mk(l):
            def rh(_m, _i, out):
                if l != LAYER or (st["proj"] is None and st["add"] is None): return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); f = h[0].float()
                if st["proj"] is not None:
                    P = st["proj"]; f = f - (f @ P.t()) @ P
                if st["add"] is not None: f = f + st["add"]
                h[0] = f.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return rh
        hooks.append(blocks[l].register_forward_hook(mk(l)))

    def collect():
        X, Yn = [], []
        for ids, rp, nds in data:
            o = model(input_ids=ids, output_hidden_states=True)
            H = o.hidden_states[LAYER + 1][0, rp].float().cpu().numpy()
            X.append(H); Yn += list(nds)
        X = np.concatenate(X); Yn = np.array(Yn)
        sel = rng.permutation(len(X))[:1200]
        return X[sel], Yn[sel]

    def expected_coord():
        """E[coord] = sum_w p(w) coord(w), averaged over readouts"""
        tot = torch.zeros(2, device=dev); m = 0
        for ids, rp, nds in data:
            p = torch.softmax(model(input_ids=ids).logits[0][rp][:, cand_t].float(), 1)
            tot += (p @ coords_t).sum(0); m += p.shape[0]
        return (tot / m).cpu().numpy()

    base_E = expected_coord()
    dev_norm = None
    rows = []
    Xc, Yc = collect()
    Bacc = np.zeros((0, cm.hidden_size))
    print(f"[{tag}] L{LAYER}  baseline E[coord] = [{base_E[0]:.3f}, {base_E[1]:.3f}]")
    print(f"\n{'rung':>5} {'cv_R2':>8} {'dE_row':>9} {'dE_col':>9} {'|dE|':>7}  interpretation")
    for r in range(1, RUNGS + 1):
        Xr = Xc - (Xc @ Bacc.T) @ Bacc if len(Bacc) else Xc
        r2, W = ridge_lono(Xr, coords[Yc], Yc, n, ALPHA)
        B = np.linalg.qr(W)[0].T                                  # 2 orthonormal rows
        if len(Bacc): B = B - (B @ Bacc.T) @ Bacc
        B = np.linalg.qr(B.T)[0].T
        if dev_norm is None:
            dev_norm = float(np.linalg.norm(Xc - Xc.mean(0), axis=1).mean())
        dE = []
        for a in range(2):
            v = torch.tensor(B[a] * dev_norm, dtype=torch.float32, device=dev)
            e = []
            for s in SCALES:
                st["add"] = s * v; e.append(expected_coord() - base_E); st["add"] = None
            dE.append(np.mean(e, 0))
        mag = float(np.abs(np.array(dE)).max())
        tagm = ("decodes AND steers" if r2 > 0.05 and mag > 0.3 else
                "decodes, does NOT steer" if r2 > 0.05 else
                "steers without decoding" if mag > 0.3 else "neither")
        print(f"{r:5} {r2:8.4f} {dE[0][0]:+9.3f} {dE[1][1]:+9.3f} {mag:7.3f}  {tagm}", flush=True)
        rows.append({"rung": r, "cv_r2": round(r2, 4),
                     "dE_axis0": [round(float(x), 4) for x in dE[0]],
                     "dE_axis1": [round(float(x), 4) for x in dE[1]],
                     "max_abs_dE": round(mag, 4), "verdict": tagm})
        Bacc = np.vstack([Bacc, B])

    # ---- controls: random directions, and DAS directions if available ----
    ctrl = {}
    rv = []
    for _ in range(3):
        q = np.linalg.qr(rng.standard_normal((cm.hidden_size, 2)))[0].T
        d2 = []
        for a in range(2):
            v = torch.tensor(q[a] * dev_norm, dtype=torch.float32, device=dev)
            e = [expected_coord() - base_E for s in SCALES
                 for _ in [st.__setitem__("add", s * v)]]
            st["add"] = None; d2.append(np.mean(e, 0))
        rv.append(float(np.abs(np.array(d2)).max()))
    ctrl["random_max_abs_dE"] = round(float(np.mean(rv)), 4)
    print(f"\nrandom-direction steering null: |dE| = {np.mean(rv):.4f}")
    if DAS_NPZ and os.path.exists(DAS_NPZ):
        z = np.load(DAS_NPZ)
        if DAS_KEY in z.files:
            R = z[DAS_KEY].astype(np.float64)
            Q = np.linalg.qr(R.T)[0].T[:2]
            d2 = []
            for a in range(min(2, len(Q))):
                v = torch.tensor(Q[a] * dev_norm, dtype=torch.float32, device=dev)
                e = []
                for s in SCALES:
                    st["add"] = s * v; e.append(expected_coord() - base_E); st["add"] = None
                d2.append(np.mean(e, 0))
            ctrl["das_max_abs_dE"] = round(float(np.abs(np.array(d2)).max()), 4)
            cosb = float(np.abs(Q[:2] @ Bacc[:2].T).max())
            ctrl["das_vs_probe_rung1_maxcos"] = round(cosb, 4)
            print(f"DAS subspace ({DAS_KEY}): |dE| = {ctrl['das_max_abs_dE']:.4f}, "
                  f"max |cos| with probe rung 1 = {cosb:.4f}")
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/orthogonal_coord_axes{OUTTAG}_{tag}.json"
    json.dump({"model": tag, "layer": LAYER, "baseline_E": base_E.tolist(),
               "rungs": rows, "controls": ctrl}, open(p_, "w"), indent=2)
    print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
