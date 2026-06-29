"""Cross-model activation patching: does patch efficacy track cross-model RSA?

For a (source A -> target B) model pair we fit a map  a_A(L_a) -> a_B(L_b)  on the
SAME (walk, step) node occurrences, then, during a target-B forward pass on HELD-OUT
walks, at a node token (block L_b, position t) we replace B's residual stream with the
reconstruction and measure the effect on B's next-step prediction:

  - neighbour accuracy : argmax over the 16 node-words is a graph neighbour of node_s
  - KL(clean || patched): forward KL over the 16 node-word next-token distribution
                          (0 = patch leaves B unchanged -> a faithful cross-model sub)

We sweep a small  layer x context-length  grid and ask whether efficacy tracks the
cross-model node-geometry RSA at that layer/context.

RECONSTRUCTION MODES (env MAP_MODES, comma list):
  ridge  : full-space ridge map  W: a_A -> a_B            (d_A x d_B params; overfits)
  pc2    : rank-2 reconstruction through the target's top-2 PC plane of node means
  best2d : rank-2 reconstruction through the target's BEST-2D geometry plane
           (top-6 PCs of node means regressed onto the graph layout coords, as in
            scripts/viz/best_2d.py). Tests whether the GEOMETRY 2-plane is the causal
            carrier -> efficacy should track RSA if so.
For 2-D modes the source predicts the target's 2 plane-coordinates (ridge a_A -> 2),
and the patch vector is  mean_B + pred_coords . plane_basis^T  (rank-2 + mean).

CONTROL (shuffled-pairing): the same map refit on PERMUTED A->B correspondence (carries
no real per-occurrence structure -> collapses toward a mean patch). real - control = signal.

Pair (default): Qwen (source) -> Llama (target), across square_grid / ring / hex. Runs on
the pod (gated 8-9B models; ungated mirrors). PRESET=smoke uses distilgpt2 for both (CPU).

Env: PRESET NWALKS(24) WLEN(400) TESTFRAC(0.34) ALPHA(1e3) MAP_MODES(pc2,best2d)
     OUTDIR DEVICE GRAPHS(square_grid,ring,hex) RELDEPTHS CKPTS
Out: <OUTDIR>/all_patch_<mode>.json  and  <OUTDIR>/slides/patch_slides_<mode>.pdf
"""
from __future__ import annotations
import os, sys, json, gc
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from align import fit_ridge, split_by_walk, r2

# ---- pair / preset -------------------------------------------------------
PRESET = os.environ.get("PRESET", "gemma_qwen")
if PRESET == "smoke":
    SOURCE = ("stub", "distilgpt2", None)
    TARGET = ("stub2", "distilgpt2", None)
else:
    SOURCE = ("Qwen", "Qwen/Qwen3-8B-Base", None)
    TARGET = ("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B")

GRAPH_KW = {
    "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
    "ring":        dict(graph_type="ring", ring_size=16),
    "hex":         dict(graph_type="hex", hex_rows=4, hex_cols=4),
}

NWALKS   = int(os.environ.get("NWALKS", "24"))
WLEN     = int(os.environ.get("WLEN", "400"))
TESTFRAC = float(os.environ.get("TESTFRAC", "0.34"))
ALPHA    = float(os.environ.get("ALPHA", "1e3"))
OUTDIR   = os.environ.get("OUTDIR", "/root/cmrun/patch" if PRESET != "smoke" else "runs/smoke_patch")
RELDEPTHS = [float(x) for x in os.environ.get("RELDEPTHS", "0.35,0.5,0.65,0.8").split(",")]
CKPTS    = [int(x) for x in os.environ.get("CKPTS", "20,60,150,300").split(",")]
MAP_MODES = [m.strip() for m in os.environ.get("MAP_MODES", "pc2,best2d").split(",") if m.strip()]
WINDOW   = 0.2
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception as e:
        if mirror:
            print(f"[{tag}] {hf} unavailable ({type(e).__name__}); using mirror {mirror}", flush=True)
            return M.load_model(mirror, cfg)
        raise


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def node_means_matrix(X, node, mask, n):
    return np.stack([X[mask & (node == k)].mean(0) if (mask & (node == k)).any()
                     else np.full(X.shape[1], np.nan) for k in range(n)])


def node_mean_rdm(X, node, mask, n):
    iu = np.triu_indices(n, 1)
    H = node_means_matrix(X, node, mask, n)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def cross_rsa(Xa, Xb, node, mask, n):
    return sp(node_mean_rdm(Xa, node, mask, n), node_mean_rdm(Xb, node, mask, n))


def pca2_plane(H):
    """Top-2 PC directions of node means -> orthonormal [d, 2] (best_2d.py top row)."""
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Vt[:2].T


def best2d_plane(H, Gc):
    """best-2D geometry plane: top-6 PCs of node means regressed onto graph layout
    coords, orthonormalised -> [d, 2] (best_2d.py bottom row)."""
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0])
    Z = (U[:, :k] * S[:k])
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]      # [k, 2]
    B = Vt[:k].T @ W                                            # [d, 2] geometry dirs
    Q, _ = np.linalg.qr(B)                                      # orthonormal plane basis
    return Q


def layer_pairs(na, nb):
    seen, out = set(), []
    for rd in RELDEPTHS:
        la, lb = round(rd * (na - 1)), round(rd * (nb - 1))
        if (la, lb) not in seen:
            seen.add((la, lb)); out.append((rd, la, lb))
    return out


def kl(p, q, eps=1e-9):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def patch_hook(pos, vec):
    def hook(_m, _i, out):
        hs = (out[0] if isinstance(out, tuple) else out).clone()
        hs[0, pos, :] = vec.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return hook


@torch.no_grad()
def next_token_probs(model, ids, pos, cand_t):
    """Renormalised distribution over the 16 candidate node-words at `pos`."""
    logits = model(input_ids=ids).logits[0]
    return torch.softmax(logits[pos][cand_t].float(), dim=0).cpu().numpy()


@torch.no_grad()
def patched_probs(model, blocks, Lb, ids, pos, vec, cand_t):
    h = blocks[Lb].register_forward_hook(patch_hook(pos, vec))
    try:
        return next_token_probs(model, ids, pos, cand_t)
    finally:
        h.remove()


def build_reconstructor(Xq, Xl, train, test, mode, B):
    """Return predict_real / predict_ctrl (source-act[N,d_a] -> target vec[N,d_b]) and
    held-out R^2. mode 'ridge' = full map; '2d' modes reconstruct through plane B[d_b,2]."""
    n_tr = int(train.sum())
    perm = RNG.permutation(n_tr)
    if B is None:                                              # full-space ridge
        rm = fit_ridge(Xq[train], Xl[train], ALPHA)
        rm_sh = fit_ridge(Xq[train], Xl[train][perm], ALPHA)
        return dict(predict_real=rm.predict, predict_ctrl=rm_sh.predict,
                    r2_train=r2(Xl[train], rm.predict(Xq[train])),
                    r2_test=r2(Xl[test], rm.predict(Xq[test])),
                    r2_test_shuf=r2(Xl[test], rm_sh.predict(Xq[test])))
    mean_l = Xl.mean(0)
    T = (Xl - mean_l) @ B                                     # [N, 2] target plane coords
    rm = fit_ridge(Xq[train], T[train], ALPHA)
    rm_sh = fit_ridge(Xq[train], T[train][perm], ALPHA)
    return dict(predict_real=lambda A: mean_l + rm.predict(A) @ B.T,
                predict_ctrl=lambda A: mean_l + rm_sh.predict(A) @ B.T,
                r2_train=r2(T[train], rm.predict(Xq[train])),
                r2_test=r2(T[test], rm.predict(Xq[test])),
                r2_test_shuf=r2(T[test], rm_sh.predict(Xq[test])))


def run_graph(gname, cfg_base, dev, modes):
    cfg = replace(cfg_base, **GRAPH_KW[gname], n_walks=NWALKS, walk_length=WLEN)
    graph = G.build_graph(cfg); n = graph.n_nodes
    walks = G.generate_walks(graph, cfg)
    words = cfg.words()
    Gc = np.array(graph.coords, float)

    print(f"[{gname}] capture source {SOURCE[0]}", flush=True)
    ma, toka = load_with_fallback(*SOURCE, cfg)
    na = ma.config.num_hidden_layers
    la_set = sorted({round(rd * (na - 1)) for rd in RELDEPTHS})
    capA = M.capture(ma, toka, walks, tuple(la_set), cfg, device=dev)
    del ma, toka; gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{gname}] capture target {TARGET[0]}", flush=True)
    mb, tokb = load_with_fallback(*TARGET, cfg)
    nb = mb.config.num_hidden_layers
    pairs = layer_pairs(na, nb)
    lb_set = sorted({lb for _, _, lb in pairs})
    capB = M.capture(mb, tokb, walks, tuple(lb_set), cfg, device=dev)
    blocks = M._decoder_blocks(mb)

    meta = capA.meta
    assert np.array_equal(meta["walk_id"], capB.meta["walk_id"])
    assert np.array_equal(meta["step"], capB.meta["step"])
    node = meta["node"]; cl = meta["context_length"]; wid = meta["walk_id"]; step = meta["step"]
    train, test = split_by_walk(meta, TESTFRAC, cfg.seed)
    row_of = {(int(w), int(s)): i for i, (w, s) in enumerate(zip(wid, step))}
    cand_t = torch.tensor([tokb(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
    eval_wids = set(int(w) for w in wid[test])

    # cross-model RSA per (pair, ckpt) -- shared across modes (the x-axis)
    rsa = {}
    for rd, la, lb in pairs:
        for C in CKPTS:
            m = (cl >= C * (1 - WINDOW)) & (cl <= C * (1 + WINDOW))
            rsa[(la, lb, C)] = cross_rsa(capA.acts[la], capB.acts[lb], node, m, n) if m.sum() >= n else float("nan")

    # reconstructors per (mode, pair)
    recon = {mode: {} for mode in modes}
    for rd, la, lb in pairs:
        Xq = capA.acts[la].astype(np.float64); Xl = capB.acts[lb].astype(np.float64)
        Hl = node_means_matrix(Xl, node, (cl >= 30), n)        # target node means (geometry plane)
        planes = {"ridge": None, "pc2": pca2_plane(Hl), "best2d": best2d_plane(Hl, Gc)}
        for mode in modes:
            f = build_reconstructor(Xq, Xl, train, test, mode, planes[mode])
            recon[mode][(la, lb)] = f
            print(f"[{gname}/{mode}] L{la}->L{lb}: R2 test={f['r2_test']:+.3f} "
                  f"(shuf={f['r2_test_shuf']:+.3f})", flush=True)

    # patching eval (clean cached once per (walk,pos); patched per mode/pair)
    acc = {mode: {(la, lb, C): {k: [] for k in
                  ("clean", "real", "ctrl", "kl_real", "kl_ctrl", "agree_real", "agree_ctrl")}
                  for _, la, lb in pairs for C in CKPTS} for mode in modes}
    for wi, wk in enumerate(walks):
        if wi not in eval_wids:
            continue
        ids = tokb(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
        spans = resolve_token_spans(tokb, wk)
        for C in CKPTS:
            s = C - 1
            if s < 0 or s > len(wk.nodes) - 2:
                continue
            pos = spans[s + 1][0] - 1
            nbrs = graph.neighbors(wk.nodes[s])
            P_clean = next_token_probs(mb, ids, pos, cand_t)
            a_clean = int(int(P_clean.argmax()) in nbrs); clean_arg = int(P_clean.argmax())
            r = row_of[(wi, s)]
            for rd, la, lb in pairs:
                a_src = capA.acts[la][r][None].astype(np.float64)
                for mode in modes:
                    f = recon[mode][(la, lb)]
                    v_real = torch.tensor(f["predict_real"](a_src)[0], device=dev)
                    v_ctrl = torch.tensor(f["predict_ctrl"](a_src)[0], device=dev)
                    P_real = patched_probs(mb, blocks, lb, ids, pos, v_real, cand_t)
                    P_ctrl = patched_probs(mb, blocks, lb, ids, pos, v_ctrl, cand_t)
                    d = acc[mode][(la, lb, C)]
                    d["clean"].append(a_clean)
                    d["real"].append(int(int(P_real.argmax()) in nbrs))
                    d["ctrl"].append(int(int(P_ctrl.argmax()) in nbrs))
                    d["kl_real"].append(kl(P_clean, P_real))
                    d["kl_ctrl"].append(kl(P_clean, P_ctrl))
                    d["agree_real"].append(int(int(P_real.argmax()) == clean_arg))
                    d["agree_ctrl"].append(int(int(P_ctrl.argmax()) == clean_arg))

    out = {mode: [] for mode in modes}
    for mode in modes:
        for rd, la, lb in pairs:
            f = recon[mode][(la, lb)]
            for C in CKPTS:
                d = acc[mode][(la, lb, C)]
                if not d["clean"]:
                    continue
                out[mode].append({
                    "graph": gname, "mode": mode, "rel_depth": rd, "L_src": la, "L_tgt": lb,
                    "context": C, "n": len(d["clean"]), "rsa": rsa[(la, lb, C)],
                    "r2_train": f["r2_train"], "r2_test": f["r2_test"], "r2_test_shuf": f["r2_test_shuf"],
                    "acc_clean": float(np.mean(d["clean"])), "acc_real": float(np.mean(d["real"])),
                    "acc_ctrl": float(np.mean(d["ctrl"])),
                    "kl_real": float(np.mean(d["kl_real"])), "kl_ctrl": float(np.mean(d["kl_ctrl"])),
                    "agree_real": float(np.mean(d["agree_real"])), "agree_ctrl": float(np.mean(d["agree_ctrl"])),
                })

    del mb, tokb, capA, capB; gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    cfg_base = replace(get_config(PRESET), device=dev)
    os.makedirs(OUTDIR, exist_ok=True)
    graphs = os.environ.get("GRAPHS", ",".join(GRAPH_KW)).split(",")
    by_mode = {mode: [] for mode in MAP_MODES}
    for g in graphs:
        out = run_graph(g, cfg_base, dev, MAP_MODES)
        for mode in MAP_MODES:
            by_mode[mode].extend(out[mode])
            json.dump(out[mode], open(f"{OUTDIR}/{g}_patch_{mode}.json", "w"), indent=2)
        print(f"[{g}] done ({', '.join(f'{m}:{len(out[m])}' for m in MAP_MODES)} cells)", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz"))
    from patch_slides import build_slides
    for mode in MAP_MODES:
        json.dump(by_mode[mode], open(f"{OUTDIR}/all_patch_{mode}.json", "w"), indent=2)
        build_slides(by_mode[mode], f"{OUTDIR}/slides/patch_slides_{mode}.pdf", SOURCE[0], TARGET[0], tag=mode)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
