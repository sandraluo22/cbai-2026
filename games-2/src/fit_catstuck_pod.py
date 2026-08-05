"""Finish the category->stuckness probe locally from resid_cache.npz.

Replicates probe_category_stuck.py's loco_r2 exactly (leave-one-category-out,
nested alpha grid on every-4th training category) but in dual form with one
eigendecomposition per (layer, fold), amortizing all alphas. Parallel over layers.
"""
import json
import os
import sys
import numpy as np

D = "/workspace/mm/games-2/runs/probe_catstuck"
ALPHAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

_z = np.load(os.path.join(D, "resid_cache.npz"))
X_ALL = _z["X"]   # (n, L, d) fp16
Y_RUN = _z["y_run"].astype(np.float64)
Y_FAM = _z["y_fam"].astype(np.float64)
Y_CAT = _z["y_cat"].astype(np.float64)
CID = _z["cat_id"]
MC = _z["mc"].astype(bool)
N, NL, DD = X_ALL.shape
print(f"states {N}, layers {NL}, dim {DD}, mc {MC.sum()}", flush=True)

rng = np.random.default_rng(2)
Y_RUN_S = rng.permutation(Y_RUN)
YF = Y_FAM[MC]; YC = Y_CAT[MC]
YF_S = rng.permutation(YF); YC_S = rng.permutation(YC)
CID_L = list(CID)
CID_MC = [c for c, m in zip(CID_L, MC) if m]


def dual_pred_multi(Ztr, ys_tr, Zte, alphas, ev=None):
    """Predictions for several targets x several alphas from one eig of the gram."""
    if ev is None:
        K = Ztr @ Ztr.T
        w, V = np.linalg.eigh(K)
    else:
        w, V = ev
    Kte = Zte @ Ztr.T                      # (nte, ntr)
    out = {}
    for a in alphas:
        inv_diag = 1.0 / (w + a)
        # coef = (K + aI)^-1 (y - ymean)  computed per target
        M = V * inv_diag                   # (ntr, ntr) scaled eigvecs
        for ti, y in enumerate(ys_tr):
            ym = y.mean()
            c = M @ (V.T @ (y - ym))
            out[(a, ti)] = Kte @ c + ym
    return out


def loco_r2_dual(X, ys, cats):
    """ys: list of targets sharing X/cats. Returns list of R2."""
    ucats = sorted(set(cats))
    cats = np.array(cats)
    yhat = [np.zeros(len(y)) for y in ys]
    mu_g, sd_g = None, None
    for hold in ucats:
        tr = cats != hold
        te = ~tr
        tc = sorted(set(cats[tr]))
        inner = set(tc[::4]) or {tc[0]}
        itr = tr & ~np.isin(cats, list(inner))
        iva = tr & np.isin(cats, list(inner))
        mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
        Zi = (X[itr] - mu) / sd
        Zv = (X[iva] - mu) / sd
        ys_i = [y[itr] for y in ys]
        pv = dual_pred_multi(Zi, ys_i, Zv, ALPHAS)
        best_a = []
        for ti, y in enumerate(ys):
            yv = y[iva]
            best, ba = -np.inf, ALPHAS[0]
            for a in ALPHAS:
                p = pv[(a, ti)]
                r2 = 1 - ((yv - p) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-9)
                if r2 > best:
                    best, ba = r2, a
            best_a.append(ba)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Zt = (X[tr] - mu) / sd
        Ze = (X[te] - mu) / sd
        K = Zt @ Zt.T
        ev = np.linalg.eigh(K)
        pt = dual_pred_multi(Zt, [y[tr] for y in ys], Ze, sorted(set(best_a)), ev=ev)
        for ti, y in enumerate(ys):
            yhat[ti][te] = pt[(best_a[ti], ti)]
    return [1 - ((y - yh) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
            for y, yh in zip(ys, yhat)]


def fit_layer(L):
    XL = X_ALL[:, L, :].astype(np.float64)
    XM = XL[MC]
    r_run = loco_r2_dual(XL, [Y_RUN, Y_RUN_S], CID_L)
    r_mc = loco_r2_dual(XM, [YF, YF_S, YC, YC_S], CID_MC)
    print(f"L{L}: famrun {r_run[0]:.3f} (sh {r_run[1]:.3f}) fam {r_mc[0]:.3f} cat {r_mc[2]:.3f}", flush=True)
    return L, r_run, r_mc


if __name__ == "__main__":
    layers = list(range(NL))
    out = {"n_states": N, "n_mc": int(MC.sum()), "n_cats": len(set(CID_L)), "layers": NL,
           "alphas": ALPHAS,
           "r2": {"famrun": [None] * NL, "fam_mass": [None] * NL, "cat_mass": [None] * NL},
           "r2_shuffled": {"famrun": [None] * NL, "fam_mass": [None] * NL, "cat_mass": [None] * NL}}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=12) as ex:
        for L, r_run, r_mc in ex.map(fit_layer, layers):
            out["r2"]["famrun"][L] = r_run[0]
            out["r2_shuffled"]["famrun"][L] = r_run[1]
            out["r2"]["fam_mass"][L] = r_mc[0]
            out["r2_shuffled"]["fam_mass"][L] = r_mc[1]
            out["r2"]["cat_mass"][L] = r_mc[2]
            out["r2_shuffled"]["cat_mass"][L] = r_mc[3]
            json.dump(out, open(os.path.join(D, "catstuck_probes_v2.json"), "w"))
    for k in ("famrun", "fam_mass", "cat_mass"):
        v = out["r2"][k]
        b = int(np.argmax(v))
        print(f"BEST {k}: R2={v[b]:.3f} @L{b}")
    print("DONE")
