"""Analysis of the control conditions (free-gen, one-way influence, real interleaved
steps) against the coupled baseline out_k2_fix. NO RSA.

Per condition/context: deep-mean R^2 (grid coords, ring coords, union modes 1-2 / 1-4)
at base vs joint_late; Procrustes base-vs-late (how much the geometry moved), within-
condition cross-context Procrustes, and behavioral stats where generation happened.

Out: printed report + out_ctrl_summary.json + ctrl_r2.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["runs/out_ctrl_freering", "runs/out_ctrl_freegrid", "runs/out_ctrl_ow_ring2grid",
         "runs/out_ctrl_ow_grid2ring", "runs/out_ctrl_real"]
BASELINE = "runs/out_k2_fix"


def norm_lap_modes(W):
    W = W.astype(float)
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(W)) - di[:, None] * W * di[None, :]
    return np.linalg.eigh(L)


def r2(Hc, F):
    F = np.atleast_2d(np.asarray(F, float).T).T
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1.0 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def unit(X):
    X = X - X.mean(0)
    return X / max(np.linalg.norm(X), 1e-12)


def psim(Aa, Bb):
    Aa, Bb = unit(Aa), unit(Bb)
    Ua, Sa, _ = np.linalg.svd(Aa, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(Bb, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    return float(np.linalg.svd(M, compute_uv=False).sum())


def main():
    zb = np.load(os.path.join(HERE, BASELINE, "nodemeans_dueling.npz"))
    nLb = int(zb["n_layers"][0])
    cg, cr = zb["coords_grid"], zb["coords_ring"]
    Au = (zb["adjacency_grid"] | zb["adjacency_ring"]).astype(float)
    _, Uu = norm_lap_modes(Au)
    FE = {"grid": cg, "ring": cr, "u12": Uu[:, 1:3], "u14": Uu[:, 1:5]}

    def H(z, key, L):
        Hm = z[f"{key}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    def deep_stats(z, nm, nL):
        out = {}
        for win in ("base", "joint_late"):
            out[win] = {f: float(np.mean([r2(H(z, f"{nm}_{win}", L), F)
                                          for L in range(24, nL)])) for f, F in FE.items()}
        out["moved"] = float(np.mean([psim(H(z, f"{nm}_base", L),
                                           H(z, f"{nm}_joint_late", L))
                                      for L in range(24, nL)]))
        return out

    report = {}
    curves = {}
    for cond in CONDS:
        path = os.path.join(HERE, cond, "nodemeans_dueling.npz")
        if not os.path.isfile(path):
            print(f"skip {cond}")
            continue
        z = np.load(path)
        names = [str(x) for x in z["ctx_names"]]
        nL = int(z["n_layers"][0])
        rep = {"ctxs": {}}
        for nm in names:
            rep["ctxs"][nm] = deep_stats(z, nm, nL)
            curves[(cond, nm)] = {
                "grid": [r2(H(z, f"{nm}_joint_late", L), cg) for L in range(nL)],
                "ring": [r2(H(z, f"{nm}_joint_late", L), cr) for L in range(nL)]}
        if len(names) == 2:
            rep["cross_late"] = float(np.mean(
                [psim(H(z, f"{names[0]}_joint_late", L), H(z, f"{names[1]}_joint_late", L))
                 for L in range(24, nL)]))
            rep["cross_base"] = float(np.mean(
                [psim(H(z, f"{names[0]}_base", L), H(z, f"{names[1]}_base", L))
                 for L in range(24, nL)]))
        lp = os.path.join(HERE, cond, "gen_log.json")
        if os.path.isfile(lp):
            log = json.load(open(lp))
            steps = [s for p in log["steps"].values() for s in p if "ring_valid" in s]
            if steps:
                occ = {}
                pair_cnt = {}
                seqs = {}
                for pk, plist in log["steps"].items():
                    seq = [s["node"] for s in plist]
                    for a, b in zip(seq, seq[1:]):
                        k = tuple(sorted((a, b)))
                        pair_cnt[k] = pair_cnt.get(k, 0) + 1
                    for x in seq:
                        occ[x] = occ.get(x, 0) + 1
                tot = sum(pair_cnt.values())
                top = max(pair_cnt.items(), key=lambda kv: kv[1])
                rep["behavior"] = {
                    "ring_valid": float(np.mean([s["ring_valid"] for s in steps])),
                    "grid_valid": float(np.mean([s["grid_valid"] for s in steps])),
                    "top_pair_frac": round(top[1] / tot, 3),
                    "occ_max": max(occ.values()), "occ_min": min(occ.values()) if len(occ) == 16 else 0}
        report[cond] = rep

    # baseline reference (coupled k2/k2 fixed vocab)
    report[BASELINE] = {"ctxs": {nm: deep_stats(zb, nm, nLb) for nm in ("grid", "ring")},
                        "cross_late": float(np.mean(
                            [psim(H(zb, "grid_joint_late", L), H(zb, "ring_joint_late", L))
                             for L in range(24, nLb)]))}

    print(json.dumps(report, indent=2))
    json.dump(report, open(os.path.join(HERE, "figs", "out_ctrl_summary.json"), "w"), indent=2)

    fig, axes = plt.subplots(1, len(curves) // 2 + 1, figsize=(4.2 * (len(curves) // 2 + 1), 4),
                             sharey=True) if False else plt.subplots(
        1, 5, figsize=(21, 4.2), sharey=True)
    panels = [("out_ctrl_freering", None), ("out_ctrl_freegrid", None),
              ("out_ctrl_ow_ring2grid", None), ("out_ctrl_ow_grid2ring", None),
              ("out_ctrl_real", None)]
    for ax, (cond, _) in zip(axes, panels):
        for (c2, nm), cv in curves.items():
            if c2 != cond:
                continue
            ls = "--" if "receiver" in nm else "-"
            ax.plot(cv["grid"], color="#023047", ls=ls,
                    label=f"{nm}: grid coords")
            ax.plot(cv["ring"], color="crimson", ls=ls, label=f"{nm}: ring coords")
        ax.set_title(cond.replace("out_ctrl_", ""), fontsize=10)
        ax.set_xlabel("layer"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0].set_ylabel(r"R$^2$ at joint_late")
    fig.suptitle("Control conditions: late-window coordinate R2 by layer")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"ctrl_r2.{ext}"), dpi=160)
    print("DONE")


if __name__ == "__main__":
    main()
