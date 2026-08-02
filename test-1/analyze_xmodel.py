"""Analysis for cross-MODEL dueling runs (run_xmodel.py). NO RSA.

Handles instances with different depths/widths (Llama 32L/4096, Qwen 36L/4096,
Gemma 42L/3584): R^2 curves are per-instance over its own layers with a deep window =
last quarter; cross-instance Procrustes pairs layers by RELATIVE depth (shape similarity
is computed in the 16-node row space, so differing hidden sizes are fine).

Env: RUN_OUT
Out: <RUN_OUT>/x_summary.json (+ printed headline)
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT")
PRIMED_WINS = ("base", "joint_early", "joint_mid", "joint_late")


def r2_from_feats(Hc, F):
    F = np.atleast_2d(np.asarray(F, float).T).T
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1.0 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def unit_shape(X):
    X = X - X.mean(0)
    return X / max(np.linalg.norm(X), 1e-12)


def procrustes_sim(A, B):
    A, B = unit_shape(A), unit_shape(B)
    Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(B, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    return float(np.linalg.svd(M, compute_uv=False).sum())


def main():
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"), allow_pickle=False)
    log = json.load(open(os.path.join(OUT, "gen_log.json")))
    names = [str(x) for x in z["ctx_names"]]
    fresh = [str(x) for x in z["fresh_names"]]
    words = [str(w) for w in z["words"]]
    n = len(words)
    nl = {nm: int(z[f"nlayers_{nm}"][0]) for nm in names + fresh}
    graphs = {}                                       # graph name -> (adjacency, coords)
    for nm in names:
        g = nm.split("-", 1)[1]
        graphs[g] = (z[f"adjacency_{nm}"].astype(bool), z[f"coords_{nm}"])
    P, T = log["npairs"], log["tgen"]
    joint = np.array([[log["steps"][f"pair{p}"][t]["node"] for t in range(T)]
                      for p in range(P)])

    A_union = np.zeros((n, n), bool)
    for A, _ in graphs.values():
        A_union |= A
    C = np.zeros((n, n))
    for p in range(P):
        for t in range(301, T):
            C[joint[p, t - 1], joint[p, t]] += 1
    W = C + C.T
    np.fill_diagonal(W, 0)
    mass = W.sum()
    stream = {g: float(W[graphs[g][0]].sum() / mass) for g in graphs}
    stream["union"] = float(W[A_union].sum() / mass)
    cnt = {}
    for p in range(P):
        for t in range(301, T):
            k = tuple(sorted((joint[p, t - 1], joint[p, t])))
            cnt[k] = cnt.get(k, 0) + 1
    tot = P * (T - 301)
    top_pairs = [(words[a], words[b], round(c / tot, 3))
                 for (a, b), c in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]]

    def H(nm, win, L):
        Hm = z[f"{nm}_{win}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    def deep(nm):
        L1 = nl[nm]
        return range(int(0.75 * L1), L1)

    # per-instance coordinate R^2, base vs late, deep-mean
    r2 = {}
    for nm in names:
        wins = {"base": "base", "late": "joint_late"}
        r2[nm] = {wl: {g: float(np.mean([r2_from_feats(H(nm, wz, L), graphs[g][1])
                                         for L in deep(nm)]))
                       for g in graphs} for wl, wz in wins.items()}
    for nm in fresh:
        r2[nm] = {"late": {g: float(np.mean([r2_from_feats(H(nm, "late", L),
                                                           graphs[g][1])
                                             for L in deep(nm)]))
                           for g in graphs}}

    # cross-instance Procrustes at matched relative depth (deep band)
    def sim_pair(a, wa, b, wb):
        fr = np.linspace(0.75, 0.97, 8)
        vals = []
        for f in fr:
            La = int(round(f * (nl[a] - 1)))
            Lb = int(round(f * (nl[b] - 1)))
            vals.append(procrustes_sim(H(a, wa, La), H(b, wb, Lb)))
        return float(np.mean(vals))

    sims = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sims[f"{a} vs {b} | base"] = sim_pair(a, "base", b, "base")
            sims[f"{a} vs {b} | late"] = sim_pair(a, "joint_late", b, "joint_late")
    for nm in names:
        tag = nm.split("-")[0]
        sims[f"{nm} vs fresh-{tag} | late"] = sim_pair(nm, "joint_late",
                                                       f"fresh-{tag}", "late")
        sims[f"{nm} | base vs late"] = sim_pair(nm, "base", nm, "joint_late")
    for i in range(len(fresh)):
        for j in range(i + 1, len(fresh)):
            sims[f"{fresh[i]} vs {fresh[j]} | late"] = \
                sim_pair(fresh[i], "late", fresh[j], "late")

    summary = {"instances": log["instances"], "stream_mass_late": stream,
               "top_oscillating_pairs": top_pairs,
               "r2_deep": r2, "procrustes_deep": sims}
    json.dump(summary, open(os.path.join(OUT, "x_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
