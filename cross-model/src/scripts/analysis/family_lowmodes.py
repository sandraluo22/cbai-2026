"""For each Markov-chain family: characterise the LOWEST normalized-Laplacian eigenmodes
(the smooth / low-frequency graph-Fourier modes), flag bipartiteness (a grid-parity-style
top mode at lambda~2), and measure which mode the MODEL's representation actually loads onto
across layers -- low-frequency (conforms to Yang et al. low-frequency bias) or a high-frequency
exception like the square grid's parity.

Uses the exact adjacency + per-layer node-means stored in nodemeans_<MODEL>_<fam>.npz.
Normalized Laplacian  Lhat = I - D^-1/2 A D^-1/2,  eigenvalues in [0,2] ascending:
mode 0 = trivial (lambda=0), mode 1 = lowest/smoothest, mode n-1 = highest frequency.
Bipartite  <=>  lambda_max = 2 (an alternating / parity mode exists at the top).

Env: DIR (markov_families dir) MODELS (comma list) OUT (json+pdf stem)
"""
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/1_decomposition/markov_families")
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
OUT = os.environ.get("OUT", "runs/axes/1_decomposition/family_lowmodes")
FAM_ORDER = ["grid", "ring", "smallworld", "tree", "sbm2", "sbm4", "er_random"]


def norm_lap(A):
    A = A.astype(float); d = A.sum(1); dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Lhat = np.eye(len(A)) - (dinv[:, None] * A * dinv[None, :])
    lam, U = np.linalg.eigh(Lhat)                      # ascending; U[:,k] mode k
    return lam, U, d


def degeneracy(lam, k, tol=1e-6):
    return int(np.sum(np.abs(lam - lam[k]) < tol))


def power_spectrum(H, U):
    """fraction of cross-node representation variance carried by each Laplacian mode."""
    Hc = H - H.mean(0)                                  # centre over nodes
    proj = U.T @ Hc                                     # (n_modes x hidden)
    p = (proj ** 2).sum(1)                              # power per mode
    return p


def analyse(model):
    res = {}
    for fam in FAM_ORDER:
        f = f"{DIR}/nodemeans_{model}_{fam}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        A = np.asarray(d["adjacency"], float)
        n = len(A)
        lam, U, deg = norm_lap(A)
        bipartite = bool(lam[-1] > 2 - 1e-6)
        nL = sum(k.startswith("layer_") for k in d.files)

        # per-layer power spectrum over non-trivial modes (1..n-1)
        best = {"layer": None, "mode": None, "frac": -1.0}
        by_layer_low = []   # fraction in lowest 3 modes, per layer
        for L in range(nL):
            H = d[f"layer_{L}"].astype(np.float32)
            p = power_spectrum(H, U)
            p_nontriv = p[1:]
            frac = p_nontriv / (p_nontriv.sum() + 1e-12)   # over modes 1..n-1
            by_layer_low.append(float(frac[:3].sum()))
            top = int(frac.argmax()) + 1                    # mode index (1-based into modes)
            if frac.max() > best["frac"]:
                best = {"layer": L, "mode": top, "frac": float(frac.max())}
        # spectrum at the peak layer (for plotting / reporting)
        Hpk = d[f"layer_{best['layer']}"].astype(np.float32)
        p = power_spectrum(Hpk, U); p_nt = p[1:]; frac_pk = (p_nt / (p_nt.sum() + 1e-12))

        dom_mode = best["mode"]                             # 1..n-1
        res[fam] = {
            "n": n, "edges": int(A.sum() // 2), "bipartite": bipartite,
            "lam_low": [float(x) for x in lam[1:4]],        # 3 lowest non-trivial eigenvalues
            "deg_low": [degeneracy(lam, k) for k in (1, 2, 3)],
            "lam_max": float(lam[-1]),
            "dom_layer": best["layer"], "dom_mode_rank": dom_mode,
            "dom_lambda": float(lam[dom_mode]), "dom_frac": best["frac"],
            "dom_is_high": bool(dom_mode >= n - 3),         # near the top of the spectrum
            "frac_pk": [float(x) for x in frac_pk],
            "lam_all": [float(x) for x in lam[1:]],
        }
    return res


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    allres = {m: analyse(m) for m in MODELS}
    json.dump(allres, open(f"{OUT}.json", "w"), indent=2)

    # ---- figure: per-family power-vs-eigenmode (Llama), low freq = left ----
    model = MODELS[0]; R = allres[model]
    fams = [f for f in FAM_ORDER if f in R]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.05 * len(fams), 2.9), sharey=True)
    for ax, fam in zip(axes, fams):
        r = R[fam]; frac = np.array(r["frac_pk"]); modes = np.arange(1, len(frac) + 1)
        cols = ["#C2410C" if (m >= r["n"] - 2) else "#1D4ED8" for m in modes]  # top = red
        ax.bar(modes, frac, color=cols, width=0.9)
        dm = r["dom_mode_rank"]
        ax.axvline(dm, color="#111827", lw=0.8, ls=":")
        bip = "bipartite" if r["bipartite"] else "not bip."
        ax.set_title(f"{fam}\n(dom mode {dm}/{r['n']-1}, {bip})", fontsize=8)
        ax.set_xlabel("mode  (low→high freq)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("frac. of rep. power", fontsize=8)
    fig.suptitle(f"{model}: which Laplacian mode the representation loads onto "
                 f"(blue=low freq, red=top/parity mode)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}_{model}.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", f"{OUT}.json", f"{OUT}_{model}.pdf/.png")

    # ---- console summary table ----
    print(f"\n{'family':11} {'n':>2} {'edges':>5} {'bip':>4} | "
          f"{'lam_low(1,2,3)':>22} lam_max | dom: rank/λ  frac  layer  low/HIGH")
    for m in MODELS:
        print(f"--- {m} ---")
        for fam in FAM_ORDER:
            if fam not in allres[m]:
                continue
            r = allres[m][fam]
            ll = ",".join(f"{x:.3f}" for x in r["lam_low"])
            hi = "HIGH" if r["dom_is_high"] else "low"
            print(f"{fam:11} {r['n']:>2} {r['edges']:>5} {str(r['bipartite'])[0]:>4} | "
                  f"{ll:>22} {r['lam_max']:.3f} | mode {r['dom_mode_rank']:>2}/"
                  f"{r['n']-1} λ={r['dom_lambda']:.2f} {r['dom_frac']*100:>4.0f}%  L{r['dom_layer']:>2}  {hi}")


if __name__ == "__main__":
    main()
