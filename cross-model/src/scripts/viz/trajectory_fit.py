"""Curve-fit the layer-correspondence 'trajectory' (bright ridge) in the
cross-model RSA heatmaps, for ALL model pairs and graphs.

Ridge: b_hat(a) = sum_b w_b b / sum_b w_b,  w_b = max(RSA(a,b) - THR, 0).
Fits (NO quadratic): linear  b = m a + c
                     sigmoid b = c + (d-c)/(1+exp(-k(a-a0)))   [4-param logistic]
Reports both weighted R^2. Per-graph slideshow + one combined montage of every
(graph x pair) panel so nothing is cherry-picked.

Version-aware. -> runs/<v>/<graph>/slides/cross_model_trajectory_fit.pdf
                 + runs/<v>/overview/trajectory_fit_montage.png
                 + runs/<v>/overview/trajectory_fit.json
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from scipy.optimize import curve_fit
from config import get_config
import graph as G
import paths as P
from heatmaps_with_null import sub_path, sp, node_means, full_rdm, MODELS, PAIRS, GRAPHS

THR = 0.16


def logistic(a, c, d, k, a0):
    return c + (d - c) / (1 + np.exp(-k * (a - a0)))


def wr2(y, yh, w):
    return float(1 - (w * (y - yh) ** 2).sum() / (w * (y - (w * y).sum() / w.sum()) ** 2).sum())


def fit_all(x, y, w):
    m, c = np.polyfit(x, y, 1, w=np.sqrt(w))
    lin = dict(coef=[float(m), float(c)], r2=wr2(y, m * x + c, w))
    sig = None
    try:
        span = x.max() - x.min()
        popt, _ = curve_fit(logistic, x, y, p0=[y.min(), y.max(), 4.0 / max(span, 1), np.median(x)],
                            sigma=1.0 / np.sqrt(w + 1e-9), absolute_sigma=False, maxfev=20000,
                            bounds=([y.min() - 50, y.min() - 50, 0, x.min() - span],
                                    [y.max() + 50, y.max() + 50, 5, x.max() + span]))
        sig = dict(coef=[float(v) for v in popt], r2=wr2(y, logistic(x, *popt), w))
    except Exception as e:
        print("  sigmoid fit failed:", type(e).__name__, flush=True)
    return lin, sig


def compute(gname, gkw):
    gr = G.build_graph(replace(get_config("gemma_qwen"), **gkw))
    n = gr.n_nodes; iu = np.triu_indices(n, 1)
    rdms = {}
    for m in MODELS:
        if not os.path.exists(sub_path(gname, m)):
            continue
        try:
            z = np.load(sub_path(gname, m), allow_pickle=False)
            layers = [int(l) for l in z["_layers"]]
            node = z["meta_node"]; mask = z["meta_context_length"] >= P.CTX_LO
            rdms[m] = (layers, {L: full_rdm(node_means(z, L, node, mask, n))[iu] for L in layers})
        except Exception as e:
            print(f"skip {gname}/{m} ({type(e).__name__})", flush=True)
    res = {}
    for A, B in PAIRS:
        if A not in rdms or B not in rdms:
            continue
        La, dA = rdms[A]; Lb, dB = rdms[B]
        H = np.array([[sp(dA[a], dB[b]) for b in Lb] for a in La])
        Lbarr = np.array(Lb, float); aa, bh, ww = [], [], []
        for i, a in enumerate(La):
            w = np.clip(H[i] - THR, 0, None)
            if w.sum() > 0:
                aa.append(a); bh.append((w * Lbarr).sum() / w.sum()); ww.append(w.max())
        if len(aa) < 5:
            continue
        aa = np.array(aa, float); bh = np.array(bh); ww = np.array(ww)
        lin, sig = fit_all(aa, bh, ww)
        rl, _ = fit_all(aa / max(La), bh / max(Lb), ww)
        res[(A, B)] = dict(H=H, La=La, Lb=Lb, aa=aa, bh=bh, lin=lin, sig=sig, rel_lin=rl)
    return res


def draw(ax, d, A, B, gname):
    La, Lb = d["La"], d["Lb"]
    im = ax.imshow(d["H"], origin="lower", aspect="auto", cmap="viridis", vmin=-0.1, vmax=1,
                   extent=[Lb[0]-.5, Lb[-1]+.5, La[0]-.5, La[-1]+.5])
    ax.scatter(d["bh"], d["aa"], c="white", s=10, edgecolor="k", lw=0.4, zorder=4)
    xs = np.linspace(d["aa"].min(), d["aa"].max(), 120)
    ax.plot(np.polyval(d["lin"]["coef"], xs), xs, "r-", lw=1.5,
            label=f"lin m={d['lin']['coef'][0]:.2f} R²={d['lin']['r2']:.2f}")
    if d["sig"]:
        ax.plot(logistic(xs, *d["sig"]["coef"]), xs, color="magenta", lw=2,
                label=f"sigmoid R²={d['sig']['r2']:.2f}")
    ax.set_xlim(Lb[0]-.5, Lb[-1]+.5); ax.set_ylim(La[0]-.5, La[-1]+.5)
    ax.set_title(f"{gname} {A}×{B}", fontsize=8)
    ax.set_xlabel(f"{B} layer", fontsize=7); ax.set_ylabel(f"{A} layer", fontsize=7)
    ax.legend(fontsize=6, loc="lower right")
    return im


def main():
    GR = list(GRAPHS.items())
    allres = {g: compute(g, kw) for g, kw in GR}
    J = {}
    for g, res in allres.items():
        if not res:
            continue
        os.makedirs(f"{P.gdir(g)}/slides", exist_ok=True)
        with PdfPages(f"{P.gdir(g)}/slides/cross_model_trajectory_fit.pdf") as pdf:
            for (A, B), d in res.items():
                fig, ax = plt.subplots(figsize=(8, 7)); draw(ax, d, A, B, g)
                fig.colorbar(ax.images[0], label="cross-model RSA"); fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
                J[f"{g}:{A}-{B}"] = dict(linear=d["lin"], sigmoid=d["sig"], rel_linear=d["rel_lin"])
                sr = d["sig"]["r2"] if d["sig"] else float("nan")
                print(f"{g} {A}×{B}: lin R²={d['lin']['r2']:.2f} sigmoid R²={sr:.2f} "
                      f"relslope={d['rel_lin']['coef'][0]:.2f}", flush=True)
        print(f"wrote {P.gdir(g)}/slides/cross_model_trajectory_fit.pdf", flush=True)

    # combined montage: rows=graphs, cols=pairs
    rows = [g for g, _ in GR if allres.get(g)]
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 5 * len(rows)), squeeze=False)
    for r, g in enumerate(rows):
        res = allres[g]
        for c, (A, B) in enumerate(PAIRS):
            a = ax[r][c]
            if (A, B) in res:
                draw(a, res[(A, B)], A, B, g)
            else:
                a.axis("off"); a.set_title(f"{g} {A}×{B}\n(no data)", fontsize=8)
    fig.suptitle(f"[{P.VERSION}] cross-model RSA ridge trajectory — all pairs/graphs "
                 "(red=linear, magenta=sigmoid)")
    fig.tight_layout()
    os.makedirs(f"{P.overview()}", exist_ok=True)
    fig.savefig(f"{P.overview()}/trajectory_fit_montage.png", dpi=130); plt.close(fig)
    json.dump(J, open(f"{P.overview()}/trajectory_fit.json", "w"), indent=1)
    print(f"wrote {P.overview()}/trajectory_fit_montage.png + .json", flush=True)


if __name__ == "__main__":
    main()
