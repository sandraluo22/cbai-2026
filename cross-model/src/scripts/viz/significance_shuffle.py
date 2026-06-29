"""Cross-model RSA significance using a PER-CELL shuffle null (the control_rsa
node-label permutation), instead of the single deep-layer threshold used by
significance_heatmaps.py.

For each (layer_A, layer_B) cell: S = raw cross-RSA - shuffle_null95(cell), where
shuffle_null95 permutes the node labels of model B's RDM at that layer and takes
the 95th pct of the resulting RSA with model A's RDM. Diverging colormap at 0;
>0 = above the shuffle null.

Version-aware. -> runs/<v>/<graph>/slides/cross_model_rsa_significance_shuffle.pdf
                 + runs/<v>/slides/all_cross_model_rsa_significance_shuffle.pdf
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P
from heatmaps_with_null import sub_path, sp, node_means, full_rdm, png_to_pdf, MODELS, PAIRS, GRAPHS

PERMS = 400


def main():
    rng = np.random.default_rng(0); all_pngs = []
    for gname, gkw in GRAPHS.items():
        if not all(os.path.exists(sub_path(gname, m)) for m in MODELS):
            print(f"skip {gname}: no acts for {P.VERSION}", flush=True); continue
        gr = G.build_graph(replace(get_config("gemma_qwen"), **gkw))
        n = gr.n_nodes; iu = np.triu_indices(n, 1)
        data = {}
        for m in MODELS:
            try:
                z = np.load(sub_path(gname, m), allow_pickle=False)
                layers = [int(l) for l in z["_layers"]]
                node = z["meta_node"]; mask = z["meta_context_length"] >= P.CTX_LO
                data[m] = {"layers": layers,
                           "rdm": {L: full_rdm(node_means(z, L, node, mask, n)) for L in layers}}
            except Exception as e:
                print(f"skip {gname}/{m} ({type(e).__name__})", flush=True)
        os.makedirs(f"{P.gdir(gname)}/slides", exist_ok=True)
        out = f"{P.gdir(gname)}/slides/cross_model_rsa_significance_shuffle.pdf"
        with PdfPages(out) as pdf:
            for A, B in PAIRS:
                if A not in data or B not in data:
                    continue
                La, Lb = data[A]["layers"], data[B]["layers"]
                rA = {a: data[A]["rdm"][a][iu] for a in La}
                # per-Lb: precompute the permuted upper-tri vectors of B's RDM
                permB = {b: np.stack([data[B]["rdm"][b][np.ix_(p, p)][iu]
                                      for p in (rng.permutation(n) for _ in range(PERMS))]) for b in Lb}
                S = np.empty((len(La), len(Lb)))
                for i, a in enumerate(La):
                    ra = rA[a]
                    for j, b in enumerate(Lb):
                        raw = sp(ra, data[B]["rdm"][b][iu])
                        nl = [sp(ra, permB[b][k]) for k in range(PERMS)]
                        S[i, j] = raw - np.percentile(nl, 95)
                vmax = max(0.05, float(np.nanmax(np.abs(S))))
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(S, origin="lower", aspect="auto", cmap="RdBu_r",
                               norm=TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax),
                               extent=[Lb[0]-.5, Lb[-1]+.5, La[0]-.5, La[-1]+.5])
                bi, bj = np.unravel_index(int(np.nanargmax(S)), S.shape)
                ax.plot(Lb[bj], La[bi], "k*", ms=12)
                frac = float((S > 0).mean()) * 100
                ax.set_xlabel(f"{B} layer"); ax.set_ylabel(f"{A} layer")
                ax.set_title(f"{gname} [{P.VERSION}] {A} vs {B}: RSA − per-cell shuffle-null95\n"
                             f"red = above shuffle null; {frac:.0f}% of cells > null", fontsize=9)
                fig.colorbar(im, label="RSA − shuffle95   (>0 significant)")
                fig.tight_layout()
                png = f"{P.gdir(gname)}/rsa_sig_shuffle_{A}_{B}.png"
                fig.savefig(png, dpi=130); all_pngs.append(png)
                pdf.savefig(fig); plt.close(fig)
                print(f"{gname} {A}x{B}: {frac:.0f}% cells > shuffle-null", flush=True)
        print(f"wrote {out}", flush=True)
    os.makedirs(f"{P.ROOT}/slides", exist_ok=True)
    if all_pngs:
        png_to_pdf(all_pngs, f"{P.ROOT}/slides/all_cross_model_rsa_significance_shuffle.pdf")
        print(f"wrote {P.ROOT}/slides/all_cross_model_rsa_significance_shuffle.pdf", flush=True)


if __name__ == "__main__":
    main()
