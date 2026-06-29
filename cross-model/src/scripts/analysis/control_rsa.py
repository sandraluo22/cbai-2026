"""Confound controls for the cross-model RSA: is the model-to-model node-geometry
similarity genuinely the in-context GRAPH, or an artifact of token identity /
occurrence frequency / position?

For each graph and model pair, at each model's grid-peak layer, we report:
  observed   : cross-model RSA of the two node-RDMs (per-node means, ctx>=300)
  token_L0   : cross-model RSA of the LAYER-0 (embedding) RDMs  -> token-identity baseline
  partial    : cross-model RSA after regressing each RDM on [own layer-0 RDM,
               frequency RDM, position RDM]  -> similarity NOT explained by those
  shuffle95  : 95th pct of a node-label permutation null  -> chance
  -graph     : cross-model RSA after regressing out the true graph-distance RDM
               (should collapse if the shared structure IS the graph)
Also each model's own grid RSA (vs graph distance) for reference.

Version-aware; run on the box that has the npz.  -> runs/<v>/overview/control_rsa.{png,json}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]
PAIRS = [("Llama", "Gemma"), ("Llama", "Qwen"), ("Gemma", "Qwen")]
HICTX, NPERM = 300, 2000


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdmv(H, iu):
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def partial(y, Cs):
    """residual of y after OLS regression on the confound vectors Cs."""
    if not Cs:
        return y
    X = np.column_stack([np.ones_like(y)] + [c.astype(float) for c in Cs])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def means(z, L, n, mask):
    node = z["meta_node"]; X = z[f"layer_{L}"].astype(np.float64)
    return np.stack([X[mask & (node == k)].mean(0) for k in range(n)])


def main():
    out = {}
    for gname, kw in GRAPHS:
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        n = gr.n_nodes; iu = np.triu_indices(n, 1)
        GD = gr.distance_matrix()[iu].astype(float)
        data = {}
        for m in P.MODELS:
            p = P.acts_path(gname, m)
            if not os.path.exists(p):
                print(f"skip {gname}/{m} (npz absent)", flush=True); continue
            try:
                z = np.load(p); node = z["meta_node"]; step = z["meta_step"]
                mask = z["meta_context_length"] >= HICTX
                freq = np.array([(mask & (node == k)).sum() for k in range(n)], float)
                pos = np.array([step[mask & (node == k)].mean() if (mask & (node == k)).any() else 0.
                                for k in range(n)], float)
                data[m] = dict(rdm=rdmv(means(z, P.peak_layer(gname, m), n, mask), iu),
                               rdm0=rdmv(means(z, 0, n, mask), iu),
                               freqR=np.abs(freq[:, None] - freq[None])[iu],
                               posR=np.abs(pos[:, None] - pos[None])[iu])
            except Exception as e:
                print(f"skip {gname}/{m} (load failed: {type(e).__name__})", flush=True); continue
        for A, B in PAIRS:
            if A not in data or B not in data:
                continue
            dA, dB = data[A], data[B]
            obs = sp(dA["rdm"], dB["rdm"])
            l0 = sp(dA["rdm0"], dB["rdm0"])
            ra = partial(dA["rdm"], [dA["rdm0"], dA["freqR"], dA["posR"]])
            rb = partial(dB["rdm"], [dB["rdm0"], dB["freqR"], dB["posR"]])
            part = sp(ra, rb)
            pg = sp(partial(dA["rdm"], [GD]), partial(dB["rdm"], [GD]))
            full = np.zeros((n, n)); full[iu] = dB["rdm"]; full += full.T
            rng = np.random.default_rng(0)
            nl = [sp(dA["rdm"], full[np.ix_(pm, pm)][iu]) for pm in (rng.permutation(n) for _ in range(NPERM))]
            key = f"{gname}:{A}-{B}"
            out[key] = dict(observed=obs, token_L0=l0, partial_confound=part,
                            minus_graph=pg, shuffle95=float(np.percentile(nl, 95)),
                            gridA=sp(dA["rdm"], GD), gridB=sp(dB["rdm"], GD))
            print(f"{key}: obs={obs:.2f} tokenL0={l0:.2f} partialConf={part:.2f} "
                  f"minusGraph={pg:.2f} shuffle95={out[key]['shuffle95']:.2f}", flush=True)

    os.makedirs(f"{P.overview()}", exist_ok=True)
    json.dump(out, open(f"{P.overview()}/control_rsa.json", "w"), indent=1)
    keys = list(out)
    if keys:
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(keys)), 5))
        x = np.arange(len(keys)); w = 0.2
        for i, (f, lab, c) in enumerate([("observed", "observed cross-RSA", "steelblue"),
                                         ("partial_confound", "partial (− token/freq/pos)", "seagreen"),
                                         ("token_L0", "layer-0 (token identity)", "orange"),
                                         ("shuffle95", "shuffle null 95%", "0.6")]):
            ax.bar(x + (i - 1.5) * w, [out[k][f] for k in keys], w, label=lab, color=c)
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("RSA"); ax.legend(fontsize=8)
        ax.set_title(f"[{P.VERSION}] cross-model RSA vs confound controls\n"
                     "genuine graph similarity: observed≈partial ≫ token-L0 & shuffle95")
        fig.tight_layout(); fig.savefig(f"{P.overview()}/control_rsa.png", dpi=150)
        print(f"wrote {P.overview()}/control_rsa.png", flush=True)


if __name__ == "__main__":
    main()
