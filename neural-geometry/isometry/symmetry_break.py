"""Attack 2 on the isometry-origins question: break the task's symmetry.

The symmetry account (Karkada et al. line): a uniform random walk on a ring is
invariant under the cyclic group; if training/ICL induces equivariant
representations, BOTH the activation and behavior manifolds must be circles
carrying the same group action, and the isometry follows from symmetry alone.

Discriminating test: bias the walk. With P(clockwise) = 0.5 + beta the cyclic
symmetry is broken (only rotation equivariance survives in a weaker form; at
large beta the chain is strongly directional). If the isometry is a symmetry
artifact it should degrade with beta; if it survives asymmetry, something
deeper aligns the two geometries.

For each beta we:
  1. generate biased ring walks (concept words, not weekdays — no semantic
     conflict wanted here)
  2. capture activations, take node means at large context
  3. read the model's next-word posterior per node (behavior manifold)
  4. compute isometry scores (activation ring vs Hellinger posterior ring)
  5. also compute the EXACT chain posterior per node (ground truth behavior
     geometry) as a reference — the model-vs-chain gap tracks how well the
     model has internalized the biased chain at all.

Usage:
  python symmetry_break.py --betas 0.0 0.1 0.2 0.3 0.45   [--smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import Config, Graph, Walk, build_graph, cm_models


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--ring-size", type=int, default=10)
    ap.add_argument("--walk-length", type=int, default=600)
    ap.add_argument("--n-walks", type=int, default=30)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.45])
    ap.add_argument("--ctx-min", type=int, default=300,
                    help="only occurrences with context >= this enter node means")
    ap.add_argument("--n-probe-prefixes", type=int, default=40,
                    help="prefixes per beta for the behavioral posterior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def biased_ring_walks(graph: Graph, cfg: Config, beta: float) -> list:
    """Random walks on the ring with P(step +1) = 0.5 + beta. Node i's neighbors
    on the ring are (i-1, i+1) mod n; beta = 0 recovers the uniform walk."""
    rng = np.random.default_rng(cfg.seed)
    n = graph.n_nodes
    walks = []
    for w in range(cfg.n_walks):
        cur = w % n
        nodes = [cur]
        for _ in range(cfg.walk_length - 1):
            step = 1 if rng.random() < 0.5 + beta else -1
            cur = (cur + step) % n
            nodes.append(cur)
        walks.append(Walk(walk_id=w, nodes=nodes,
                          words=[graph.words[v] for v in nodes]))
    return walks


def exact_chain_posterior(n: int, beta: float) -> np.ndarray:
    """One-step transition rows of the biased ring chain: [n, n]."""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i + 1) % n] = 0.5 + beta
        P[i, (i - 1) % n] = 0.5 - beta
    return P


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    kw = dict(name="symbreak", graph_type="ring", ring_size=args.ring_size,
              word_set="concepts", walk_length=args.walk_length,
              n_walks=args.n_walks, seed=args.seed,
              model_a=args.model, model_b=args.model, device=args.device,
              capture_layers_a=(args.layer,), capture_layers_b=(args.layer,),
              align_layer_a=args.layer, align_layer_b=args.layer)
    if args.smoke:
        kw.update(model_a="distilgpt2", model_b="distilgpt2", device="cpu",
                  dtype="float32", walk_length=80, n_walks=6, ring_size=6,
                  capture_layers_a=(3,), capture_layers_b=(3,),
                  align_layer_a=3, align_layer_b=3)
        args.layer, args.ctx_min, args.n_probe_prefixes = 3, 40, 10
        args.betas = [0.0, 0.3]
    cfg = Config(**kw)

    graph = build_graph(cfg)
    n = graph.n_nodes
    model, tok = cm_models.load_model(cfg.model_a, cfg)
    word_ids = nglib.node_first_token_ids(tok, graph.words)

    results = []
    for beta in args.betas:
        walks = biased_ring_walks(graph, cfg, beta)
        capture = cm_models.capture(model, tok, walks, (args.layer,), cfg)
        acts = capture.acts[args.layer]
        nodes, ctx = capture.meta["node"], capture.meta["context_length"]

        means = nglib.node_means(acts, nodes, ctx >= args.ctx_min, n)
        assert not np.isnan(means).any(), f"beta={beta}: node coverage incomplete"
        ring = nglib.fit_ring(means)

        # model behavior: mean posterior per node over late-context prefixes
        post_rows = [[] for _ in range(n)]
        mass_all = []
        rng = np.random.default_rng(args.seed + 1)
        probes = 0
        while probes < args.n_probe_prefixes:
            wk = walks[int(rng.integers(len(walks)))]
            s = int(rng.integers(args.ctx_min, cfg.walk_length))
            prefix = " ".join(wk.words[:s])
            cur = wk.nodes[s - 1]
            p, mass = nglib.next_word_posterior(model, tok, prefix, word_ids,
                                                cfg.device)
            post_rows[cur].append(p); mass_all.append(mass); probes += 1
        model_post = np.stack([
            np.mean(r, axis=0) if r else np.full(n, np.nan) for r in post_rows])
        covered = ~np.isnan(model_post).any(axis=1)

        chain_post = exact_chain_posterior(n, beta)

        iso_model = nglib.isometry_scores(
            means[covered], nglib.hellinger_embed(model_post[covered]))
        iso_chain = nglib.isometry_scores(means, nglib.hellinger_embed(chain_post))
        # model-vs-chain behavioral fidelity (has the model learned the bias?)
        tv = float(np.abs(model_post[covered] - chain_post[covered]).sum(1).mean() / 2)

        rec = {"beta": beta, "n_nodes_covered": int(covered.sum()),
               "circularity": ring.circularity,
               "var_explained_2d": ring.var_explained,
               "mean_word_mass": float(np.mean(mass_all)),
               "model_chain_tv": tv,
               **{f"iso_model_{k}": v for k, v in iso_model.items()},
               **{f"iso_chain_{k}": v for k, v in iso_chain.items()}}
        results.append(rec)
        print(f"[symbreak] beta={beta:.2f}  iso(model)={iso_model['dist_spearman']:.3f}  "
              f"iso(chain)={iso_chain['dist_spearman']:.3f}  "
              f"circ={ring.circularity:.3f}  TV(model,chain)={tv:.3f}")

    payload = {"config": dataclasses.asdict(cfg), "layer": args.layer,
               "results": results}
    (out / "symmetry_break.json").write_text(json.dumps(payload, indent=2))
    plot(results, out / "symmetry_break.pdf")
    print(f"[symbreak] -> {out/'symmetry_break.json'}, {out/'symmetry_break.pdf'}")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = [r["beta"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, [r["iso_model_dist_spearman"] for r in results], "o-",
            label="isometry (model behavior)")
    ax.plot(betas, [r["iso_chain_dist_spearman"] for r in results], "s--",
            label="isometry (exact chain)")
    ax.plot(betas, [r["circularity"] for r in results], "^:", label="circularity")
    ax.set(xlabel="walk bias beta  (P(+1) = 0.5 + beta)", ylabel="score",
           title="does the isometry survive symmetry breaking?")
    ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
