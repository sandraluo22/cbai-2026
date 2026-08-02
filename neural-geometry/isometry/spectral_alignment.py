"""Attack 3 on the isometry-origins question: the spectral/Markov account.

Hypothesis: for a random walk on a graph, BOTH manifolds are spectral
embeddings of the same operator. The optimal predictive representation of the
walk is spanned by low-frequency graph-Laplacian eigenvectors (for a ring,
exactly the Fourier pair that makes a circle — cf. the Dirichlet-energy story
in Park et al.), and the posterior geometry is governed by the same transition
matrix. If both the activation node-means and the behavior posteriors align
with those eigenfunctions, the isometry isn't a coincidence of training: it is
inherited from the task's transition structure.

Per graph (ring / grid / prism — different spectra on purpose), we compute:
  1. Laplacian eigenvectors of the graph
  2. activation node-means (large context) -> alignment of their top-k PCA
     subspace with the span of the first k nontrivial eigenvectors
     (mean principal angle), plus per-eigenvector R^2
  3. model posteriors per node -> same alignment on the behavior side, and
     act<->beh isometry as the consistency check
  4. diffusion-map coordinates of the exact chain as the analytic reference

The spectral account uniquely predicts that changing the graph's spectrum
changes BOTH geometries the SAME way — which the cross-graph comparison tests.

Usage:
  python spectral_alignment.py --graphs ring grid prism   [--smoke]
Reuses cross-model graph builders; concept words (no semantic priors).
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
from nglib import Config, build_graph, generate_walks, cm_models


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--graphs", nargs="+", default=["ring", "grid", "prism"])
    ap.add_argument("--k", type=int, default=4,
                    help="subspace dimension compared (top-k PCA vs first k "
                         "nontrivial Laplacian eigenvectors)")
    ap.add_argument("--walk-length", type=int, default=600)
    ap.add_argument("--n-walks", type=int, default=30)
    ap.add_argument("--ctx-min", type=int, default=300)
    ap.add_argument("--n-probe-prefixes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def laplacian_eigs(graph) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the (combinatorial) graph Laplacian, ascending.
    Returns (eigenvalues, eigenvectors[:, i])."""
    n = graph.n_nodes
    A = np.zeros((n, n))
    for i, nbrs in enumerate(graph.adjacency):
        for j in nbrs:
            A[i, j] = 1.0
    L = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(L)
    return w, V


def transition_matrix(graph) -> np.ndarray:
    n = graph.n_nodes
    P = np.zeros((n, n))
    for i, nbrs in enumerate(graph.adjacency):
        for j in nbrs:
            P[i, j] = 1.0 / len(nbrs)
    return P


def subspace_alignment(X: np.ndarray, eigvecs: np.ndarray, k: int) -> dict:
    """Alignment of the top-k PCA subspace of centered X ([n_nodes, d]) with
    span(eigvecs[:, 1:k+1]) — note eigvec 0 is constant and drops out under
    centering. Both live in node-space: PCA scores are [n_nodes, k]."""
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :k]                                  # node-space PCA coords
    E = eigvecs[:, 1:k + 1]
    angles = nglib.principal_angles(scores, E)
    # per-eigenvector R^2: how much of eigvec e is captured by the PCA subspace
    Q, _ = np.linalg.qr(scores)
    r2 = [float(np.linalg.norm(Q.T @ E[:, i]) ** 2 /
                np.linalg.norm(E[:, i]) ** 2) for i in range(k)]
    return {"mean_principal_angle_deg": float(np.degrees(angles).mean()),
            "per_eigvec_r2": r2,
            "pca_spectrum_top8": [float(s) for s in (S[:8] ** 2 / (S ** 2).sum())]}


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)

    base = dict(model_a=args.model, model_b=args.model, device=args.device,
                seed=args.seed, walk_length=args.walk_length, n_walks=args.n_walks,
                word_set="concepts",
                capture_layers_a=(args.layer,), capture_layers_b=(args.layer,),
                align_layer_a=args.layer, align_layer_b=args.layer)
    graph_kw = {"ring": dict(graph_type="ring", ring_size=12),
                "grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
                "prism": dict(graph_type="prism", prism_k=7)}
    if args.smoke:
        base.update(model_a="distilgpt2", model_b="distilgpt2", device="cpu",
                    dtype="float32", walk_length=80, n_walks=8,
                    capture_layers_a=(3,), capture_layers_b=(3,),
                    align_layer_a=3, align_layer_b=3)
        args.layer, args.ctx_min, args.n_probe_prefixes = 3, 40, 12
        args.graphs = ["ring"]
        graph_kw["ring"] = dict(graph_type="ring", ring_size=6)

    model, tok = cm_models.load_model(base["model_a"],
                                      Config(name="tmp", **base))

    all_results = {}
    for gname in args.graphs:
        cfg = Config(name=f"spectral_{gname}", **{**base, **graph_kw[gname]})
        graph = build_graph(cfg)
        n = graph.n_nodes
        k = min(args.k, n - 2)
        print(f"[spectral] {gname}: n={n}, k={k}")

        walks = generate_walks(graph, cfg)
        capture = cm_models.capture(model, tok, walks, (args.layer,), cfg)
        acts = capture.acts[args.layer]
        nodes, ctx = capture.meta["node"], capture.meta["context_length"]
        means = nglib.node_means(acts, nodes, ctx >= args.ctx_min, n)
        assert not np.isnan(means).any(), f"{gname}: node coverage incomplete"

        evals, evecs = laplacian_eigs(graph)
        act_align = subspace_alignment(means, evecs, k)

        # behavior: model posteriors per node from late-context prefixes
        word_ids = nglib.node_first_token_ids(tok, graph.words)
        post_rows = [[] for _ in range(n)]
        for _ in range(args.n_probe_prefixes):
            wk = walks[int(rng.integers(len(walks)))]
            s = int(rng.integers(args.ctx_min, cfg.walk_length))
            prefix = " ".join(wk.words[:s])
            p, _ = nglib.next_word_posterior(model, tok, prefix, word_ids,
                                             cfg.device)
            post_rows[wk.nodes[s - 1]].append(p)
        model_post = np.stack([
            np.mean(r, axis=0) if r else np.full(n, np.nan) for r in post_rows])
        covered = ~np.isnan(model_post).any(axis=1)
        beh_pts = nglib.hellinger_embed(model_post[covered])

        beh_align = subspace_alignment(beh_pts, evecs[covered], k) \
            if covered.sum() > k + 1 else None
        chain_pts = nglib.hellinger_embed(transition_matrix(graph))
        chain_align = subspace_alignment(chain_pts, evecs, k)
        iso = nglib.isometry_scores(means[covered], beh_pts) \
            if covered.sum() >= 4 else None

        all_results[gname] = {
            "n_nodes": n, "k": k,
            "laplacian_eigenvalues": [float(v) for v in evals],
            "activation_vs_laplacian": act_align,
            "behavior_vs_laplacian": beh_align,
            "exact_chain_vs_laplacian": chain_align,
            "act_beh_isometry": iso,
            "nodes_covered": int(covered.sum()),
        }
        print(f"[spectral] {gname}: act-Laplacian angle="
              f"{act_align['mean_principal_angle_deg']:.1f} deg, "
              f"beh angle="
              f"{beh_align['mean_principal_angle_deg']:.1f} deg"
              if beh_align else f"[spectral] {gname}: behavior undercovered")

    payload = {"layer": args.layer, "graphs": all_results}
    (out / "spectral_alignment.json").write_text(json.dumps(payload, indent=2))
    print(f"[spectral] -> {out/'spectral_alignment.json'}")


if __name__ == "__main__":
    main()
