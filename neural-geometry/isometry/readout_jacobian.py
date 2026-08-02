"""Attack 1 on the isometry-origins question: interrogate the readout map.

The behavior manifold is the image of the activation manifold under the
"readout" — everything downstream of layer L, ending in softmax. The
activation<->behavior isometry (arXiv 2605.05115) holds locally iff that map,
restricted to the manifold's tangent plane, is a scaled orthogonal map
(conformal), AND off-manifold directions are relatively suppressed.

This script measures exactly that, at each node of a fitted ring:

  - tangent amplification  |f(h + eps*t) - f(h - eps*t)| / (2*eps)
    along the ring tangent t and the in-plane radial direction
  - off-manifold amplification along random directions orthogonal to the plane
  - conformality: 2x2 Gram matrix of the map on the tangent plane ->
    singular-value ratio (1.0 = perfectly conformal)

Readout modes:
  --mode patch   (default) honest readout: replace the final-token residual at
                 layer L in a real context and rerun downstream blocks
  --mode lens    logit-lens (final norm + unembed only) — the cheap variant;
                 if lens already shows conformality, the unembedding alone
                 explains the isometry and nothing deeper is needed.
                 THE DIFFERENCE BETWEEN THE TWO MODES IS ITSELF THE RESULT:
                 it localizes where isometry is enforced (unembedding vs
                 downstream computation).

Outputs are Hellinger-embedded (sqrt-prob) so "output distance" is a proper
information geometry, matching nglib.isometry_scores.

Requires a completed shuffled-weekday capture (walk_capture.npz +
pretrained_ring.npz) — reuse it rather than re-capturing:

  python readout_jacobian.py --capture-dir ../shuffled-weekday/runs \
      --layer 26 [--mode patch|lens] [--ring incontext|pretrained]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import DAYS_PERMUTED, build_graph, generate_walks, cm_models
from nglib import Config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="../shuffled-weekday/runs")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--mode", choices=["patch", "lens"], default="patch")
    ap.add_argument("--ring", choices=["incontext", "pretrained"], default="incontext")
    ap.add_argument("--ctx-depth", type=int, default=800,
                    help="context bin used for the in-context ring / patch context")
    ap.add_argument("--eps-frac", type=float, default=0.02,
                    help="finite-difference step as a fraction of ring radius")
    ap.add_argument("--n-offplane", type=int, default=8,
                    help="random off-plane directions per node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def load_ring_points(args, cap_dir: Path, layer: int):
    """Node-mean points [7, d] for the chosen ring."""
    if args.ring == "pretrained":
        pre = np.load(cap_dir / "pretrained_ring.npz")
        key = f"layer_{layer}" if f"layer_{layer}" in pre.files else \
            [k for k in pre.files if k.startswith("layer_")][-1]
        return np.asarray(pre[key], dtype=np.float64)
    cap = cm_models.load_capture(str(cap_dir / "walk_capture.npz"))
    layer = layer if layer in cap.acts else sorted(cap.acts)[-1]
    acts, nodes, ctx = cap.acts[layer], cap.meta["node"], cap.meta["context_length"]
    lo, hi = args.ctx_depth * 0.75, args.ctx_depth * 1.25
    means = nglib.node_means(acts, nodes, (ctx >= lo) & (ctx <= hi), 7)
    assert not np.isnan(means).any(), "incomplete node coverage in chosen ctx bin"
    return means


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cap_dir = Path(args.capture_dir)

    cfg_kw = dict(name="readout_jac", graph_type="ring", ring_size=7,
                  word_set="days", model_a=args.model, model_b=args.model,
                  device=args.device, seed=args.seed,
                  walk_length=max(args.ctx_depth, 10), n_walks=4)
    if args.smoke:
        cfg_kw.update(model_a="distilgpt2", model_b="distilgpt2", device="cpu",
                      dtype="float32", walk_length=40)
        args.layer, args.ctx_depth = 3, 30
    cfg = Config(**cfg_kw)

    points = load_ring_points(args, cap_dir, args.layer)
    ring = nglib.fit_ring(points)
    d = points.shape[1]
    eps = args.eps_frac * float(ring.radii.mean())

    model, tok = cm_models.load_model(cfg.model_a, cfg)
    day_ids = nglib.node_first_token_ids(tok, DAYS_PERMUTED)

    # patch mode needs a real context to rerun; use one long walk prefix per node,
    # truncated where that node occurs closest to ctx_depth.
    contexts = {}
    if args.mode == "patch":
        graph = build_graph(cfg)
        walks = generate_walks(graph, cfg)
        for n in range(7):
            best = None
            for wk in walks:
                for s in range(min(args.ctx_depth, len(wk.nodes)), 0, -1):
                    if wk.nodes[s - 1] == n:
                        best = " ".join(wk.words[:s]); break
                if best:
                    break
            assert best, f"node {n} never occurred in the probe walks"
            contexts[n] = best

    def readout(node: int, h: np.ndarray) -> np.ndarray:
        """h (full residual, [d]) -> Hellinger-embedded restricted posterior."""
        if args.mode == "lens":
            p = nglib.logit_lens_posterior(model, h, day_ids)
        else:
            p, _ = nglib.patched_forward_posterior(
                model, tok, contexts[node], args.layer, h, "replace",
                day_ids, cfg.device)
        return nglib.hellinger_embed(p[None, :])[0]

    def direction_gain(node: int, h0: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Central-difference image of unit direction v under the readout:
        (f(h0+eps v) - f(h0-eps v)) / (2 eps). Returns the image vector."""
        return (readout(node, h0 + eps * v) - readout(node, h0 - eps * v)) / (2 * eps)

    records = []
    for n in range(7):
        h0 = points[n]
        t = ring.tangent(n)
        r = ring.radial(n)

        jt = direction_gain(n, h0, t)          # image of tangent
        jr = direction_gain(n, h0, r)          # image of in-plane radial

        # conformality on the tangent plane: singular values of the 2-column
        # image matrix [jt jr] (the map's action restricted to the plane)
        J2 = np.stack([jt, jr], axis=1)
        sv = np.linalg.svd(J2, compute_uv=False)
        conformality = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0

        # off-plane control: random unit directions orthogonal to the plane
        off_gains = []
        for _ in range(args.n_offplane):
            v = rng.standard_normal(d)
            v -= ring.plane @ (ring.plane.T @ v)        # project out the plane
            v /= np.linalg.norm(v)
            off_gains.append(float(np.linalg.norm(direction_gain(n, h0, v))))

        rec = {
            "node": n, "day": DAYS_PERMUTED[n],
            "gain_tangent": float(np.linalg.norm(jt)),
            "gain_radial": float(np.linalg.norm(jr)),
            "gain_offplane_mean": float(np.mean(off_gains)),
            "gain_offplane_std": float(np.std(off_gains)),
            "conformality_sv_ratio": conformality,
        }
        records.append(rec)
        print(f"[jac] node {n} ({DAYS_PERMUTED[n]:9s}) "
              f"tan={rec['gain_tangent']:.4f} rad={rec['gain_radial']:.4f} "
              f"off={rec['gain_offplane_mean']:.4f} conf={conformality:.3f}")

    gains_t = [r["gain_tangent"] for r in records]
    summary = {
        "mode": args.mode, "ring": args.ring, "layer": args.layer,
        "ctx_depth": args.ctx_depth, "eps": eps,
        # isometry-up-to-scale predicts tangent gain ~constant around the ring:
        "tangent_gain_cv": float(np.std(gains_t) / np.mean(gains_t)),
        "mean_conformality": float(np.mean([r["conformality_sv_ratio"] for r in records])),
        # on-manifold amplification relative to off-manifold:
        "onoff_ratio": float(np.mean(gains_t) /
                             np.mean([r["gain_offplane_mean"] for r in records])),
        "records": records,
    }
    path = out / f"readout_jacobian_{args.mode}_{args.ring}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"[jac] tangent-gain CV={summary['tangent_gain_cv']:.3f}  "
          f"conformality={summary['mean_conformality']:.3f}  "
          f"on/off ratio={summary['onoff_ratio']:.2f}  -> {path}")


if __name__ == "__main__":
    main()
