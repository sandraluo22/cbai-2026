"""Causal arbitration: at a given context depth, which ring does steering move?

At depth s we have two candidate activation rings through the current day's
representation: the PRETRAINED weekday ring (fit from pretrained_ring.npz) and
the IN-CONTEXT ring (fit from walk_capture.npz occurrences near depth s). We
add +/- alpha * (unit tangent of each ring at the current node) to the final
token's residual at layer L, rerun the forward pass, and measure how the
restricted next-day posterior ROTATES under each ring's coordinate system
(circular-mean position over that ring's cyclic order).

Predictions:
  - early context: pretrained-tangent steering rotates behavior along the
    SEMANTIC cycle; in-context-tangent steering does little.
  - late context : in-context-tangent steering rotates behavior along the
    PERMUTED cycle; pretrained-tangent steering does little.
  - the causal handoff depth may differ from the representational transition
    (analyze_shift.py) -- that gap, if any, is the finding.

Usage:
  python steer_arbitration.py --depths 30 100 300 800 --alphas 4 8 16
  (expects runs/walk_capture.npz + runs/pretrained_ring.npz; needs the model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import DAYS_PERMUTED, build_graph, generate_walks, cm_models, \
    semantic_day_cycle

from capture_ctx import make_cfg

IN_CONTEXT_CYCLE = list(range(7))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--walk-length", type=int, default=800)
    ap.add_argument("--n-walks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depths", type=int, nargs="+", default=[30, 100, 300, 800])
    ap.add_argument("--alphas", type=float, nargs="+", default=[4.0, 8.0, 16.0],
                    help="steering magnitudes, in units of the ring RADIUS "
                         "fraction: delta = alpha/10 * radius * tangent")
    ap.add_argument("--n-steer-walks", type=int, default=6)
    ap.add_argument("--layers", type=int, nargs="+", default=[26])  # make_cfg compat
    return ap.parse_args()


def circ_delta(a: float, b: float) -> float:
    """Signed angular difference a-b wrapped to (-pi, pi]."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def main():
    args = parse_args()
    cfg = make_cfg(args)
    out = Path(args.out)
    if args.smoke:
        args.depths, args.alphas, args.n_steer_walks = [10, 30], [4.0], 2
        args.layer = 3

    cap = cm_models.load_capture(str(out / "walk_capture.npz"))
    layer = args.layer if args.layer in cap.acts else sorted(cap.acts)[-1]
    acts, nodes, ctx = cap.acts[layer], cap.meta["node"], cap.meta["context_length"]

    pre = np.load(out / "pretrained_ring.npz")
    pre_ring = nglib.fit_ring(pre[f"layer_{layer}"]) if f"layer_{layer}" in pre.files \
        else nglib.fit_ring(pre[[k for k in pre.files if k.startswith("layer_")][-1]])
    sem_cycle = semantic_day_cycle()

    graph = build_graph(cfg)
    walks = generate_walks(graph, cfg)[: args.n_steer_walks]
    model, tok = cm_models.load_model(cfg.model_a, cfg)
    day_ids = nglib.node_first_token_ids(tok, DAYS_PERMUTED)

    results = []
    for s in args.depths:
        # in-context ring fit near this depth
        mask = (ctx >= s * 0.75) & (ctx <= s * 1.25)
        means = nglib.node_means(acts, nodes, mask, 7)
        if np.isnan(means).any():
            print(f"[steer] depth {s}: incomplete node coverage, skipping")
            continue
        ic_ring = nglib.fit_ring(means)

        for wk in walks:
            if s > len(wk.words):
                continue
            prefix = " ".join(wk.words[:s])
            cur = wk.nodes[s - 1]
            base_p, base_mass = nglib.next_word_posterior(
                model, tok, prefix, day_ids, cfg.device)
            base_pos = {
                "ic": nglib.circular_mean_position(base_p, IN_CONTEXT_CYCLE),
                "sem": nglib.circular_mean_position(base_p, sem_cycle),
            }
            for ring_name, ring in (("incontext", ic_ring), ("pretrained", pre_ring)):
                tangent = ring.tangent(cur)
                radius = float(ring.radii.mean())
                for alpha in args.alphas:
                    for sign in (+1, -1):
                        delta = sign * (alpha / 10.0) * radius * tangent
                        p, mass = nglib.patched_forward_posterior(
                            model, tok, prefix, layer, delta, "add",
                            day_ids, cfg.device)
                        results.append({
                            "depth": s, "walk_id": wk.walk_id, "node": cur,
                            "ring": ring_name, "alpha": alpha, "sign": sign,
                            "rot_incontext": circ_delta(
                                nglib.circular_mean_position(p, IN_CONTEXT_CYCLE),
                                base_pos["ic"]),
                            "rot_semantic": circ_delta(
                                nglib.circular_mean_position(p, sem_cycle),
                                base_pos["sem"]),
                            "day_mass": mass, "base_day_mass": base_mass,
                        })
        done = [r for r in results if r["depth"] == s]
        if done:
            for ring_name in ("incontext", "pretrained"):
                sel = [r for r in done if r["ring"] == ring_name]
                mic = np.mean([abs(r["rot_incontext"]) for r in sel])
                msem = np.mean([abs(r["rot_semantic"]) for r in sel])
                print(f"[steer] depth={s:4d} {ring_name:10s} "
                      f"|rot_ic|={mic:.3f}  |rot_sem|={msem:.3f}")

    (out / "steer_arbitration.json").write_text(json.dumps(results, indent=2))
    print(f"[steer] {len(results)} interventions -> {out/'steer_arbitration.json'}")


if __name__ == "__main__":
    main()
