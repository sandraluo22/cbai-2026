"""Analyze how the activation manifold shifts as the in-context ring is learned.

Per context-length bin, this fits the activation ring (7 node-mean points ->
2D plane + circle) and computes, as functions of context depth:

  H1 rigid relabeling   -> plane stays aligned with the PRETRAINED weekday
                           plane (small principal angles to pretrained_ring)
                           while the cyclic ORDER flips semantic -> permuted.
  H2 subspace competition -> principal angle to the pretrained plane GROWS
                           while angle to the final (fully-learned) plane
                           SHRINKS; both orders partially expressed mid-way.
  H3 smooth deformation -> order agreement moves gradually with no sharp
                           subspace rotation.

Also tracks the ISOMETRY between activation and behavior manifolds per bin:
  - act ring (node means) vs walk-continuation posterior ring (Hellinger)
  - and, for the transition window, whether behavior tracks the OLD ring, the
    NEW ring, or interpolates (the sharpest test of the paper's isometry claim
    under manifold competition).

Inputs : runs/walk_capture.npz, runs/pretrained_ring.npz, runs/behavior.npz
Outputs: runs/shift_metrics.json, runs/shift_summary.pdf

Usage: python analyze_shift.py [--layer 26] [--out runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import cm_models, semantic_day_cycle

IN_CONTEXT_CYCLE = list(range(7))          # node i is adjacent to i+1 by design


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--bins", type=int, nargs="+",
                    default=[10, 20, 30, 50, 75, 100, 150, 200, 300, 450, 600, 800])
    ap.add_argument("--bin-width", type=float, default=0.25,
                    help="relative half-width of each context bin")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    cap = cm_models.load_capture(str(out / "walk_capture.npz"))
    layer = args.layer if args.layer in cap.acts else sorted(cap.acts)[-1]
    acts = cap.acts[layer]
    nodes, ctx = cap.meta["node"], cap.meta["context_length"]

    pre = np.load(out / "pretrained_ring.npz")
    pre_key = f"layer_{layer}" if f"layer_{layer}" in pre.files else \
        [k for k in pre.files if k.startswith("layer_")][-1]
    pre_ring = nglib.fit_ring(pre[pre_key])
    sem_cycle = semantic_day_cycle()

    beh = np.load(out / "behavior.npz", allow_pickle=True)
    cont = beh["probe"] == "cont"

    # final-bin ("fully learned") reference plane
    last_lo, last_hi = args.bins[-1] * (1 - args.bin_width), args.bins[-1] * (1 + args.bin_width)
    final_means = nglib.node_means(acts, nodes, (ctx >= last_lo) & (ctx <= last_hi), 7)
    final_ring = nglib.fit_ring(final_means)

    records = []
    for c in args.bins:
        lo, hi = c * (1 - args.bin_width), c * (1 + args.bin_width)
        mask = (ctx >= lo) & (ctx <= hi)
        means = nglib.node_means(acts, nodes, mask, 7)
        if np.isnan(means).any():
            print(f"[analyze] bin {c}: some node unobserved, skipping")
            continue
        ring = nglib.fit_ring(means)

        rec = {
            "ctx": c,
            "n_occ": int(mask.sum()),
            "circularity": ring.circularity,
            "var_explained_2d": ring.var_explained,
            # subspace trajectory
            "angle_to_pretrained_deg": float(np.degrees(
                nglib.principal_angles(ring.plane, pre_ring.plane)).mean()),
            "angle_to_final_deg": float(np.degrees(
                nglib.principal_angles(ring.plane, final_ring.plane)).mean()),
            # which cyclic order does the fitted ring realize?
            "order_incontext": nglib.cyclic_order_agreement(ring.angles, IN_CONTEXT_CYCLE),
            "order_semantic": nglib.cyclic_order_agreement(ring.angles, sem_cycle),
            # how much does the pretrained ring persist INSIDE this bin's acts?
            # (project node means onto the pretrained plane and re-check order)
            "order_semantic_in_pretrained_plane": nglib.cyclic_order_agreement(
                nglib.fit_ring((means - means.mean(0)) @ pre_ring.plane @ pre_ring.plane.T
                               + means.mean(0)).angles, sem_cycle),
        }

        # behavior manifold at this depth: mean continuation posterior per node
        bmask = cont & (beh["step"] == c)
        if bmask.sum() >= 5:
            bp = np.full((7, 7), np.nan)
            for n in range(7):
                sel = bmask & (beh["node"] == n)
                if sel.any():
                    bp[n] = beh["posterior"][sel].mean(axis=0)
            if not np.isnan(bp).any():
                bpts = nglib.hellinger_embed(bp)
                iso = nglib.isometry_scores(means, bpts)
                rec.update({f"iso_{k}": v for k, v in iso.items()})
                b_ring = nglib.fit_ring(bpts)
                rec["beh_order_incontext"] = nglib.cyclic_order_agreement(
                    b_ring.angles, IN_CONTEXT_CYCLE)
                rec["beh_order_semantic"] = nglib.cyclic_order_agreement(
                    b_ring.angles, sem_cycle)
                rec["beh_day_mass"] = float(beh["day_mass"][bmask].mean())

        records.append(rec)
        print(f"[analyze] ctx={c:4d}  ang_pre={rec['angle_to_pretrained_deg']:5.1f}  "
              f"ang_final={rec['angle_to_final_deg']:5.1f}  "
              f"ord_ic={rec['order_incontext']:.2f}  ord_sem={rec['order_semantic']:.2f}")

    (out / "shift_metrics.json").write_text(json.dumps(
        {"layer": layer, "records": records}, indent=2))
    plot(records, out / "shift_summary.pdf")
    print(f"[analyze] wrote {out/'shift_metrics.json'} and {out/'shift_summary.pdf'}")


def plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = [r["ctx"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(cs, [r["angle_to_pretrained_deg"] for r in records], "o-",
            label="to pretrained plane")
    ax.plot(cs, [r["angle_to_final_deg"] for r in records], "s-",
            label="to final (learned) plane")
    ax.set(xscale="log", xlabel="context length (words)",
           ylabel="mean principal angle (deg)", title="subspace trajectory")
    ax.legend()

    ax = axes[1]
    ax.plot(cs, [r["order_incontext"] for r in records], "o-", label="in-context order")
    ax.plot(cs, [r["order_semantic"] for r in records], "s-", label="semantic order")
    ax.axhline(2 / 7, color="gray", ls=":", label="chance")
    ax.set(xscale="log", xlabel="context length", ylabel="cyclic order agreement",
           title="which ring is expressed (activations)")
    ax.legend()

    ax = axes[2]
    have = [r for r in records if "iso_dist_spearman" in r]
    if have:
        ax.plot([r["ctx"] for r in have], [r["iso_dist_spearman"] for r in have],
                "o-", label="act-beh dist Spearman")
        ax.plot([r["ctx"] for r in have], [r["beh_order_incontext"] for r in have],
                "s--", label="behavior: in-context order")
        ax.plot([r["ctx"] for r in have], [r["beh_order_semantic"] for r in have],
                "^--", label="behavior: semantic order")
    ax.set(xscale="log", xlabel="context length", ylabel="score",
           title="isometry through the transition")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
