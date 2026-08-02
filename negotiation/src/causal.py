"""The causal step: is A's opponent representation USED, not just present?

Take the probe's alpha-direction w in A's activations (probes.py, best layer
unless cfg.causal_layer overrides), set A against a FIXED mid-alpha opponent
(cfg.causal_alpha), and steer A's generations by gamma * w_hat at that layer
for gamma in cfg.causal_gammas. If injecting "opponent is greedy" shifts A's
counteroffers the way genuinely greedy opponents do (reference curve fitted
on the corpus transcripts: A's behavior vs B's true alpha), the representation
is mechanism, not epiphenomenon.

Behavioral readouts per episode: A's mean counter-demand (A's own share when
countering) and A's accept rate. Capture is OFF (we only need behavior) and
the capture pass would be unsteered anyway.

Outputs: causal_transcripts/, causal_results.json, causal_shift.pdf/png.

Run:  python src/causal.py --preset default
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config          # noqa: E402
from baselines import load_transcripts         # noqa: E402
from game import play_episode                  # noqa: E402
from modeling import Steering, load_model      # noqa: E402


def a_behavior(turns) -> dict:
    counters = [t for t in turns if t["a_action"] == "counter"]
    return {
        "counter_demand": (float(np.mean([t["a_counter_share"]
                                          for t in counters]))
                           if counters else np.nan),
        "accept_rate": float(np.mean([t["a_action"] == "accept"
                                      for t in turns])),
    }


def corpus_reference(cfg: Config, n_bins: int = 5):
    """A's behavior vs B's TRUE alpha, from the main corpus: the yardstick the
    steered runs are compared against."""
    recs = load_transcripts(cfg)
    alphas = np.array([r["alpha"] for r in recs])
    beh = [a_behavior(r["turns"]) for r in recs]
    bins = np.linspace(cfg.alpha_lo, cfg.alpha_hi, n_bins + 1)
    ref = []
    for k in range(n_bins):
        sel = (alphas >= bins[k]) & (alphas < bins[k + 1] + (k == n_bins - 1))
        cd = [beh[i]["counter_demand"] for i in np.where(sel)[0]
              if not np.isnan(beh[i]["counter_demand"])]
        ar = [beh[i]["accept_rate"] for i in np.where(sel)[0]]
        ref.append({"alpha_mid": float((bins[k] + bins[k + 1]) / 2),
                    "n": int(sel.sum()),
                    "counter_demand": float(np.mean(cd)) if cd else np.nan,
                    "accept_rate": float(np.mean(ar)) if ar else np.nan})
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_dir = cfg.run_dir()

    w = np.load(os.path.join(run_dir, "probe_directions.npz"))["w"]
    r2 = np.load(os.path.join(run_dir, "probe_r2.npz"))["r2"]
    layer_hs = int(np.argmax(r2[:, -1])) if cfg.causal_layer < 0 \
        else cfg.causal_layer                       # hidden_states index
    if layer_hs < 1:
        raise SystemExit("best probe layer is the embedding layer; pick "
                         "an explicit cfg.causal_layer")
    block = layer_hs - 1   # Steering hooks block l -> hidden_states index l+1
    print(f"steering A along probe direction at hidden_states index "
          f"{layer_hs} (decoder block {block})")

    torch.manual_seed(cfg.seed + 7)
    model, tok = load_model(cfg)
    steer_vecs = None
    if cfg.tier == 2:
        from steering import load_direction
        steer_vecs = load_direction(cfg)

    tdir = os.path.join(run_dir, "causal_transcripts")
    os.makedirs(tdir, exist_ok=True)

    results = {"layer_hs": layer_hs, "alpha_opponent": cfg.causal_alpha,
               "gammas": {}, "reference": corpus_reference(cfg)}
    for gamma in cfg.causal_gammas:
        per_ep = []
        for j in range(cfg.causal_episodes):
            idx = 900_000 + int(round(gamma * 1000)) % 100_000 + j
            a_steer = Steering(model, w, [block], coef=gamma) \
                if gamma != 0.0 else None
            ep = play_episode(model, tok, cfg, idx, cfg.causal_alpha,
                              steer_vecs=steer_vecs, a_steer=a_steer,
                              capture=False, verbalize=False)
            rec = ep.to_json()
            with open(os.path.join(
                    tdir, f"ep_g{gamma:+05.1f}_{j:03d}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            per_ep.append(a_behavior(rec["turns"]))
        cd = [b["counter_demand"] for b in per_ep
              if not np.isnan(b["counter_demand"])]
        summary = {
            "n": len(per_ep),
            "counter_demand_mean": float(np.mean(cd)) if cd else np.nan,
            "counter_demand_sem": (float(np.std(cd) / np.sqrt(len(cd)))
                                   if len(cd) > 1 else np.nan),
            "accept_rate_mean": float(np.mean([b["accept_rate"]
                                               for b in per_ep])),
        }
        results["gammas"][f"{gamma:+.1f}"] = summary
        print(f"gamma={gamma:+5.1f}: counter_demand="
              f"{summary['counter_demand_mean']:.1f} "
              f"accept_rate={summary['accept_rate_mean']:.2f}")

    with open(os.path.join(run_dir, "causal_results.json"), "w") as f:
        json.dump(results, f, indent=1)

    # plot: steered A vs the corpus reference curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ref = results["reference"]
    for ax, key, label in [
            (axes[0], "counter_demand", "A's mean counter-demand"),
            (axes[1], "accept_rate", "A's accept rate")]:
        ax.plot([r["alpha_mid"] for r in ref], [r[key] for r in ref],
                marker="o", color="gray",
                label="corpus: vs B's true $\\alpha$")
        gammas = sorted(results["gammas"], key=float)
        vals = [results["gammas"][g][key + "_mean"] for g in gammas]
        for g, v in zip(gammas, vals):
            ax.axhline(v, ls="--", lw=1)
            ax.annotate(f"$\\gamma$={g}", xy=(cfg.alpha_lo, v), fontsize=7,
                        va="bottom")
        ax.set_xlabel("B's true $\\alpha$ (reference) ")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    fig.suptitle(f"steering A along the opponent-greed probe direction "
                 f"(layer {layer_hs}, opponent $\\alpha$="
                 f"{cfg.causal_alpha}) -- {cfg.name}")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"causal_shift.{ext}"),
                    bbox_inches="tight", dpi=150)
    print(f"wrote causal_results.json and causal_shift -> {run_dir}")


if __name__ == "__main__":
    main()
