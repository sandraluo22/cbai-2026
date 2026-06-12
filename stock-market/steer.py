"""Activation STEERING (activation addition) — the scalable cousin of patch.py.

Where patch.py OVERWRITES the receiver's activations with the donor's, steering
ADDS a scaled direction and sweeps the coefficient alpha. Here the direction is
the natural per-pair social contrast at each layer:

    d_L = act(social_high) - act(social_low)            (at marker / social tokens)

We add alpha * d_L to the social_LOW run and ask: as alpha grows, does the choice
move toward what the social_high run chose (and toward the shifted target)?
  * alpha=0  -> the unsteered low run (control)
  * alpha=1  -> approximately reconstructs the high activation (~ patching)
  * alpha>1  -> over-amplified social evidence, beyond the natural +delta shift.

This tests whether a single linear "social" axis can CAUSALLY drive the decision
even though the raw +-delta counterfactual could not. Mirrors patch.py's two
position sets ("marker", "social_target") and its (fixed) pair keying.

    python steer.py --out results/scoped --alphas 0,1,2,4,8,16
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import env as E
import prompt as P
from patch import _ec_from_row, _social_target_positions


def run_steering(cfg: dict, out_root: Path, alphas: list[float], layers: list[int] | None):
    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    rows = [json.loads(l) for l in (out_root / "trials.jsonl").read_text().splitlines()]
    pairs = defaultdict(dict)
    for r in rows:
        if r["mode"] == "counterfactual" and r["pair_id"]:
            cell_key = (r["pair_id"], r["lmbda"], r["w"], r["tau"], r["seed"])
            pairs[cell_key][r["arm"]] = r

    layer_list = layers if layers is not None else list(range(runner.n_layers))
    out = (out_root / "steering.jsonl").open("w")
    n = cfg["n_companies"]
    for cell_key, arms in pairs.items():
        if "low" not in arms or "high" not in arms:
            continue
        pid = cell_key[0]
        lo, hi = arms["low"], arms["high"]
        ec = _ec_from_row(hi, cfg)
        p_lo = P.render(np.array(lo["private"]), np.array(lo["social"]), ec)
        p_hi = P.render(np.array(hi["private"]), np.array(hi["social"]), ec)
        o_lo, o_hi = runner.run(p_lo), runner.run(p_hi)
        a_lo = P.parse_action(o_lo.text, n)["company"]
        a_hi = P.parse_action(o_hi.text, n)["company"]
        aligned = (o_lo.seq_len == o_hi.seq_len)
        cell = {"lmbda": hi["lmbda"], "w": hi["w"], "tau": hi["tau"], "seed": hi["seed"]}
        tgt = hi["target"]

        # position sets defined on the (low) prompt we steer; must align with high
        pos_sets = {"marker": [o_lo.marker_pos],
                    "social_target": _social_target_positions(runner, p_lo, tgt, n)}
        for pset_name, positions in pos_sets.items():
            if not positions or not aligned:
                out.write(json.dumps({"pair_id": pid, **cell, "position_set": pset_name,
                                      "aligned": aligned, "skipped": True}) + "\n")
                continue
            # contrast direction per layer at these positions: high - low
            a_hi_acts = runner.capture_positions(p_hi, positions)   # (L+1, P, hidden)
            a_lo_acts = runner.capture_positions(p_lo, positions)
            diff = a_hi_acts - a_lo_acts                            # (L+1, P, hidden)
            for layer in layer_list:
                for alpha in alphas:
                    steers = [{"layer": layer, "positions": positions,
                               "vector": diff[layer + 1], "alpha": alpha}]
                    txt = runner.run_steered(p_lo, steers)
                    a_st = P.parse_action(txt, n)["company"]
                    out.write(json.dumps({
                        "pair_id": pid, **cell, "layer": layer, "alpha": alpha,
                        "position_set": pset_name, "aligned": True, "target": tgt,
                        "action_low": a_lo, "action_high": a_hi, "action_steered": a_st,
                        "moved": bool(a_st != a_lo),
                        "moved_to_high": bool(a_st == a_hi and a_hi != a_lo),
                        "moved_to_target": bool(a_st == tgt)}) + "\n")
    out.close()
    print(f"steering done -> {out_root}/steering.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    ap.add_argument("--alphas", default="0,1,2,4,8,16",
                    help="comma-separated steering coefficients")
    ap.add_argument("--layers", default="",
                    help="comma-separated layers to steer (default: all)")
    args = ap.parse_args()
    out_root = Path(args.out)
    cfg = yaml.safe_load((out_root / "config.yaml").read_text())
    alphas = [float(a) for a in args.alphas.split(",") if a != ""]
    layers = [int(l) for l in args.layers.split(",")] if args.layers else None
    run_steering(cfg, out_root, alphas, layers)


if __name__ == "__main__":
    main()
