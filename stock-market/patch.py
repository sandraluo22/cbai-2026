"""Activation patching — the load-bearing causal measurement.

For each counterfactual pair (social_low vs social_high), patch the DONOR (low)
activations into the RECEIVER (high) run, layer by layer, at two position sets:
  * "marker"        — the Decision ":" token (the default belief anchor), and
  * "social_target" — the social-evidence tokens for the shifted company.
Record whether the receiver's choice FLIPS toward the donor's choice. This
localizes WHERE the social channel changes the decision, without assuming the
colon holds the belief.

Reads the experiment's trials.jsonl (which stores each arm's private/social
arrays) and writes patching.jsonl keyed by (pair_id, layer, position_set).

Alignment note: patching requires the low/high prompts to have equal token length
and aligned positions. They differ only in the target's social values; if a value
crosses a digit boundary the token counts can differ — such pairs are detected and
skipped with a logged `aligned=false` row.
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


def _ec_from_row(row, cfg) -> E.EnvConfig:
    return E.EnvConfig(n_companies=cfg["n_companies"], T=cfg["T"], sigma_p=row["sigma_p"],
                       sigma_s=row["sigma_s"], tau=row["tau"], w=row["w"],
                       theta_scale=cfg["theta_scale"], prior_std=cfg["prior_std"],
                       allow_withdraw=cfg["allow_withdraw"], seed=row["seed"])


def _social_target_positions(runner, prompt, target, n) -> list[int]:
    """Token indices of the EXTERNAL block row for `target` company."""
    from prompt import _company_label
    _, _, offsets = runner._encode(prompt)
    # locate the social block, then the target company's row within it
    soc0 = prompt.rfind("EXTERNAL readings")
    row_label = f"Company {_company_label(target)}:"
    c0 = prompt.find(row_label, soc0)
    if c0 < 0:
        return []
    c1 = prompt.find("\n", c0)
    c1 = c1 if c1 > 0 else len(prompt)
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0]


def run_patching(cfg: dict, out_root: Path):
    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    rows = [json.loads(l) for l in (out_root / "trials.jsonl").read_text().splitlines()]
    pairs = defaultdict(dict)
    for r in rows:
        if r["mode"] == "counterfactual" and r["pair_id"]:
            # pair_id alone (s{seed}_t{target}_d{delta}) collides across config
            # cells because it omits lambda/w/tau — without the full cell key the
            # dict would keep only the LAST cell's arms per pair_id. Key on the cell.
            cell_key = (r["pair_id"], r["lmbda"], r["w"], r["tau"], r["seed"])
            pairs[cell_key][r["arm"]] = r

    out = (out_root / "patching.jsonl").open("w")
    n = cfg["n_companies"]
    for cell_key, arms in pairs.items():
        pid = cell_key[0]
        if "low" not in arms or "high" not in arms:
            continue
        lo, hi = arms["low"], arms["high"]
        ec = _ec_from_row(hi, cfg)
        p_lo = P.render(np.array(lo["private"]), np.array(lo["social"]), ec)
        p_hi = P.render(np.array(hi["private"]), np.array(hi["social"]), ec)
        o_lo, o_hi = runner.run(p_lo), runner.run(p_hi)
        a_lo = P.parse_action(o_lo.text, n)["company"]
        a_hi = P.parse_action(o_hi.text, n)["company"]
        aligned = (o_lo.seq_len == o_hi.seq_len)

        pos_sets = {"marker": [o_hi.marker_pos],
                    "social_target": _social_target_positions(runner, p_hi, hi["target"], n)}
        cell = {"lmbda": hi["lmbda"], "w": hi["w"], "tau": hi["tau"], "seed": hi["seed"]}
        for pset_name, positions in pos_sets.items():
            if not positions or not aligned:
                out.write(json.dumps({"pair_id": pid, **cell, "position_set": pset_name,
                                      "aligned": aligned, "skipped": True}) + "\n")
                continue
            donor = runner.capture_positions(p_lo, positions)     # (L+1, P, hidden)
            for layer in range(runner.n_layers):
                patches = [{"layer": layer, "positions": positions,
                            "values": donor[layer + 1]}]          # +1: skip embedding index
                patched_txt = runner.run_patched(p_hi, patches)
                a_patched = P.parse_action(patched_txt, n)["company"]
                out.write(json.dumps({
                    "pair_id": pid, **cell, "layer": layer, "position_set": pset_name,
                    "aligned": True, "target": hi["target"],
                    "action_low": a_lo, "action_high": a_hi, "action_patched": a_patched,
                    "flipped": bool(a_patched != a_hi),
                    "flipped_to_low": bool(a_patched == a_lo and a_lo != a_hi)}) + "\n")
    out.close()
    print(f"patching done -> {out_root}/patching.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp")
    args = ap.parse_args()
    out_root = Path(args.out)
    cfg = yaml.safe_load((out_root / "config.yaml").read_text())
    run_patching(cfg, out_root)


if __name__ == "__main__":
    main()
