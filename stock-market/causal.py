"""Corrected causal localization: patching AND steering at the DECISION position
with a graded logit readout.

Why this supersedes patch.py/steer.py for the causal claim:
  * Site: those anchored on the 'Decision :' colon, but with use_chat_template=true
    the colon is followed by the assistant-header tokens, so it is NOT the position
    that emits the decision. Intervening there is a near no-op (verified). Here we
    intervene at the LAST prompt token — the position whose next-token distribution
    IS the decision.
  * Readout: those recorded only greedy argmax flips, which a robust model rarely
    shows. Here we read the per-company LOGIT (max over each letter's candidate
    tokens), a graded measure that reveals sub-threshold causal effects.
  * Alignment: those required low/high to be token-aligned (52/64 pairs were skipped
    when a shifted social value changed token counts). Intervening only at each
    prompt's own last position needs no cross-prompt alignment, so all pairs are used.

For each counterfactual pair we set donor = the arm with MORE social evidence on the
target, receiver = the arm with LESS. We then, layer by layer:
  * PATCH the donor's decision-position residual into the receiver, and
  * STEER the receiver by alpha * (donor - receiver) at the decision position,
and record the change in the target company's logit. A positive shift means "more
social evidence for the target causally raised the model's preference for it."

    python causal.py --out results/scoped --alphas 0,0.5,1,2,4,8
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
from patch import _ec_from_row


def run_causal(cfg: dict, out_root: Path, alphas: list[float]):
    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    n = cfg["n_companies"]
    cids = runner.company_token_ids(n)
    rows = [json.loads(l) for l in (out_root / "trials.jsonl").read_text().splitlines()]
    pairs = defaultdict(dict)
    for r in rows:
        if r["mode"] == "counterfactual" and r["pair_id"]:
            pairs[(r["pair_id"], r["lmbda"], r["w"], r["tau"], r["seed"])][r["arm"]] = r

    out = (out_root / "causal.jsonl").open("w")
    for cell_key, arms in pairs.items():
        if "low" not in arms or "high" not in arms:
            continue
        pid = cell_key[0]
        lo, hi = arms["low"], arms["high"]
        tgt = hi["target"]
        # orient by ACTUAL social magnitude on the target (delta can be + or -)
        lo_soc = np.array(lo["social"])[tgt].mean()
        hi_soc = np.array(hi["social"])[tgt].mean()
        donor, recv = (hi, lo) if hi_soc >= lo_soc else (lo, hi)
        ec = _ec_from_row(hi, cfg)
        p_donor = P.render(np.array(donor["private"]), np.array(donor["social"]), ec)
        p_recv = P.render(np.array(recv["private"]), np.array(recv["social"]), ec)

        o_recv, o_donor = runner.run(p_recv, generate=False), runner.run(p_donor, generate=False)
        base_recv = runner.forward_logits(p_recv, company_token_ids=cids)[1]
        base_donor = runner.forward_logits(p_donor, company_token_ids=cids)[1]
        recv_last, donor_last = o_recv.seq_len - 1, o_donor.seq_len - 1
        # decision-position residual, all layers, for donor and receiver
        D = runner.capture_positions(p_donor, [donor_last])   # (L+1, 1, H)
        Rv = runner.capture_positions(p_recv, [recv_last])
        cell = {"lmbda": hi["lmbda"], "w": hi["w"], "tau": hi["tau"], "seed": hi["seed"]}

        for layer in range(runner.n_layers):
            # PATCH donor decision residual -> receiver
            _, comp_p = runner.forward_logits(
                p_recv, company_token_ids=cids,
                patches=[{"layer": layer, "positions": [recv_last], "values": D[layer + 1]}])
            # STEER receiver by alpha*(donor-receiver)
            vec = D[layer + 1] - Rv[layer + 1]
            steer_rows = []
            for a in alphas:
                _, comp_s = runner.forward_logits(
                    p_recv, company_token_ids=cids,
                    steers=[{"layer": layer, "positions": [recv_last], "vector": vec, "alpha": a}])
                steer_rows.append({"alpha": a,
                                   "dlogit_target": float(comp_s[tgt] - base_recv[tgt]),
                                   "argmax": int(np.argmax(comp_s))})
            out.write(json.dumps({
                "pair_id": pid, **cell, "layer": layer, "target": tgt,
                "base_recv_target_logit": float(base_recv[tgt]),
                "base_donor_target_logit": float(base_donor[tgt]),
                "base_recv_argmax": int(np.argmax(base_recv)),
                "base_donor_argmax": int(np.argmax(base_donor)),
                "patch_dlogit_target": float(comp_p[tgt] - base_recv[tgt]),
                "patch_argmax": int(np.argmax(comp_p)),
                "steer": steer_rows}) + "\n")
    out.close()
    print(f"causal done -> {out_root}/causal.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    ap.add_argument("--alphas", default="0,0.5,1,2,4,8")
    args = ap.parse_args()
    out_root = Path(args.out)
    cfg = yaml.safe_load((out_root / "config.yaml").read_text())
    alphas = [float(a) for a in args.alphas.split(",") if a != ""]
    run_causal(cfg, out_root, alphas)


if __name__ == "__main__":
    main()
