"""Generate the data needed for the lambda-hat analysis (Steps 1-6).

The behavioral logs hold only the discrete choice and only SOCIAL-shift pairs.
This produces the missing pieces with a forward-only GPU pass:
  * a CONTINUOUS readout y_model = (target company logit) - (mean company logit)
    at the decision token (centered so it is a clean relative valuation), and
  * single-channel counterfactual shifts of BOTH channels (social and private),
    at multiple signed magnitudes Δ ∈ {±1,±2,±4}, with the other channel held
    fixed — exactly what the paired-slope estimator and the |Δ|-linearity check need.

Per condition cell (lambda->sigma_s, tau), seed, target: one base arm plus, for
each channel and Δ, one shifted arm. Rows are written to <out>/lam_trials.jsonl.

    python lam_data.py --framing new --out results/scoped_market
    python lam_data.py --framing old --out results/scoped
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, yaml

import env as E
import prompt as P
from cossim import render_old   # the old "second analyst" framing


def sigma_s_for_lambda(lam, sp):
    lam = min(max(lam, 1e-6), 1 - 1e-6)
    return sp * math.sqrt((1 - lam) / lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--framing", choices=["old", "new"], required=True)
    ap.add_argument("--config", default="configs/scoped_market.yaml")
    ap.add_argument("--deltas", default="-4,-2,-1,1,2,4")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    deltas = [float(d) for d in args.deltas.split(",")]
    render = (lambda pr, so, ec: P.render(pr, so, ec)) if args.framing == "new" else render_old

    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    n = cfg["n_companies"]
    cids = runner.company_token_ids(n)

    def y_of(state, ec, target):
        prm = render(state.private, state.social, ec)
        _, comp = runner.forward_logits(prm, company_token_ids=cids)
        return float(comp[target] - comp.mean()), comp.tolist()

    out_path = Path(args.out) / "lam_trials.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w")
    sp = cfg["sigma_p"]
    for lam in cfg["lambdas"]:
        ss = sigma_s_for_lambda(lam, sp)
        for tau in cfg["taus"]:
            ec = E.EnvConfig(n_companies=n, T=cfg["T"], sigma_p=sp, sigma_s=ss, tau=tau,
                             w=0.0, theta_scale=cfg["theta_scale"], prior_std=cfg["prior_std"],
                             allow_withdraw=False)
            cond = {"lmbda": lam, "sigma_p": sp, "sigma_s": ss, "tau": tau, "w": 0.0}
            for seed in cfg["seeds"]:
                for target in range(n):
                    st = E.make_state(ec, seed)
                    y0, c0 = y_of(st, ec, target)
                    p0 = float(st.private[target].mean()); s0 = float(st.social[target].mean())
                    pid = f"L{lam}_t{tau}_s{seed}_c{target}"
                    f.write(json.dumps({**cond, "seed": seed, "target": target, "pair_id": pid,
                                        "arm": "base", "shifted": "none", "delta": 0.0,
                                        "y_model": y0, "y_base": y0, "p": p0, "s": s0}) + "\n")
                    for ch in ["social", "private"]:
                        for d in deltas:
                            st2 = st.copy()
                            if ch == "social":
                                st2.social[target] = st.social[target] + d
                            else:
                                st2.private[target] = st.private[target] + d
                            yd, _ = y_of(st2, ec, target)
                            p = float(st2.private[target].mean()); s = float(st2.social[target].mean())
                            f.write(json.dumps({**cond, "seed": seed, "target": target, "pair_id": pid,
                                                "arm": "shift", "shifted": ch, "delta": d,
                                                "y_model": yd, "y_base": y0, "p": p, "s": s}) + "\n")
    f.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
