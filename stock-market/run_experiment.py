"""Sweep lambda x tau x w x trial-mode x seeds; generate counterfactual pairs;
capture activations; write the log table.

Safeguards (per spec):
  * --dry-run prints the job list + forward-pass estimate and exits (no model).
  * --smoke runs ONE trial AND one counterfactual pair end-to-end and prints the
    decision-line tokenization, marker position, captured span, activation shapes,
    the prompt diff for the pair, and one patching step — for you to verify.
  * a full run exceeding `max_forward_passes` refuses unless --yes is passed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

import env as E
import prompt as P


def sigma_s_for_lambda(lmbda: float, sigma_p: float) -> float:
    """Invert lambda = (1/ss^2)/(1/sp^2 + 1/ss^2)  ->  ss = sp * sqrt((1-l)/l)."""
    lmbda = min(max(lmbda, 1e-6), 1 - 1e-6)
    return sigma_p * math.sqrt((1 - lmbda) / lmbda)


def cells(cfg: dict):
    for lam in cfg["lambdas"]:
        ss = sigma_s_for_lambda(lam, cfg["sigma_p"])
        for tau in cfg["taus"]:
            for w in cfg["ws"]:
                for seed in cfg["seeds"]:
                    yield dict(lmbda=lam, sigma_p=cfg["sigma_p"], sigma_s=ss,
                               tau=tau, w=w, seed=seed)


def env_cfg(cell, cfg) -> E.EnvConfig:
    return E.EnvConfig(n_companies=cfg["n_companies"], T=cfg["T"],
                       sigma_p=cell["sigma_p"], sigma_s=cell["sigma_s"], tau=cell["tau"],
                       w=cell["w"], theta_scale=cfg["theta_scale"], prior_std=cfg["prior_std"],
                       allow_withdraw=cfg["allow_withdraw"], seed=cell["seed"])


def trials_for_cell(cell, cfg) -> list[E.Trial]:
    ec = env_cfg(cell, cfg)
    out: list[E.Trial] = []
    targets = cfg["targets"][: min(2, len(cfg["targets"]))]   # subsample to bound cost
    for mode in cfg["trial_modes"]:
        for target in targets:
            if mode == "neutral":
                out.append(E.neutral_trial(ec, cell["seed"], target))
            elif mode == "disagreement":
                for d in cfg["deltas"]:
                    out.append(E.disagreement_trial(ec, cell["seed"], target, d))
            elif mode == "counterfactual":
                for d in cfg["deltas"]:
                    lo, hi = E.counterfactual_pair(ec, cell["seed"], target, d)
                    out.extend([lo, hi])
    return out


def estimate(cfg: dict) -> tuple[int, int]:
    n_cells = sum(1 for _ in cells(cfg))
    per_cell = len(trials_for_cell(next(iter(cells(cfg))), cfg))
    return n_cells, n_cells * per_cell


# --------------------------------------------------------------------------- #
def rational_row(trial: E.Trial, ec: E.EnvConfig) -> dict:
    er, et, ec_post = E.expected_reward(trial.state.private, trial.state.social, ec)
    return {"E_theta": et.tolist(), "E_c": ec_post.tolist(), "E_reward": er.tolist(),
            "rational_action": int(np.argmax(er)),
            "rational_eff_social_weight": E.rational_effective_social_weight(
                ec, trial.state.private.shape[1], trial.state.social.shape[1])}


def smoke(cfg: dict):
    """One trial + one counterfactual pair end-to-end, with verification prints."""
    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    cell = next(iter(cells(cfg)))
    ec = env_cfg(cell, cfg)

    # single neutral trial
    tr = E.neutral_trial(ec, cell["seed"], target=0)
    prm = P.render(tr.state.private, tr.state.social, ec)
    out = runner.run(prm)
    pl, sl = runner.block_token_lengths(prm)
    print("=== SMOKE: single trial ===")
    print(f"lambda={cell['lmbda']} tau={cell['tau']} w={cell['w']}")
    print(f"PRIVATE block tokens={pl}  SOCIAL block tokens={sl}  residual={pl - sl}")
    print(f"decision line tokens: {runner.tokenizer.convert_ids_to_tokens(out.input_ids[out.decision_span[0]:out.decision_span[1] + 1])}")
    print(f"marker_pos={out.marker_pos}  decision_span={out.decision_span}  seq_len={out.seq_len}")
    print(f"activation shape (marker): {out.activations.shape}   span: {out.span_activations.shape}")
    print(f"model output: {out.text!r}  parsed={P.parse_action(out.text, ec.n_companies)}")

    # one counterfactual pair + one patch step
    lo, hi = E.counterfactual_pair(ec, cell["seed"], target=2, delta=4.0)
    p_lo = P.render(lo.state.private, lo.state.social, ec)
    p_hi = P.render(hi.state.private, hi.state.social, ec)
    o_lo, o_hi = runner.run(p_lo), runner.run(p_hi)
    print("\n=== SMOKE: counterfactual pair (target C, delta +4) ===")
    print(f"low  action={P.parse_action(o_lo.text, ec.n_companies)}  high action={P.parse_action(o_hi.text, ec.n_companies)}")
    # diff between prompts
    import difflib
    diff = [l for l in difflib.unified_diff(p_lo.splitlines(), p_hi.splitlines(), lineterm="") if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    print("prompt diff:\n  " + "\n  ".join(diff[:8]))
    # patch layer 16 marker activation from low -> high, see if choice flips
    mid = runner.n_layers // 2
    patches = [{"layer": mid, "positions": [o_hi.marker_pos],
                "values": o_lo.activations[mid + 1][None, :]}]
    patched = runner.run_patched(p_hi, patches)
    print(f"patch L{mid} marker (low->high): patched action={P.parse_action(patched, ec.n_companies)}")
    print("\nSmoke OK.")


def full_run(cfg: dict, out_root: Path):
    from model_runner import ModelRunner, RunnerConfig
    runner = ModelRunner(RunnerConfig(**cfg["model"]))
    out_root.mkdir(parents=True, exist_ok=True)
    acts_dir = out_root / "acts"; acts_dir.mkdir(exist_ok=True)
    logf = (out_root / "trials.jsonl").open("w")
    tid = 0
    for cell in cells(cfg):
        ec = env_cfg(cell, cfg)
        for tr in trials_for_cell(cell, cfg):
            prm = P.render(tr.state.private, tr.state.social, ec)
            pl, sl = runner.block_token_lengths(prm)
            out = runner.run(prm)
            np.save(acts_dir / f"{tid:06d}.npy", out.activations)
            np.save(acts_dir / f"{tid:06d}_span.npy", out.span_activations)
            priv, soc = E.channel_estimates(tr.state.private, tr.state.social) \
                if hasattr(E, "channel_estimates") else (tr.state.private.mean(1), tr.state.social.mean(1))
            row = {"trial_id": tid, **cell, "mode": tr.mode, "target": tr.target,
                   "delta": tr.delta, "pair_id": tr.pair_id, "arm": tr.arm,
                   "block_tokens_private": pl, "block_tokens_social": sl,
                   "private": tr.state.private.tolist(), "social": tr.state.social.tolist(),
                   "theta": tr.state.theta.tolist(), "c": tr.state.c.tolist(),
                   **rational_row(tr, ec),
                   "private_implied": priv.tolist(), "social_implied": soc.tolist(),
                   "output": out.text, "parsed": P.parse_action(out.text, ec.n_companies),
                   "marker_pos": out.marker_pos, "decision_span": list(out.decision_span),
                   "acts_path": str(acts_dir / f"{tid:06d}.npy")}
            logf.write(json.dumps(row) + "\n")
            tid += 1
    logf.close()
    (out_root / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"wrote {tid} trials -> {out_root}/trials.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a run exceeding max_forward_passes")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    n_cells, n_fwd = estimate(cfg)
    print(f"=== {n_cells} config cells, ~{n_fwd} focal forward passes (capture; patching adds more) ===")
    if args.dry_run:
        print("--dry-run: exiting before model load."); return
    if args.smoke:
        smoke(cfg); return
    if n_fwd > cfg["max_forward_passes"] and not args.yes:
        print(f"REFUSING: {n_fwd} > max_forward_passes={cfg['max_forward_passes']}. "
              f"Re-run with --yes to proceed (or shrink the grid)."); return
    full_run(cfg, Path(cfg["output_root"]))


if __name__ == "__main__":
    main()
