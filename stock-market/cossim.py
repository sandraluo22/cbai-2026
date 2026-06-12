"""Cosine similarity between the OLD ("second analyst") and NEW ("market trend")
social-steering directions.

For each matched counterfactual pair (same seed/env in both runs; only the prompt
framing differs), the steering direction at layer L is
    d_L = act(donor) - act(receiver)        at the DECISION (last) token,
donor = the arm with MORE social evidence on the target. We recompute d_L under
both framings and report cos(d_L^old, d_L^new), averaged over pairs, per layer.
High cos => same axis, weaker response (framing changed magnitude, not direction);
low cos => the framing rotated the social representation.

Also computes the same at the MARKER (colon) token from the saved acts as a
cross-check (the probe/representational site vs the causal/steering site).

    python cossim.py   # expects results/scoped (old) and results/scoped_market (new)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import prompt as P
from prompt import _readings_block, MARKER, PRIVATE_LABEL, SOCIAL_LABEL
import env as E
from patch import _ec_from_row


def render_old(private, social, ec):
    """The pre-edit framing that produced results/scoped (second-analyst wording)."""
    n = private.shape[0]
    companies = ", ".join(P._company_label(i) for i in range(n))
    framing = (
        f"You are an investor choosing where to invest among {n} companies: "
        f"{companies}. Each has a hidden value. You have two evidence sources, "
        f"shown in the same format: your own {PRIVATE_LABEL} readings, and the "
        f"{SOCIAL_LABEL} readings from other analysts. Higher is better.\n")
    pblock = _readings_block(PRIVATE_LABEL, private, n)
    sblock = _readings_block(SOCIAL_LABEL, social, n)
    instruct = (f"\nWeigh the two sources and invest in ONE company ({companies}). "
                f"Respond with ONLY the single company letter you choose "
                f"(one of {companies}) — no other words.\n")
    return f"{framing}\n{pblock}\n\n{sblock}\n{instruct}\n{MARKER}"


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def main():
    import yaml
    from model_runner import ModelRunner, RunnerConfig
    cfg = yaml.safe_load(Path("results/scoped_market/config.yaml").read_text())
    runner = ModelRunner(RunnerConfig(**cfg["model"]))

    def load_pairs(path):
        d = {}
        for r in map(json.loads, open(path)):
            if r["mode"] != "counterfactual":
                continue
            k = (r["pair_id"], r["lmbda"], r["w"], r["tau"], r["seed"])
            d.setdefault(k, {})[r["arm"]] = r
        return d
    po = load_pairs("results/scoped/trials.jsonl")
    pn = load_pairs("results/scoped_market/trials.jsonl")

    dec_cos, mk_cos = [], []   # per pair: (L,) arrays
    for k in po:
        if k not in pn or "low" not in po[k] or "high" not in po[k]:
            continue
        lo, hi = po[k]["low"], po[k]["high"]
        tgt = hi["target"]
        # donor = arm with MORE social on target (identical across framings)
        more = "high" if np.array(hi["social"])[tgt].mean() >= np.array(lo["social"])[tgt].mean() else "low"
        less = "low" if more == "high" else "high"
        ec = _ec_from_row(hi, cfg)

        def dec_dir(run, rowdict):
            don, rec = rowdict[more], rowdict[less]
            rend = render_old if run == "old" else (lambda pr, so, e: P.render(pr, so, e))
            pd = rend(np.array(don["private"]), np.array(don["social"]), ec)
            pr = rend(np.array(rec["private"]), np.array(rec["social"]), ec)
            ld = runner._encode(pd)[1]["input_ids"].shape[1] - 1
            lr = runner._encode(pr)[1]["input_ids"].shape[1] - 1
            D = runner.capture_positions(pd, [ld])[:, 0, :]   # (L+1, H)
            R = runner.capture_positions(pr, [lr])[:, 0, :]
            return D - R
        do = dec_dir("old", po[k]); dn = dec_dir("new", pn[k])
        dec_cos.append([cos(do[L], dn[L]) for L in range(do.shape[0])])

        # marker direction from saved acts (donor - receiver), per run
        def mk_dir(base, rowdict):
            don = np.load(f"{base}/acts/{rowdict[more]['trial_id']:06d}.npy")
            rec = np.load(f"{base}/acts/{rowdict[less]['trial_id']:06d}.npy")
            return don - rec                                  # (33, H)
        mo = mk_dir("results/scoped", po[k]); mn = mk_dir("results/scoped_market", pn[k])
        mk_cos.append([cos(mo[L], mn[L]) for L in range(mo.shape[0])])

    dec = np.array(dec_cos); mk = np.array(mk_cos)
    print(f"matched pairs: {len(dec)}")
    print("\ncos(old, new) social direction — mean over pairs, by layer:")
    print(f"{'L':>3} | {'decision-tok':>12} | {'marker-tok':>10}")
    for L in range(mk.shape[1]):
        dv = f"{np.nanmean(dec[:, L]):+.3f}" if L < dec.shape[1] else "   -  "
        print(f"{L:>3} | {dv:>12} | {np.nanmean(mk[:, L]):+.3f}")
    print(f"\nOVERALL mean cos — decision-tok: {np.nanmean(dec):+.3f}   marker-tok: {np.nanmean(mk):+.3f}")
    print(f"mid/late layers (15-31) decision-tok: {np.nanmean(dec[:, 15:]):+.3f}")
    np.savez("results/cossim_old_vs_new.npz", decision=dec, marker=mk)
    print("saved -> results/cossim_old_vs_new.npz")


if __name__ == "__main__":
    main()
