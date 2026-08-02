"""Aggregate trust-inference results: stats, CSV, and plots.

Reads the raw results JSON written by ``run.py`` (a list of trial records, each
with a parsed ``response``) and produces:

* primary.png   — trust mass vs. demonstrated accuracy, with a Spearman
                  monotonicity test (does trust rise with the track record?)
* dose.png      — trust *discrimination* (high-acc minus low-acc mass) vs. the
                  number of verified claims (dose condition)
* label.png     — label-override: how far track record moves trust away from the
                  label-only prior (labels + baseline conditions)
* justification.png — rate at which justifications cite the track record, per condition
* summary.csv   — one row per (trial, source)
* summary.json  — the headline metrics

Pure numpy + matplotlib (Agg) — no torch, no scipy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) and "results" in data else data


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson correlation of ranks (no scipy)."""
    if len(x) < 3:
        return float("nan")

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a))
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        avg = sums / counts
        return avg[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# flatten to per-(trial, source) rows
# --------------------------------------------------------------------------- #
def to_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        resp = r.get("response", {})
        trust_by_key = resp.get("trust_by_key", {})
        for s in r["sources"]:
            rows.append({
                "trial_id": r["trial_id"],
                "condition": r["condition"],
                "seed": r["seed"],
                "source_key": s["key"],
                "label": s["label"],
                "position": s["position"],
                "demonstrated_accuracy": s["demonstrated_accuracy"],
                "n_claims": s["n_claims"],
                "error_magnitude": s["error_magnitude"],
                "final_value": s["final_value"],
                "trust": trust_by_key.get(s["key"]),
                "parse_ok": resp.get("parse_ok", False),
                "confidence": resp.get("confidence"),
                "references_track_record": resp.get("references_track_record", False),
                "crossed": r.get("params", {}).get("crossed"),
            })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def primary(rows, outdir) -> dict:
    """Trust mass vs demonstrated accuracy + monotonicity test."""
    pts = [(rw["demonstrated_accuracy"], rw["trust"]) for rw in rows
           if rw["demonstrated_accuracy"] is not None and rw["trust"] is not None
           and rw["parse_ok"]]
    if not pts:
        return {"n": 0}
    acc = np.array([p[0] for p in pts])
    trust = np.array([p[1] for p in pts])
    rho = _spearman(acc, trust)

    # binned means for a monotonicity check across accuracy levels
    levels = sorted(set(np.round(acc, 3)))
    means = [float(trust[np.isclose(acc, lv)].mean()) for lv in levels]
    monotone = all(b >= a - 1e-9 for a, b in zip(means, means[1:]))

    plt.figure(figsize=(6, 4.5))
    jitter = (np.random.default_rng(0).random(len(acc)) - 0.5) * 0.01
    plt.scatter(acc + jitter, trust, alpha=0.3, s=18, color="#4C72B0",
                label="per source")
    plt.plot(levels, means, "o-", color="#C44E52", lw=2, ms=8,
             label="mean per accuracy level")
    plt.xlabel("demonstrated accuracy (fraction correct in log)")
    plt.ylabel("trust mass on final claim")
    plt.title(f"Trust vs demonstrated accuracy\nSpearman rho = {rho:.3f}"
              f"  |  monotone across levels: {monotone}")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "primary.png"), dpi=130)
    plt.close()
    return {"n": len(pts), "spearman_rho": rho, "levels": levels,
            "mean_trust_per_level": means, "monotone": monotone}


def dose(rows, outdir) -> dict:
    """Discrimination (high_acc - low_acc trust) vs number of verified claims."""
    by_trial = defaultdict(dict)
    for rw in rows:
        if rw["condition"] != "dose" or not rw["parse_ok"] or rw["trust"] is None:
            continue
        by_trial[rw["trial_id"]][rw["source_key"]] = rw
    per_n = defaultdict(list)
    for tid, d in by_trial.items():
        if "high_acc" in d and "low_acc" in d:
            n = d["high_acc"]["n_claims"]
            per_n[n].append(d["high_acc"]["trust"] - d["low_acc"]["trust"])
    if not per_n:
        return {"n": 0}
    ns = sorted(per_n)
    means = [float(np.mean(per_n[n])) for n in ns]
    sems = [float(np.std(per_n[n]) / max(1, np.sqrt(len(per_n[n])))) for n in ns]

    plt.figure(figsize=(6, 4.5))
    plt.errorbar(ns, means, yerr=sems, fmt="o-", color="#55A868", lw=2, ms=8,
                 capsize=4)
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.xlabel("number of verified claims (dose)")
    plt.ylabel("trust(high-acc) - trust(low-acc)")
    plt.title("Dose-response: trust discrimination vs evidence")
    plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dose.png"), dpi=130)
    plt.close()
    return {"n": sum(len(v) for v in per_n.values()),
            "dose_levels": ns, "mean_discrimination": means}


def label_override(rows, outdir) -> dict:
    """How far track record shifts trust off the label-only prior.

    label prior: mean trust on the FORUM-labelled source in the baseline (no
    track record). In the 'labels/crossed' trials the high-accuracy source wears
    that same forum label, so its trust *above* the prior is the track-record pull.
    """
    forum_prior = [rw["trust"] for rw in rows
                   if rw["condition"] == "baseline" and rw["source_key"] == "forum_label"
                   and rw["parse_ok"] and rw["trust"] is not None]
    # crossed labels trials: high_acc carries the forum label
    crossed_hi = [rw["trust"] for rw in rows
                  if rw["condition"] == "labels" and rw["crossed"]
                  and rw["source_key"] == "high_acc"
                  and rw["parse_ok"] and rw["trust"] is not None]
    aligned_hi = [rw["trust"] for rw in rows
                  if rw["condition"] == "labels" and rw["crossed"] is False
                  and rw["source_key"] == "high_acc"
                  and rw["parse_ok"] and rw["trust"] is not None]

    prior = float(np.mean(forum_prior)) if forum_prior else float("nan")
    crossed_m = float(np.mean(crossed_hi)) if crossed_hi else float("nan")
    aligned_m = float(np.mean(aligned_hi)) if aligned_hi else float("nan")
    override = crossed_m - prior  # >0 ⇒ track record overrides the bad label

    labels = ["label prior\n(forum, no record)",
              "high-acc + forum label\n(crossed)",
              "high-acc + peer label\n(aligned)"]
    vals = [prior, crossed_m, aligned_m]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, vals, color=["#8172B3", "#C44E52", "#55A868"])
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                     ha="center")
    plt.ylabel("trust mass on the high-accuracy source")
    plt.ylim(0, 1.05)
    plt.title(f"Label override = {override:+.2f}\n"
              "(trust the forum-labelled high-accuracy source gains over the label prior)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "label.png"), dpi=130)
    plt.close()
    return {"forum_label_prior": prior, "crossed_high_acc_trust": crossed_m,
            "aligned_high_acc_trust": aligned_m, "label_override": override,
            "n_baseline": len(forum_prior), "n_crossed": len(crossed_hi)}


def justification_coding(rows, outdir) -> dict:
    """Per-condition rate at which justifications reference the track record."""
    by_cond = defaultdict(list)
    seen = set()
    for rw in rows:
        if not rw["parse_ok"]:
            continue
        if rw["trial_id"] in seen:        # one justification per trial
            continue
        seen.add(rw["trial_id"])
        by_cond[rw["condition"]].append(1 if rw["references_track_record"] else 0)
    conds = sorted(by_cond)
    rates = [float(np.mean(by_cond[c])) for c in conds]

    plt.figure(figsize=(6.5, 4))
    plt.bar(conds, rates, color="#4C72B0")
    plt.ylabel("fraction of justifications citing track record")
    plt.ylim(0, 1.05)
    plt.title("Justification coding by condition")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "justification.png"), dpi=130)
    plt.close()
    return {c: r for c, r in zip(conds, rates)}


# --------------------------------------------------------------------------- #
def analyze(results_path: str, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    results = load_results(results_path)
    rows = to_rows(results)
    write_csv(rows, os.path.join(outdir, "summary.csv"))

    n_total = len(results)
    n_parsed = sum(1 for r in results if r.get("response", {}).get("parse_ok"))
    summary = {
        "results_path": results_path,
        "n_trials": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "primary": primary(rows, outdir),
        "dose": dose(rows, outdir),
        "label_override": label_override(rows, outdir),
        "justification_track_record_rate": justification_coding(rows, outdir),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_summary(s: dict) -> None:
    print("\n=== TRUST-2 SUMMARY ===")
    print(f"trials: {s['n_trials']}  parsed: {s['n_parsed']} "
          f"({s['parse_rate']*100:.0f}%)")
    p = s["primary"]
    if p.get("n"):
        print(f"primary: Spearman rho = {p['spearman_rho']:.3f}  "
              f"monotone={p['monotone']}  (n={p['n']})")
    d = s["dose"]
    if d.get("n"):
        print(f"dose: levels={d['dose_levels']}  "
              f"discrimination={[round(x,2) for x in d['mean_discrimination']]}")
    lo = s["label_override"]
    if not np.isnan(lo.get("label_override", float("nan"))):
        print(f"label override = {lo['label_override']:+.2f} "
              f"(prior={lo['forum_label_prior']:.2f}, "
              f"crossed high-acc={lo['crossed_high_acc_trust']:.2f})")
    print(f"justification track-record rate: {s['justification_track_record_rate']}")
    print("=======================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze trust-2 results.")
    ap.add_argument("results", help="path to results JSON from run.py")
    ap.add_argument("--outdir", default=None, help="plot/CSV output dir "
                    "(default: alongside results)")
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(os.path.dirname(args.results) or ".",
                                         "analysis")
    s = analyze(args.results, outdir)
    _print_summary(s)
    print(f"wrote plots + CSV to {outdir}")
