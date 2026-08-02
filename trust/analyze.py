"""Analyze a completed run: estimate the model's per-round trust trajectory lambda_hat_t,
compare to the learned-precision Bayesian curve, and compute every headline statistic.

lambda_hat_t : per round t, regress the model's estimate on the two source estimates
               across companies x games (history controlled by conditioning on t):
                   model_est ~ b0 + bA * a + bB * b
               lambda_hat_t = bB / (bA + bB)  (weight on the ACCURATE source b).
               Fresh draws each round give the variation that identifies the weights.
               Bootstrap over GAMES for CIs. Degenerate guard: if bA+bB ~ 0 we flag it
               and report raw slopes instead of the ratio.

Writes analysis.json + per-round/summary CSVs into the run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import env as E
import bayes as B


# --------------------------------------------------------------------------- #
def load(out: Path):
    rows = [json.loads(l) for l in (out / "rounds.jsonl").read_text().splitlines()]
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    return rows, cfg


def _fit_lambda(a, b, y):
    """Collinearity-robust convex-weight estimator. With est ~ (1-λ)*a + λ*b + c, regress
    (y - a) on (b - a): the single slope IS λ, the weight on the accurate source b. This
    avoids the explosive βB/(βA+βB) ratio of the two-regressor form when a,b are nearly
    collinear (both ≈ θ + noise). Returns (lambda_hat, bA=1-λ, bB=λ, degenerate).
    Degenerate when a≈b (no spread to separate the sources)."""
    m = ~(np.isnan(a) | np.isnan(b) | np.isnan(y))
    a, b, y = a[m], b[m], y[m]
    d = b - a
    if a.size < 3 or np.var(d) < 1e-6:
        return np.nan, np.nan, np.nan, True
    X = np.column_stack([np.ones_like(d), d])
    beta, *_ = np.linalg.lstsq(X, y - a, rcond=None)
    lam = float(beta[1])
    return lam, float(1.0 - lam), lam, False


def _by_game_round(rows):
    """rows -> dict[(seed)] -> dict[t] -> (a[], b[], y[]) for one condition's rows."""
    g = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    for r in rows:
        a, b, y = g[r["seed"]][r["t"]]
        a.append(r["a"]); b.append(r["b"]); y.append(r["model_est"])
    return {s: {t: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
                for t, v in tv.items()} for s, tv in g.items()}


def lambda_trajectory(rows, T, n_boot=2000, rng=None):
    """Per-round lambda_hat with bootstrap CIs (resampling games)."""
    rng = rng or np.random.default_rng(0)
    bg = _by_game_round(rows)
    seeds = list(bg.keys())
    lam = np.full(T, np.nan); lo = np.full(T, np.nan); hi = np.full(T, np.nan)
    bA_t = np.full(T, np.nan); bB_t = np.full(T, np.nan); degen = np.zeros(T, bool)
    def _cat(samp, k, t):
        arrs = [bg[s][t][k] for s in samp if t in bg[s]]
        return np.concatenate(arrs) if arrs else np.array([])

    for t in range(T):
        A, Bv, Y = _cat(seeds, 0, t), _cat(seeds, 1, t), _cat(seeds, 2, t)
        lam[t], bA_t[t], bB_t[t], degen[t] = _fit_lambda(A, Bv, Y)
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(seeds, size=len(seeds), replace=True)
            l, *_ = _fit_lambda(_cat(samp, 0, t), _cat(samp, 1, t), _cat(samp, 2, t))
            if not np.isnan(l):
                boots.append(l)
        if boots:
            lo[t], hi[t] = np.percentile(boots, [2.5, 97.5])
    return dict(lambda_hat=lam, lo=lo, hi=hi, bA=bA_t, bB=bB_t, degenerate=degen)


def bayes_curve(cond, cfg, T, n_boot=2000, rng=None):
    """Mean learned-precision Bayesian trust_pre curve over the condition's games + CI."""
    rng = rng or np.random.default_rng(1)
    ec = E.EnvConfig(M=cfg["env"]["M"], T=T, mu=cfg["env"]["mu"],
                     theta_scale=cfg["env"]["theta_scale"], sigma_B=cfg["env"]["sigma_B"],
                     gap=cond["gap"])
    prior = B.ReputationPrior(**cfg["prior"])
    curves = []
    for s in range(cond["n_games"]):
        g = E.make_game(ec, seed=cfg["seed_base"] + s)
        curves.append(B.bayes_trajectory(g, prior)["trust_pre"])
    curves = np.array(curves)
    mean = curves.mean(0)
    boots = np.array([curves[rng.integers(0, len(curves), len(curves))].mean(0)
                      for _ in range(n_boot // 4)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(w=mean, lo=lo, hi=hi, oracle=ec.oracle_trust_B,
                prior=prior.trust_B())


def flip_round(series, thresh=0.5):
    idx = np.where(np.asarray(series) >= thresh)[0]
    return int(idx[0] + 1) if idx.size else None


def rmse(rows):
    e = [r["model_est"] - r["theta"] for r in rows if not np.isnan(r["model_est"])]
    return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")


# --------------------------------------------------------------------------- #
def analyze(out: Path):
    rows, cfg = load(out)
    T = cfg["env"]["T"]
    n_boot = cfg.get("bootstrap", 2000)
    conds = {c["name"]: c for c in json.loads((out / "conditions.json").read_text())}
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    result = {"conditions": {}, "T": T}
    rng = np.random.default_rng(0)
    for name, c in conds.items():
        crows = by_cond.get(name, [])
        entry = {**c}
        if c["info_level"] == "no_advisor":
            entry["rmse"] = rmse(crows)
            result["conditions"][name] = entry
            continue
        lam = lambda_trajectory(crows, T, n_boot=n_boot, rng=rng)
        bay = bayes_curve(c, cfg, T, n_boot=n_boot, rng=rng)
        fm, fb = flip_round(lam["lambda_hat"]), flip_round(bay["w"])
        entry.update(
            lambda_hat=lam["lambda_hat"].tolist(), lambda_lo=lam["lo"].tolist(),
            lambda_hi=lam["hi"].tolist(), bA=lam["bA"].tolist(), bB=lam["bB"].tolist(),
            degenerate=lam["degenerate"].tolist(),
            bayes_w=bay["w"].tolist(), bayes_lo=bay["lo"].tolist(),
            bayes_hi=bay["hi"].tolist(), oracle=bay["oracle"], prior=bay["prior"],
            flip_model=fm, flip_bayes=fb,
            lag=(None if fm is None or fb is None else fm - fb),
            anchor_lambda=float(lam["lambda_hat"][0]),
            final_lambda=float(np.nanmean(lam["lambda_hat"][-3:])),
            mean_overtrust=float(np.nanmean(np.array(bay["w"]) - lam["lambda_hat"])),
            rmse=rmse(crows))
        result["conditions"][name] = entry

    # comprehension summary
    comp_path = out / "comprehension.json"
    if comp_path.exists():
        comp = json.loads(comp_path.read_text())
        def ok(c):
            return ("A" in c.get("reputable_source", "")) and c.get("fresh_items_each_round")
        result["comprehension"] = {"n": len(comp),
                                   "frac_correct": float(np.mean([ok(c) for c in comp])) if comp else None,
                                   "raw": comp}

    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    _write_csvs(out, result)
    print(f"wrote {out}/analysis.json (+ CSVs)")
    return result


def _write_csvs(out: Path, result: dict):
    import csv
    # per-round trajectory
    with (out / "trajectory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "kind", "gap", "rep_strength", "info_level", "round",
                    "lambda_hat", "lambda_lo", "lambda_hi", "bayes_w", "oracle"])
        for name, c in result["conditions"].items():
            if "lambda_hat" not in c:
                continue
            for t in range(result["T"]):
                w.writerow([name, c["kind"], c["gap"], c["rep_strength"], c["info_level"],
                            t + 1, c["lambda_hat"][t], c["lambda_lo"][t], c["lambda_hi"][t],
                            c["bayes_w"][t], c["oracle"]])
    # condition summary
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "kind", "gap", "rep_strength", "info_level",
                    "flip_model", "flip_bayes", "lag", "anchor_lambda", "final_lambda",
                    "oracle", "mean_overtrust", "rmse"])
        for name, c in result["conditions"].items():
            w.writerow([name, c["kind"], c["gap"], c["rep_strength"], c["info_level"],
                        c.get("flip_model"), c.get("flip_bayes"), c.get("lag"),
                        c.get("anchor_lambda"), c.get("final_lambda"), c.get("oracle"),
                        c.get("mean_overtrust"), c.get("rmse")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    args = ap.parse_args()
    analyze(Path(args.out))


if __name__ == "__main__":
    main()
