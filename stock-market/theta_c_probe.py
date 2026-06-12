"""Probe the marker activations for the TRUE latents theta (fundamental) and c
(crowd target) of the target company, alongside the observable means p, s.

Uses the same ridge_r2_oos primitive as analyze.py. Reports per-layer R^2 for:
  theta  : true hidden fundamental of the target
  c      : crowd target of the target  (== theta when tau=0)
  p      : private-implied estimate  (mean private reading)  -- for reference
  s      : social-implied estimate  (mean social reading)    -- for reference
Also reports theta/c on the tau=1.5 subset only, where c != theta is meaningful.

    python theta_c_probe.py --out results/scoped
    python theta_c_probe.py --out results/scoped_market
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analyze import ridge_r2_oos


def probes(A, targets):
    res = {k: [] for k in targets}
    for L in range(A.shape[1]):
        X = A[:, L, :]
        for k, y in targets.items():
            res[k].append(ridge_r2_oos(X, y))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    args = ap.parse_args()
    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "trials.jsonl").read_text().splitlines()]

    A, th, c, p, s, tau = [], [], [], [], [], []
    for r in rows:
        A.append(np.load(r["acts_path"]))
        t = r["target"]
        th.append(r["theta"][t]); c.append(r["c"][t])
        p.append(r["private_implied"][t]); s.append(r["social_implied"][t])
        tau.append(r["tau"])
    A = np.stack(A); th = np.array(th); c = np.array(c)
    p = np.array(p); s = np.array(s); tau = np.array(tau)
    nL = A.shape[1]

    full = probes(A, {"theta": th, "c": c, "private_implied": p, "social_implied": s})
    m = tau > 0    # tau=1.5 subset: c and theta genuinely differ
    sub = probes(A[m], {"theta": th[m], "c": c[m]})

    print(f"{out.name}: {len(th)} trials ({m.sum()} with tau>0)\n")
    print("Per-layer probe R^2 (all trials):")
    print(f"{'L':>3} | {'theta':>7} | {'c':>7} | {'p(priv)':>8} | {'s(soc)':>8} || tau>0: {'theta':>7} | {'c':>7}")
    print("-" * 72)
    for L in range(nL):
        print(f"{L:>3} | {full['theta'][L]:>7.3f} | {full['c'][L]:>7.3f} | "
              f"{full['private_implied'][L]:>8.3f} | {full['social_implied'][L]:>8.3f} || "
              f"       {sub['theta'][L]:>7.3f} | {sub['c'][L]:>7.3f}")

    # csv
    with open(out / "theta_c_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "theta", "c", "private_implied", "social_implied",
                    "theta_tau1.5", "c_tau1.5"])
        for L in range(nL):
            w.writerow([L, full["theta"][L], full["c"][L], full["private_implied"][L],
                        full["social_implied"][L], sub["theta"][L], sub["c"][L]])

    # plot
    with PdfPages(out / "theta_c_probes.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, col, ls in [("theta", "#4c72b0", "-"), ("c", "#c44e52", "-"),
                           ("private_implied", "#4c72b0", "--"), ("social_implied", "#c44e52", "--")]:
            ax.plot(full[k], ls, color=col, marker="o", ms=3,
                    label={"theta": "θ true fundamental", "c": "c crowd target",
                           "private_implied": "p private mean (ref)",
                           "social_implied": "s social mean (ref)"}[k])
        ax.set(title=f"{out.name}: decoding TRUE latents (θ, c) vs observable means (p, s)",
               xlabel="layer", ylabel="out-of-sample R²")
        ax.axhline(0, color="grey", lw=.5); ax.legend(); ax.grid(lw=.3, alpha=.4)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sub["theta"], "-o", ms=3, color="#4c72b0", label="θ (τ=1.5 only)")
        ax.plot(sub["c"], "-o", ms=3, color="#c44e52", label="c (τ=1.5 only)")
        ax.set(title=f"{out.name}: θ vs c when they differ (τ=1.5 subset)",
               xlabel="layer", ylabel="out-of-sample R²")
        ax.axhline(0, color="grey", lw=.5); ax.legend(); ax.grid(lw=.3, alpha=.4)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"\nsaved {out}/theta_c_probe.csv and {out}/theta_c_probes.pdf")


if __name__ == "__main__":
    main()
