"""lambda-hat analysis (Steps 1-6): revealed social weight vs Bayesian optimum.

Reads <out>/lam_trials.jsonl (from lam_data.py) and produces:
  * <out>/lambda_hat.csv  - per-condition table
  * <out>/lambda_hat.png  - lambda_hat vs lambda* with identity line + CIs
and prints a written readout.

Estimators per condition (= sigma_p, sigma_s, tau; w does not enter prompt or lambda*):
  Step 1 paired (primary, causal): slope of y_model in each channel from
     single-channel shifts, beta = mean (y_shift - y_base)/Delta; lam_paired = b_s/(b_p+b_s)
  Step 2 regression (cross-check): y_model ~ b0 + b_p p + b_s s (OLS); lam_reg = b_s/(b_p+b_s)
  Step 3 lambda* = (sigma_s^2+tau^2)^-1 / (sigma_p^-2 + (sigma_s^2+tau^2)^-1)
  Step 4 bootstrap CIs (resample base-groups for paired; rows for regression)
  Step 6 sanity: regression R^2, slope constancy across |Delta|, scale, degeneracy.
"""
from __future__ import annotations
import argparse, json, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)


def lam_star(sigma_p, sigma_s, tau):
    prec_p = 1.0 / sigma_p ** 2
    prec_s = 1.0 / (sigma_s ** 2 + tau ** 2)
    return prec_s / (prec_p + prec_s)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def paired_betas(rows):
    """beta_p, beta_s = mean single-channel slope (y_shift - y_base)/Delta."""
    sl = {"social": [], "private": []}
    for r in rows:
        if r["arm"] == "shift" and r["delta"] != 0:
            sl[r["shifted"]].append((r["y_model"] - r["y_base"]) / r["delta"])
    bp = float(np.mean(sl["private"])) if sl["private"] else float("nan")
    bs = float(np.mean(sl["social"])) if sl["social"] else float("nan")
    return bp, bs


def ratio(bp, bs):
    denom = bp + bs
    return bs / denom if abs(denom) > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "lam_trials.jsonl").read_text().splitlines()]

    conds = defaultdict(list)
    for r in rows:
        conds[(r["sigma_p"], r["sigma_s"], r["tau"])].append(r)

    table = []
    for (sp, ss, tau), rs in sorted(conds.items()):
        lam = rs[0]["lmbda"]
        ls = lam_star(sp, ss, tau)

        # Step 1 paired
        bp, bs = paired_betas(rs)
        lam_paired = ratio(bp, bs)

        # Step 2 regression over all arms in condition
        p = np.array([r["p"] for r in rs]); s = np.array([r["s"] for r in rs])
        y = np.array([r["y_model"] for r in rs])
        X = np.vstack([np.ones_like(p), p, s]).T
        beta, r2 = ols(X, y)
        bp_r, bs_r = beta[1], beta[2]
        lam_reg = ratio(bp_r, bs_r)
        corr_ps = float(np.corrcoef(p, s)[0, 1])

        # Step 6 slope constancy across |Delta|
        bymag = defaultdict(lambda: {"social": [], "private": []})
        for r in rs:
            if r["arm"] == "shift":
                bymag[abs(r["delta"])][r["shifted"]].append((r["y_model"] - r["y_base"]) / r["delta"])
        lin = {m: {ch: float(np.mean(v[ch])) for ch in v} for m, v in sorted(bymag.items())}

        # Step 4 bootstrap (paired: resample base-groups; reg: resample rows)
        groups = defaultdict(list)
        for r in rs:
            groups[(r["seed"], r["target"])].append(r)
        gkeys = list(groups)
        lp_bs, lr_bs, dl_bs = [], [], []
        for _ in range(args.nboot):
            gk = [gkeys[i] for i in RNG.integers(0, len(gkeys), len(gkeys))]
            rb = [r for k in gk for r in groups[k]]
            bpb, bsb = paired_betas(rb)
            lp_bs.append(ratio(bpb, bsb)); dl_bs.append(ratio(bpb, bsb) - ls)
            pb = np.array([r["p"] for r in rb]); sb = np.array([r["s"] for r in rb])
            yb = np.array([r["y_model"] for r in rb])
            betab, _ = ols(np.vstack([np.ones_like(pb), pb, sb]).T, yb)
            lr_bs.append(ratio(betab[1], betab[2]))

        def ci(a):
            a = np.array(a); a = a[np.isfinite(a)]
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2
        lp_ci, lr_ci, dl_ci = ci(lp_bs), ci(lr_bs), ci(dl_bs)

        degenerate = abs(bp + bs) < 1e-6
        table.append(dict(lmbda=lam, sigma_p=sp, sigma_s=round(ss, 3), tau=tau, w=0.0,
                          lam_star=ls, lam_paired=lam_paired, lam_paired_lo=lp_ci[0], lam_paired_hi=lp_ci[1],
                          lam_reg=lam_reg, lam_reg_lo=lr_ci[0], lam_reg_hi=lr_ci[1],
                          dlam=lam_paired - ls, dlam_lo=dl_ci[0], dlam_hi=dl_ci[1],
                          beta_p=bp, beta_s=bs, beta_p_reg=float(bp_r), beta_s_reg=float(bs_r),
                          reg_R2=r2, corr_ps=corr_ps, degenerate=degenerate, linearity=lin))

    # ---- save CSV ----
    cols = ["lmbda", "sigma_p", "sigma_s", "tau", "w", "lam_star", "lam_paired",
            "lam_paired_lo", "lam_paired_hi", "lam_reg", "lam_reg_lo", "lam_reg_hi",
            "dlam", "dlam_lo", "dlam_hi", "beta_p", "beta_s", "reg_R2", "corr_ps", "degenerate"]
    with open(out / "lambda_hat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in table:
            w.writerow(t)

    # ---- Step 5 plot ----
    xs = np.array([t["lam_star"] for t in table]); ys = np.array([t["lam_paired"] for t in table])
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="rational (λ̂=λ*)")
    for t in table:
        ax.errorbar(t["lam_star"], t["lam_paired"],
                    yerr=[[t["lam_paired"] - t["lam_paired_lo"]], [t["lam_paired_hi"] - t["lam_paired"]]],
                    fmt="o", ms=7, capsize=3, color="#c44e52")
        ax.annotate(f"λ={t['lmbda']},τ={t['tau']}", (t["lam_star"], t["lam_paired"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    # trend fit
    if len(xs) >= 2:
        b1, b0 = np.polyfit(xs, ys, 1)
        xx = np.linspace(0, max(0.9, xs.max() + .05), 50)
        ax.plot(xx, b0 + b1 * xx, "-", color="#4c72b0", lw=1.6,
                label=f"trend  λ̂ = {b0:.2f} + {b1:.2f}·λ*")
    ax.set(xlim=(0, .9), ylim=(0, .9), xlabel="Bayesian optimal weight  λ*",
           ylabel="model revealed weight  λ̂ (paired)",
           title=f"{out.name}: revealed vs optimal social weight")
    ax.legend(loc="upper left", fontsize=9); ax.grid(lw=.3, alpha=.4)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out / "lambda_hat.png", dpi=140); plt.close(fig)

    # ---- written readout ----
    print(f"\n===== {out.name}: lambda-hat analysis =====")
    print(f"{'cond':>14} | {'λ*':>5} | {'λ̂_pair (95% CI)':>22} | {'λ̂_reg':>6} | "
          f"{'Δλ (CI)':>20} | {'β_p':>6} {'β_s':>6} | {'R²':>5} {'r(p,s)':>6}")
    for t in table:
        print(f"λ={t['lmbda']} τ={t['tau']:>3} | {t['lam_star']:>5.2f} | "
              f"{t['lam_paired']:>6.3f} [{t['lam_paired_lo']:>5.2f},{t['lam_paired_hi']:>5.2f}] | "
              f"{t['lam_reg']:>6.3f} | "
              f"{t['dlam']:>+5.2f} [{t['dlam_lo']:>+4.2f},{t['dlam_hi']:>+4.2f}] | "
              f"{t['beta_p']:>6.3f} {t['beta_s']:>6.3f} | {t['reg_R2']:>5.2f} {t['corr_ps']:>+6.2f}")
    if len(xs) >= 2:
        b1, b0 = np.polyfit(xs, ys, 1)
        print(f"\nStep 5 trend:  λ̂ = {b0:.3f} + {b1:.3f}·λ*   "
              f"(slope≈1 ⇒ tracks reliability; slope≈0 ⇒ ignores it)")
    over = sum(t["dlam"] > 0 for t in table); under = sum(t["dlam"] < 0 for t in table)
    print(f"over-weights social in {over}/{len(table)} conditions, under in {under}/{len(table)}")
    print("\nStep 6 — slope constancy across |Δ| (should be ~constant if integration is linear):")
    for t in table:
        lin = t["linearity"]
        ss_ = "  ".join(f"|Δ|={m}: s={lin[m]['social']:+.3f}/p={lin[m]['private']:+.3f}" for m in sorted(lin))
        flag = ""
        mags = sorted(lin)
        if len(mags) >= 2:
            for ch in ["social", "private"]:
                a, b = lin[mags[0]][ch], lin[mags[-1]][ch]
                if abs(a) > 1e-6 and abs((b - a) / a) > 0.30:
                    flag += f" [{ch} slope nonconstant]"
        print(f"  λ={t['lmbda']} τ={t['tau']}: {ss_}{flag}")
        if t["degenerate"]:
            print(f"    WARNING degenerate: β_p+β_s≈0, λ̂ undefined.")
    print(f"\nsaved {out}/lambda_hat.csv and {out}/lambda_hat.png")


if __name__ == "__main__":
    main()
