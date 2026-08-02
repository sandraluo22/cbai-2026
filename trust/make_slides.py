"""Build the deliverable: one landscape PDF slideshow (matplotlib PdfPages), one figure
per slide with a title + a 1-2 sentence "how to read it" caption. Also dumps individual
PNGs. Run after analyze.py.

    python make_slides.py --out results/scoped
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

C_MODEL, C_BAYES, C_ORACLE = "#dd8452", "#4c72b0", "#55a868"


def load(out: Path):
    res = json.loads((out / "analysis.json").read_text())
    cfg = json.loads(json.dumps(__import__("yaml").safe_load((out / "config.yaml").read_text())))
    return res, cfg


def _names(cfg):
    d = cfg["design"]
    a = d["anchor_gap"]
    return dict(anchor=a, main_rep=d["main_rep"], main_info=d["main_info"],
               canonical=f"gap{a}_{d['main_rep']}_{d['main_info']}",
               gaps=d["gaps"], rep_strengths=d["rep_strengths"],
               info_levels=d["info_levels"], robustness=d.get("robustness", []))


def _caption(fig, text):
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=9, style="italic",
             wrap=True)


def _traj(ax, c, label, color, show_ci=True):
    T = len(c["lambda_hat"]); x = np.arange(1, T + 1)
    lam = np.array(c["lambda_hat"], float)
    ax.plot(x, lam, "-o", ms=3, color=color, label=label)
    if show_ci:
        ax.fill_between(x, c["lambda_lo"], c["lambda_hi"], color=color, alpha=0.15)


# --------------------------------------------------------------------------- #
def s1_title(cfg, N):
    fig = plt.figure(figsize=(13.33, 7.5)); fig.clf()
    d = cfg["design"]; e = cfg["env"]
    sigA = d["anchor_gap"] * e["sigma_B"]
    lines = [
        ("Trust", 22, "bold"),
        ("In-context trust-learning under a reputation vs. accuracy conflict", 14, "normal"),
        ("", 8, "normal"),
        ("Question: when a source's STATED REPUTATION conflicts with its OBSERVED ACCURACY,", 12, "normal"),
        ("does a frozen-weight LLM learn to trust the accurate source — and how fast vs. a rational learner?", 12, "normal"),
        ("", 6, "normal"),
        (f"Model: {cfg['model']['model_name']} ({cfg['model'].get('backend', 'anthropic')})   |   "
         f"Source A = reputable but NOISIER,  Source B = unestablished but ACCURATE", 11, "normal"),
        (f"Accuracy gaps swept: sigma_A/sigma_B = {d['gaps']}  (sigma_B={e['sigma_B']})   |   "
         f"Reputation strengths: {d['rep_strengths']}", 11, "normal"),
        (f"M={e['M']} companies/round,  T={e['T']} rounds,  ~{d['games']} games/condition   |   "
         f"info-ladder: {d['info_levels']}", 11, "normal"),
        (f"{N} per-(round,company) observations   |   {date.today().isoformat()}", 10, "normal"),
    ]
    y = 0.82
    for txt, sz, w in lines:
        fig.text(0.5, y, txt, ha="center", fontsize=sz, fontweight=w)
        y -= 0.058 if sz >= 14 else 0.05
    _caption(fig, "Setup overview. Reputation is verbal track-record framing only; it never updates and is decoupled from reward.")
    return fig


def s2_method():
    fig, ax = plt.subplots(figsize=(13.33, 7.5)); ax.axis("off")
    ax.set_title("Method — per-round in-context loop", fontsize=15, fontweight="bold")
    steps = [
        "1.  Build round-t prompt:  persistent reputation header  +  RAW recap of past rounds (advisor record only)  +  this round's two source estimates",
        "2.  Model outputs a numeric estimate per company (structured output) — the estimate IS the recommendation",
        "3.  Reveal the true value for each company   ->   accuracy evidence accumulates against the stale reputation",
        "4.  Extend history; proceed to round t+1     (one API call per round, T calls per game)",
        "",
        "Reputation vs. accuracy CONFLICT:  the header says A is trustworthy, but B's revealed errors are smaller every round.",
        "",
        "Normative baseline (LEARNED precision):  a Bayesian that does NOT know the noise levels, starts from a reputation-derived",
        "prior favoring A, and updates each source's precision from realized errors:  pi_hat = (nu0 + n) / (nu0*sig0^2 + S).",
        "Trust on B:  w = pi_hat_B / (pi_hat_A + pi_hat_B)  ->  migrates A->B as evidence accrues  (oracle asymptote uses the TRUE sigmas).",
        "",
        "Model trust readout:  per round t, regress model_est ~ bA*a + bB*b across companies x games;  lambda_hat_t = bB/(bA+bB).",
    ]
    y = 0.86
    for s in steps:
        ax.text(0.04, y, s, fontsize=11, transform=ax.transAxes, family="monospace")
        y -= 0.066
    _caption(fig, "How one game runs and how the model's revealed trust (lambda_hat) and the rational trust (w) are computed.")
    return fig


def s3_headline(res, nm):
    c = res["conditions"][nm["canonical"]]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    T = len(c["lambda_hat"]); x = np.arange(1, T + 1)
    _traj(ax, c, "model  λ̂ₜ (revealed trust on accurate source)", C_MODEL)
    ax.plot(x, c["bayes_w"], "--s", ms=3, color=C_BAYES, label="learned-precision Bayesian  wₜ")
    ax.fill_between(x, c["bayes_lo"], c["bayes_hi"], color=C_BAYES, alpha=0.12)
    ax.axhline(c["oracle"], color=C_ORACLE, ls=":", label=f"oracle asymptote ({c['oracle']:.2f})")
    ax.axhline(0.5, color="gray", lw=.6, ls="-")
    ax.set(xlabel="round", ylabel="trust weight on the ACCURATE source (B)", ylim=(0, 1.02),
           title=f"Headline trust trajectory  (gap={nm['anchor']}, {nm['main_rep']} reputation, {nm['main_info']} info)")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    _caption(fig, "Both curves start anchored on the reputable-but-wrong source (low) and should climb toward the oracle. "
                  "Model below Bayesian = reputation-stickiness; shaded = 95% bootstrap CI.")
    return fig


def s_sources_trace(out: Path):
    """Embed the precomputed per-round source/truth/model trace (sources_trace.png),
    if present in the run directory. Returns None when the PNG hasn't been generated."""
    img_path = out / "sources_trace.png"
    if not img_path.exists():
        return None
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    ax.set_title("Per-round trace — sources, truth, and model estimate",
                 fontsize=14, fontweight="bold")
    _caption(fig, "Anchor condition: each round shows Source A (reputable, noisy), Source B (new, accurate), "
                  "the truth θ, and the model's estimate. Bottom panel: pooled |error| per source vs. the model.")
    return fig


def s_sources_values(out: Path):
    """Embed the precomputed raw-value trace (sources_values.png) — θ, Source A, Source B,
    and the model estimate per round for one game — if present in the run directory."""
    img_path = out / "sources_values.png"
    if not img_path.exists():
        return None
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    ax.set_title("Raw values — Source A vs. truth vs. Source B vs. model estimate",
                 fontsize=14, fontweight="bold")
    _caption(fig, "Single game (anchor condition): the model's estimate weaves between the two sources rather than "
                  "hugging either — the ~0.5 blend the regression reports. Sources sit in a narrow band vs. θ's round-to-round swings.")
    return fig


def s_weight_trace(out: Path):
    """Embed the implied-weight trace (weight_trace.png) — the linear, full-resolution
    view of the model's weight on B per round — if present in the run directory."""
    img_path = out / "weight_trace.png"
    if not img_path.exists():
        return None
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    ax.set_title("Implied weight on the accurate source — the uncompressed view",
                 fontsize=14, fontweight="bold")
    _caption(fig, "Weight (model−a)/(b−a) is linear in the weight, unlike |error|. The model jumps off the reputation "
                  "anchor (≈0) to an EQUAL blend (≈0.5) and plateaus there — short of the accuracy-justified 0.80.")
    return fig


def s_errors_trace(out: Path):
    """Embed the signed-error trace (errors_trace.png) — model−θ, A−θ, B−θ per round for
    one game — if present in the run directory."""
    img_path = out / "errors_trace.png"
    if not img_path.exists():
        return None
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    ax.set_title("Signed error — model−θ vs A−θ vs B−θ",
                 fontsize=14, fontweight="bold")
    _caption(fig, "Single game with θ subtracted out: the noisy source A−θ swings widest, the accurate B−θ hugs zero, "
                  "and the model−θ rides between them — the ≈0.5 blend, not a B-follower. (Pooled signed error is ≈0; all sources are unbiased.)")
    return fig


def s4_stickiness(res, nm):
    c = res["conditions"][nm["canonical"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    # left: model vs bayes with anchor + crossing annotations
    T = len(c["lambda_hat"]); x = np.arange(1, T + 1)
    ax1.plot(x, c["lambda_hat"], "-o", ms=3, color=C_MODEL, label="model λ̂ₜ")
    ax1.plot(x, c["bayes_w"], "--s", ms=3, color=C_BAYES, label="Bayesian wₜ")
    ax1.axhline(0.5, color="gray", lw=.6)
    for fr, col, lab in [(c["flip_model"], C_MODEL, "model flip"), (c["flip_bayes"], C_BAYES, "Bayes flip")]:
        if fr:
            ax1.axvline(fr, color=col, ls=":", alpha=.7)
    ax1.set(xlabel="round", ylabel="trust on B", ylim=(0, 1.02), title="Anchoring + crossing")
    ax1.legend(loc="lower right"); ax1.grid(alpha=.3)
    # right: bars for anchor lambda, lag, mean over-trust
    lag = c["lag"] if c["lag"] is not None else 0
    vals = [c["anchor_lambda"], c["mean_overtrust"], (lag / T)]
    labs = [f"start anchor λ̂₁\n({c['anchor_lambda']:.2f})",
            f"mean over-trust of A\n(w−λ̂ = {c['mean_overtrust']:+.2f})",
            f"flip lag / T\n({c['lag']} rounds behind)" if c["lag"] is not None else "flip lag\n(n/a)"]
    ax2.bar(range(3), vals, color=[C_MODEL, "#c44e52", "#8172b3"])
    ax2.set_xticks(range(3)); ax2.set_xticklabels(labs, fontsize=9)
    ax2.axhline(0, color="k", lw=.6); ax2.set(title="Reputation-stickiness metrics")
    ax2.grid(axis="y", alpha=.3)
    _caption(fig, "Stickiness = the model starts more anchored on A than warranted (left) and trails the rational A→B migration "
                  "(positive over-trust, rounds-behind on the right).")
    return fig


def s5_gap(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    gaps, fm, fb, orc = [], [], [], []
    for g in nm["gaps"]:
        c = res["conditions"].get(f"gap{g}_{nm['main_rep']}_{nm['main_info']}")
        if not c:
            continue
        gaps.append(g); orc.append(c["oracle"])
        fm.append(c["flip_model"] if c["flip_model"] else np.nan)
        fb.append(c["flip_bayes"] if c["flip_bayes"] else np.nan)
    x = np.arange(len(gaps)); width = .36
    ax.bar(x - width/2, fb, width, color=C_BAYES, label="Bayesian flip round")
    ax.bar(x + width/2, fm, width, color=C_MODEL, label="model flip round")
    for i, g in enumerate(gaps):
        ax.text(i, 0.3, f"oracle\n{orc[i]:.2f}", ha="center", fontsize=8, color=C_ORACLE)
    ax.set_xticks(x); ax.set_xticklabels([f"gap={g}\nσ_A={g*12:.0f}" for g in gaps])
    ax.set(ylabel="round trust first crosses 0.5  (NaN bar = never within T)",
           title="DIFFICULTY (accuracy-gap) sweep — where trust transitions, model vs. Bayes")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    _caption(fig, "THE key plot: as the accuracy gap widens (easier to tell sources apart) both flip sooner. "
                  "Taller model bars than Bayes = the model needs more evidence to overcome reputation.")
    return fig


def s6_repstrength(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    cmap = {"faint": "#8c8c8c", "moderate": C_MODEL, "strong": "#c44e52"}
    ref = None
    for rs in nm["rep_strengths"]:
        name = (nm["canonical"] if rs == nm["main_rep"]
                else f"gap{nm['anchor']}_{rs}_{nm['main_info']}")
        c = res["conditions"].get(name)
        if not c:
            continue
        ref = c
        x = np.arange(1, len(c["lambda_hat"]) + 1)
        ax.plot(x, c["lambda_hat"], "-o", ms=3, color=cmap.get(rs, "k"), label=f"{rs} reputation")
    if ref:
        ax.plot(np.arange(1, len(ref["bayes_w"]) + 1), ref["bayes_w"], "--", color=C_BAYES, label="Bayesian wₜ")
        ax.axhline(ref["oracle"], color=C_ORACLE, ls=":")
    ax.axhline(0.5, color="gray", lw=.6)
    ax.set(xlabel="round", ylabel="trust on B", ylim=(0, 1.02),
           title=f"Reputation-STRENGTH effect (gap={nm['anchor']}): how persistent authority slows the A→B flip")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    _caption(fig, "More persistently asserted reputation (faint→moderate→strong) should push the model's trajectory lower / later.")
    return fig


def s7_asymptote(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    gaps, fin, orc = [], [], []
    for g in nm["gaps"]:
        c = res["conditions"].get(f"gap{g}_{nm['main_rep']}_{nm['main_info']}")
        if not c:
            continue
        gaps.append(g); fin.append(c["final_lambda"]); orc.append(c["oracle"])
    x = np.arange(len(gaps)); width = .36
    ax.bar(x - width/2, orc, width, color=C_ORACLE, label="accuracy-justified (oracle)")
    ax.bar(x + width/2, fin, width, color=C_MODEL, label="model final λ̂ (last 3 rounds)")
    ax.set_xticks(x); ax.set_xticklabels([f"gap={g}" for g in gaps])
    ax.set(ylabel="trust on B", ylim=(0, 1.02),
           title="Asymptote — final model trust vs. the accuracy-justified weight")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    _caption(fig, "Gap between green (justified) and orange (model) at the end = residual deference to reputation that never resolves.")
    return fig


def s8_infoladder(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    cmap = {"lean": C_MODEL, "self_history": "#8172b3", "summary": "#55a868", "full": "#c44e52"}
    ref = None
    for il in nm["info_levels"]:
        name = (nm["canonical"] if il == nm["main_info"]
                else f"gap{nm['anchor']}_{nm['main_rep']}_{il}")
        c = res["conditions"].get(name)
        if not c:
            continue
        ref = c
        x = np.arange(1, len(c["lambda_hat"]) + 1)
        ax.plot(x, c["lambda_hat"], "-o", ms=3, color=cmap.get(il, "k"), label=il)
    if ref:
        ax.plot(np.arange(1, len(ref["bayes_w"]) + 1), ref["bayes_w"], "--", color=C_BAYES, label="Bayesian wₜ")
        ax.axhline(ref["oracle"], color=C_ORACLE, ls=":")
    ax.axhline(0.5, color="gray", lw=.6)
    ax.set(xlabel="round", ylabel="trust on B", ylim=(0, 1.02),
           title=f"Information-ladder (gap={nm['anchor']}): lean / +self-history / +summary / full")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    _caption(fig, "Localizes the bottleneck: if +summary (handed reliability) beats lean, it's an INFERENCE limit; "
                  "if even full fails, it's DEFERENCE; if +self-history shifts it, anchoring.")
    return fig


def s9_robustness(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    c0 = res["conditions"][nm["canonical"]]
    x = np.arange(1, len(c0["lambda_hat"]) + 1)
    ax.plot(x, c0["lambda_hat"], "-o", ms=3, color="k", label="canonical")
    variants = {"swap": ("gap%s_robust_swaplabels", C_MODEL, "swap labels"),
                "order": ("gap%s_robust_order", "#55a868", "swap order"),
                "paraphrase": ("gap%s_robust_paraphrase", "#8172b3", "paraphrase")}
    for rv in nm["robustness"]:
        pat, col, lab = variants[rv]
        c = res["conditions"].get(pat % nm["anchor"])
        if not c:
            continue
        ax.plot(np.arange(1, len(c["lambda_hat"]) + 1), c["lambda_hat"], "-^", ms=3,
                color=col, alpha=.8, label=lab)
    ax.plot(x, c0["bayes_w"], "--", color=C_BAYES, label="Bayesian wₜ")
    ax.axhline(0.5, color="gray", lw=.6)
    ax.set(xlabel="round", ylabel="trust on B (accurate source, env terms)", ylim=(0, 1.02),
           title=f"Surface robustness (gap={nm['anchor']}): label-swap / order / paraphrase")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    _caption(fig, "Trajectories should overlap — the A→B migration shouldn't depend on which letter is reputable, the listing order, or wording.")
    return fig


def s10_table(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5)); ax.axis("off")
    ax.set_title("Summary — flip round (model vs Bayes), lag, final trust, verdict", fontsize=14, fontweight="bold")
    head = ["condition", "gap", "rep", "info", "flip\nmodel", "flip\nBayes", "lag", "final\nλ̂", "oracle", "verdict"]
    rows = []
    for name, c in res["conditions"].items():
        if "lambda_hat" not in c:
            if c.get("info_level") == "no_advisor":
                rows.append([name, c["gap"], c["rep_strength"], c["info_level"], "-", "-", "-",
                             "-", "-", f"RMSE={c['rmse']:.0f} (necessity)"])
            continue
        lag = c["lag"]
        verdict = ("learns→B" if (c["flip_model"] and c["flip_model"] <= (c["flip_bayes"] or 0) + 2)
                   else "sticky/slow" if c["flip_model"] else "no flip in T")
        rows.append([name, c["gap"], c["rep_strength"], c["info_level"],
                     c["flip_model"] or "—", c["flip_bayes"] or "—",
                     f"{lag:+d}" if lag is not None else "—",
                     f"{c['final_lambda']:.2f}", f"{c['oracle']:.2f}", verdict])
    tbl = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
    _caption(fig, "lag = model flip round minus Bayesian flip round (positive = behind). final λ̂ vs oracle shows residual deference.")
    return fig


def s11_takeaways(res, nm):
    fig, ax = plt.subplots(figsize=(13.33, 7.5)); ax.axis("off")
    ax.set_title("Takeaways", fontsize=16, fontweight="bold")
    c = res["conditions"][nm["canonical"]]
    na = next((v for v in res["conditions"].values() if v.get("info_level") == "no_advisor"), {})
    comp = res.get("comprehension", {})
    flips = [(v["gap"], v["flip_model"], v["flip_bayes"]) for v in res["conditions"].values()
             if v.get("kind") == "main"]
    learns = c["flip_model"] is not None
    bullets = [
        f"• Does it learn to trust the accurate source?  {'YES' if learns else 'NO'} — at gap={nm['anchor']} the model's revealed "
        f"trust crosses 0.5 at round {c['flip_model']} (Bayes: {c['flip_bayes']}).",
        f"• Reputation-stickiness: starts anchored at λ̂₁={c['anchor_lambda']:.2f} (prior favors A) and over-trusts A by "
        f"{c['mean_overtrust']:+.2f} on average vs. the rational curve; flip lag = {c['lag']} rounds.",
        f"• Where the transition sits vs. Bayes across gaps (model vs Bayes flip round): "
        f"{', '.join(f'gap{g}:{fm}/{fb}' for g, fm, fb in sorted(flips))}.",
        f"• Asymptote: final λ̂={c['final_lambda']:.2f} vs accuracy-justified {c['oracle']:.2f} — "
        f"{'residual deference remains' if c['final_lambda'] < c['oracle'] - 0.05 else 'roughly reaches the justified weight'}.",
        f"• Inference vs. deference: compare lean vs. +summary on S8 — if handing the model running accuracy helps, the bottleneck "
        f"is inference; if not, it's deference.",
        f"• Sanity: no-advisor RMSE={na.get('rmse', float('nan')):.0f} (advisors must add info); "
        f"comprehension correct in {comp.get('frac_correct', float('nan')):.0%} of probes.",
        "• Caveats: scoped run (~20 games/condition → wider CIs than the 1000-game target); thinking disabled; "
        "single model; synthetic Gaussian world.",
    ]
    y = 0.84
    for b in bullets:
        ax.text(0.04, y, b, fontsize=10.5, transform=ax.transAxes, wrap=True, va="top")
        y -= 0.108
    _caption(fig, "Headline conclusions; see the summary table (S10) and CSVs for the full per-condition numbers.")
    return fig


# --------------------------------------------------------------------------- #
def build(out: Path):
    res, cfg = load(out)
    nm = _names(cfg)
    N = sum(1 for _ in (out / "rounds.jsonl").read_text().splitlines())
    figs = [
        ("01_title", s1_title(cfg, N)),
        ("02_method", s2_method()),
        ("03_headline_trajectory", s3_headline(res, nm)),
        ("03b_sources_trace", s_sources_trace(out)),
        ("03c_sources_values", s_sources_values(out)),
        ("03d_weight_trace", s_weight_trace(out)),
        ("03e_errors_trace", s_errors_trace(out)),
        ("04_stickiness", s4_stickiness(res, nm)),
        ("05_gap_sweep", s5_gap(res, nm)),
        ("06_reputation_strength", s6_repstrength(res, nm)),
        ("07_asymptote", s7_asymptote(res, nm)),
        ("08_information_ladder", s8_infoladder(res, nm)),
        ("09_robustness", s9_robustness(res, nm)),
        ("10_summary_table", s10_table(res, nm)),
        ("11_takeaways", s11_takeaways(res, nm)),
    ]
    figs = [(name, fig) for name, fig in figs if fig is not None]  # drop absent optional slides
    png_dir = out / "slides_png"; png_dir.mkdir(exist_ok=True)
    pdf_path = out / "slideshow.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, fig in figs:
            fig.subplots_adjust(bottom=0.12, top=0.9)
            fig.savefig(png_dir / f"{name}.png", dpi=130)
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path}  (+ {len(figs)} PNGs in {png_dir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
