"""Extra descriptive plots for a completed run (default results/scoped).

These summarize the BEHAVIORAL side of the sweep (the model's single-shot
investment choices) plus an illustrative view of the evidence the model saw
across the T reading rounds. Run after run_experiment.py has produced
trials.jsonl (+ analysis.json from analyze.py for the probe panel).

    python make_plots.py --out results/scoped
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(out: Path):
    rows = [json.loads(l) for l in (out / "trials.jsonl").read_text().splitlines()]
    analysis = {}
    if (out / "analysis.json").exists():
        analysis = json.loads((out / "analysis.json").read_text())
    return rows, analysis


# --------------------------------------------------------------------------- #
# 1. Revealed vs rational social weight, per (lambda, w, tau)                  #
# --------------------------------------------------------------------------- #
def plot_revealed_weight(analysis, out_dir: Path):
    rw = analysis.get("revealed_weight", {})
    if not rw:
        return
    items = sorted(rw.values(), key=lambda d: (d["lambda"], d["tau"], d["w"]))
    labels = [f"λ={d['lambda']}\nw={d['w']}\nτ={d['tau']}" for d in items]
    lam_hat = [d["lambda_hat"] for d in items]
    rat = [d["rational_eff_weight"] for d in items]
    x = np.arange(len(items)); width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - width / 2, rat, width, label="rational (optimal) social weight", color="#4c72b0")
    ax.bar(x + width / 2, lam_hat, width, label="model revealed weight  λ̂", color="#dd8452")
    ax.set(title="What the optimal decision SHOULD weight on social vs. what the model DID",
           ylabel="effective weight on social evidence", ylim=(0, 1.05))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(); ax.grid(axis="y", lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(out_dir / "revealed_vs_rational_weight.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Counterfactual causal effect on BEHAVIOR: does shifting social move the    #
#    choice toward the target? (low arm = social unshifted, high = shifted)     #
# --------------------------------------------------------------------------- #
def plot_counterfactual_effect(rows, out_dir: Path):
    pairs = defaultdict(dict)
    for r in rows:
        if r["mode"] == "counterfactual" and r["pair_id"]:
            # NOTE: pair_id alone collides across cells (it omits lambda/w/tau),
            # so key on the full config cell to keep pairs distinct.
            key = (r["pair_id"], r["lmbda"], r["w"], r["tau"])
            pairs[key][r["arm"]] = r
    # group by (lambda, sign of delta)
    buckets = defaultdict(lambda: {"chose_tgt_low": 0, "chose_tgt_high": 0, "flip": 0, "n": 0})
    for arms in pairs.values():
        if "low" not in arms or "high" not in arms:
            continue
        lo, hi = arms["low"], arms["high"]
        tgt = hi["target"]
        sign = "+Δ (social ↑)" if hi["delta"] > 0 else "−Δ (social ↓)"
        key = (hi["lmbda"], sign)
        b = buckets[key]
        b["n"] += 1
        b["chose_tgt_low"] += int(lo["parsed"]["company"] == tgt)
        b["chose_tgt_high"] += int(hi["parsed"]["company"] == tgt)
        b["flip"] += int(lo["parsed"]["company"] != hi["parsed"]["company"])
    keys = sorted(buckets)
    labels = [f"λ={k[0]}\n{k[1]}" for k in keys]
    p_low = [buckets[k]["chose_tgt_low"] / buckets[k]["n"] for k in keys]
    p_high = [buckets[k]["chose_tgt_high"] / buckets[k]["n"] for k in keys]
    flip = [buckets[k]["flip"] / buckets[k]["n"] for k in keys]
    x = np.arange(len(keys)); width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width, p_low, width, label="P(chose target) | social unshifted", color="#55a868")
    ax.bar(x, p_high, width, label="P(chose target) | social shifted", color="#c44e52")
    ax.bar(x + width, flip, width, label="P(choice changed)", color="#8172b3")
    ax.set(title="Counterfactual: causal effect of shifting the target's SOCIAL evidence on the choice",
           ylabel="rate over pairs", ylim=(0, 1.05))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", lw=.3, alpha=.5)
    total_flips = sum(buckets[k]["flip"] for k in keys)
    total_pairs = sum(buckets[k]["n"] for k in keys)
    ax.text(0.5, 0.82, f"behavioral choice changed in {total_flips} / {total_pairs} pairs\n"
            f"(shifting social by ±4 — larger than θ-scale=3 — never flipped the one-shot choice)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0c060"))
    fig.tight_layout(); fig.savefig(out_dir / "counterfactual_effect.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Did the model pick the rational (reward-maximizing) company? by condition #
# --------------------------------------------------------------------------- #
def plot_accuracy(rows, out_dir: Path):
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["lmbda"], r["tau"], r["w"])
        buckets[key][1] += 1
        if r["parsed"]["company"] == r["rational_action"]:
            buckets[key][0] += 1
    keys = sorted(buckets)
    labels = [f"λ={k[0]}\nτ={k[1]}\nw={k[2]}" for k in keys]
    acc = [buckets[k][0] / buckets[k][1] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(range(len(keys)), acc, color="#4c72b0")
    overall = sum(b[0] for b in buckets.values()) / sum(b[1] for b in buckets.values())
    ax.axhline(overall, color="k", ls="--", lw=1, label=f"overall = {overall:.2f}")
    ax.axhline(0.25, color="grey", ls=":", lw=1, label="chance (1/4)")
    ax.set(title="Agreement with the rational (reward-maximizing) choice, by condition",
           ylabel="fraction matching rational action", ylim=(0, 1.05))
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(); ax.grid(axis="y", lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(out_dir / "rational_agreement.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Per-layer probe R^2 (re-plot with peak annotations)                       #
# --------------------------------------------------------------------------- #
def plot_layer_probes(analysis, out_dir: Path):
    probes = analysis.get("layer_probes", {})
    if not probes:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = {"private_implied": "#4c72b0", "social_implied": "#dd8452", "E_reward": "#55a868"}
    for k, ys in probes.items():
        ys = np.array(ys)
        ax.plot(ys, "-o", ms=3, label=k, color=colors.get(k))
        peak = int(np.argmax(ys))
        ax.scatter([peak], [ys[peak]], s=60, facecolors="none", edgecolors=colors.get(k))
    ax.set(title="Per-layer linear-probe R²  (where each signal becomes decodable)",
           xlabel="residual-stream layer  (0 = embedding … 32 = final)",
           ylabel="out-of-sample R²")
    ax.axhline(0, color="grey", lw=.5); ax.legend(); ax.grid(lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(out_dir / "layer_probes_annotated.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. The evidence the model actually saw, across the T reading rounds          #
#    (closest thing to "over timesteps": readings r1..rT per company)          #
# --------------------------------------------------------------------------- #
def plot_evidence_rounds(rows, out_dir: Path):
    # pick a clean neutral trial to illustrate
    tr = next((r for r in rows if r["mode"] == "neutral"), rows[0])
    priv = np.array(tr["private"]); soc = np.array(tr["social"])
    n, T = priv.shape
    chosen = tr["parsed"]["company"]; rational = tr["rational_action"]
    rounds = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.2), sharey=True)
    for i in range(n):
        ax = axes[i]
        ax.plot(rounds, priv[i], "-o", color="#4c72b0", label="PERSONAL (private)")
        ax.plot(rounds, soc[i], "-s", color="#dd8452", label="EXTERNAL (social)")
        ax.axhline(tr["theta"][i], color="#4c72b0", ls=":", lw=1, alpha=.7)
        ax.axhline(tr["c"][i], color="#dd8452", ls=":", lw=1, alpha=.7)
        comp = chr(ord("A") + i)
        tag = ""
        if i == chosen:
            tag += "  ← MODEL CHOSE"
        if i == rational:
            tag += "  (rational)"
        ax.set_title(f"Company {comp}{tag}", fontsize=9,
                     color="#c44e52" if i == chosen else "k")
        ax.set_xlabel("reading round"); ax.grid(lw=.3, alpha=.5)
        if i == 0:
            ax.set_ylabel("reading value"); ax.legend(fontsize=7)
    fig.suptitle(f"Evidence shown across the {T} rounds  "
                 f"(λ={tr['lmbda']}, w={tr['w']}, τ={tr['tau']}, seed={tr['seed']}; "
                 f"dotted = true θ / crowd c)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "evidence_over_rounds.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Does the chosen company track its private or its social estimate?         #
# --------------------------------------------------------------------------- #
def plot_choice_drivers(rows, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, lam in zip(axes, sorted({r["lmbda"] for r in rows})):
        sub = [r for r in rows if r["lmbda"] == lam and r["parsed"]["company"] is not None]
        # rank of chosen company by private vs social implied estimate (1 = best)
        for r in sub:
            ch = r["parsed"]["company"]
            pi = np.array(r["private_implied"]); si = np.array(r["social_implied"])
            # higher implied = more attractive; plot chosen company's two estimates
            ax.scatter(pi[ch], si[ch], s=18, alpha=.5, color="#4c72b0")
        lim = [-8, 8]
        ax.plot(lim, lim, color="grey", ls="--", lw=.8)
        ax.set(title=f"λ={lam}", xlabel="private-implied value of CHOSEN company",
               xlim=lim, ylim=lim)
        ax.grid(lw=.3, alpha=.5)
    axes[0].set_ylabel("social-implied value of chosen company")
    fig.suptitle("Chosen company's private vs. social estimate  "
                 "(points above the diagonal = social looked better than private)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "choice_drivers.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Corrected causal results (causal.py): patching + steering at the decision  #
#    position, read out as a change in the target company's logit.              #
# --------------------------------------------------------------------------- #
def plot_causal(out: Path, out_dir: Path):
    f = out / "causal.jsonl"
    if not f.exists():
        return
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    lambdas = sorted({r["lmbda"] for r in rows})
    colors = {lambdas[0]: "#4c72b0", lambdas[-1]: "#c44e52"}

    # (a) PATCHING depth curve: mean patch dlogit(target) by layer, per lambda,
    #     with the natural counterfactual effect (donor-recv base logit) as ceiling.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for lam in lambdas:
        sub = [r for r in rows if r["lmbda"] == lam]
        layers = sorted({r["layer"] for r in sub})
        by = {L: [] for L in layers}
        for r in sub:
            by[r["layer"]].append(r["patch_dlogit_target"])
        mean = [float(np.mean(by[L])) for L in layers]
        ax.plot(layers, mean, "-o", ms=3, color=colors[lam], label=f"λ={lam}  patch Δlogit(target)")
        ceil = float(np.mean([r["base_donor_target_logit"] - r["base_recv_target_logit"] for r in sub]))
        ax.axhline(ceil, color=colors[lam], ls="--", lw=1, alpha=.7,
                   label=f"λ={lam}  full-counterfactual ceiling")
    ax.axhline(0, color="grey", lw=.5)
    ax.set(title="Patching the decision-position residual (donor→receiver): causal effect by layer",
           xlabel="layer patched", ylabel="Δ logit of target company")
    ax.legend(fontsize=8); ax.grid(lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(out_dir / "causal_patching_by_layer.png", dpi=140); plt.close(fig)

    # (b) STEERING dose-response: mean steer dlogit(target) vs alpha, averaged over
    #     ALL layers (robust), per lambda. Shows saturation then off-distribution collapse.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for lam in lambdas:
        sub = [r for r in rows if r["lmbda"] == lam]
        bylα = defaultdict(list)
        for r in sub:
            for s in r["steer"]:
                bylα[s["alpha"]].append(s["dlogit_target"])
        alphas = sorted(bylα)
        mean = [float(np.mean(bylα[a])) for a in alphas]
        ax.plot(alphas, mean, "-o", ms=5, color=colors[lam], label=f"λ={lam}  (mean over all layers)")
    ax.axhline(0, color="grey", lw=.5)
    ax.set(title="Steering the decision position by α·(more-social − less-social): dose-response",
           xlabel="steering coefficient α", ylabel="Δ logit of target company")
    ax.legend(fontsize=9); ax.grid(lw=.3, alpha=.5)
    ax.annotate("graded effect saturates ~α2–4,\nthen collapses off-distribution",
                xy=(0.5, 0.04), xycoords="axes fraction", fontsize=8, color="#555")
    fig.tight_layout(); fig.savefig(out_dir / "causal_steering_dose.png", dpi=140); plt.close(fig)

    # (c) argmax-flip rate from steering vs alpha (does it ever flip the choice?)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for lam in lambdas:
        sub = [r for r in rows if r["lmbda"] == lam]
        bylα = defaultdict(lambda: [0, 0])
        for r in sub:
            for s in r["steer"]:
                bylα[s["alpha"]][1] += 1
                bylα[s["alpha"]][0] += int(s["argmax"] == r["target"])
        alphas = sorted(bylα)
        rate = [bylα[a][0] / bylα[a][1] for a in alphas]
        ax.plot(alphas, rate, "-o", ms=4, color=colors[lam], label=f"λ={lam}")
    ax.set(title="Does steering ever make the model PICK the target? (argmax = target)",
           xlabel="steering coefficient α", ylabel="P(argmax == target), over all layers·pairs")
    ax.legend(); ax.grid(lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(out_dir / "causal_steering_flip.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped")
    args = ap.parse_args()
    out = Path(args.out)
    out_dir = out / "plots"; out_dir.mkdir(parents=True, exist_ok=True)
    rows, analysis = load(out)
    plot_revealed_weight(analysis, out_dir)
    plot_counterfactual_effect(rows, out_dir)
    plot_accuracy(rows, out_dir)
    plot_layer_probes(analysis, out_dir)
    plot_evidence_rounds(rows, out_dir)
    plot_choice_drivers(rows, out_dir)
    plot_causal(out, out_dir)
    print(f"wrote plots -> {out_dir}/")
    for p in sorted(out_dir.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
