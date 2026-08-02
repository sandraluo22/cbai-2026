"""MULTI — analysis over the four games' transcripts (local, no GPU).

Reads every *_transcript.jsonl under RUN_DIR (default runs/multi), groups by the
`game` field of the meta record, and writes one PDF + one summary JSON per
transcript into an `analysis/` sibling directory.

Shared panels (every game — possible because records are standardized):
  1. belief quality: per-round mean KL(model belief || each logged exact reference)
     and mean belief mass on the truth vs the references' mass on the truth;
  2. calibration: belief top-mass vs correctness of the top pick.

Game-specific panels:
  corrupt : trust separation — mean P(source reliable) for corrupted vs honest
            sources by round (and by episode when PERSIST reputations can form);
            claim-vs-source dissociation scatter (P(claim true) vs P(source
            reliable) for the corrupted seat).
  priors  : consensus classification — final-round group mean belief compared to
            post_pool (Bayesian pooling), post_flat (prior washout), the linear
            average of the seats' own post_bayes, and each seat's post_bayes
            (dominance); reports the nearest by KL. Plus prior persistence: per-seat
            KL(belief || own post_bayes) over rounds.
  cfact   : factual vs counterfactual probe accuracy by round (does discussion of
            hypotheticals contaminate factual beliefs); agreement-vote accuracy.
  dynamic : staleness — belief mass on the CURRENT true location vs mass on the
            true location L hours ago (lag profile -> effective belief age);
            stale-probe accuracy ("where was it at hour 1"); trust separation when
            logged (UNIFIED).

Usage:  python runs/multi/multi_analysis.py            # everything under runs/multi
        python runs/multi/multi_analysis.py <transcript.jsonl> [...]
"""
from __future__ import annotations

import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import multi_core as MC

RUN_DIR = MC.env("RUN_DIR", "runs/multi")


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    meta = rows[0]
    el = [r for r in rows if r["type"] in ("elicit", "elicit_replay")]
    return meta, rows, el


def by_round(el):
    out = {}
    for r in el:
        out.setdefault(r["round"], []).append(r)
    return out


def mass_on(dist, key):
    return dist.get(key, 0.0)


def top(dist):
    return max(dist, key=dist.get)


# ---------------------------------------------------------------------------
# shared panels
# ---------------------------------------------------------------------------
def panel_quality(ax_kl, ax_mass, el):
    rounds = sorted(by_round(el))
    refs = sorted({k for r in el for k in r.get("refs", {})})
    for ref in refs:
        kl = [np.mean([MC.dict_kl(r["belief"], r["refs"][ref])
                       for r in by_round(el)[t] if ref in r.get("refs", {})])
              for t in rounds]
        ax_kl.plot(rounds, kl, "-o", ms=3, label=f"KL(belief || {ref})")
        m = [np.mean([mass_on(r["refs"][ref], r["truth"])
                      for r in by_round(el)[t] if ref in r.get("refs", {})])
             for t in rounds]
        ax_mass.plot(rounds, m, "--", lw=1, label=f"{ref} on truth")
    bm = [np.mean([mass_on(r["belief"], r["truth"]) for r in by_round(el)[t]])
          for t in rounds]
    ax_mass.plot(rounds, bm, "-o", ms=4, color="k", label="model belief on truth")
    ax_kl.set_xlabel("round"); ax_kl.set_ylabel("KL (nats)")
    ax_kl.set_title("belief vs exact references", fontsize=10)
    ax_kl.legend(fontsize=7); ax_kl.grid(alpha=.3)
    ax_mass.set_xlabel("round"); ax_mass.set_ylabel("probability mass on truth")
    ax_mass.set_ylim(0, 1); ax_mass.set_title("truth tracking", fontsize=10)
    ax_mass.legend(fontsize=7); ax_mass.grid(alpha=.3)


def panel_calibration(ax, el):
    conf = np.array([max(r["belief"].values()) for r in el])
    ok = np.array([float(top(r["belief"]) == r["truth"]) for r in el])
    bins = np.linspace(0, 1, 6)
    ctr, acc, cnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum():
            ctr.append((lo + hi) / 2); acc.append(ok[m].mean()); cnt.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.plot(ctr, acc, "-o")
    for x, y, c in zip(ctr, acc, cnt):
        ax.annotate(str(c), (x, y), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("belief top-mass"); ax.set_ylabel("top-pick accuracy")
    ax.set_title("calibration", fontsize=10); ax.grid(alpha=.3)


# ---------------------------------------------------------------------------
# game-specific panels
# ---------------------------------------------------------------------------
def panel_trust(ax, el, key="p_source_reliable", truthkey="truth_source_honest"):
    rounds = sorted(by_round(el))
    hon, cor = [], []
    for t in rounds:
        h, c = [], []
        for r in by_round(el)[t]:
            for who, p in r.get(key, {}).items():
                (h if r.get(truthkey, {}).get(who, True) else c).append(p)
        hon.append(np.mean(h) if h else np.nan)
        cor.append(np.mean(c) if c else np.nan)
    ax.plot(rounds, hon, "-o", ms=3, color="tab:green", label="honest sources")
    ax.plot(rounds, cor, "-o", ms=3, color="tab:red", label="corrupted source")
    ax.set_xlabel("round"); ax.set_ylabel("mean P(source reliable)")
    ax.set_ylim(0, 1); ax.set_title("trust separation", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    return {"trust_honest_final": float(hon[-1]) if hon else None,
            "trust_corrupt_final": float(cor[-1]) if cor else None}


def panel_corrupt(pdf, meta, el, summary):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    summary.update(panel_trust(ax[0], el))
    # claim-vs-source dissociation for the corrupted seat
    xs, ys = [], []
    for r in el:
        for who, honest in r.get("truth_source_honest", {}).items():
            if not honest and who in r.get("p_claim_true", {}):
                xs.append(r["p_source_reliable"][who]); ys.append(r["p_claim_true"][who])
    ax[1].scatter(xs, ys, s=12, alpha=.5)
    ax[1].plot([0, 1], [0, 1], ":", color="gray")
    ax[1].set_xlabel("P(source reliable)"); ax[1].set_ylabel("P(claim true)")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].set_title("corrupted seat: source vs claim", fontsize=10); ax[1].grid(alpha=.3)
    if xs:
        summary["claim_source_corr"] = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None
    # per-episode trust in the corrupted seat (reputation formation under PERSIST)
    eps = sorted({r["episode"] for r in el})
    cur = [np.mean([p for r in el if r["episode"] == e
                    for who, p in r.get("p_source_reliable", {}).items()
                    if not r.get("truth_source_honest", {}).get(who, True)] or [np.nan])
           for e in eps]
    ax[2].plot(eps, cur, "-o", ms=4, color="tab:red")
    ax[2].set_xlabel("episode"); ax[2].set_ylabel("P(corrupted source reliable)")
    ax[2].set_ylim(0, 1); ax[2].grid(alpha=.3)
    ax[2].set_title("reputation across episodes" + (" (PERSIST)" if meta.get("persist")
                                                    else " (identities reshuffled)"), fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def panel_priors(pdf, meta, el, summary):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # prior persistence: per-seat KL(belief || own post_bayes) by round
    rounds = sorted(by_round(el))
    for name in meta["agents"]:
        sel = [r for r in el if r["agent"] == name]
        if not sel:
            continue
        kl = [np.mean([MC.dict_kl(r["belief"], r["refs"]["post_bayes"])
                       for r in sel if r["round"] == t]) for t in rounds]
        ax[0].plot(rounds, kl, "-o", ms=3, label=name)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("KL(belief || own Bayes posterior)")
    ax[0].set_title("does discussion move agents off their own prior?", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # consensus classification at the final round, per episode
    last = max(rounds)
    labels = ["post_pool", "post_flat", "linear_avg", "dominance"]
    counts = {k: 0 for k in labels}
    for e in sorted({r["episode"] for r in el}):
        fin = [r for r in el if r["round"] == last and r["episode"] == e]
        if not fin:
            continue
        keys = list(fin[0]["belief"])
        group = {k: float(np.mean([r["belief"][k] for r in fin])) for k in keys}
        cands = {"post_pool": fin[0]["refs"]["post_pool"],
                 "post_flat": fin[0]["refs"]["post_flat"],
                 "linear_avg": {k: float(np.mean([r["refs"]["post_bayes"][k] for r in fin]))
                                for k in keys}}
        kls = {k: MC.dict_kl(group, v) for k, v in cands.items()}
        dom = min((MC.dict_kl(group, r["refs"]["post_bayes"]) for r in fin))
        kls["dominance"] = dom
        counts[min(kls, key=kls.get)] += 1
    ax[1].bar(range(len(labels)), [counts[k] for k in labels], color="tab:blue")
    ax[1].set_xticks(range(len(labels)), labels, fontsize=8)
    ax[1].set_ylabel("episodes"); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("nearest aggregation rule to the final group belief", fontsize=10)
    summary["consensus_counts"] = counts
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def panel_cfact(pdf, meta, el, summary):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    rounds = sorted(by_round(el))
    fact = [np.mean([(r["p_fact_on"] > 0.5) == bool(r["fact_truth"])
                     for r in by_round(el)[t]]) for t in rounds]
    cf = [np.mean([(r["p_cfact_on"] > 0.5) == bool(r["cfact_truth"])
                   for r in by_round(el)[t]]) for t in rounds]
    ax[0].plot(rounds, fact, "-o", ms=4, label="factual probe")
    ax[0].plot(rounds, cf, "-o", ms=4, label="counterfactual probe")
    ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("round"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("does hypothetical talk contaminate facts?", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    summary["fact_acc"] = float(np.mean(fact)); summary["cfact_acc"] = float(np.mean(cf))
    # agreement votes vs claim truth (from msg records)
    ax[1].axis("off")
    txt = ["agreement votes vs ground truth of the claim:"]
    msgs = summary.pop("_msgs", [])
    votes_ok, votes_all = 0, 0
    for m in msgs:
        for who, agree in m.get("votes", {}).items():
            votes_all += 1
            votes_ok += int(agree == bool(m["truth_of_claim"]))
    if votes_all:
        summary["vote_acc"] = votes_ok / votes_all
        txt.append(f"vote accuracy: {votes_ok}/{votes_all} = {votes_ok / votes_all:.2f}")
    ax[1].text(0.02, 0.9, "\n".join(txt), fontsize=10, va="top", family="monospace")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def panel_dynamic(pdf, meta, rows, el, summary):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # lag profile: belief mass on the true location L hours ago
    traj = {r["episode"]: r["traj"] for r in rows if r["type"] == "obs"}
    lags = list(range(0, 5))
    prof = []
    for L in lags:
        vals = [mass_on(r["belief"], traj[r["episode"]][max(0, r["round"] - L)])
                for r in el if r["round"] - L >= 0]
        prof.append(np.mean(vals) if vals else np.nan)
    ax[0].plot(lags, prof, "-o", ms=5)
    ax[0].set_xlabel("lag L (hours ago)"); ax[0].set_ylabel("belief mass on truth at t-L")
    ax[0].set_title("effective belief age", fontsize=10); ax[0].grid(alpha=.3)
    summary["lag_profile"] = [float(x) for x in prof]
    summary["belief_age"] = int(np.nanargmax(prof))
    # stale probe: was-at-hour-1 accuracy by round (past kept separate from present?)
    rounds = sorted(by_round(el))
    sacc = [np.mean([top(r["stale_belief"]) == r["stale_truth"] for r in by_round(el)[t]])
            for t in rounds]
    nacc = [np.mean([top(r["belief"]) == r["truth"] for r in by_round(el)[t]])
            for t in rounds]
    ax[1].plot(rounds, nacc, "-o", ms=3, label="now (top pick correct)")
    ax[1].plot(rounds, sacc, "-o", ms=3, label="hour-1 probe correct")
    ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("hour"); ax[1].set_ylabel("accuracy")
    ax[1].set_title("'is true' vs 'was true'", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    if any("p_source_reliable" in r for r in el):
        summary.update(panel_trust(ax[2], el))
    else:
        ax[2].axis("off")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
def analyze(path):
    meta, rows, el = load(path)
    game = meta.get("game", "?")
    outdir = os.path.join(os.path.dirname(path), "analysis")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0].replace("_transcript", "")
    summary = {"transcript": path, "game": game, "n_elicits": len(el)}
    if not el:
        print(f"[analysis] {path}: no elicit records, skipped")
        return
    with PdfPages(os.path.join(outdir, stem + ".pdf")) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        panel_quality(ax[0], ax[1], el)
        panel_calibration(ax[2], el)
        fig.suptitle(f"MULTI/{game} — {stem}", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        rounds = sorted(by_round(el))
        summary["final_mass_on_truth"] = float(
            np.mean([mass_on(r["belief"], r["truth"]) for r in by_round(el)[rounds[-1]]]))
        if game == "corrupt":
            panel_corrupt(pdf, meta, el, summary)
        elif game == "priors":
            panel_priors(pdf, meta, el, summary)
        elif game == "cfact":
            summary["_msgs"] = [r for r in rows if r["type"] == "msg"]
            panel_cfact(pdf, meta, el, summary)
        elif game == "dynamic":
            panel_dynamic(pdf, meta, rows, el, summary)
    with open(os.path.join(outdir, stem + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] {game}: {stem} -> {outdir}/  "
          f"final mass-on-truth {summary['final_mass_on_truth']:.2f}")


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(RUN_DIR, "**", "*_transcript.jsonl"), recursive=True))
    if not paths:
        print(f"[analysis] no transcripts under {RUN_DIR}")
    for p in paths:
        try:
            analyze(p)
        except Exception as e:
            print(f"[analysis] {p}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
