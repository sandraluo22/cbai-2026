"""Control 2 -- text-only baselines. A's probes must beat these for
"A represents B" to mean more than "the transcript reveals B".

Two baselines, both predicting alpha from the transcript prefix up to turn t
(held out by episode, same split protocol as the probes):

  behavioral : ridge on hand-built features of B's observable play
               (mean/last/min/max of B's demands, accept rates, reject rate)
               -- an explicit fitted behavioral model of B's offers.
  tfidf      : ridge on TF-IDF character+word n-grams of the canonical
               transcript rendering -- "whatever the surface text reveals".

Output: baselines_r2.npz (r2_behavioral, r2_tfidf: [n_rounds]) and
baselines_curves.pdf/png overlaying both against the best probe layer.

Run:  python src/baselines.py --preset default
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config              # noqa: E402
from episodes import transcripts_dir               # noqa: E402
from game import render_transcript                 # noqa: E402
from probes import split_by_episode                # noqa: E402


def load_transcripts(cfg: Config):
    tdir = transcripts_dir(cfg)
    files = sorted(f for f in os.listdir(tdir) if f.startswith("ep_"))
    recs = []
    for fname in files:
        with open(os.path.join(tdir, fname)) as f:
            recs.append(json.load(f))
    if not recs:
        raise SystemExit(f"no transcripts in {tdir}; run episodes.py first")
    return recs


def behavioral_features(rec: dict, upto: int, pie: int) -> np.ndarray:
    turns = rec["turns"][:upto]
    demands = np.array([t["b_offer_keep"] for t in turns], dtype=np.float64)
    n = len(turns)
    counters = [t for t in turns if t["a_action"] == "counter"]
    rejected = [t for t in counters if t["b_verdict"] == "reject"]
    return np.array([
        demands.mean() / pie,
        demands[-1] / pie,
        demands.min() / pie,
        demands.max() / pie,
        demands.std() / pie,
        len(counters) / n,                                   # A countered
        (len(rejected) / len(counters)) if counters else 0.0,  # B rejected
        np.mean([t["b_points"] for t in turns]) / pie,       # B's realized take
    ])


def r2_by_turn_behavioral(recs, cfg: Config) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    y = np.array([r["alpha"] for r in recs])
    tr, te = split_by_episode(len(recs), cfg.test_frac, cfg.seed)
    out = np.zeros(cfg.n_rounds)
    for t in range(1, cfg.n_rounds + 1):
        X = np.stack([behavioral_features(r, t, cfg.pie) for r in recs])
        reg = RidgeCV(alphas=cfg.baseline_ridge_alphas).fit(X[tr], y[tr])
        out[t - 1] = reg.score(X[te], y[te])
    return out


def r2_by_turn_tfidf(recs, cfg: Config) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    y = np.array([r["alpha"] for r in recs])
    tr, te = split_by_episode(len(recs), cfg.test_frac, cfg.seed)
    out = np.zeros(cfg.n_rounds)
    for t in range(1, cfg.n_rounds + 1):
        texts = [render_transcript(r["turns"], t, cfg.pie) for r in recs]
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              max_features=20000, sublinear_tf=True)
        X = vec.fit_transform([texts[i] for i in tr]).toarray()
        X_te = vec.transform([texts[i] for i in te]).toarray()
        reg = RidgeCV(alphas=cfg.baseline_ridge_alphas).fit(X, y[tr])
        out[t - 1] = reg.score(X_te, y[te])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_dir = cfg.run_dir()

    recs = load_transcripts(cfg)
    print(f"{len(recs)} transcripts")
    r2_beh = r2_by_turn_behavioral(recs, cfg)
    r2_tfi = r2_by_turn_tfidf(recs, cfg)
    np.savez(os.path.join(run_dir, "baselines_r2.npz"),
             r2_behavioral=r2_beh, r2_tfidf=r2_tfi)
    print("behavioral R2 by turn:", " ".join(f"{v:+.2f}" for v in r2_beh))
    print("tf-idf     R2 by turn:", " ".join(f"{v:+.2f}" for v in r2_tfi))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    turns = np.arange(1, cfg.n_rounds + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(turns, r2_beh, marker="o", label="behavioral model of B's offers")
    ax.plot(turns, r2_tfi, marker="s", label="TF-IDF transcript text")
    p = os.path.join(run_dir, "probe_r2.npz")
    if os.path.exists(p):
        r2 = np.load(p)["r2"]
        best_l = int(np.argmax(r2[:, -1]))
        ax.plot(turns, r2[best_l], marker="^", lw=2.5,
                label=f"A's residual probe (layer {best_l})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("turn (round)")
    ax.set_ylabel("held-out R$^2$")
    ax.set_title(f"probes vs text-only baselines ({cfg.name})")
    ax.legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"baselines_curves.{ext}"),
                    bbox_inches="tight", dpi=150)
    print(f"wrote baselines_r2.npz and baselines_curves -> {run_dir}")


if __name__ == "__main__":
    main()
