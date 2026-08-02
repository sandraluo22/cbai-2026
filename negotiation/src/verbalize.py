"""Control 3 -- probe vs verbalization: does A know more about B than it can
say? The verbalized guesses were collected DURING episode generation (game.py
forks A's context at cfg.verbalize_rounds and asks for a 0-100 greediness
estimate; the fork never pollutes the main context). Here we score them
against alpha and against the probes at the same rounds.

For comparability everything is reported as held-out R^2 of a 1-D linear
recalibration (guess -> alpha), computed on the SAME test episodes as the
probes. Raw Pearson r is printed too.

Output: verbalize_r2.npz + verbalize_curves.pdf/png.

Run:  python src/verbalize.py --preset default
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config              # noqa: E402
from baselines import load_transcripts     # noqa: E402
from probes import split_by_episode        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_dir = cfg.run_dir()

    recs = load_transcripts(cfg)
    rounds = sorted(r for r in cfg.verbalize_rounds if r > 0)
    y = np.array([r["alpha"] for r in recs])
    tr, te = split_by_episode(len(recs), cfg.test_frac, cfg.seed)

    r2s, rs, n_ok = [], [], []
    for rnd in rounds:
        g = np.full(len(recs), np.nan)
        for i, rec in enumerate(recs):
            for v in rec.get("verbalized", []):
                if v["round"] == rnd and v["guess"] is not None:
                    g[i] = v["guess"]
        ok = ~np.isnan(g)
        n_ok.append(int(ok.sum()))
        tr_ok = [i for i in tr if ok[i]]
        te_ok = [i for i in te if ok[i]]
        if len(tr_ok) < 3 or len(te_ok) < 2:
            r2s.append(np.nan)
            rs.append(np.nan)
            continue
        # 1-D linear recalibration fit on train, scored on test
        b, a = np.polyfit(g[tr_ok], y[tr_ok], 1)
        pred = b * g[te_ok] + a
        resid = y[te_ok] - pred
        r2s.append(1 - resid.var() / y[te_ok].var())
        rs.append(float(np.corrcoef(g[ok], y[ok])[0, 1]))

    np.savez(os.path.join(run_dir, "verbalize_r2.npz"),
             rounds=np.array(rounds), r2=np.array(r2s),
             pearson=np.array(rs), n_valid=np.array(n_ok))
    for rnd, r2, r, n in zip(rounds, r2s, rs, n_ok):
        print(f"round {rnd:2d}: verbalized R2={r2:+.3f}  pearson={r:+.3f}  "
              f"(n_valid={n}/{len(recs)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, r2s, marker="o", label="A's verbalized guess")
    p = os.path.join(run_dir, "probe_r2.npz")
    if os.path.exists(p):
        z = np.load(p)
        r2, turns = z["r2"], z["turns"]
        best_l = int(np.argmax(r2[:, -1]))
        ax.plot(turns, r2[best_l], marker="^", lw=2.5,
                label=f"A's residual probe (layer {best_l})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("turn (round)")
    ax.set_ylabel("held-out R$^2$")
    ax.set_title(f"introspection gap: probe vs verbalized ({cfg.name})")
    ax.legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"verbalize_curves.{ext}"),
                    bbox_inches="tight", dpi=150)
    print(f"wrote verbalize_r2.npz and verbalize_curves -> {run_dir}")


if __name__ == "__main__":
    main()
