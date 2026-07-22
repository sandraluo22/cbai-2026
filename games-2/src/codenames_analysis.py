"""Quantify memory vs memoryless, and fit spymaster->guesser regressions, for the
open-clue Codenames runs.

Games are PAIRED across modes: game `gi` uses the same seed (=> same targets) in both
the memoryless and memory runs, so we can compare them game-by-game.

Reads the two transcripts:
  runs/codenames/llm_codenames_open/      (memoryless)
  runs/codenames/llm_codenames_open_mem/  (spymaster remembers past clues)

Produces:
  1. A paired memory-effect table (turns-to-complete, recovery, clue-repeat rate,
     coupling, adaptivity) with Wilcoxon signed-rank p-values.
  2. OLS regressions (game-clustered SEs) of a GUESSER outcome on SPYMASTER signal +
     a memory dummy -- the spymaster->guesser link, and how memory shifts it.
  3. codenames_analysis.pdf : paired turns, recovery~round per mode, and a
     Frisch-Waugh partial-regression of recovery on adaptivity (round controlled).

Usage:  python src/codenames_analysis.py
Out:    runs/codenames/codenames_analysis.{pdf,json}  (+ printed report)
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import statsmodels.formula.api as smf

RUNS = "runs/codenames"
MODES = {"llm_codenames_open": 0, "llm_codenames_open_mem": 1}   # dir -> memory flag
TRANSCRIPT = "game_llm_open_LlamaInst_vs_QwenInst_transcript.jsonl"


def load_rounds():
    recs = []
    for d, mem in MODES.items():
        path = os.path.join(RUNS, d, TRANSCRIPT)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        seen = {}                                          # (pair,game) -> set of clues so far
        for r in rows:
            key = (r["pair"], r["game"])
            s = seen.setdefault(key, set())
            is_rep = r["clue"] in s
            s.add(r["clue"])
            recs.append({
                "memory": mem, "pair": r["pair"], "spymaster": r["spymaster"],
                "guesser": r["guesser"], "game": r["game"], "round": r["round"] + 1,
                "coupling": r["coupling"]["kl"], "adaptivity": r["adaptivity"]["kl"],
                "recovery": r["target_mass"], "n_correct": int(sum(r["correct"])),
                "clue": r["clue"], "is_repeat": int(is_rep),
                "spy_qwen": int(r["spymaster"] == "QwenInst"),
            })
    df = pd.DataFrame(recs)
    df["game_uid"] = df["memory"].astype(str) + "|" + df["pair"] + "|" + df["game"].astype(str)
    return df


def per_game(df):
    g = df.groupby(["memory", "pair", "game", "spy_qwen"]).agg(
        turns=("round", "max"),
        final_recovery=("recovery", "last"),
        mean_coupling=("coupling", "mean"),
        mean_adaptivity=("adaptivity", "mean"),
        repeat_rate=("is_repeat", "mean"),
    ).reset_index()
    return g


def memory_paired(g):
    """Paired mem vs no-mem per (pair, game). Returns a table of effects + Wilcoxon p."""
    wide = g.pivot_table(index=["pair", "game"], columns="memory",
                         values=["turns", "final_recovery", "mean_coupling",
                                 "mean_adaptivity", "repeat_rate"])
    out = []
    for metric in ["turns", "final_recovery", "repeat_rate", "mean_coupling", "mean_adaptivity"]:
        a = wide[(metric, 0)].values          # memoryless
        b = wide[(metric, 1)].values          # memory
        diff = b - a
        # Wilcoxon needs some nonzero diffs
        try:
            p = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = 1.0
        out.append({"metric": metric, "nomem_mean": a.mean(), "mem_mean": b.mean(),
                    "mean_diff(mem-nomem)": diff.mean(), "wilcoxon_p": p, "n_pairs": len(diff)})
    return pd.DataFrame(out), wide


def fwl_partial(df, y="recovery", x="adaptivity", controls=("round", "spy_qwen")):
    """Frisch-Waugh-Lovell partial regression of y on x controlling for `controls`,
    per memory mode. Returns dict mode -> (rx, ry, slope, p)."""
    res = {}
    for mem in (0, 1):
        d = df[df.memory == mem]
        cform = " + ".join(controls)
        ry = smf.ols(f"{y} ~ {cform}", data=d).fit().resid
        rx = smf.ols(f"{x} ~ {cform}", data=d).fit().resid
        fit = smf.ols("ry ~ rx", data=pd.DataFrame({"ry": ry, "rx": rx})).fit()
        res[mem] = (rx.values, ry.values, float(fit.params["rx"]), float(fit.pvalues["rx"]))
    return res


def main():
    df = load_rounds()
    g = per_game(df)
    print("=" * 78)
    print(f"loaded {len(df)} rounds  |  {len(g)} games "
          f"({df.memory.sum()} mem rounds, {(df.memory == 0).sum()} no-mem rounds)")

    # ---- 1. memory vs memoryless (paired) ----
    tab, wide = memory_paired(g)
    print("\n[1] MEMORY vs MEMORYLESS  (paired by pair+game, n=16; Wilcoxon signed-rank)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n    turns-to-complete by role ordering (mem - nomem):")
    for pair in sorted(g.pair.unique()):
        sub = wide.loc[pair] if pair in wide.index.get_level_values(0) else None
        a = g[(g.pair == pair) & (g.memory == 0)]["turns"].mean()
        b = g[(g.pair == pair) & (g.memory == 1)]["turns"].mean()
        print(f"      {pair}:  nomem={a:.2f}  mem={b:.2f}  diff={b - a:+.2f}")

    # ---- 2. regressions: spymaster -> guesser, with memory dummy ----
    print("\n[2] SPYMASTER -> GUESSER regressions (per-round; SE clustered by game)")
    models = {
        "recovery ~ round + adaptivity + memory + spy_qwen":
            "recovery ~ round + adaptivity + memory + spy_qwen",
        "recovery ~ round + adaptivity*memory + spy_qwen  (does memory shift the slope?)":
            "recovery ~ round + adaptivity*memory + spy_qwen",
        "coupling ~ round + adaptivity + memory + spy_qwen":
            "coupling ~ round + adaptivity + memory + spy_qwen",
    }
    reg_summary = {}
    for label, formula in models.items():
        m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["game_uid"]})
        print(f"\n  MODEL: {label}")
        coefs = pd.DataFrame({"coef": m.params, "se": m.bse, "t": m.tvalues, "p": m.pvalues})
        print(coefs.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"    R2={m.rsquared:.3f}  n={int(m.nobs)}")
        reg_summary[label] = {"r2": float(m.rsquared),
                              "coef": {k: float(v) for k, v in m.params.items()},
                              "p": {k: float(v) for k, v in m.pvalues.items()}}

    # ---- 3. Frisch-Waugh partial: recovery ~ adaptivity | round, spy_qwen, per mode ----
    fwl = fwl_partial(df)
    print("\n[3] spymaster->guesser PARTIAL effect  (recovery on adaptivity, round+spymaster controlled)")
    for mem, lab in ((0, "no-mem"), (1, "mem  ")):
        _, _, slope, p = fwl[mem]
        print(f"      {lab}: partial slope = {slope:+.4f}  (p={p:.3f})")

    # ---- figure ----
    os.makedirs(RUNS, exist_ok=True)
    figpath = os.path.join(RUNS, "codenames_analysis.pdf")
    with PdfPages(figpath) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        # (a) paired turns
        pairs = sorted(g.pair.unique())
        cols = {pairs[0]: "tab:blue", pairs[1]: "tab:orange"}
        for pair in pairs:
            for gi in sorted(g.game.unique()):
                a = g[(g.pair == pair) & (g.game == gi) & (g.memory == 0)]["turns"]
                b = g[(g.pair == pair) & (g.game == gi) & (g.memory == 1)]["turns"]
                if len(a) and len(b):
                    ax[0].plot([0, 1], [a.values[0], b.values[0]], color=cols[pair], alpha=.35, lw=1)
            ma = g[(g.pair == pair) & (g.memory == 0)]["turns"].mean()
            mb = g[(g.pair == pair) & (g.memory == 1)]["turns"].mean()
            ax[0].plot([0, 1], [ma, mb], "-o", color=cols[pair], lw=3, label=f"{pair}")
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["memoryless", "memory"])
        ax[0].set_ylabel("turns to find all targets"); ax[0].set_title("(a) Efficiency: paired per game", fontsize=10)
        ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        # (b) recovery ~ round per mode
        for mem, c, lab in ((0, "tab:red", "memoryless"), (1, "tab:green", "memory")):
            d = df[df.memory == mem]
            jit = d["round"] + np.random.default_rng(mem).normal(0, .05, len(d))
            ax[1].scatter(jit, d["recovery"], s=10, alpha=.25, color=c)
            fit = smf.ols("recovery ~ round", data=d).fit()
            xs = np.linspace(d["round"].min(), d["round"].max(), 20)
            ax[1].plot(xs, fit.params["Intercept"] + fit.params["round"] * xs, color=c, lw=2.4,
                       label=f"{lab}: slope={fit.params['round']:+.3f}")
        ax[1].set_xlabel("round"); ax[1].set_ylabel("recovery (target mass)")
        ax[1].set_title("(b) Guesser recovery vs round", fontsize=10); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        # (c) FWL partial: recovery ~ adaptivity | round
        for mem, c, lab in ((0, "tab:red", "memoryless"), (1, "tab:green", "memory")):
            rx, ry, slope, p = fwl[mem]
            ax[2].scatter(rx, ry, s=10, alpha=.25, color=c)
            xs = np.linspace(rx.min(), rx.max(), 20)
            ax[2].plot(xs, slope * xs, color=c, lw=2.4, label=f"{lab}: slope={slope:+.3f} (p={p:.2f})")
        ax[2].set_xlabel("adaptivity  (residualized on round, spymaster)")
        ax[2].set_ylabel("recovery  (residualized)")
        ax[2].set_title("(c) spymaster→guesser partial effect\n(recovery on adaptivity, round controlled)", fontsize=10)
        ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
        fig.suptitle("Open-clue Codenames: memory vs memoryless, and spymaster→guesser regression", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
    print(f"\nwrote {figpath}")

    json.dump({"memory_paired": tab.to_dict(orient="records"), "regressions": reg_summary,
               "fwl_partial": {("nomem" if k == 0 else "mem"): {"slope": v[2], "p": v[3]} for k, v in fwl.items()}},
              open(os.path.join(RUNS, "codenames_analysis.json"), "w"), indent=2)
    print(f"wrote {os.path.join(RUNS, 'codenames_analysis.json')}")


if __name__ == "__main__":
    main()
