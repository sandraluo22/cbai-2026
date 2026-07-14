"""Steering-summary figure: |Δ mass on each cut's + side| vs dose, per model, WITH a random-vector
baseline (mean over cuts ± std). Reads acs_<model>_<graph>.json (the *_rand version that
carries out['random']).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/3_causal/axis_cut_sweep_rand")
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
ONLY = os.environ.get("CUTS", "")                     # comma list to keep; empty = all
SHOWRAND = os.environ.get("SHOWRAND", "1") == "1"     # draw the random-vector baseline?
OUT = os.environ.get("OUT", f"{DIR}/acs_summary.pdf")
CUTCOL = {"x": "#F97316", "y": "#EAB308", "diagonal": "#14B8A6",
          "anti-diagonal": "#2563EB", "parity": "#7C3AED"}


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        d = json.load(open(f"{DIR}/acs_{m}_{GS}.json"))
        doses = d["doses"]; xs = list(range(len(doses) + 1))              # 0 = clean
        keep = ONLY.split(",") if ONLY else None
        for cname, cd in d["cuts"].items():
            if keep and cname not in keep: continue
            sw = cd["sweep"]; base = sw["clean"]["mass_plus"]
            y = [0.0] + [abs(sw[f"{dd:g}"]["mass_plus"] - base) for dd in doses]
            ax.plot(xs, y, "o-", color=CUTCOL.get(cname), lw=1.8, ms=4, label=cname)
        if SHOWRAND and "random" in d:
            r = d["random"]
            ym = np.array([0.0] + [r[f"{dd:g}"]["abs_dmass_mean"] for dd in doses])
            ys = np.array([0.0] + [r[f"{dd:g}"]["abs_dmass_std"] for dd in doses])
            ax.plot(xs, ym, "--", color="k", lw=1.6, label="random vector")
            ax.fill_between(xs, ym - ys, ym + ys, color="k", alpha=0.12)
        ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"{dd:g}" for dd in doses], fontsize=7)
        ax.set_xlabel("steer dose (axis σ)"); ax.set_title(m, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    axes[0].set_ylabel("|Δ mass on cut's + side|  (vs clean)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Steering each grid cut vs dose"
                 + ("  (random-vector baseline ≈ 0)" if SHOWRAND else ""), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
