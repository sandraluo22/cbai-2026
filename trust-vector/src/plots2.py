"""Two figures for protocol v2: the cosine heatmap, and projection vs conversation depth.

FIG 1  cosine between every v2 family's direction, diagonal = split-half reliability
       (the ceiling), so an off-diagonal cell can be read against what that row can
       possibly reach. Families are ordered by FORM (one-clause / elaborated / game /
       story) rather than by dimension, because form is what the matrix clusters on.

FIG 2  the conversation measurement, per turn rather than end-to-end. The scam panel
       is the one to look at: the projection is tracked at every depth, with the turn
       where the caller asks for the passcode marked, so a drop (or its absence) at
       that specific point is visible instead of being averaged into an endpoint.

env: OUT (../out) TAG (full)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
TAG = os.environ.get("TAG", "full")

ORDER = ["direct_a", "comp_a", "hon_a", "rel_a",
         "direct_b", "comp_b", "hon_b", "rel_b", "warmth_b",
         "relational", "game_a", "game_b",
         "story_trust", "story_comp", "story_hon", "story_rel",
         # same property, four different ways of ASKING the model for a story:
         # spelled-out operationalisation vs bare request, with and without the
         # "don't use the word" ban
         "story_trust@acct", "story_trust@story",
         "story_trust@acctnb", "story_trust@storynb"]
GROUP = {"one-clause": ORDER[0:4], "elaborated": ORDER[4:9],
         "relational / game": ORDER[9:12], "model-written": ORDER[12:16],
         "generation-prompt variants": ORDER[16:20]}

HELP_LAB = ["opening\nclaim", '"oh no,\nwhat happened"', "tells you to\nhang up + call back",
            '"called back,\nso this is real"', "explains the\ndeclined charges",
            '"matches the\nalert I got"', "new card, says\nwe never ask for codes",
            '"showing in\nthe app"', "case number in\nsecure inbox", '"thanks!"']
SCAM_LAB = ["opening\nclaim", '"oh no,\nwhat happened"', "urgency:\ntwo minutes",
            '"should I call\nthe number?"', "refuses\nverification", '"okay,\nI\'m here"',
            "ASKS FOR\nPASSCODE", '"bank says it\nnever asks"', "pressures\nagain",
            '"thanks!"']


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def fig_heat():
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    layers = [int(x) for x in z["layers"]]
    fams = [f for f in ORDER if f"{f}.{TAG}--last--full" in z.files]
    V = lambda f, l: z[f"{f}.{TAG}--last--full"][layers.index(l)]        # noqa: E731
    H = lambda f, l, h: z[f"{f}.{TAG}--last--{h}"][layers.index(l)]      # noqa: E731
    usable = [l for l in layers if l > 0]
    best = max(usable, key=lambda l: np.mean([cos(H(f, l, "h0"), H(f, l, "h1"))
                                              for f in fams]))
    n = len(fams)
    M = np.zeros((n, n))
    for i, a in enumerate(fams):
        for j, b in enumerate(fams):
            M[i, j] = (cos(H(a, best, "h0"), H(a, best, "h1")) if i == j
                       else cos(V(a, best), V(b, best)))
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(fams, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(fams, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.62 else "black")
    # group separators — the clusters are by FORM, so show where the forms change
    edges, k = [], 0
    for g, fs in GROUP.items():
        k += len([f for f in fs if f in fams])
        edges.append(k)
    for e in edges[:-1]:
        ax.axhline(e - 0.5, color="k", lw=1.4)
        ax.axvline(e - 0.5, color="k", lw=1.4)
    ax.set_title(f"protocol v2 — cosine between candidate directions, layer {best}\n"
                 "read at an appended bare name token; diagonal = split-half "
                 "reliability (the ceiling)\n"
                 "black lines separate one-clause / elaborated / relational+game / "
                 "model-written", fontsize=11)
    fig.colorbar(im, ax=ax, label="cosine similarity", fraction=0.046)
    fig.tight_layout()
    p = os.path.join(OUT, f"v2_heatmap_{TAG}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}  (layer {best})")
    return best


def fig_depth():
    p_json = os.path.join(OUT, "project.json")
    if not os.path.exists(p_json):
        print("[fig] no project.json yet")
        return
    r = json.load(open(p_json))
    keys = sorted({k.rsplit("_", 1)[0] for k in r if k.endswith("_scam")})
    layers = sorted({int(k.split("_L")[1]) for k in keys})
    L = layers[len(layers) // 2]
    ks = [k for k in keys if k.endswith(f"_L{L}")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, tag, labs in ((axes[0], "helpful", HELP_LAB), (axes[1], "scam", SCAM_LAB)):
        for k in ks:
            fam = k.split("_L")[0]
            style = "--" if fam.startswith(("comp", "warmth", "hon_", "rel_")) else "-"
            ax.plot(r[f"{k}_{tag}"], marker="o", ms=4, ls=style, label=fam)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=7.5, rotation=45, ha="right")
        ax.set_xlabel("conversation depth (turns included)")
        ax.grid(alpha=0.25)
        ax.set_title(f"{tag} conversation")
    axes[1].axvline(6, color="crimson", lw=1.6, ls=":")
    axes[1].annotate("passcode requested", xy=(6, axes[1].get_ylim()[1]),
                     xytext=(6.05, axes[1].get_ylim()[1] * 0.97), color="crimson",
                     fontsize=9, va="top")
    axes[0].set_ylabel("projection onto the candidate direction\n"
                       "(z units of the stimulus-set spread)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"does the read-out move as the conversation goes on?  layer {L}  "
                 "(dashed = component/decoy directions, solid = trust directions)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "v2_depth.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}  (layer {L})")


if __name__ == "__main__":
    fig_heat()
    fig_depth()
    print("PLOTS2_DONE")
