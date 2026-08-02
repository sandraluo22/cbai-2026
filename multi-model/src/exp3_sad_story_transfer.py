"""Exp3 -- Emotion transfer via CONTEXT, not steering: tell Llama a sad story,
THEN have it generate the walk (no residual-stream surgery). Does the sadness
ride into Qwen through the walk tokens?

This is the prompt-based counterpart to Exp2. Instead of adding a sadness vector
to Llama's residual stream (which also wrecked the walk, validity 0.53->0.27),
we prepend a genuinely sad passage (`common.SAD_STORY`) as natural-language
context and let Llama generate the walk while reading it. Control = a
length-matched neutral passage (`common.NEUTRAL_STORY`). Everything downstream
mirrors Exp2:

  1. Llama generates the walk (constrained to 16 node words) with the story in
     context. Two conditions: sad_story / neutral_story. Records gen validity ->
     we expect prompting to preserve the walk far better than steering did.
  2. Feed each generated walk to Qwen (ONLY the walk, no story) and measure, per
     Qwen layer, the mean projection of Qwen's residual onto Qwen's sadness
     direction. sad_story vs neutral_story.
  3. Story: Qwen free-generates from the walk; score sadness (Qwen-sadness
     projection + sad-word fraction).
  4. Context-length sweep over how many walk steps Qwen sees.

Requires the emotion vectors from build_emotion_vectors.py (Qwen at least).
Env: PRESET GRAPH NSEED XCTX GSTEPS TEMP MEASURE_LO/HI STORY_TOK CTX_GRID RUN_DIR DEVICE
Out: <RUN_DIR>/exp3_sad_story_transfer.json + .pdf
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C  # noqa: E402
import graph as G   # noqa: E402

GRAPH = os.environ.get("GRAPH", "square_grid")
NSEED = int(os.environ.get("NSEED", "6" if C.PRESET != "smoke" else "3"))
XCTX = int(os.environ.get("XCTX", "80" if C.PRESET != "smoke" else "15"))
GSTEPS = int(os.environ.get("GSTEPS", "220" if C.PRESET != "smoke" else "40"))
TEMP = float(os.environ.get("TEMP", "1.0"))
STORY_TOK = int(os.environ.get("STORY_TOK", "100" if C.PRESET != "smoke" else "20"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
_CTX_DEFAULT = "10,25,50,100,220" if C.PRESET != "smoke" else "5,15,40"
CTX_GRID = [int(x) for x in os.environ.get("CTX_GRID", _CTX_DEFAULT).split(",")]

# reuse Exp2's helpers verbatim (band / sadness_by_layer / qwen_story)
from exp2_sadness_transfer import band, sadness_by_layer, qwen_story  # noqa: E402

STORIES = {"neutral_story": C.NEUTRAL_STORY, "sad_story": C.SAD_STORY}


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg(GRAPH, n_walks=max(NSEED, 8), walk_length=max(XCTX, max(CTX_GRID) + 5), device=dev)
    graph, n, coords = C.build_grid(cfg)
    seeds = G.generate_walks(graph, cfg)[:NSEED]

    sad_q, meta_q = C.load_emotion_vectors("Qwen", RUN_DIR)
    sad_q = sad_q[meta_q["sadness_idx"]]                       # (L,H)

    out = {"graph": GRAPH, "n_nodes": n, "ctx_grid": CTX_GRID, "nseed": NSEED,
           "xctx": XCTX, "gsteps": GSTEPS, "story_tok": STORY_TOK,
           "prefixes": {k: v for k, v in STORIES.items()},
           "gen_behaviour": {}, "walk_sadness": {}, "story": {}, "ctx_sweep": {}}

    # ---- 1. Llama generates walks under each story prefix ----
    print("[exp3] loading Llama", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(ltok, graph, dev)
    walks = {k: [] for k in STORIES}
    beh = {k: [] for k in STORIES}
    for cond, prefix in STORIES.items():
        for si, seed in enumerate(seeds):
            nodes, b = C.generate_walk(llama, ltok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                       temp=TEMP, rng=np.random.default_rng(3000 + si), prefix=prefix)
            walks[cond].append(C.mkwalk(nodes, graph))
            beh[cond].append(b)
    out["gen_behaviour"] = {c: {"nbr_mass": float(np.nanmean([x["nbr_mass"] for x in beh[c]])),
                                "validity": float(np.nanmean([x["validity"] for x in beh[c]]))}
                            for c in STORIES}
    print(f"[exp3] gen behaviour: {out['gen_behaviour']}", flush=True)
    C.free(llama, ltok)

    # ---- 2. + 3. Qwen sadness read-out + story ----
    print("[exp3] loading Qwen", flush=True)
    qwen, qtok = C.load_model("Qwen", cfg)
    meas_layers = band(C.n_layers(qwen), "MEASURE_LO", "MEASURE_HI")
    sad_dir_q = {L: C.unit(sad_q[L]) for L in meas_layers}
    story_layer = meas_layers[len(meas_layers) // 2]

    for cond in STORIES:
        out["walk_sadness"][cond] = sadness_by_layer(qwen, qtok, walks[cond], sad_dir_q, dev)
    peakL = max(meas_layers, key=lambda L: out["walk_sadness"]["sad_story"][L] - out["walk_sadness"]["neutral_story"][L])
    out["walk_sadness"]["peak_layer"] = int(peakL)
    print(f"[exp3] walk sadness Δ(sad-neutral) peaks at Qwen L{peakL}: "
          f"{out['walk_sadness']['sad_story'][peakL]-out['walk_sadness']['neutral_story'][peakL]:+.4f}", flush=True)

    for cond in STORIES:
        proj, frac, stories = [], [], []
        for wk in walks[cond]:
            story, _ = qwen_story(qwen, qtok, wk.text, dev, STORY_TOK)
            stories.append(story)
            frac.append(C.sad_word_fraction(story))
            if story.strip():
                p = C.project_residuals_on_dir(qwen, qtok, story, dev, {story_layer: sad_dir_q[story_layer]})
                proj.append(p[story_layer])
        out["story"][cond] = {"proj": float(np.nanmean(proj)) if proj else float("nan"),
                              "sad_word_frac": float(np.nanmean(frac)),
                              "story_layer": int(story_layer), "samples": stories[:2]}
    print(f"[exp3] story sad-word frac neutral={out['story']['neutral_story']['sad_word_frac']:.4f} "
          f"sad={out['story']['sad_story']['sad_word_frac']:.4f}", flush=True)

    # ---- 4. context-length sweep ----
    for cond in STORIES:
        rows = []
        for cl in CTX_GRID:
            projs, fracs = [], []
            for wk in walks[cond]:
                sub = C.mkwalk(wk.nodes[:cl], graph)
                pd = C.project_residuals_on_dir(qwen, qtok, sub.text, dev, {peakL: sad_dir_q[peakL]})
                projs.append(pd[peakL])
                story, _ = qwen_story(qwen, qtok, sub.text, dev, STORY_TOK)
                fracs.append(C.sad_word_fraction(story))
            rows.append({"ctx": cl, "walk_proj": float(np.nanmean(projs)),
                         "story_sad_frac": float(np.nanmean(fracs))})
        out["ctx_sweep"][cond] = rows
    C.free(qwen, qtok)

    json.dump(out, open(os.path.join(RUN_DIR, "exp3_sad_story_transfer.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp3_sad_story_transfer.pdf"))
    print(f"[exp3] DONE -> {RUN_DIR}/exp3_sad_story_transfer.json", flush=True)


def make_fig(out, path):
    cols = {"neutral_story": "k", "sad_story": "tab:red"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        ws = out["walk_sadness"]
        layers = sorted(int(k) for k in ws["neutral_story"])
        for cond, c in cols.items():
            ax[0].plot(layers, [ws[cond][L] if L in ws[cond] else ws[cond][str(L)] for L in layers],
                       "-o", ms=3, color=c, label=cond)
        ax[0].axvline(ws["peak_layer"], color="tab:red", ls=":", lw=1)
        ax[0].set_xlabel("Qwen layer"); ax[0].set_ylabel("mean proj onto Qwen sadness dir")
        ax[0].set_title("Qwen sadness projection over the walk", fontsize=10); ax[0].legend(fontsize=8)
        for cond, c in cols.items():
            rows = out["ctx_sweep"][cond]
            ax[1].plot([r["ctx"] for r in rows], [r["walk_proj"] for r in rows], "-o", ms=4, color=c, label=cond)
        ax[1].set_xlabel("Llama walk steps Qwen sees"); ax[1].set_ylabel(f"proj @ L{ws['peak_layer']}")
        ax[1].set_title("Sadness vs context length", fontsize=10); ax[1].legend(fontsize=8)
        for cond, c in cols.items():
            rows = out["ctx_sweep"][cond]
            ax[2].plot([r["ctx"] for r in rows], [r["story_sad_frac"] for r in rows], "-o", ms=4, color=c, label=cond)
        ax[2].set_xlabel("Llama walk steps Qwen sees"); ax[2].set_ylabel("Qwen story sad-word fraction")
        ax[2].set_title("Qwen story sadness vs context length", fontsize=10); ax[2].legend(fontsize=8)
        b = out["gen_behaviour"]
        fig.suptitle(f"[{out['graph']}] Exp3 sad-STORY transfer — prime Llama with a sad story (no steering) → Qwen.  "
                     f"Llama walk validity neutral={b['neutral_story']['validity']:.2f}/sad={b['sad_story']['validity']:.2f} "
                     f"(vs Exp2 steering 0.53→0.27).  Story sad-word frac "
                     f"neutral={out['story']['neutral_story']['sad_word_frac']:.3f}/sad={out['story']['sad_story']['sad_word_frac']:.3f}",
                     fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
