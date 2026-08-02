"""Exp2 -- Emotion TRANSFER: steer SADNESS into Llama while it generates the
walk, feed the walk to Qwen, and ask whether the sadness rides along.

Requires the emotion vectors from build_emotion_vectors.py (Llama + Qwen) in
RUN_DIR.

Pipeline:
  1. Steer Llama's residual stream along its SADNESS direction (unit sadness
     vector at a band of layers, dose = SDOSE x typical residual norm) while it
     free-generates the walk (still constrained to the 16 node words). Also a
     clean (unsteered) condition. -> `sad_walk`, `clean_walk`.
  2. Feed each generated walk to Qwen and measure, per Qwen layer, the mean
     projection of Qwen's residual (at the node tokens) onto QWEN's OWN sadness
     direction. Does steering Llama sad raise Qwen's sadness projection?
  3. STORY: give Qwen the generated walk as a prefix and let it free-generate a
     short continuation. Score the story's sadness two ways: (a) mean projection
     of Qwen's residual over the story tokens onto Qwen's sadness direction; (b)
     a model-independent sad-word fraction. sad-context vs clean-context.
  4. CONTEXT-LENGTH SWEEP: repeat (2)+(3) while varying how many steps of Llama's
     generated walk Qwen sees (CTX_GRID). Does more Llama-sad-context make Qwen
     sadder / its story sadder?

Env: PRESET GRAPH(square_grid) NSEED(6) XCTX(80) GSTEPS(220) TEMP(1.0)
     SDOSE(8) STEER_LO/STEER_HI (Llama steer-layer band; default mid third)
     MEASURE_LO/MEASURE_HI (Qwen measure-layer band; default mid third)
     STORY_TOK(100) CTX_GRID(10,25,50,100,220) RUN_DIR DEVICE
Out: <RUN_DIR>/exp2_sadness_transfer.json + .pdf
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

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

GRAPH = os.environ.get("GRAPH", "square_grid")
NSEED = int(os.environ.get("NSEED", "6" if C.PRESET != "smoke" else "3"))
XCTX = int(os.environ.get("XCTX", "80" if C.PRESET != "smoke" else "15"))
GSTEPS = int(os.environ.get("GSTEPS", "220" if C.PRESET != "smoke" else "40"))
TEMP = float(os.environ.get("TEMP", "1.0"))
# Steering dose, expressed as a MULTIPLE of the typical residual-stream norm at
# the steered layer (added vector norm = SDOSE x ref_norm x unit(sadness)). ~0.6
# is a firm push that mostly preserves walk validity; sweep {0.25,0.5,1,2} on the
# real models (a too-large dose randomizes the walk -> validity collapses).
SDOSE = float(os.environ.get("SDOSE", "0.6"))
STORY_TOK = int(os.environ.get("STORY_TOK", "100" if C.PRESET != "smoke" else "20"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
_CTX_DEFAULT = "10,25,50,100,220" if C.PRESET != "smoke" else "5,15,40"
CTX_GRID = [int(x) for x in os.environ.get("CTX_GRID", _CTX_DEFAULT).split(",")]


def band(nL, lo_env, hi_env):
    """Layer band; default = middle third of the model."""
    lo = os.environ.get(lo_env)
    hi = os.environ.get(hi_env)
    if lo is not None and hi is not None:
        return list(range(int(lo), int(hi) + 1))
    return list(range(nL // 3, 2 * nL // 3))


def steer_map(sad_clean, layers, dose, ref_norm_by_layer):
    """layer -> dose*ref_norm[L]*unit(sadness_vec[L]). Scaling by the typical
    residual norm keeps the dose comparable across layers/models."""
    m = {}
    for L in layers:
        m[L] = (dose * ref_norm_by_layer[L] * C.unit(sad_clean[L])).astype(np.float32)
    return m


@torch.no_grad() if torch is not None else (lambda f: f)
def residual_norms(model, tok, graph, walk, dev, layers):
    """Typical per-token residual L2 norm at each layer (for dose scaling)."""
    blocks = C.decoder_blocks(model)
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    ids = tok(walk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
    grabbed.clear()
    model(input_ids=ids)
    for h in hs:
        h.remove()
    return {L: float(grabbed[L][0].float().norm(dim=-1).mean().cpu()) for L in layers}


def sadness_by_layer(qwen, qtok, walks, sad_dir_by_layer, dev, node_tokens=True):
    """Mean projection of Qwen residual onto its sadness direction, per layer,
    averaged over walks. Uses ALL tokens (node walks are all node tokens)."""
    accum = {L: [] for L in sad_dir_by_layer}
    for wk in walks:
        proj = C.project_residuals_on_dir(qwen, qtok, wk.text, dev, sad_dir_by_layer)
        for L, v in proj.items():
            accum[L].append(v)
    return {L: float(np.nanmean(vs)) for L, vs in accum.items()}


@torch.no_grad() if torch is not None else (lambda f: f)
def qwen_story(qwen, qtok, prefix_text, dev, max_new):
    """Free-generate a short continuation from the walk prefix (unconstrained)."""
    enc = qtok(prefix_text, add_special_tokens=True, return_tensors="pt").to(dev)
    ids = enc["input_ids"]
    gen = qwen.generate(ids, attention_mask=enc.get("attention_mask"),
                        max_new_tokens=max_new, do_sample=True, temperature=1.0,
                        top_p=0.95, pad_token_id=qtok.eos_token_id)
    new_ids = gen[0, ids.shape[1]:]
    story = qtok.decode(new_ids, skip_special_tokens=True)
    return story, ids.shape[1]


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg(GRAPH, n_walks=max(NSEED, 8), walk_length=max(XCTX, max(CTX_GRID) + 5), device=dev)
    graph, n, coords = C.build_grid(cfg)
    seeds = G.generate_walks(graph, cfg)[:NSEED]

    # emotion vectors (built earlier)
    sad_l, meta_l = C.load_emotion_vectors("Llama", RUN_DIR)
    sad_q, meta_q = C.load_emotion_vectors("Qwen", RUN_DIR)
    si_l, si_q = meta_l["sadness_idx"], meta_q["sadness_idx"]
    sad_l = sad_l[si_l]                                  # (L,H) Llama sadness
    sad_q = sad_q[si_q]                                  # (L,H) Qwen sadness

    out = {"graph": GRAPH, "n_nodes": n, "sdose": SDOSE, "ctx_grid": CTX_GRID,
           "nseed": NSEED, "xctx": XCTX, "gsteps": GSTEPS, "story_tok": STORY_TOK,
           "gen_behaviour": {}, "walk_sadness": {}, "story": {}, "ctx_sweep": {}}

    # ---- 1. Llama generates clean + sad-steered walks ----
    print("[exp2] loading Llama", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    nL_l = C.n_layers(llama)
    steer_layers = band(nL_l, "STEER_LO", "STEER_HI")
    rn = residual_norms(llama, ltok, graph, C.mkwalk(seeds[0].nodes[:XCTX], graph), dev, steer_layers)
    smap = steer_map(sad_l, steer_layers, SDOSE, rn)
    print(f"[exp2] Llama steer band L{steer_layers[0]}..{steer_layers[-1]} dose={SDOSE}", flush=True)

    cand = C.candidate_token_ids(ltok, graph, dev)
    walks = {"clean": [], "sad": []}
    beh = {"clean": [], "sad": []}
    for si, seed in enumerate(seeds):
        for cond, steer in (("clean", None), ("sad", smap)):
            nodes, b = C.generate_walk(llama, ltok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                       temp=TEMP, rng=np.random.default_rng(2000 + si), steer=steer)
            walks[cond].append(C.mkwalk(nodes, graph))
            beh[cond].append(b)
    out["gen_behaviour"] = {c: {"nbr_mass": float(np.nanmean([x["nbr_mass"] for x in beh[c]])),
                                "validity": float(np.nanmean([x["validity"] for x in beh[c]]))}
                            for c in walks}
    print(f"[exp2] gen behaviour: {out['gen_behaviour']}", flush=True)
    C.free(llama, ltok)

    # ---- 2. + 3. Qwen sadness read-out + story ----
    print("[exp2] loading Qwen", flush=True)
    qwen, qtok = C.load_model("Qwen", cfg)
    nL_q = C.n_layers(qwen)
    meas_layers = band(nL_q, "MEASURE_LO", "MEASURE_HI")
    sad_dir_q = {L: C.unit(sad_q[L]) for L in meas_layers}
    story_layer = meas_layers[len(meas_layers) // 2]

    # full-walk sadness projection per Qwen layer, clean vs sad
    for cond in ("clean", "sad"):
        out["walk_sadness"][cond] = sadness_by_layer(qwen, qtok, walks[cond], sad_dir_q, dev)
    peakL = max(meas_layers, key=lambda L: out["walk_sadness"]["sad"][L] - out["walk_sadness"]["clean"][L])
    out["walk_sadness"]["peak_layer"] = int(peakL)
    print(f"[exp2] walk sadness Δ(sad-clean) peaks at Qwen L{peakL}: "
          f"{out['walk_sadness']['sad'][peakL]-out['walk_sadness']['clean'][peakL]:+.4f}", flush=True)

    # story: prefix = full generated walk; measure story sadness two ways
    for cond in ("clean", "sad"):
        proj, frac = [], []
        stories = []
        for wk in walks[cond]:
            story, _ = qwen_story(qwen, qtok, wk.text, dev, STORY_TOK)
            stories.append(story)
            frac.append(C.sad_word_fraction(story))
            if story.strip():
                p = C.project_residuals_on_dir(qwen, qtok, story, dev, {story_layer: sad_dir_q[story_layer]})
                proj.append(p[story_layer])
        out["story"][cond] = {"proj": float(np.nanmean(proj)) if proj else float("nan"),
                              "sad_word_frac": float(np.nanmean(frac)),
                              "story_layer": int(story_layer),
                              "samples": stories[:2]}
    print(f"[exp2] story sad-word frac  clean={out['story']['clean']['sad_word_frac']:.4f} "
          f"sad={out['story']['sad']['sad_word_frac']:.4f}", flush=True)

    # ---- 4. context-length sweep ----
    for cond in ("clean", "sad"):
        rows = []
        for cl in CTX_GRID:
            projs, fracs = [], []
            for wk in walks[cond]:
                sub = C.mkwalk(wk.nodes[:cl], graph)
                pd = C.project_residuals_on_dir(qwen, qtok, sub.text, dev,
                                                {peakL: sad_dir_q[peakL]})
                projs.append(pd[peakL])
                story, _ = qwen_story(qwen, qtok, sub.text, dev, STORY_TOK)
                fracs.append(C.sad_word_fraction(story))
            rows.append({"ctx": cl, "walk_proj": float(np.nanmean(projs)),
                         "story_sad_frac": float(np.nanmean(fracs))})
        out["ctx_sweep"][cond] = rows
    C.free(qwen, qtok)

    json.dump(out, open(os.path.join(RUN_DIR, "exp2_sadness_transfer.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp2_sadness_transfer.pdf"))
    print(f"[exp2] DONE -> {RUN_DIR}/exp2_sadness_transfer.json", flush=True)


def make_fig(out, path):
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        # (1) per-Qwen-layer sadness projection, clean vs sad
        ws = out["walk_sadness"]
        layers = sorted(int(k) for k in ws["clean"])
        ax[0].plot(layers, [ws["clean"][str(L)] if str(L) in ws["clean"] else ws["clean"][L] for L in layers],
                   "-o", ms=3, color="k", label="clean walk")
        ax[0].plot(layers, [ws["sad"][str(L)] if str(L) in ws["sad"] else ws["sad"][L] for L in layers],
                   "-o", ms=3, color="tab:red", label="sad-steered walk")
        ax[0].axvline(ws["peak_layer"], color="tab:red", ls=":", lw=1)
        ax[0].set_xlabel("Qwen layer"); ax[0].set_ylabel("mean proj onto Qwen sadness dir")
        ax[0].set_title("Qwen sadness projection over the walk", fontsize=10); ax[0].legend(fontsize=8)
        # (2) context-length sweep: walk projection at peak layer
        for cond, c in (("clean", "k"), ("sad", "tab:red")):
            rows = out["ctx_sweep"][cond]
            ax[1].plot([r["ctx"] for r in rows], [r["walk_proj"] for r in rows], "-o", ms=4, color=c, label=cond)
        ax[1].set_xlabel("Llama walk steps Qwen sees"); ax[1].set_ylabel(f"proj @ L{out['walk_sadness']['peak_layer']}")
        ax[1].set_title("Sadness vs context length", fontsize=10); ax[1].legend(fontsize=8)
        # (3) context-length sweep: story sad-word fraction
        for cond, c in (("clean", "k"), ("sad", "tab:red")):
            rows = out["ctx_sweep"][cond]
            ax[2].plot([r["ctx"] for r in rows], [r["story_sad_frac"] for r in rows], "-o", ms=4, color=c, label=cond)
        ax[2].set_xlabel("Llama walk steps Qwen sees"); ax[2].set_ylabel("Qwen story sad-word fraction")
        ax[2].set_title("Qwen story sadness vs context length", fontsize=10); ax[2].legend(fontsize=8)
        b = out["gen_behaviour"]
        fig.suptitle(f"[{out['graph']}] Exp2 sadness transfer — steer Llama sad (dose={out['sdose']}) → Qwen.  "
                     f"Llama gen validity clean={b['clean']['validity']:.2f}/sad={b['sad']['validity']:.2f}. "
                     f"Story sad-word frac clean={out['story']['clean']['sad_word_frac']:.3f}/"
                     f"sad={out['story']['sad']['sad_word_frac']:.3f}", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
