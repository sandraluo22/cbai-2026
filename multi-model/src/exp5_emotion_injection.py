"""Exp5 -- Does EMOTION cross by activation INJECTION (linear map), the way grid
geometry does?

Emotion failed to cross as tokens (Exp2/3/4) -- even a classifier on the reader's
walk activations is at chance, and it's null even Llama->Llama, so the emotion is
absent from the neutral-word walk itself. But a linear injection map has DIRECT
access to Llama's residual stream, where emotion IS present (the emotion vectors
were built from it). So injection could carry emotion even though tokens can't.
This is the missing cell of the 2x2 {grid, emotion} x {tokens, activations}.

Two tests, both using a ridge alignment map W: Llama@L_A -> Qwen@L_B fit on paired
per-occurrence residuals from real walks (a genuine residual-space alignment, not
the rank-16 node-mean map):

  A. GEOMETRIC axis transport. Push Llama's sadness / joy emotion directions
     through W and cosine them against Qwen's OWN sadness / joy directions.
     matched (sad->sad) vs mismatched (sad->joy) is the specificity control.
     Per matched-depth layer pair. If matched >> mismatched > 0, the map that
     aligns the grid ALSO carries the emotion axis.

  B. CAUSAL steer. At the best pair, take the TRANSPORTED sadness direction (a
     Qwen-space vector that came from Llama), add it to Qwen's residual while Qwen
     free-generates from a neutral walk prefix, and measure the story's sadness
     (Qwen-sad projection + sad-word fraction). Baselines: no steer, Qwen's OWN
     sadness vector (upper bound), a random direction (floor). If transported ~
     own-Qwen, emotion crosses by injection.

Requires emotion_vectors_{Llama,Qwen}.npz in RUN_DIR.
Env: PRESET GRAPH NWALKS(8) WLEN(300) CTXLO(100) ALPHA(1e4) RELDEPTHS
     STEER_DOSE(6) STORY_TOK(100) NSTORY(8) RUN_DIR DEVICE
Out: <RUN_DIR>/exp5_emotion_injection.json + .pdf
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
from models import resolve_token_spans  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "8" if C.PRESET != "smoke" else "4"))
WLEN = int(os.environ.get("WLEN", "300" if C.PRESET != "smoke" else "40"))
CTXLO = int(os.environ.get("CTXLO", "100" if C.PRESET != "smoke" else "5"))
ALPHA = float(os.environ.get("ALPHA", "1e4"))
STEER_DOSE = float(os.environ.get("STEER_DOSE", "6.0"))
STORY_TOK = int(os.environ.get("STORY_TOK", "100" if C.PRESET != "smoke" else "20"))
NSTORY = int(os.environ.get("NSTORY", "8" if C.PRESET != "smoke" else "3"))
RELDEPTHS = [float(x) for x in os.environ.get("RELDEPTHS", "0.25,0.4,0.55,0.7,0.85").split(",")]
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
SAD, JOY = 25, 17                                      # emotion-vector row indices


@torch.no_grad() if torch is not None else (lambda f: f)
def capture_occ(model, tok, walks, layers, dev, ctxlo):
    """Per-occurrence residuals (last-subword token, context >= ctxlo) at each
    layer in `layers`. Returns {L: (n_occ, H)}, occurrences in a fixed order that
    is identical across models (same walks, same ctx filter)."""
    blocks = C.decoder_blocks(model)
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    rows = {L: [] for L in layers}
    try:
        for wk in walks:
            ids = tok(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk)
            cl = np.arange(1, len(wk.nodes) + 1)
            use = [t[-1] for s, t in enumerate(spans) if cl[s] >= ctxlo]
            grabbed.clear()
            model(input_ids=ids)
            for L in layers:
                rows[L].append(grabbed[L][0][use].float().cpu().numpy())
    finally:
        for h in hs:
            h.remove()
    return {L: np.concatenate(rows[L], 0) for L in layers}


def fit_ridge(XA, XB, alpha):
    """Ridge map Llama->Qwen on standardized/centered per-occurrence acts. Returns
    a prep that transports a *direction* d (a difference vector) via
    transport_dir()."""
    muA = XA.mean(0); sdA = XA.std(0) + 1e-6
    Xs = (XA - muA) / sdA
    muB = XB.mean(0); Yc = XB - muB
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)     # U(n,r) S(r) Vt(r,d)
    coef = (S / (S ** 2 + alpha))[:, None] * (U.T @ Yc)   # (r, d_B)
    return {"sdA": sdA, "V": Vt.T, "coef": coef}          # W = (V @ coef); r@W = (r/sdA @ V) @ coef


def transport_dir(prep, d):
    """Map a direction from Llama@L_A into Qwen@L_B space (mean offsets cancel for
    a difference vector)."""
    return ((d / prep["sdA"]) @ prep["V"]) @ prep["coef"]


def cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg(GRAPH, n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph, n, coords = C.build_grid(cfg)
    walks = G.generate_walks(graph, cfg)

    emo_l, ml = C.load_emotion_vectors("Llama", RUN_DIR)
    emo_q, mq = C.load_emotion_vectors("Qwen", RUN_DIR)
    nLA_emo, nLB_emo = ml["L"], mq["L"]

    # matched-relative-depth layer pairs
    def layers_for(nL):
        return [max(1, min(nL - 1, int(round(rd * nL)))) for rd in RELDEPTHS]

    LA_list = layers_for(nLA_emo)
    LB_list = layers_for(nLB_emo)
    pairs = list(zip(RELDEPTHS, LA_list, LB_list))

    # ---- capture paired per-occurrence residuals ----
    print(f"[exp5] Llama: capture {len(walks)} walks @ LA={LA_list}", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    XA = capture_occ(llama, ltok, walks, LA_list, dev, CTXLO)
    C.free(llama, ltok)
    print(f"[exp5] Qwen: capture @ LB={LB_list}", flush=True)
    qwen, qtok = C.load_model("Qwen", cfg)
    XB = capture_occ(qwen, qtok, walks, LB_list, dev, CTXLO)
    n_occ = next(iter(XA.values())).shape[0]
    print(f"[exp5] paired occurrences: {n_occ}", flush=True)

    # ---- A. geometric transport ----
    transport = []
    for rd, LA, LB in pairs:
        prep = fit_ridge(XA[LA], XB[LB], ALPHA)
        sad_t = transport_dir(prep, emo_l[SAD, LA])
        joy_t = transport_dir(prep, emo_l[JOY, LA])
        rec = {
            "reldepth": rd, "LA": LA, "LB": LB,
            "sad_matched": cos(sad_t, emo_q[SAD, LB]),      # sad_Llama -> sad_Qwen
            "sad_mismatched": cos(sad_t, emo_q[JOY, LB]),   # sad_Llama -> joy_Qwen (control)
            "joy_matched": cos(joy_t, emo_q[JOY, LB]),
            "joy_mismatched": cos(joy_t, emo_q[SAD, LB]),
        }
        transport.append(rec)
        print(f"[exp5] rd={rd} LA{LA}->LB{LB}: sad matched={rec['sad_matched']:+.3f} "
              f"mismatched={rec['sad_mismatched']:+.3f} | joy matched={rec['joy_matched']:+.3f}", flush=True)

    best = max(transport, key=lambda r: r["sad_matched"])
    LA_b, LB_b = best["LA"], best["LB"]
    prep_b = fit_ridge(XA[LA_b], XB[LB_b], ALPHA)
    sad_transported = C.unit(transport_dir(prep_b, emo_l[SAD, LA_b]))   # Qwen-space, from Llama
    sad_own = C.unit(emo_q[SAD, LB_b])
    rng = np.random.default_rng(0)
    rand_dir = C.unit(rng.standard_normal(sad_own.shape))
    print(f"[exp5] best transport pair LA{LA_b}->LB{LB_b}: sad matched={best['sad_matched']:+.3f}; "
          f"cos(transported, own Qwen sad)={cos(sad_transported, sad_own):+.3f}", flush=True)

    # ---- B. causal steer of Qwen with the transported direction ----
    # typical residual norm at LB_b for dose scaling
    ref = np.linalg.norm(XB[LB_b], axis=1).mean()
    meas_layer = min(nLB_emo - 1, LB_b + 4)
    sad_meas = {meas_layer: C.unit(emo_q[SAD, meas_layer])}
    prefixes = [C.mkwalk(wk.nodes[:120], graph).text for wk in walks[:NSTORY]]

    def story_sadness(steer_vec):
        smap = {LB_b: (STEER_DOSE * ref * C.unit(steer_vec)).astype(np.float32)} if steer_vec is not None else None
        projs, fracs = [], []
        blocks = C.decoder_blocks(qwen)
        for pref in prefixes:
            handles = C.steer_hooks(blocks, dev, smap) if smap else []
            try:
                enc = qtok(pref, add_special_tokens=True, return_tensors="pt").to(dev)
                gen = qwen.generate(enc["input_ids"], attention_mask=enc.get("attention_mask"),
                                    max_new_tokens=STORY_TOK, do_sample=True, temperature=1.0,
                                    top_p=0.95, pad_token_id=qtok.eos_token_id)
                story = qtok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            finally:
                for h in handles:
                    h.remove()
            fracs.append(C.sad_word_fraction(story))
            if story.strip():
                projs.append(C.project_residuals_on_dir(qwen, qtok, story, dev, sad_meas)[meas_layer])
        return {"sad_word_frac": float(np.nanmean(fracs)),
                "sad_proj": float(np.nanmean(projs)) if projs else float("nan")}

    causal = {
        "none":        story_sadness(None),
        "transported": story_sadness(sad_transported),   # Llama sad, mapped into Qwen
        "own_qwen":    story_sadness(sad_own),            # Qwen's own sad (upper bound)
        "random":      story_sadness(rand_dir),           # floor
    }
    for k, v in causal.items():
        print(f"[exp5] steer={k:11s} sad_word_frac={v['sad_word_frac']:.4f} sad_proj={v['sad_proj']:+.3f}", flush=True)
    C.free(qwen, qtok)

    out = {"graph": GRAPH, "n_occ": int(n_occ), "alpha": ALPHA, "pairs": transport,
           "best_pair": {"LA": LA_b, "LB": LB_b, "sad_matched": best["sad_matched"],
                         "cos_transported_vs_ownQwen": cos(sad_transported, sad_own)},
           "steer_dose": STEER_DOSE, "meas_layer": int(meas_layer), "causal": causal}
    json.dump(out, open(os.path.join(RUN_DIR, "exp5_emotion_injection.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp5_emotion_injection.pdf"))
    print(f"[exp5] DONE -> {RUN_DIR}/exp5_emotion_injection.json", flush=True)


def make_fig(out, path):
    P = out["pairs"]
    rd = [p["reldepth"] for p in P]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(rd, [p["sad_matched"] for p in P], "-o", color="tab:red", label="sad→sad (matched)")
        ax[0].plot(rd, [p["sad_mismatched"] for p in P], "--o", color="tab:red", alpha=.5, label="sad→joy (control)")
        ax[0].plot(rd, [p["joy_matched"] for p in P], "-o", color="tab:green", label="joy→joy (matched)")
        ax[0].axhline(0, color=".7", lw=.6); ax[0].set_ylim(-0.6, 1.0)
        ax[0].set_xlabel("relative depth (matched L_A, L_B)"); ax[0].set_ylabel("cosine after transport")
        ax[0].set_title("A. Does W transport the emotion axis? (Llama dir → Qwen dir)", fontsize=9)
        ax[0].legend(fontsize=8)
        cz = out["causal"]; order = ["none", "random", "transported", "own_qwen"]
        cols = ["0.6", "tab:blue", "tab:red", "k"]
        ax[1].bar(range(4), [cz[k]["sad_word_frac"] for k in order], color=cols)
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels(order, fontsize=8)
        ax[1].set_ylabel("Qwen story sad-word fraction")
        bp = out["best_pair"]
        ax[1].set_title(f"B. Steer Qwen@L{bp['LB']} with transported sad dir\n"
                        f"(cos vs own-Qwen sad={bp['cos_transported_vs_ownQwen']:+.2f})", fontsize=9)
        fig.suptitle(f"[{out['graph']}] Exp5 — does emotion cross by activation injection? "
                     f"transport map fit on {out['n_occ']} paired occ, α={out['alpha']:g}", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
