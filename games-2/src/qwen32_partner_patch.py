"""CAUSAL localization of the co-adaptation (partner-reading) circuit in the word-
convergence game -- the head-resolved analog of the KL-coupling instrument.

Setup (denoising / interchange patch). For a real game state we build two prompts to
player A that are IDENTICAL except the partner's last revealed word:
    CLEAN     : "...Round t: the other player said <REAL>, you said <s>. \n My word:"
    CORRUPTED : "...Round t: the other player said <GHOST>, you said <s>. \n My word:"
<REAL> = the actual co-adaptive partner word; <GHOST> = a word from the NON-ADAPTIVE
fixed-partner (ghost) pool. Both are single tokens, so the two prompts align position-
for-position. Behaviour metric at the answer position:
    M = logit(REAL) - logit(GHOST)
The recently-mentioned partner word is elevated by the partner-reading heads, so
M_clean >> M_corrupted; the SIZE of that gap is exactly this state's coupling. We then
patch each attention head's o_proj INPUT columns (head h owns [h*hd:(h+1)*hd]) at the
answer position from the clean run into the corrupted run and measure how much of the
gap it restores:
    restoration = (M_patched - M_corr) / (M_clean - M_corr)
Heads with high mean restoration ARE the circuit that reads the partner and drives the
co-adaptive move -- i.e. what makes 'real partner' differ from 'ghost'.

Env: MODEL(QwenInst32) N_STATES(24) MARGIN(1.0) SEED(0) LAYERS(all) RUN_DIR DEVICE
Out: <RUN_DIR>/qwen32_partner_patch.json  (+ .pdf heatmap; top heads printed)
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
N_STATES = int(os.environ.get("N_STATES", "24"))
MARGIN = float(os.environ.get("MARGIN", "1.0"))       # keep states whose clean-corrupt gap exceeds this (real coupling)
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/game-1/qwen32/qwen32_partner_patch")
COADAPT = os.environ.get("COADAPT", "runs/game-1/qwen32/qwen32_pca_w2v/qwen32_pca_transcript.json")
GHOSTTR = os.environ.get("GHOSTTR", "runs/game-1/qwen32/qwen32_fixed_partner/qwen32_fixed_transcript.json")

OPEN_PROMPT = ("You are playing a word game with another player. Each round, you both say one word at the "
               "same time, then the two words are revealed. You win the round only if you both said the same "
               "word. You may not say any word that has already been said by either player.")


def build_prompt(tok, hist):
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}." for k, (o, s) in enumerate(hist))
    return LA._render(tok, OPEN_PROMPT + ((" " + lines) if lines else "")) + "\nMy word:"


def rollouts_from_coadapt(path):
    """Segment the co-adaptive transcript into rollouts (turn resets to 1) and return,
    per rollout, A's history of (other_word, self_word) pairs -- start pair first."""
    rows = json.load(open(path))
    P1, P2 = MODEL + "_1", MODEL + "_2"
    outs, cur = [], None
    for r in rows:
        if r["turn"] == 1 or "start" in r:
            if cur:
                outs.append(cur)
            sa, sb = r["start"]                        # P1 start, P2 start
            cur = [(sb, sa)]                           # A=P1: "other said sb, you said sa"
        cur.append((r["picks"][P2], r["picks"][P1]))
    if cur:
        outs.append(cur)
    return outs


def one_tok(tok, w):
    return len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1


def first_id(tok, w):
    return tok(" " + w, add_special_tokens=False)["input_ids"][0]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    nL = model.config.num_hidden_layers
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH
    print(f"[patch] {MODEL}: {nL} layers, {nH} heads, head_dim {hd} (o_proj in {nH*hd})", flush=True)

    # ---- ghost word pool (single-token, non-adaptive) ----
    ghost_rows = json.load(open(GHOSTTR))
    ghost_pool = sorted({r["ghost"] for r in ghost_rows if one_tok(tok, r["ghost"])})
    rng = np.random.default_rng(SEED)

    # ---- assemble states: (hist with REAL last word, hist with GHOST last word) ----
    states = []
    for roll in rollouts_from_coadapt(COADAPT):
        for i in range(1, len(roll)):                 # state = history up through pair i (>=2 partner reveals)
            o_real, s = roll[i]
            if not one_tok(tok, o_real):
                continue
            o_ghost = str(rng.choice(ghost_pool))
            if o_ghost == o_real:
                continue
            hist = list(roll[: i + 1])
            hist_ghost = hist[:-1] + [(o_ghost, s)]
            states.append({"clean": build_prompt(tok, hist), "corr": build_prompt(tok, hist_ghost),
                           "id_real": first_id(tok, o_real), "id_ghost": first_id(tok, o_ghost),
                           "real": o_real, "ghost": o_ghost})
    rng.shuffle(states)

    # ---- keep states whose clean/corrupt gap is a real coupling signal (batched screen) ----
    @torch.no_grad()
    def metric(prompts, id_real, id_ghost):
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        lg = model(**enc).logits[:, -1, :].float()    # left-padded -> answer at -1
        ii = torch.arange(len(prompts))
        return (lg[ii, id_real] - lg[ii, id_ghost]).cpu().numpy(), enc

    kept = []
    for s in states:
        if len(kept) >= N_STATES:
            break
        idr, idg = torch.tensor([s["id_real"]]), torch.tensor([s["id_ghost"]])
        mc, _ = metric([s["clean"]], idr, idg)
        mk, _ = metric([s["corr"]], idr, idg)
        if float(mc - mk) > MARGIN:                    # real word elevated over ghost -> usable coupling
            s["m_clean"], s["m_corr"] = float(mc), float(mk)
            kept.append(s)
    print(f"[patch] kept {len(kept)}/{N_STATES} states (gap>{MARGIN}); "
          f"mean gap {np.mean([s['m_clean']-s['m_corr'] for s in kept]):.2f}", flush=True)

    clean_prompts = [s["clean"] for s in kept]
    corr_prompts = [s["corr"] for s in kept]
    id_real = torch.tensor([s["id_real"] for s in kept])
    id_ghost = torch.tensor([s["id_ghost"] for s in kept])
    m_clean = np.array([s["m_clean"] for s in kept])
    m_corr = np.array([s["m_corr"] for s in kept])
    denom = np.maximum(m_clean - m_corr, 1e-6)

    # ---- cache CLEAN o_proj-input at the answer position, every layer ----
    cache = {}

    def cap_hook(L):
        def hook(mod, args):
            cache[L] = args[0][:, -1, :].detach().clone()
        return hook

    handles = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(cap_hook(L)) for L in range(nL)]
    with torch.no_grad():
        enc_clean = tok(clean_prompts, return_tensors="pt", padding=True).to(dev)
        model(**enc_clean)
    for h in handles:
        h.remove()

    enc_corr = tok(corr_prompts, return_tensors="pt", padding=True).to(dev)
    ii = torch.arange(len(kept))

    @torch.no_grad()
    def patched_metric(L, cols):
        z = cache[L]

        def hook(mod, args):
            x = args[0].clone()
            x[:, -1, cols] = z[:, cols]
            return (x,) + args[1:]
        hh = model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook)
        try:
            lg = model(**enc_corr).logits[:, -1, :].float()
        finally:
            hh.remove()
        m = (lg[ii, id_real] - lg[ii, id_ghost]).cpu().numpy()
        return float(np.mean((m - m_corr) / denom))

    @torch.no_grad()
    def patched_group(heads):
        """Patch a SET of (layer, head) o_proj-input columns (answer pos) together."""
        bylayer = {}
        for L, h in heads:
            bylayer.setdefault(L, []).append(h)
        hooks = []
        for L, hs in bylayer.items():
            z = cache[L]
            cols = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in hs])
            cols = torch.tensor(cols, device=z.device)

            def mk(z=z, cols=cols):
                def hook(mod, args):
                    x = args[0].clone(); x[:, -1, cols] = z[:, cols]
                    return (x,) + args[1:]
                return hook
            hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk()))
        try:
            lg = model(**enc_corr).logits[:, -1, :].float()
        finally:
            for h in hooks:
                h.remove()
        m = (lg[ii, id_real] - lg[ii, id_ghost]).cpu().numpy()
        return float(np.mean((m - m_corr) / denom))

    # ---- VALIDATE: do the top-K heads jointly restore the gap, and beat random-K? ----
    if os.environ.get("VALIDATE"):
        top = json.load(open(os.path.join(RUN_DIR, "qwen32_partner_patch.json")))["top_heads"]
        rgen = np.random.default_rng(SEED + 7)
        allheads = [(L, h) for L in range(nL) for h in range(nH)]
        print("[patch] VALIDATE joint restoration (top-K vs random-K):", flush=True)
        val = {}
        for K in [4, 8, 16, 32]:
            topK = [(L, h) for L, h, _ in top[:K]]
            rndK = [allheads[i] for i in rgen.choice(len(allheads), K, replace=False)]
            rt, rr = patched_group(topK), patched_group(rndK)
            val[K] = {"top": rt, "rand": rr}
            print(f"    K={K:2d}:  top {rt:+.3f}   random {rr:+.3f}", flush=True)
        json.dump(val, open(os.path.join(RUN_DIR, "qwen32_partner_patch_validate.json"), "w"), indent=1)
        print(f"[patch] VALIDATE done -> {RUN_DIR}", flush=True); return

    layers = [int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else list(range(nL))

    # ---- LAYER sweep first: patch the WHOLE attn output (all heads) at the answer pos ----
    RL = np.full(nL, np.nan, np.float32)
    for L in layers:
        RL[L] = patched_metric(L, slice(0, nH * hd))
    order = np.argsort(-np.nan_to_num(RL))
    print("[patch] LAYER sweep (whole-attn restoration) top 10:", flush=True)
    for L in order[:10]:
        print(f"    layer {int(L):2d}: {RL[L]:+.3f}", flush=True)
    if os.environ.get("LAYER_SWEEP"):                  # diagnostic-only: stop before the head sweep
        json.dump({"model": MODEL, "n_states": len(kept), "layer_restore": RL.tolist(),
                   "m_clean_mean": float(m_clean.mean()), "m_corr_mean": float(m_corr.mean())},
                  open(os.path.join(RUN_DIR, "qwen32_partner_layer_sweep.json"), "w"), indent=1)
        print(f"[patch] LAYER_SWEEP done -> {RUN_DIR}", flush=True); return
    R = np.full((nL, nH), np.nan, np.float32)
    for L in layers:
        for h in range(nH):
            R[L, h] = patched_metric(L, slice(h * hd, (h + 1) * hd))
        top = np.nanargmax(R[L]); print(f"[patch] layer {L:2d}: best head {top} restore {R[L, top]:.3f}", flush=True)

    flat = [(L, h, float(R[L, h])) for L in layers for h in range(nH)]
    flat.sort(key=lambda x: -x[2])
    print("\n[patch] TOP 15 HEADS (layer, head, restoration frac):")
    for L, h, v in flat[:15]:
        print(f"    L{L}H{h}: {v:+.3f}", flush=True)

    json.dump({"model": MODEL, "n_states": len(kept), "margin": MARGIN,
               "m_clean_mean": float(m_clean.mean()), "m_corr_mean": float(m_corr.mean()),
               "restoration": R.tolist(), "top_heads": flat[:40]},
              open(os.path.join(RUN_DIR, "qwen32_partner_patch.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"{MODEL}: partner-word patch restoration (real vs ghost), n={len(kept)} states")
    fig.colorbar(im, label="fraction of coupling gap restored")
    for L, h, v in flat[:8]:
        ax.add_patch(plt.Rectangle((h - .5, L - .5), 1, 1, fill=False, ec="k", lw=1.5))
    fig.tight_layout(); fig.savefig(os.path.join(RUN_DIR, "qwen32_partner_patch.pdf"))
    print(f"[patch] DONE -> {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
