"""FULL-GRID version of the bridge-head test: zero-ablate EVERY (layer, head) at the
answer position and measure d_bridge / d_cue on the kept cue-pair items (batched: one
forward per head, all items at once). Renders a slideshow of heatmaps:

  page 1: mean d_bridge over items   (drop in bridge-word logit when head is ablated)
  page 2: mean d_cue                 (drop in cue-word logits — copy contribution)
  page 3: bridge specificity d_bridge - d_cue
  page 4+: per-item d_bridge heatmaps
Top-32 partner-patch restoration heads are outlined on every page.

Env: MODEL(QwenInst32) PATCH_JSON BRIDGE_JSON(runs/qwen32_bridge_heads.json)
     OUT_NPZ(runs/qwen32_bridge_allheads.npz) OUT_PDF(runs/qwen32_bridge_allheads.pdf)
     DEVICE
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import llm_agents as LA
from qwen32_bridge_heads import OPEN_PROMPT, prompt_for, first_id

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON",
                            "runs/game-1/qwen32/qwen32_partner_patch/qwen32_partner_patch.json")
BRIDGE_JSON = os.environ.get("BRIDGE_JSON", "runs/qwen32_bridge_heads.json")
OUT_NPZ = os.environ.get("OUT_NPZ", "runs/qwen32_bridge_allheads.npz")
OUT_PDF = os.environ.get("OUT_PDF", "runs/qwen32_bridge_allheads.pdf")


def render(dB, dC, items, top_heads, out_pdf):
    """dB, dC: (nL, nH, nI). Slideshow of heatmaps with restoration heads outlined."""
    def page(pdf, M, title, cmap="RdBu_r"):
        fig, ax = plt.subplots(figsize=(11, 10))
        v = np.percentile(np.abs(M), 99.5) or 1e-6
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-v, vmax=v, origin="lower")
        for l, h in top_heads:
            ax.add_patch(Rectangle((h - .5, l - .5), 1, 1, fill=False, lw=1.2, ec="black"))
        ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with PdfPages(out_pdf) as pdf:
        page(pdf, dB.mean(-1), "mean d_bridge (logit drop of BRIDGE word when head ablated) "
                               "— boxes = top-32 restoration heads")
        page(pdf, dC.mean(-1), "mean d_cue (logit drop of CUE words — copy contribution)")
        page(pdf, dB.mean(-1) - dC.mean(-1), "bridge specificity: d_bridge − d_cue")
        for i, it in enumerate(items):
            page(pdf, dB[:, :, i], f"d_bridge — item {i}: {it['said']} + {it['you']} → {it['bridge']}")
    print(f"[allheads] wrote {out_pdf}", flush=True)


def main():
    import torch
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    nL = model.config.num_hidden_layers
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH

    items = json.load(open(BRIDGE_JSON))["kept_items"]
    prompts = [prompt_for(tok, it["said"], it["you"]) for it in items]
    bids = torch.tensor([first_id(tok, it["bridge"]) for it in items], device=dev)
    sids = torch.tensor([first_id(tok, it["said"]) for it in items], device=dev)
    yids = torch.tensor([first_id(tok, it["you"]) for it in items], device=dev)
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    nI = len(items)
    ii = torch.arange(nI, device=dev)

    state = {"head": None}
    def make_pre(layer):
        def pre(_m, args):
            if state["head"] is None or state["head"][0] != layer:
                return None
            x = args[0].clone()
            h = state["head"][1]
            x[:, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    @torch.no_grad()
    def batch_logits():
        return model(**enc).logits[:, -1, :].float()      # left-padded -> answer at -1

    state["head"] = None
    base = batch_logits()
    bB = base[ii, bids]; bS = base[ii, sids]; bY = base[ii, yids]

    dB = np.zeros((nL, nH, nI), dtype=np.float32)
    dC = np.zeros((nL, nH, nI), dtype=np.float32)
    for l in range(nL):
        for h in range(nH):
            state["head"] = (l, h)
            lg = batch_logits()
            dB[l, h] = (bB - lg[ii, bids]).cpu().numpy()
            dC[l, h] = (((bS - lg[ii, sids]) + (bY - lg[ii, yids])) / 2).cpu().numpy()
        print(f"[allheads] layer {l + 1}/{nL} done", flush=True)

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    order = np.argsort(R.flatten())[::-1][:32]
    top_heads = [(int(i // nH), int(i % nH)) for i in order]
    np.savez_compressed(OUT_NPZ, d_bridge=dB, d_cue=dC,
                        items=json.dumps(items), top_heads=np.array(top_heads))
    print(f"[allheads] wrote {OUT_NPZ}", flush=True)

    mB = dB.mean(-1)
    best = np.argsort(mB.flatten())[::-1][:10]
    print("[allheads] top-10 d_bridge heads overall:", flush=True)
    for i in best:
        l, h = int(i // nH), int(i % nH)
        star = " <-- top-32 restoration" if (l, h) in set(top_heads) else ""
        print(f"[allheads]   L{l} H{h}  d_bridge {mB[l, h]:+.4f}  d_cue {dC.mean(-1)[l, h]:+.4f}{star}",
              flush=True)
    render(dB, dC, items, top_heads, OUT_PDF)


if __name__ == "__main__":
    main()
