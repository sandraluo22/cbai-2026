"""Causal ('propagated') logit lens for the convergence direction: inject the layer-L
direction into the residual stream at layer L, let the REMAINING layers process it,
then read the output logits. delta = logits(patched) - logits(clean). Top tokens =
what moving along the direction at layer L promotes AFTER propagation through the rest
of the network -- more faithful than unembedding a mid-layer direction directly.

Averaged over a few representative game base-prompts. Magnitude = SCALE x the clean
last-position residual norm at layer L.

Env: MODEL(QwenInst32) DIR_NPZ LAYERS(24,40,48,56,64) SCALE(6) TOPK(12)
"""
from __future__ import annotations
import os
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
DIR_NPZ = os.environ.get("DIR_NPZ", "runs/game1_qwen32_pca_w2v/qwen32_convergence_dir.npz")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "24,40,48,56,64").split(",")]
SCALE = float(os.environ.get("SCALE", "6"))
TOPK = int(os.environ.get("TOPK", "12"))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    blocks = model.model.layers
    z = np.load(DIR_NPZ, allow_pickle=True)
    D1, D2 = z["dir1"], z["dir2"]; P1, P2 = [str(p) for p in z["players"]]

    # a few representative base prompts (game states ending in "My word:")
    hists = [[("cloud", "sand")],
             [("river", "sand"), ("sky", "stone")],
             [("moon", "star"), ("night", "sun"), ("dawn", "sky")]]
    prompts = [G.build_prompt(tok, h, {w for pr in h for w in pr}) for h in hists]
    ids_list = [tok(p, return_tensors="pt").input_ids.to(dev) for p in prompts]

    @torch.no_grad()
    def clean_logits_and_norm(ids, L):
        hs = model(ids, output_hidden_states=True)
        return hs.logits[0, -1].float(), float(hs.hidden_states[L + 1][0, -1].norm())

    def add_hook(vec):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[:, -1, :] = h[:, -1, :] + vec.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    @torch.no_grad()
    def patched_logits(ids, L, dvec, mag):
        vec = torch.tensor(dvec / (np.linalg.norm(dvec) + 1e-9), device=dev) * mag
        hd = blocks[L].register_forward_hook(add_hook(vec))
        try:
            return model(ids).logits[0, -1].float()
        finally:
            hd.remove()

    def lens(D, L):
        deltas = []
        for ids in ids_list:
            clean, norm = clean_logits_and_norm(ids, L)
            patched = patched_logits(ids, L, D[L], SCALE * norm)
            deltas.append((patched - clean).cpu())
        d = torch.stack(deltas).mean(0)
        top = torch.topk(d, TOPK).indices.tolist(); bot = torch.topk(-d, TOPK).indices.tolist()
        clean = lambda ids: [tok.decode([i]).strip() for i in ids]
        return clean(top), clean(bot)

    print(f"[prop-lens] {MODEL}; inject dir @L, propagate through remaining layers, read logits "
          f"(SCALE={SCALE}, oriented toward converged end)\n")
    for L in LAYERS:
        t1, b1 = lens(D1, L)
        t2, _ = lens(D2, L)
        print(f"=== layer {L} — {P1} ===")
        print(f"  promotes (converged end): {', '.join(w for w in t1 if w)}")
        print(f"  suppresses (early end)  : {', '.join(w for w in b1 if w)}")
        print(f"  ({P2} shares {len(set(t1)&set(t2))}/{TOPK} promoted)\n")


if __name__ == "__main__":
    main()
