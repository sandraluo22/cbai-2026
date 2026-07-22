"""Logit-lens the convergence direction: project each layer's direction vector through
the model's final norm + unembedding to see which TOKENS it promotes (positive end =
what 'converging' moves toward) and suppresses (negative end).

Caveat: applying a mid-layer direction through the FINAL unembedding is the standard
logit-lens approximation -- most faithful at deep layers.

Env: MODEL(QwenInst32) DIR_NPZ LAYERS(24,40,48,56,64) TOPK(12)
"""
from __future__ import annotations
import os
import numpy as np
import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
DIR_NPZ = os.environ.get("DIR_NPZ", "runs/game1_qwen32_pca_w2v/qwen32_convergence_dir.npz")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "24,40,48,56,64").split(",")]
TOPK = int(os.environ.get("TOPK", "12"))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    WU = model.get_output_embeddings().weight            # (vocab, hidden)
    gamma = None
    for path in ("model.norm.weight",):                  # final RMSNorm scale
        try:
            gamma = dict(model.named_parameters())[path]
        except KeyError:
            pass
    z = np.load(DIR_NPZ, allow_pickle=True)
    D1, D2 = z["dir1"], z["dir2"]
    P1, P2 = [str(p) for p in z["players"]]

    def lens(dvec):
        v = torch.tensor(dvec, device=dev, dtype=WU.dtype)
        if gamma is not None:
            v = v * gamma                                # apply final-norm scaling
        scores = (WU @ v).float().cpu()
        top = torch.topk(scores, TOPK).indices.tolist()
        bot = torch.topk(-scores, TOPK).indices.tolist()
        clean = lambda ids: [tok.decode([i]).strip() for i in ids]
        return clean(top), clean(bot)

    print(f"[lens] {MODEL}; direction oriented toward INCREASING turn (converged end)\n")
    for L in LAYERS:
        t1, b1 = lens(D1[L])
        print(f"=== layer {L} — {P1} ===")
        print(f"  promotes (converged end): {', '.join(w for w in t1 if w)[:200]}")
        print(f"  suppresses (early end)  : {', '.join(w for w in b1 if w)[:200]}")
        # sanity: Qwen2 should be ~identical
        t2, _ = lens(D2[L])
        shared = len(set(t1) & set(t2))
        print(f"  ({P2} shares {shared}/{TOPK} promoted tokens with {P1})\n")


if __name__ == "__main__":
    main()
