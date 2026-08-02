"""H2a: induction scores for every (layer, query-head) of Qwen3-32B, and overlap with
the top partner-patch restoration heads — is the "partner-reading" circuit just
generic induction/copy machinery?

Induction score of a head = mean attention from position t in the SECOND copy of a
repeated random-token sequence to position t-L+1 (the token AFTER the previous
occurrence of the current token). Prev-token score (attention to t-1) for reference.
Requires eager attention (output_attentions).

Env: MODEL(QwenInst32) PATCH_JSON L_HALF(64) N_SEQ(24) BATCH(8) TOPN(32) SEED(0)
     OUT_JSON DEVICE
"""
from __future__ import annotations
import os
import json
import numpy as np
from llm_agents import SPEC

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON",
                            "runs/game-1/qwen32/qwen32_partner_patch/qwen32_partner_patch.json")
L_HALF = int(os.environ.get("L_HALF", "64"))
N_SEQ = int(os.environ.get("N_SEQ", "24"))
BATCH = int(os.environ.get("BATCH", "8"))
TOPN = int(os.environ.get("TOPN", "32"))
SEED = int(os.environ.get("SEED", "0"))
OUT_JSON = os.environ.get("OUT_JSON", "runs/qwen32_induction_overlap.json")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    hf, mirror = SPEC[MODEL]
    name = hf or mirror
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    print(f"[ind] {MODEL} eager: {nL} layers x {nH} q-heads", flush=True)

    rng = np.random.default_rng(SEED)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ind = np.zeros((nL, nH)); prev = np.zeros((nL, nH)); nb = 0
    with torch.no_grad():
        for b0 in range(0, N_SEQ, BATCH):
            bs = min(BATCH, N_SEQ - b0)
            x = rng.integers(1000, 50000, size=(bs, L_HALF))
            ids = np.concatenate([np.full((bs, 1), bos), x, x], axis=1)
            out = model(torch.tensor(ids, device=dev), output_attentions=True)
            # positions: bos=0, copy1=1..L, copy2=L+1..2L; query t in copy2 (skip first)
            qs = np.arange(L_HALF + 2, 2 * L_HALF + 1)
            for li, att in enumerate(out.attentions):     # (bs, nH, T, T)
                a = att.float().cpu().numpy()
                ind[li] += a[:, :, qs, qs - L_HALF + 1].mean(axis=(0, 2))
                prev[li] += a[:, :, qs, qs - 1].mean(axis=(0, 2))
            nb += 1
            print(f"[ind] batch {nb} done", flush=True)
    ind /= nb; prev /= nb

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    flat = np.argsort(R.flatten())[::-1][:TOPN]
    heads = [(int(i // R.shape[1]), int(i % R.shape[1])) for i in flat]
    ind_flat = ind.flatten()
    pct = lambda v: float((ind_flat < v).mean())
    top = [{"layer": l, "head": h, "restoration": float(R[l, h]),
            "induction": float(ind[l, h]), "induction_pctile": pct(ind[l, h]),
            "prev_token": float(prev[l, h])} for l, h in heads]
    n_top5 = sum(t["induction_pctile"] >= 0.95 for t in top)
    best_ind = np.argsort(ind_flat)[::-1][:10]
    print(f"[ind] top-{TOPN} restoration heads: mean induction "
          f"{np.mean([t['induction'] for t in top]):.4f} vs all-head mean {ind.mean():.4f}; "
          f"{n_top5}/{TOPN} are top-5% induction heads", flush=True)
    json.dump({"model": MODEL, "l_half": L_HALF, "n_seq": N_SEQ,
               "induction": ind.tolist(), "prev_token": prev.tolist(),
               "top_restoration_heads": top, "n_in_top5pct_induction": n_top5,
               "top10_induction_heads": [[int(i // nH), int(i % nH), float(ind_flat[i])]
                                          for i in best_ind]},
              open(OUT_JSON, "w"))
    print("[ind] wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
