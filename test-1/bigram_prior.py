"""Measure Llama's pretrained bigram prior between all concept-word pairs in the
cross-model WORDS pool (the 'paper tiger' problem). For each single-token word w1,
run [BOS, " w1"] and record the next-token logits of every candidate " w2". Saved raw;
the assignment search normalizes offline.

Out: <OUTDIR>/bigram_prior.npz  (words, logits [n, n])
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config, WORDS
import models as M

OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out"))
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
    model = tok = None
    for name in MODEL_CANDS:
        try:
            print(f"loading {name}", flush=True); model, tok = M.load_model(name, cfg); break
        except Exception as e:
            print(f"  failed: {e}", flush=True)
    bos = tok.bos_token_id

    words, tids = [], []
    for w in WORDS:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            words.append(w); tids.append(ids[0])
        else:
            print(f"skip multi-token: {w} -> {ids}", flush=True)
    n = len(words)
    ids = torch.tensor([[bos, t] for t in tids], device=DEVICE)
    out = model(input_ids=ids)
    lg = out.logits[:, -1, :].float()                       # [n, vocab]
    L = lg[:, torch.tensor(tids, device=DEVICE)].cpu().numpy()   # [n, n] w1 -> w2 logits
    np.savez_compressed(os.path.join(OUTDIR, "bigram_prior.npz"),
                        words=np.array(words), logits=L)
    print(f"DONE n={n} -> {OUTDIR}/bigram_prior.npz", flush=True)


if __name__ == "__main__":
    main()
