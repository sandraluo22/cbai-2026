"""Fresh-context control: what geometry does the jointly-generated stream ALONE induce?

Rebuilds each pair's 600-word joint stream from out/gen_log.json, feeds it to a BLANK
Llama-3.1-8B context (no grid/ring priming), and captures per-node mean residuals at every
decoder block in the same word-step windows as the joint phase of the main run
(early 0-100, mid 100-300, late 300-600). If the fresh geometry matches the primed
contexts' joint_late geometry, the converged hybrid is fully determined by the joint
stream; any surplus own-graph structure in the primed contexts is hysteresis.

Env: OUTDIR(out) CM_SRC DEVICE(cuda)
Out: <OUTDIR>/nodemeans_fresh.npz
"""
from __future__ import annotations
import os, sys, json
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import models as M

OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out"))
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WINDOWS = {"early": (0, 100), "mid": (100, 300), "late": (300, 600)}


@torch.no_grad()
def main():
    log = json.load(open(os.path.join(OUTDIR, "gen_log.json")))
    words = log["words"]; n = len(words)
    P, T = log["npairs"], log["tgen"]
    joint = np.array([[log["steps"][f"pair{p}"][t]["node"] for t in range(T)]
                      for p in range(P)])

    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
    model = tok = None
    for name in MODEL_CANDS:
        try:
            print(f"loading {name}", flush=True); model, tok = M.load_model(name, cfg); break
        except Exception as e:
            print(f"  failed: {e}", flush=True)
    assert model is not None
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    bos = tok.bos_token_id
    cand = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1
        cand.append(ids[0])

    grabbed = {}
    def mk(L):
        def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]

    nsum = {w: {L: np.zeros((n, cm.hidden_size)) for L in range(nL)} for w in WINDOWS}
    ncnt = {w: np.zeros(n) for w in WINDOWS}
    try:
        for p in range(P):
            ids = torch.tensor([[bos] + [cand[nd] for nd in joint[p]]], device=DEVICE)
            grabbed.clear()
            try:
                model(input_ids=ids, logits_to_keep=1)
            except TypeError:
                model(input_ids=ids)
            for wname, (lo, hi) in WINDOWS.items():
                pos = list(range(1 + lo, 1 + hi))
                nds = list(joint[p][lo:hi])
                for L in range(nL):
                    rows = grabbed[L][0][pos].float().cpu().numpy()
                    np.add.at(nsum[wname][L], nds, rows)
                np.add.at(ncnt[wname], nds, 1.0)
            print(f"fresh pair {p} done", flush=True)
    finally:
        for h in handles:
            h.remove()

    save = {"n_layers": np.array([nL]), "words": np.array(words)}
    for wname in WINDOWS:
        cnt = np.maximum(ncnt[wname], 1.0)
        for L in range(nL):
            save[f"fresh_{wname}_layer_{L}"] = (nsum[wname][L] / cnt[:, None]).astype(np.float16)
        save[f"fresh_{wname}_ncnt"] = ncnt[wname]
    path = os.path.join(OUTDIR, "nodemeans_fresh.npz")
    np.savez_compressed(path, **save)
    print(f"DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
