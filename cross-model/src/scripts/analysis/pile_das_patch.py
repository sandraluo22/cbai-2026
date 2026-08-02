"""Causal test of the DAS parity subspace on NATURAL text. Ablate (project out) the learned rank-r subspace
of L14H26's output on Pile documents and measure the per-token next-token loss delta vs the clean pass.
If the grid-parity subspace really is a reused word-boundary feature, the loss damage should concentrate on
the same word_initial/continuation token classes that the projection analysis flagged — and beat a random
subspace of the same rank in the same head.

Conditions: das_r1, das_r16 (from the SAVE_R rotations npz), rand1, rand16 (fixed-seed orthonormal controls).

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NDOCS(150) MAXTOK(512) TOPK(25) WIN(14)
     DATASET(NeelNanda/pile-10k) R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_<model>.npz)
     RGRID(4x4) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_das_patch<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, heapq
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
NDOCS = int(os.environ.get("NDOCS", "150")); MAXTOK = int(os.environ.get("MAXTOK", "512"))
TOPK = int(os.environ.get("TOPK", "25")); WIN = int(os.environ.get("WIN", "14"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
R_NPZ = os.environ.get("R_NPZ", f"runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
RGRID = os.environ.get("RGRID", "4x4"); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)

    rz = np.load(R_NPZ)
    rng = np.random.default_rng(SEED)
    def orth(mat):
        q, _ = np.linalg.qr(mat.T); return q.T[:mat.shape[0]]
    subs = {"das_r1": rz[f"{RGRID}_R1"], "das_r16": rz[f"{RGRID}_R16"],
            "rand1": orth(rng.standard_normal((1, hd))), "rand16": orth(rng.standard_normal((16, hd)))}
    Pmats = {k: torch.tensor(v.T @ v, dtype=torch.float32, device=dev) for k, v in subs.items()}  # hd x hd projectors

    state = {"P": None}
    def patch_pre(_m, args):
        if state["P"] is not None:
            x = args[0].clone(); s = x[0, :, csl].float()
            x[0, :, csl] = (s - s @ state["P"]).to(x.dtype)
            return (x,) + tuple(args[1:])
    ph = proj.register_forward_pre_hook(patch_pre)

    def losses(ids):
        lg = model(input_ids=ids).logits[0].float()
        lsm = torch.log_softmax(lg[:-1], -1)
        return -lsm[torch.arange(ids.shape[1] - 1), ids[0, 1:]].cpu().numpy()   # loss[t] = NLL of token t+1

    import string as _string
    def categorize(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"
    def ctx_str(idl, t):
        a = max(0, t - WIN)
        return (tok.decode(idl[a:t + 1]) + "⟦" + tok.decode(idl[t + 1:t + 2]) + "⟧").replace("\n", "⏎")[-110:]

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)
    conds = list(subs)
    catsum = {c: {k: [0.0, 0] for k in ("word_initial", "continuation", "punct", "digit")} for c in conds}
    allsum = {c: [0.0, 0] for c in conds}; heaps = {c: [] for c in conds}; push = 0; cnt = 0
    def add(heap, key, ctx):
        nonlocal push; push += 1
        if len(heap) < TOPK: heapq.heappush(heap, (key, push, ctx))
        elif key > heap[0][0]: heapq.heapreplace(heap, (key, push, ctx))
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] < 12: continue
        idl = ids[0].tolist(); pieces = tok.convert_ids_to_tokens(idl)
        state["P"] = None; base = losses(ids)
        cats = [categorize(pieces[t + 1]) for t in range(len(idl) - 1)]     # category of the PREDICTED token
        for c in conds:
            state["P"] = Pmats[c]; d = losses(ids) - base
            for t in range(3, len(idl) - 1):
                catsum[c][cats[t]][0] += float(d[t]); catsum[c][cats[t]][1] += 1
                allsum[c][0] += float(d[t]); allsum[c][1] += 1
            for t in np.argsort(d[3:])[::-1][:5] + 3:
                add(heaps[c], float(d[t]), {"dloss": round(float(d[t]), 3), "ctx": ctx_str(idl, int(t)), "doc": cnt})
        state["P"] = None
        cnt += 1
        if cnt % 25 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    ph.remove()

    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "r_npz": R_NPZ, "rgrid": RGRID, "ndocs": cnt,
           "mean_dloss": {c: round(allsum[c][0] / max(allsum[c][1], 1), 5) for c in conds},
           "by_category": {c: {k: {"mean_dloss": round(v[0] / max(v[1], 1), 5), "n": v[1]}
                               for k, v in catsum[c].items()} for c in conds},
           "top_damaged": {c: [x for _, _, x in sorted(heaps[c], reverse=True)] for c in conds}}
    p = f"{OUTDIR}/pile_das_patch{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)
    for c in conds:
        print(f"[{c}] mean dloss={out['mean_dloss'][c]:+.4f}  by cat: "
              + "  ".join(f"{k}={v['mean_dloss']:+.4f}" for k, v in out['by_category'][c].items()), flush=True)


if __name__ == "__main__":
    main()
