"""Dump per-token DAS-subspace firing over Pile documents for the transcript viewer: for each token,
the fraction of L14H26's output energy inside the rank-16 DAS parity subspace (share16), the signed
projection on the rank-1 DAS direction (aligned to proto_delta), and the head-output norm. One JSON
with token strings so a standalone HTML page can render highlighted transcripts.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NDOCS(100) MAXTOK(512) DATASET(NeelNanda/pile-10k)
     R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_<model>.npz) RGRID(4x4)
     DAS_NPZ(runs/axes/4_circuits/das/das_grid_patch_<model>_L<l>H<h>.npz) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_das_transcripts<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
NDOCS = int(os.environ.get("NDOCS", "100")); MAXTOK = int(os.environ.get("MAXTOK", "512"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
R_NPZ = os.environ.get("R_NPZ", f"runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
RGRID = os.environ.get("RGRID", "4x4")
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L{HEAD_LAYER}H{HEAD_IDX}.npz")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)

    R16 = torch.tensor(np.load(R_NPZ)[f"{RGRID}_R16"], dtype=torch.float32, device=dev)
    dz = np.load(DAS_NPZ)
    das1 = dz["global_R1"][0].astype(np.float64); das1 /= np.linalg.norm(das1)
    proto = dz["proto_delta"].astype(np.float64)
    if das1 @ (proto / np.linalg.norm(proto)) < 0: das1 = -das1
    d1 = torch.tensor(das1, dtype=torch.float32, device=dev)

    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hk = proj.register_forward_pre_hook(cap)

    import string as _string
    def catc(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "p"
        if core and all(ch.isdigit() for ch in core): return "d"
        return "w" if wi else "c"

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)
    docs = []; cnt = 0
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        idl = ids[0].tolist(); zc.clear(); model(input_ids=ids)
        Z = zc["z"][0, :, csl].float()
        tot = (Z ** 2).sum(1)
        sh16 = ((Z @ R16.t()).pow(2).sum(1) / tot.clamp_min(1e-6)).cpu().numpy()
        p1 = (Z @ d1).cpu().numpy()
        pieces = tok.convert_ids_to_tokens(idl)
        toks = [tok.decode([i]) for i in idl]
        meta = ex.get("meta", {}); src = meta.get("pile_set_name", "") if isinstance(meta, dict) else ""
        docs.append({"i": cnt, "src": src,
                     "t": toks[1:],                                      # skip BOS
                     "s": [round(float(x), 3) for x in sh16[1:]],
                     "p": [round(float(x), 2) for x in p1[1:]],
                     "c": "".join(catc(pc) for pc in pieces[1:])})
        cnt += 1
        if cnt % 25 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    hk.remove()
    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "rgrid": RGRID, "ndocs": cnt,
           "expected_random_share16": round(16 / hd, 4), "docs": docs}
    p = f"{OUTDIR}/pile_das_transcripts{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
