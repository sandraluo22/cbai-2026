"""Per-token DAS firing over MANY Pile documents (v3): parity subspaces PLUS the coordinate
subspace (L21H10 rank-16, rot180-trained). Per token: share of head-output energy in (a) the single-head rank-16 subspace (L14H26),
(b) the jointly-trained PAIR subspaces (L14H26 + L14H19, energy pooled over both heads), (c) the
CONCAT rank-32 subspace over [L2H26; L14H26; L14H19] (384-d), plus the signed rank-1 DAS projection.

Env: GEN_MODEL(Llama) NDOCS(1000) MAXTOK(384) DATASET(NeelNanda/pile-10k)
     R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_<model>.npz)
     PAIR_NPZ(runs/axes/4_circuits/parity/das_multihead_block_L14H26-L14H19_save_<model>.npz)
     CAT_NPZ(runs/axes/4_circuits/parity/das_multihead_concat_L2H26-L14H26-L14H19_save_<model>.npz)
     DAS_NPZ(runs/axes/4_circuits/das/das_grid_patch_<model>_L14H26.npz) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_das_transcripts3<OUTTAG>_<model>.json
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
NDOCS = int(os.environ.get("NDOCS", "1000")); MAXTOK = int(os.environ.get("MAXTOK", "384"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
R_NPZ = os.environ.get("R_NPZ", f"{P}/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
PAIR_NPZ = os.environ.get("PAIR_NPZ", f"{P}/das_multihead_block_L14H26-L14H19_save_{GEN_MODEL}.npz")
CAT_NPZ = os.environ.get("CAT_NPZ", f"{P}/das_multihead_concat_L2H26-L14H26-L14H19_save_{GEN_MODEL}.npz")
COORD_NPZ = os.environ.get("COORD_NPZ", f"{P}/das_coord_R_{GEN_MODEL}.npz")
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L14H26.npz")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    _, hd = attn_proj(blocks[14], cm)
    sl = {"a": slice(26 * hd, 27 * hd), "b": slice(19 * hd, 20 * hd), "e": slice(26 * hd, 27 * hd), "u": slice(10 * hd, 11 * hd)}  # a=14:26 b=14:19 e=2:26 u=21:10

    R16 = torch.tensor(np.load(R_NPZ)["4x4_R16"], dtype=torch.float32, device=dev)
    pz = np.load(PAIR_NPZ)["4x4_r16"]                     # (2, 16*128) flattened per head
    Ra = torch.tensor(pz[0].reshape(16, hd), dtype=torch.float32, device=dev)
    Rb = torch.tensor(pz[1].reshape(16, hd), dtype=torch.float32, device=dev)
    Rcat = torch.tensor(np.load(CAT_NPZ)["4x4_r32"], dtype=torch.float32, device=dev)   # (32, 384)
    Rco = torch.tensor(np.load(COORD_NPZ)["4x4_R16"], dtype=torch.float32, device=dev)
    dz = np.load(DAS_NPZ)
    das1 = dz["global_R1"][0].astype(np.float64); das1 /= np.linalg.norm(das1)
    proto = dz["proto_delta"].astype(np.float64)
    if das1 @ (proto / np.linalg.norm(proto)) < 0: das1 = -das1
    d1 = torch.tensor(das1, dtype=torch.float32, device=dev)

    caps = {}
    hooks = []
    for L in (2, 14, 21):
        proj = attn_proj(blocks[L], cm)[0]
        def mk(L):
            def hh(_m, args): caps[L] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mk(L)))

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
        idl = ids[0].tolist(); caps.clear(); model(input_ids=ids)
        za = caps[14][0, :, sl["a"]].float(); zb = caps[14][0, :, sl["b"]].float(); ze = caps[2][0, :, sl["e"]].float(); zu = caps[21][0, :, sl["u"]].float()
        ea = (za ** 2).sum(1); eb = (zb ** 2).sum(1); ee = (ze ** 2).sum(1)
        s16 = ((za @ R16.t()).pow(2).sum(1) / ea.clamp_min(1e-6)).cpu().numpy()
        p1 = (za @ d1).cpu().numpy()
        qpair = (((za @ Ra.t()).pow(2).sum(1) + (zb @ Rb.t()).pow(2).sum(1)) / (ea + eb).clamp_min(1e-6)).cpu().numpy()
        zcat = torch.cat([ze, za, zb], dim=1)
        rcat = ((zcat @ Rcat.t()).pow(2).sum(1) / (zcat ** 2).sum(1).clamp_min(1e-6)).cpu().numpy()
        kco = ((zu @ Rco.t()).pow(2).sum(1) / (zu ** 2).sum(1).clamp_min(1e-6)).cpu().numpy()
        pieces = tok.convert_ids_to_tokens(idl)
        meta = ex.get("meta", {}); src = meta.get("pile_set_name", "") if isinstance(meta, dict) else ""
        docs.append({"i": cnt, "src": src, "t": [tok.decode([i]) for i in idl[1:]],
                     "s": [round(float(x), 3) for x in s16[1:]],
                     "p": [round(float(x), 2) for x in p1[1:]],
                     "q": [round(float(x), 3) for x in qpair[1:]],
                     "r": [round(float(x), 3) for x in rcat[1:]],
                     "k": [round(float(x), 3) for x in kco[1:]],
                     "c": "".join(catc(pc) for pc in pieces[1:])})
        cnt += 1
        if cnt % 100 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    for h in hooks: h.remove()
    out = {"model": tag, "ndocs": cnt, "maxtok": MAXTOK,
           "chance": {"s": round(16 / hd, 4), "q": round(32 / (2 * hd), 4), "r": round(32 / (3 * hd), 4), "k": round(16 / hd, 4)},
           "docs": docs}
    pth = f"{OUTDIR}/pile_das_transcripts3{OUTTAG}_{tag}.json"
    json.dump(out, open(pth, "w"), ensure_ascii=False)
    print(f"DONE -> {pth}", flush=True)


if __name__ == "__main__":
    main()
