"""CONTROL: is the per-token firing profile of the DAS subspaces SPECIFIC to those subspaces, or does any
random subspace of the same head show equally reliable token structure?

For each Pile token we compute the energy share of the head's output inside:
  real_par   — the DAS parity rank-16 subspace of L14H26
  real_coord — the DAS coordinate rank-16 subspace of L21H10 (rot180-trained)
  randA_0..N — N random 16-dim subspaces of L14H26   (same head as parity)
  randB_0..N — N random 16-dim subspaces of L21H10   (same head as coord)
Per-token sums are accumulated SEPARATELY for even- and odd-indexed documents, so split-half reliability
of each subspace's token profile can be computed offline. Also accumulates token-level cross-products so
correlations between every pair of subspaces can be computed exactly (tests whether the parity/coord
"different populations" result is a subspace effect or merely a different-head effect).

Env: GEN_MODEL(Llama) NDOCS(1000) MAXTOK(384) NRAND(8) SEED(0) DATASET(NeelNanda/pile-10k)
     R_NPZ / COORD_NPZ as in pile_das_transcripts3  OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_random_subspace_control<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
from collections import defaultdict
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
NDOCS = int(os.environ.get("NDOCS", "1000")); MAXTOK = int(os.environ.get("MAXTOK", "384"))
NRAND = int(os.environ.get("NRAND", "8")); SEED = int(os.environ.get("SEED", "0"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
R_NPZ = os.environ.get("R_NPZ", f"{P}/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
COORD_NPZ = os.environ.get("COORD_NPZ", f"{P}/das_coord_R_{GEN_MODEL}.npz")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    _, hd = attn_proj(blocks[14], cm)
    slA = slice(26 * hd, 27 * hd)      # L14H26 (parity head)
    slB = slice(10 * hd, 11 * hd)      # L21H10 (coord head)

    rng = np.random.default_rng(SEED)
    def rand_sub(): return np.linalg.qr(rng.standard_normal((hd, 16)))[0].T
    subsA = [("real_par", np.load(R_NPZ)["4x4_R16"])] + [(f"randA_{i}", rand_sub()) for i in range(NRAND)]
    subsB = [("real_coord", np.load(COORD_NPZ)["4x4_R16"])] + [(f"randB_{i}", rand_sub()) for i in range(NRAND)]
    names = [n for n, _ in subsA] + [n for n, _ in subsB]
    A = torch.tensor(np.stack([v for _, v in subsA]), dtype=torch.float32, device=dev)   # [SA,16,hd]
    B = torch.tensor(np.stack([v for _, v in subsB]), dtype=torch.float32, device=dev)
    nS = len(names)
    print(f"[{tag}] {nS} subspaces: {names}", flush=True)

    caps = {}
    hooks = []
    for L in (14, 21):
        proj = attn_proj(blocks[L], cm)[0]
        def mk(L):
            def hh(_m, args): caps[L] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mk(L)))

    half_sum = [defaultdict(lambda: np.zeros(nS)), defaultdict(lambda: np.zeros(nS))]
    half_cnt = [defaultdict(int), defaultdict(int)]
    ssum = np.zeros(nS); scross = np.zeros((nS, nS)); ntok = 0

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True); cnt = 0
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        idl = ids[0].tolist(); caps.clear(); model(input_ids=ids)
        za = caps[14][0, :, slA].float(); zb = caps[21][0, :, slB].float()
        ea = (za ** 2).sum(1).clamp_min(1e-6); eb = (zb ** 2).sum(1).clamp_min(1e-6)
        shA = torch.einsum("srd,td->str", A, za).pow(2).sum(2) / ea            # [SA, T]
        shB = torch.einsum("srd,td->str", B, zb).pow(2).sum(2) / eb
        Sh = torch.cat([shA, shB], 0)[:, 1:].cpu().numpy()                     # [nS, T-1], skip BOS
        h = cnt % 2
        for j, i in enumerate(idl[1:]):
            t = tok.decode([i]); t = t if len(t) <= 14 else t[:14]
            half_sum[h][t] += Sh[:, j]; half_cnt[h][t] += 1
        ssum += Sh.sum(1); scross += Sh @ Sh.T; ntok += Sh.shape[1]
        cnt += 1
        if cnt % 200 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    for hk in hooks: hk.remove()

    toks = sorted(set(half_cnt[0]) | set(half_cnt[1]))
    out = {"model": tag, "ndocs": cnt, "ntok": int(ntok), "names": names, "nrand": NRAND,
           "mean": (ssum / ntok).round(5).tolist(),
           "cross": (scross / ntok).round(6).tolist(),
           "tokens": {t: {"n0": half_cnt[0].get(t, 0), "n1": half_cnt[1].get(t, 0),
                          "s0": half_sum[0][t].round(4).tolist() if t in half_sum[0] else None,
                          "s1": half_sum[1][t].round(4).tolist() if t in half_sum[1] else None}
                      for t in toks if half_cnt[0].get(t, 0) >= 20 and half_cnt[1].get(t, 0) >= 20}}
    p = f"{OUTDIR}/pile_random_subspace_control{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print(f"DONE -> {p}  ({len(out['tokens'])} tokens kept)", flush=True)


if __name__ == "__main__":
    main()
