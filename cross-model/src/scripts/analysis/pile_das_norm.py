"""How much does L14H26's output FIRE inside the DAS parity subspace on natural text? No ablation —
just measure, per Pile token, the fraction of the head-output energy that lands in the rank-16 (and
rank-1) DAS subspace: share_r = ||P_r z||^2 / ||z||^2. A random rank-r subspace captures r/hd of the
energy in expectation (0.125 for r=16), so shares above that mean the head is genuinely firing along
the parity subspace. Aggregated by word-boundary token category, with NRANDSUB random-subspace nulls
carried through the same pass, plus top-firing contexts.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NDOCS(200) MAXTOK(512) TOPK(30) WIN(14)
     DATASET(NeelNanda/pile-10k) R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_<model>.npz)
     RGRID(4x4) NRANDSUB(16) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_das_norm<OUTTAG>_<model>.json
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
NDOCS = int(os.environ.get("NDOCS", "200")); MAXTOK = int(os.environ.get("MAXTOK", "512"))
TOPK = int(os.environ.get("TOPK", "30")); WIN = int(os.environ.get("WIN", "14"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
R_NPZ = os.environ.get("R_NPZ", f"runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
RGRID = os.environ.get("RGRID", "4x4"); NRANDSUB = int(os.environ.get("NRANDSUB", "16"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)

    rz = np.load(R_NPZ); rng = np.random.default_rng(SEED)
    R16 = torch.tensor(rz[f"{RGRID}_R16"], dtype=torch.float32, device=dev)
    R1 = torch.tensor(rz[f"{RGRID}_R1"], dtype=torch.float32, device=dev)
    rands = []
    for _ in range(NRANDSUB):
        q, _r = np.linalg.qr(rng.standard_normal((hd, 16)))
        rands.append(torch.tensor(q.T, dtype=torch.float32, device=dev))

    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hk = proj.register_forward_pre_hook(cap)

    import string as _string
    def categorize(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"
    def ctx_str(idl, t):
        a = max(0, t - WIN)
        return (tok.decode(idl[a:t]) + "⟦" + tok.decode(idl[t:t + 1]) + "⟧").replace("\n", "⏎")[-100:]

    cats = ("word_initial", "continuation", "punct", "digit")
    # sums of energy per category: total, das16, das1, and per random subspace
    E = {k: {"tot": 0.0, "das16": 0.0, "das1": 0.0, "rand": np.zeros(NRANDSUB), "n": 0} for k in cats}
    share_vals = {k: [] for k in cats}          # token-level share_16 for distribution stats
    top = []; push = 0
    def add(key, ctx):
        nonlocal push; push += 1
        if len(top) < TOPK: heapq.heappush(top, (key, push, ctx))
        elif key > top[0][0]: heapq.heapreplace(top, (key, push, ctx))

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True); cnt = 0
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        idl = ids[0].tolist(); zc.clear(); model(input_ids=ids)
        Z = zc["z"][0, :, csl].float()
        tot = (Z ** 2).sum(1)
        e16 = (Z @ R16.t()).pow(2).sum(1); e1 = (Z @ R1.t()).pow(2).sum(1)
        er = torch.stack([(Z @ q.t()).pow(2).sum(1) for q in rands])          # [NRANDSUB, T]
        pieces = tok.convert_ids_to_tokens(idl)
        totn = tot.cpu().numpy(); e16n = e16.cpu().numpy(); e1n = e1.cpu().numpy(); ern = er.cpu().numpy()
        for t in range(3, len(idl)):
            c = categorize(pieces[t]); e = E[c]
            e["tot"] += float(totn[t]); e["das16"] += float(e16n[t]); e["das1"] += float(e1n[t])
            e["rand"] += ern[:, t]; e["n"] += 1
            if totn[t] > 1e-6:
                sh = float(e16n[t] / totn[t]); share_vals[c].append(sh)
                add(sh, {"share16": round(sh, 3), "norm": round(float(np.sqrt(totn[t])), 2),
                         "cat": c, "ctx": ctx_str(idl, t), "doc": cnt})
        cnt += 1
        if cnt % 50 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    hk.remove()

    def catrow(k):
        e = E[k]; rs = e["rand"] / max(e["tot"], 1e-9)
        return {"n": e["n"],
                "energy_share_das16": round(e["das16"] / max(e["tot"], 1e-9), 4),
                "energy_share_das1": round(e["das1"] / max(e["tot"], 1e-9), 4),
                "rand16_share_mean": round(float(rs.mean()), 4), "rand16_share_max": round(float(rs.max()), 4),
                "token_share16_median": round(float(np.median(share_vals[k])), 4) if share_vals[k] else None,
                "token_share16_p90": round(float(np.percentile(share_vals[k], 90)), 4) if share_vals[k] else None}
    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "r_npz": R_NPZ, "rgrid": RGRID, "ndocs": cnt,
           "expected_random_share_r16": round(16 / hd, 4), "expected_random_share_r1": round(1 / hd, 4),
           "by_category": {k: catrow(k) for k in cats},
           "top_firing_share16": [c for _, _, c in sorted(top, reverse=True)]}
    p = f"{OUTDIR}/pile_das_norm{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)
    for k in cats:
        r = out["by_category"][k]
        print(f"  {k:13} share16={r['energy_share_das16']:.3f} (rand {r['rand16_share_mean']:.3f}, exp 0.125) "
              f"share1={r['energy_share_das1']:.4f} (exp 0.0078)  n={r['n']}", flush=True)


if __name__ == "__main__":
    main()
