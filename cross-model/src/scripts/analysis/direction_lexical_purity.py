"""Does averaging the DAS direction over word->node assignments remove its LEXICAL component?

The problem: with a fixed word assignment, node parity is a deterministic function of the current token,
so a direction that merely separates 8 words from the other 8 scores as a perfect "parity" direction.
Evidence: with EVERY attention head mean-ablated (no cross-position information at all) the residual
coefficient still separates the parity classes, because the residual then holds only the current token's
embedding + MLPs.

The fix: train the direction under several random word->node assignments and average. The lexical split
differs per assignment and cancels; the structural (in-context) component survives.

This script sign-aligns and averages the per-assignment directions, then for each candidate direction
measures, on a HELD-OUT word assignment never used in training:
    sep_full    parity-class separation of the residual coefficient, normal forward pass
    sep_static  the same with all attention mean-ablated  -> the lexical/static floor
    lexical_fraction = sep_static / sep_full     (lower = purer in-context direction)
plus pairwise cosines between the per-assignment directions (how much structure is shared at all).

Env: GEN_MODEL(Llama) LAYER(14) WPS("0,1,2,3,4,5") HELDOUT_WP(11) NPZ_GLOB(auto)
     K(4) NWALKS(3) WLEN(1200) CTXLO(800) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/direction_lexical_purity_<model>.json  (+ averaged direction .npy)
"""
from __future__ import annotations
import os, json, glob
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
LAYER = int(os.environ.get("LAYER", "14"))
WPS = [int(x) for x in os.environ.get("WPS", "0,1,2,3,4,5").split(",")]
HELDOUT_WPS = [int(x) for x in os.environ.get("HELDOUT_WPS", "11").split(",")]
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH

    # ---- gather per-assignment directions ----
    dirs = {}
    for wp in WPS:
        f = f"{P}/das_multihead_resid_wp{wp}_L{LAYER}_save_{GEN_MODEL}.npz"
        if not os.path.exists(f):
            print(f"[warn] missing {f}"); continue
        z = np.load(f)
        if "4x4_r1" not in z.files: continue
        v = z["4x4_r1"][0].astype(np.float64); dirs[f"wp{wp}"] = v / np.linalg.norm(v)
    assert len(dirs) >= 2, f"need >=2 per-assignment directions, found {len(dirs)}"
    keys = list(dirs)
    # sign-align to the first, then average
    ref = dirs[keys[0]]
    aligned = {k: (v if v @ ref >= 0 else -v) for k, v in dirs.items()}
    cos = {f"{keys[i]}|{keys[j]}": round(float(aligned[keys[i]] @ aligned[keys[j]]), 3)
           for i in range(len(keys)) for j in range(i + 1, len(keys))}
    avg = np.mean([aligned[k] for k in keys], 0); avg /= np.linalg.norm(avg)
    np.save(f"{OUTDIR}/lexfree_r1_{tag}.npy", avg)
    print(f"[{tag}] {len(keys)} directions; pairwise cos mean={np.mean(list(cos.values())):+.3f}", flush=True)

    cands = dict(aligned); cands["AVERAGED"] = avg
    seed_stable = f"{P}/seed_stable_r1_{tag}.npy"
    if os.path.exists(seed_stable):
        v = np.load(seed_stable).astype(np.float64); cands["seed_stable(old)"] = v / np.linalg.norm(v)
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(cm.hidden_size); cands["random"] = r / np.linalg.norm(r)

    # ---- held-out word assignments (loop) ----
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    base_graph = graph

    per_ho = {}
    abl = {"on": False}
    def killer(l):
        def hh(_m, args):
            if not abl["on"]: return
            x = args[0].clone(); x[0] = x[0].mean(0, keepdim=True)
            return (x,) + tuple(args[1:])
        return hh
    hooks = [attn_proj(blocks[l], cm)[0].register_forward_pre_hook(killer(l)) for l in range(nL)]

    hooks_installed = True
    for HO in HELDOUT_WPS:
      rw = np.random.default_rng(9000 + HO)
      sel = rw.permutation(len(_config.WORDS))[:n]
      words = [_config.WORDS[i] for i in sel]
      graph = replace(base_graph, words=words)
      wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
      print(f"[{tag}] held-out wp{HO}: {words[:4]} …", flush=True)
      walks = G.generate_walks(graph, cfg)
      data = []
      for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s2 for s2 in range(len(wk.nodes) - 1) if s2 + 1 >= CTXLO]
        data.append((ids, torch.tensor([s2 + 1 for s2 in steps], device=dev),
                     np.array([col[wk.nodes[s2]] for s2 in steps])))

      def coefs(ablate, data=data):
        abl["on"] = ablate
        H, Y = [], []
        for ids, rp, y in data:
            o = model(input_ids=ids, output_hidden_states=True)
            H.append(o.hidden_states[LAYER + 1][0, rp].float().cpu().numpy()); Y.append(y)
        abl["on"] = False
        return np.concatenate(H), np.concatenate(Y)
      Hf, Y = coefs(False)
      Hs, _ = coefs(True)
      res = {}
      for nm, v in cands.items():
        cf = Hf @ v; cs = Hs @ v
        sf = float(cf[Y > 0].mean() - cf[Y < 0].mean())
        ss = float(cs[Y > 0].mean() - cs[Y < 0].mean())
        if sf < 0: sf, ss = -sf, -ss          # orient so the in-context separation is positive
        res[nm] = {"sep_full": round(sf, 4), "sep_static": round(ss, 4),
                   "lexical_fraction": round(float(ss / sf), 3) if abs(sf) > 1e-6 else None,
                   "in_context_sep": round(float(sf - ss), 4),
                   "abs_r_full": round(float(abs(np.corrcoef(cf, Y)[0, 1])), 3),
                   "abs_r_static": round(float(abs(np.corrcoef(cs, Y)[0, 1])), 3)}
      per_ho[f"wp{HO}"] = res
    for hk in hooks: hk.remove()
    names = list(cands)
    print(f"\nMEAN over {len(HELDOUT_WPS)} held-out assignments")
    print(f"{'direction':18} {'|r|_full':>9} {'|r|_static':>11} {'lex_frac':>9} {'sep_full':>9}")
    agg = {}
    for nm in names:
        rf = float(np.mean([per_ho[k][nm]["abs_r_full"] for k in per_ho]))
        rs = float(np.mean([per_ho[k][nm]["abs_r_static"] for k in per_ho]))
        sf = float(np.mean([per_ho[k][nm]["sep_full"] for k in per_ho]))
        lf = float(np.mean([per_ho[k][nm]["lexical_fraction"] or 0 for k in per_ho]))
        agg[nm] = {"abs_r_full": round(rf, 3), "abs_r_static": round(rs, 3),
                   "lexical_fraction": round(lf, 3), "sep_full": round(sf, 3),
                   "abs_r_full_sd": round(float(np.std([per_ho[k][nm]["abs_r_full"] for k in per_ho])), 3)}
        print(f"{nm:18} {rf:9.3f} {rs:11.3f} {lf:9.3f} {sf:9.3f}")
    out = {"model": tag, "layer": LAYER, "wps": WPS, "heldout_wps": HELDOUT_WPS,
           "per_heldout": per_ho, "aggregate": agg,
           "pairwise_cos": cos, "mean_pairwise_cos": round(float(np.mean(list(cos.values()))), 3),
           }
    p = f"{OUTDIR}/direction_lexical_purity_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
