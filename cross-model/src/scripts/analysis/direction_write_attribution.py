"""WHO writes the parity direction — and is it the same machinery in natural text as in the grid task?

The residual stream is an exact additive sum: h_L = emb + sum_{l<=L} (attn_l + mlp_l), and attn_l splits
per head as z_lh @ W_O[h]. So the coefficient h_L . v decomposes EXACTLY into per-head and per-MLP
contributions: contribution(l,h) = z_lh . (W_O[h]^T v). No approximation, no probing.

We compute that decomposition twice — on Pile text and on in-context grid walks — for the seed-stable
parity direction, the single-run parity direction, and random controls. Then:
  - which heads write the direction in each setting (ranked, with concentration/participation ratio)
  - the correlation between the natural-text profile and the grid profile  <-- the mechanism question
  - whether the real direction's writer profile is more concentrated than a random direction's
If the same heads write it in both settings, the machinery is genuinely shared. If different heads write
the same direction, then it is one subspace re-used by distinct circuits — which is itself a clean answer.

Env: GEN_MODEL(Llama) LAYER(14) NDOCS(120) MAXTOK(320) K(4) NWALKS(6) CTXLO(1000) WLEN(1300)
     PAR_NPZ STABLE_NPY NRAND(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/direction_write_attribution<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
from graph import Walk
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
LAYER = int(os.environ.get("LAYER", "14"))
NDOCS = int(os.environ.get("NDOCS", "120")); MAXTOK = int(os.environ.get("MAXTOK", "320"))
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "6"))
CTXLO = int(os.environ.get("CTXLO", "1000")); WLEN = int(os.environ.get("WLEN", "1300"))
NRAND = int(os.environ.get("NRAND", "3")); SEED = int(os.environ.get("SEED", "0"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
PAR_NPZ = os.environ.get("PAR_NPZ", f"{P}/das_multihead_resid_L{LAYER}_save_{GEN_MODEL}.npz")
STABLE_NPY = os.environ.get("STABLE_NPY", f"{P}/seed_stable_r1_{GEN_MODEL}.npy")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); dm = cm.hidden_size
    nH = cm.num_attention_heads; hd = getattr(cm, "head_dim", None) or dm // nH
    rng = np.random.default_rng(SEED)

    dirs = {}
    if os.path.exists(STABLE_NPY):
        v = np.load(STABLE_NPY).astype(np.float64); dirs["par_stable_r1"] = v / np.linalg.norm(v)
    z = np.load(PAR_NPZ)
    if "4x4_r1" in z.files:
        v = z["4x4_r1"][0].astype(np.float64); dirs["par_r1"] = v / np.linalg.norm(v)
    for i in range(NRAND):
        v = rng.standard_normal(dm); dirs[f"rand_{i}"] = v / np.linalg.norm(v)
    names = list(dirs)
    Vt = {n: torch.tensor(v, dtype=torch.float32, device=dev) for n, v in dirs.items()}
    print(f"[{tag}] directions: {names}", flush=True)

    # precompute per-layer, per-head projection vectors  w[l,h] = W_O[l][:, h].T @ v   -> [nH, hd]
    Wv = {}
    for n in names:
        rows = []
        for l in range(LAYER + 1):
            W = attn_proj(blocks[l], cm)[0].weight.detach().float()          # [dm, nH*hd]
            rows.append((W.t() @ Vt[n]).view(nH, hd))                        # [nH, hd]
        Wv[n] = torch.stack(rows)                                            # [L+1, nH, hd]

    caps = {}
    hooks = []
    for l in range(LAYER + 1):
        proj = attn_proj(blocks[l], cm)[0]
        def mkz(l):
            def hh(_m, args): caps[("z", l)] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mkz(l)))
        def mkm(l):
            def hh(_m, _i, out): caps[("m", l)] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hooks.append(blocks[l].mlp.register_forward_hook(mkm(l)))

    def decompose(ids, keep):
        """returns per-direction dict of head [L+1,nH], mlp [L+1], emb scalar, total — summed over `keep`."""
        caps.clear()
        o = model(input_ids=ids, output_hidden_states=True)
        kt = torch.tensor(keep, device=dev)
        out = {}
        for n in names:
            Zh = torch.stack([caps[("z", l)][0, kt].float().view(len(keep), nH, hd) for l in range(LAYER + 1)])
            head = (Zh * Wv[n][:, None, :, :]).sum(-1)                                                  # [L+1,T,nH]
            mlp = torch.stack([caps[("m", l)][0, kt].float() @ Vt[n] for l in range(LAYER + 1)])        # [L+1,T]
            emb = o.hidden_states[0][0, kt].float() @ Vt[n]
            tot = o.hidden_states[LAYER + 1][0, kt].float() @ Vt[n]
            out[n] = {"head": head.cpu().numpy(), "mlp": mlp.cpu().numpy(),
                      "emb": emb.cpu().numpy(), "tot": tot.cpu().numpy()}
        return out

    acc = {n: {"head": np.zeros((LAYER + 1, nH)), "head_sq": np.zeros((LAYER + 1, nH)),
               "mlp": np.zeros(LAYER + 1), "emb": 0.0, "tot": [], "recon": [], "npos": 0}
           for n in names}
    accg = {n: {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else (0.0 if not isinstance(v, list) else []))
                for k, v in acc[n].items()} for n in names}

    def absorb(store, dec):
        for n in names:
            d = dec[n]
            store[n]["head"] += d["head"].sum(1); store[n]["head_sq"] += (d["head"] ** 2).sum(1)
            store[n]["mlp"] += d["mlp"].sum(1); store[n]["emb"] += float(d["emb"].sum())
            store[n]["tot"].append(d["tot"])
            store[n]["recon"].append(d["head"].sum((0, 2)) + d["mlp"].sum(0) + d["emb"])
            store[n]["npos"] += len(d["tot"])

    # ---------- natural text ----------
    from datasets import load_dataset
    nd = 0
    for ex in load_dataset(DATASET, split="train", streaming=True):
        if nd >= NDOCS: break
        t = ex["text"]
        if not t or len(t) < 40: continue
        ids = tok(t, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] < 16: continue
        absorb(acc, decompose(ids, list(range(3, ids.shape[1]))))
        nd += 1
        if nd % 30 == 0: print(f"[pile] {nd}/{NDOCS}", flush=True)

    # ---------- grid walks ----------
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    for wk in G.generate_walks(graph, cfg):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk)
        keep = [spans[s][-1] for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if not keep: continue
        for i in range(0, len(keep), 400):
            absorb(accg, decompose(ids, keep[i:i + 400]))
    print(f"[grid] {accg[names[0]]['npos']} node positions", flush=True)

    def profile(store, n):
        h = store[n]["head"] / max(store[n]["npos"], 1)                 # mean signed contribution
        hs = store[n]["head_sq"] / max(store[n]["npos"], 1)
        sd = np.sqrt(np.maximum(hs - h ** 2, 0))
        w = np.abs(h).ravel(); w = w / (w.sum() + 1e-12)
        pr = float(1.0 / (w ** 2).sum())
        tot = np.concatenate(store[n]["tot"]); rec = np.concatenate(store[n]["recon"])
        return h, sd, pr, float(np.corrcoef(tot, rec)[0, 1]), float(np.abs(tot - rec).mean() / (np.abs(tot).mean() + 1e-9))

    res = {}
    for n in names:
        hp, sdp, prp, cp, ep = profile(acc, n)
        hg, sdg, prg, cg, eg = profile(accg, n)
        flat_p, flat_g = hp.ravel(), hg.ravel()
        r_pearson = float(np.corrcoef(flat_p, flat_g)[0, 1])
        rk = lambda x: np.argsort(np.argsort(-np.abs(x)))
        r_spear = float(np.corrcoef(rk(flat_p), rk(flat_g))[0, 1])
        topp = np.argsort(-np.abs(flat_p))[:10]; topg = np.argsort(-np.abs(flat_g))[:10]
        fmt = lambda i, arr: {"head": f"L{i//nH}H{i%nH}", "contrib": round(float(arr[i]), 4)}
        res[n] = {
            "reconstruction_check": {"pile_corr": round(cp, 5), "pile_rel_err": round(ep, 5),
                                     "grid_corr": round(cg, 5), "grid_rel_err": round(eg, 5)},
            "participation_ratio_pile": round(prp, 1), "participation_ratio_grid": round(prg, 1),
            "profile_corr_pile_vs_grid_pearson": round(r_pearson, 3),
            "profile_corr_pile_vs_grid_spearman_absrank": round(r_spear, 3),
            "top_writers_pile": [fmt(int(i), flat_p) for i in topp],
            "top_writers_grid": [fmt(int(i), flat_g) for i in topg],
            "mlp_share_pile": round(float(np.abs(acc[n]["mlp"] / max(acc[n]["npos"], 1)).sum() /
                                          (np.abs(flat_p).sum() + np.abs(acc[n]["mlp"] / max(acc[n]["npos"], 1)).sum() + 1e-9)), 3),
        }
        print(f"  {n:14} PR pile={prp:5.1f} grid={prg:5.1f} | corr(pile,grid) r={r_pearson:+.3f} "
              f"rank_r={r_spear:+.3f} | recon r={cp:.4f}", flush=True)
        print(f"      pile writers: " + ", ".join(f"{d['head']}({d['contrib']:+.3f})" for d in res[n]["top_writers_pile"][:6]), flush=True)
        print(f"      grid writers: " + ", ".join(f"{d['head']}({d['contrib']:+.3f})" for d in res[n]["top_writers_grid"][:6]), flush=True)
    for h in hooks: h.remove()

    out = {"model": tag, "layer": LAYER, "n_heads": nH, "ndocs": nd,
           "grid_positions": int(accg[names[0]]["npos"]), "directions": res}
    p = f"{OUTDIR}/direction_write_attribution{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
