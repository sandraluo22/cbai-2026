"""WHAT does the residual parity direction do in natural text? Per-token damage profile of ablating it,
with the controls the earlier (inert-direction) analyses lacked.

For every predicted token we record the loss increase caused by projecting out the direction, plus token
features, and test one PRE-REGISTERED hypothesis motivated by the attention analysis: L14H19 sends ~97% of
its attention to previous occurrences of the current token, so ablating the parity machinery should hurt
most on tokens that REPEAT earlier tokens in the document. Because harder tokens have more room to get
worse, the repeat effect is measured WITHIN baseline-loss quintiles.

Reported per condition (real parity r1/r8 + rank-matched random directions):
  - split-half reliability of the per-token-string damage profile (docs split even/odd)
  - mean damage by token category and by repeat status, quintile-matched on baseline loss
  - correlation with baseline loss, token position, and prior-occurrence count
  - top-damaged contexts

Env: GEN_MODEL(Llama) LAYER(14) NDOCS(300) MAXTOK(384) NRAND(3) TOPK(30) WIN(12) SEED(0)
     PAR_NPZ(runs/.../das_multihead_resid_L14_save_<model>.npz) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_parity_damage_profile<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, heapq
from dataclasses import replace
from collections import defaultdict
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
LAYER = int(os.environ.get("LAYER", "14"))
NDOCS = int(os.environ.get("NDOCS", "300")); MAXTOK = int(os.environ.get("MAXTOK", "384"))
NRAND = int(os.environ.get("NRAND", "3")); TOPK = int(os.environ.get("TOPK", "30"))
WIN = int(os.environ.get("WIN", "12")); SEED = int(os.environ.get("SEED", "0"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
PAR_NPZ = os.environ.get("PAR_NPZ", f"{P}/das_multihead_resid_L{LAYER}_save_{GEN_MODEL}.npz")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); dm = cm.hidden_size
    rng = np.random.default_rng(SEED)

    z = np.load(PAR_NPZ)
    subs = {}
    for r in (1, 8):
        k = f"4x4_r{r}"
        if k in z.files:
            q, _ = np.linalg.qr(z[k].astype(np.float64).T); subs[f"par_r{r}"] = q.T[:r]
    for r in (1, 8):
        for i in range(NRAND):
            subs[f"rand{r}_{i}"] = np.linalg.qr(rng.standard_normal((dm, r)))[0].T
    names = list(subs)
    Q = {n: torch.tensor(v, dtype=torch.float32, device=dev) for n, v in subs.items()}
    print(f"[{tag}] conditions: {names}", flush=True)

    state = {"Q": None}
    def rh(_m, _i, out):
        if state["Q"] is None: return out
        h = out[0] if isinstance(out, tuple) else out
        x = h[0].float(); q = state["Q"]
        h = h.clone(); h[0] = (x - (x @ q.t()) @ q).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hook = blocks[LAYER].register_forward_hook(rh)

    import string as _string
    def catc(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"

    from datasets import load_dataset
    docs = []
    for ex in load_dataset(DATASET, split="train", streaming=True):
        if len(docs) >= NDOCS: break
        t = ex["text"]
        if t and len(t) >= 40: docs.append(t)

    D = {n: [] for n in names}          # per-token damage, flat
    base = []; cat = []; rep = []; nprev = []; pos = []; half = []; strs = []; ctxs = []
    for di, text in enumerate(docs):
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] < 12: continue
        idl = ids[0].tolist(); pieces = tok.convert_ids_to_tokens(idl)
        state["Q"] = None
        lg = model(input_ids=ids).logits[0].float()
        lsm = torch.log_softmax(lg[:-1], -1)
        b = -lsm[torch.arange(ids.shape[1] - 1), ids[0, 1:]].cpu().numpy()
        seen = defaultdict(int)
        for t in range(len(idl) - 1): seen[idl[t]] += 0     # init
        counts = defaultdict(int); prevn = []
        for t in range(len(idl) - 1):
            counts[idl[t]] += 1
            prevn.append(counts[idl[t + 1]])                # prior occurrences of the PREDICTED token
        keep = list(range(3, len(idl) - 1))
        base += [float(b[t]) for t in keep]
        cat += [catc(pieces[t + 1]) for t in keep]
        rep += [1 if prevn[t] > 0 else 0 for t in keep]
        nprev += [int(prevn[t]) for t in keep]
        pos += [t for t in keep]
        half += [di % 2] * len(keep)
        strs += [tok.decode([idl[t + 1]])[:14] for t in keep]
        ctxs += [(di, t)] * len(keep)
        for n in names:
            state["Q"] = Q[n]
            l = model(input_ids=ids).logits[0].float()
            ls = torch.log_softmax(l[:-1], -1)
            d = (-ls[torch.arange(ids.shape[1] - 1), ids[0, 1:]].cpu().numpy()) - b
            D[n] += [float(d[t]) for t in keep]
        state["Q"] = None
        if (di + 1) % 50 == 0: print(f"[{tag}] {di+1}/{len(docs)} docs", flush=True)
    hook.remove()

    base = np.array(base); cat = np.array(cat); rep = np.array(rep)
    nprev = np.array(nprev); pos = np.array(pos); half = np.array(half); strs = np.array(strs)
    qs = np.quantile(base, [0.2, 0.4, 0.6, 0.8]); qbin = np.digitize(base, qs)
    print(f"\n{len(base)} tokens; {rep.mean():.1%} are repeats of an earlier token\n", flush=True)

    res = {}
    for n in names:
        d = np.array(D[n])
        # split-half reliability of the per-token-string damage profile
        prof = {}
        for h in (0, 1):
            m = half == h; s = defaultdict(list)
            for st, v in zip(strs[m], d[m]): s[st].append(v)
            prof[h] = {k: float(np.mean(v)) for k, v in s.items() if len(v) >= 15}
        common = sorted(set(prof[0]) & set(prof[1]))
        r_sh = float(np.corrcoef([prof[0][k] for k in common], [prof[1][k] for k in common])[0, 1]) if len(common) > 20 else None
        # repeat effect, matched on baseline-loss quintile
        eff = []
        for qb in range(5):
            m = qbin == qb
            if (rep[m] == 1).sum() > 30 and (rep[m] == 0).sum() > 30:
                eff.append(float(d[m & (rep == 1)].mean() - d[m & (rep == 0)].mean()))
        res[n] = {"mean": round(float(d.mean()), 5), "sd": round(float(d.std()), 5),
                  "splithalf_r": None if r_sh is None else round(r_sh, 3), "n_tokens_profiled": len(common),
                  "by_cat": {c: round(float(d[cat == c].mean()), 5) for c in ("word_initial", "continuation", "punct", "digit")},
                  "repeat_effect_quintile_matched": round(float(np.mean(eff)), 5) if eff else None,
                  "repeat_effect_raw": round(float(d[rep == 1].mean() - d[rep == 0].mean()), 5),
                  "corr_baseline_loss": round(float(np.corrcoef(d, base)[0, 1]), 3),
                  "corr_nprev": round(float(np.corrcoef(d, nprev)[0, 1]), 3),
                  "corr_position": round(float(np.corrcoef(d, pos)[0, 1]), 3)}
        print(f"  {n:12} mean={res[n]['mean']:+.5f} splithalf_r={res[n]['splithalf_r']} "
              f"repeat_effect(matched)={res[n]['repeat_effect_quintile_matched']} "
              f"corr_base={res[n]['corr_baseline_loss']:+.2f} corr_nprev={res[n]['corr_nprev']:+.3f}", flush=True)

    # cross-condition: do real and random damage the same tokens?
    prof_all = {}
    for n in names:
        d = np.array(D[n]); s = defaultdict(list)
        for st, v in zip(strs, d): s[st].append(v)
        prof_all[n] = {k: float(np.mean(v)) for k, v in s.items() if len(v) >= 30}
    keys = sorted(set.intersection(*[set(p) for p in prof_all.values()]))
    Mx = {}
    for a in names:
        for b2 in names:
            if a < b2:
                Mx[f"{a}|{b2}"] = round(float(np.corrcoef([prof_all[a][k] for k in keys],
                                                          [prof_all[b2][k] for k in keys])[0, 1]), 3)
    # top damaged contexts for the real r1
    dr = np.array(D["par_r1"]); top = np.argsort(dr)[::-1][:TOPK]
    tops = [{"dloss": round(float(dr[i]), 3), "tok": strs[i], "cat": str(cat[i]),
             "repeat": int(rep[i]), "base": round(float(base[i]), 2)} for i in top]

    out = {"model": tag, "layer": LAYER, "ndocs": len(docs), "n_tokens": int(len(base)),
           "frac_repeat": round(float(rep.mean()), 4), "conditions": res,
           "profile_cross_correlation": Mx, "top_damaged_par_r1": tops,
           "profile_keys": len(keys)}
    p = f"{OUTDIR}/pile_parity_damage_profile{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
