"""Does COMPOSING two independent graded semantic axes invoke a GRID-like structure? We build a
lightness x hue grid of colour terms ("dark red", "light green", ...) -- two axes that each vary
monotonically -- and test whether the model's representation lays them out as a 2D lattice, exactly the
way we tested the abstract 4x4 in-context grid:

  grid_2d RSA  : rep distances vs 2D coordinate (lightness,hue) distance   -- overall grid fit
  best2d RSA   : supervised fit of top PCs -> (lightness,hue) coords        -- how planar/grid the top variance is
  lightness RSA / hue RSA : each axis alone                                 -- do BOTH contribute (grid) or one dominate (entangled)?
  linear decode R^2 per axis                                               -- is each axis linearly readable?
Plus a 2D embedding (best-2D projection) drawn as a lattice so you can SEE whether composition is grid-like.

Env: GEN_MODEL(Llama) LIGHT(dark,medium,light) HUE(red,orange,yellow,green) OUTDIR DEVICE
Out: <OUTDIR>/color_grid_<model>.json (+ figure via viz/color_grid_plot.py)
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
LIGHT = os.environ.get("LIGHT", "dark,medium,light").split(",")           # axis 1 (rows)
HUE = os.environ.get("HUE", "red,orange,yellow,green").split(",")         # axis 2 (cols)
CARRIER = os.environ.get("CARRIER", "The colour {item}")
NAME = os.environ.get("NAME", "color")                                    # output tag: general 2-axis composition
OUTDIR = os.environ.get("OUTDIR", "runs/axes/6_geometry")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def rdm(H): iu = np.triu_indices(H.shape[0], 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def best2d(H, C):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, C - C.mean(0), rcond=None)[0]
    return Z @ W


def decode_r2(H, y, k=6):
    """leave-one-out R^2 from top-k PCs -- honest (full-dim probe overfits with n<<d)."""
    n = len(y); Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False); Z = U[:, :k] * S[:k]
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        w = np.linalg.lstsq(np.c_[Z[tr], np.ones(len(tr))], y[tr] - y.mean(), rcond=None)[0]
        pred[i] = np.r_[Z[i], 1] @ w
    yc = y - y.mean(); return float(1 - ((yc - pred) ** 2).sum() / ((yc ** 2).sum() + 1e-12))


@torch.no_grad()
def extract(model, tok, blocks, nL, items, dev):
    H = {L: [] for L in range(nL)}
    for it in items:
        text = CARRIER.format(item=it)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = torch.tensor([enc["input_ids"]], device=dev); offs = enc["offset_mapping"]
        a = text.index(it); b = a + len(it)
        toks = [t for t, (o0, o1) in enumerate(offs) if o0 is not None and o1 > o0 and o0 < b and o1 > a]
        last = toks[-1]; g = {}
        def mk(L):
            def hh(_m, _i, out): g[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
        model(input_ids=ids)
        for h in hs: h.remove()
        for L in range(nL): H[L].append(g[L][0, last].float().cpu().numpy())
    return {L: np.array(H[L]) for L in range(nL)}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    nL_light, nH_hue = len(LIGHT), len(HUE); n = nL_light * nH_hue
    items, coords = [], []
    for li, l in enumerate(LIGHT):
        for hi, h in enumerate(HUE):
            items.append(f"{l} {h}"); coords.append((li, hi))
    coords = np.array(coords, float)
    Dlight = np.abs(coords[:, 0][:, None] - coords[:, 0][None])[np.triu_indices(n, 1)]
    Dhue = np.abs(coords[:, 1][:, None] - coords[:, 1][None])[np.triu_indices(n, 1)]
    Dgrid = np.linalg.norm(coords[:, None] - coords[None], axis=2)[np.triu_indices(n, 1)]      # 2D coord distance

    model, tok = load_with_fallback(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    print(f"[{tag}] {nL_light}x{nH_hue} colour grid: {items}", flush=True)
    H = extract(model, tok, blocks, nL, items, dev)

    per = []
    for L in range(nL):
        Hc = H[L] - H[L].mean(0); R = rdm(Hc)
        per.append({"layer": L, "grid_2d": sp(R, Dgrid), "lightness": sp(R, Dlight), "hue": sp(R, Dhue),
                    "best2d": sp(rdm(best2d(H[L], coords)), Dgrid),
                    "r2_light": decode_r2(H[L], coords[:, 0]), "r2_hue": decode_r2(H[L], coords[:, 1])})
    Lstar = max(range(nL), key=lambda L: per[L]["best2d"])
    emb = best2d(H[Lstar], coords)                                                              # 2D layout at best layer
    best = per[Lstar]
    print(f"[{tag}] best layer L{Lstar}: grid_2d RSA={best['grid_2d']:.2f} best2d={best['best2d']:.2f} "
          f"| lightness={best['lightness']:.2f} hue={best['hue']:.2f} | R2 light={best['r2_light']:.2f} hue={best['r2_hue']:.2f}", flush=True)

    out = {"model": tag, "name": NAME, "light": LIGHT, "hue": HUE, "items": items, "coords": coords.tolist(),
           "best_layer": Lstar, "best": best, "per_layer": per, "embedding": emb.tolist()}
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/color_grid_{NAME}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
