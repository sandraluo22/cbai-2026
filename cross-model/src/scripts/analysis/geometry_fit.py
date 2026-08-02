"""Geometry battery -- FIT stage. For each concept space, extract per-item representations and score them
(RSA) against every candidate structural distance matrix (line/arc, cycle, simplex, tree, product, helix).
The geometry the model actually uses is the one whose distance matrix best matches the representation
distances. We report, per space, the best-fitting geometry vs the hypothesised one, and (for helices)
whether the helix beats its pure-cycle / pure-line components, (for products) whether the factorized
product beats a single axis.

Env: GEN_MODEL(Llama) OUTDIR DEVICE
Out: <OUTDIR>/geometry_fit_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from geometry_spaces import SPACES, candidate_dmats

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/6_geometry")
INTENDED = {"arc": "line", "simplex": "simplex", "tree": "tree", "product": "product", "helix": "helix"}


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def sp_rsa(a, b):
    iu = np.triu_indices(a.shape[0], 1); x, y = a[iu], b[iu]
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


@torch.no_grad()
def extract(model, tok, blocks, nL, carrier, items, dev):
    """per-layer per-item residual at the item's last token in its carrier template."""
    H = {L: [] for L in range(nL)}
    for it in items:
        text = carrier.format(item=it)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = torch.tensor([enc["input_ids"]], device=dev); offs = enc["offset_mapping"]
        a = text.index(it); b = a + len(it)
        toks = [t for t, (o0, o1) in enumerate(offs) if o0 is not None and o1 > o0 and o0 < b and o1 > a]
        last = toks[-1]
        grabbed = {}
        def mk(L):
            def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
        model(input_ids=ids)
        for h in hs: h.remove()
        for L in range(nL): H[L].append(grabbed[L][0, last].float().cpu().numpy())
    return {L: np.array(H[L]) for L in range(nL)}


def rep_dists(Hc):
    return np.linalg.norm(Hc[:, None] - Hc[None], axis=2)


def equidistance(Rd):
    """1 - CV of off-diagonal distances: 1.0 = perfectly equidistant (regular simplex)."""
    iu = np.triu_indices(Rd.shape[0], 1); d = Rd[iu]
    return float(1 - d.std() / (d.mean() + 1e-9))


def eff_dim(Hc):
    """participation ratio of the singular-value spectrum (how many dims the points spread over)."""
    s = np.linalg.svd(Hc, compute_uv=False) ** 2
    return float((s.sum() ** 2) / ((s ** 2).sum() + 1e-12))


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = load_with_fallback(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    print(f"[{tag}] geometry fit over {len(SPACES)} spaces", flush=True)

    out = {"model": tag, "spaces": {}}
    for name, sp in SPACES.items():
        cands = candidate_dmats(sp); intended = INTENDED[sp["family"]]
        H = extract(model, tok, blocks, nL, sp["carrier"], sp["items"], dev)
        best = {g: (-1.0, -1) for g in cands}; eq_best = (-9.0, -1)
        for L in range(nL):
            Hc = H[L] - H[L].mean(0); Rd = rep_dists(Hc)
            for g, D in cands.items():
                r = sp_rsa(Rd, D)
                if r > best[g][0]: best[g] = (r, L)
            eq = equidistance(Rd)
            if eq > eq_best[0]: eq_best = (eq, L, eff_dim(Hc))
        rsa = {g: round(best[g][0], 3) for g in cands}
        winner = (max(rsa, key=rsa.get) if rsa else "simplex")
        rec = {"family": sp["family"], "intended": intended, "n": len(sp["items"]), "rsa": rsa,
               "best_geom": winner, "intended_rsa": rsa.get(intended),
               "intended_wins": (winner == intended) if intended != "simplex" else (eq_best[0] > 0.85),
               "equidistance": round(eq_best[0], 3), "eff_dim": round(eq_best[2], 2),
               "best_layer": best[intended][1] if intended in best else eq_best[1]}
        out["spaces"][name] = rec
        rec["graph_rsa"] = {g: rsa[g] for g in ("ring", "grid", "hex") if g in rsa}
        gtxt = " ".join(f"{g}={rsa[g]}" for g in ("ring", "grid", "hex") if g in rsa) or "(no ring/grid/hex layout)"
        print(f"  {name:12} [{sp['family']:7}] {gtxt:38} | own-geom {intended}={rec['intended_rsa']} equidist={rec['equidistance']}", flush=True)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/geometry_fit_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
