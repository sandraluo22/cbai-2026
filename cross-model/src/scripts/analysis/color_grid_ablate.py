"""Causal test that lightness x hue is a real 2D grid with INDEPENDENT axes (robust to the decodability
overfitting worry). Two behavioural tasks -- "which is darker?" (lightness) and "which is more red?"
(hue) -- and we project out the lightness axis or the hue axis from the residual stream and re-measure
BOTH. A clean 2D grid predicts a DOUBLE DISSOCIATION: ablating lightness breaks only the lightness task,
ablating hue breaks only the hue task; a random direction breaks neither. Behaviour can't overfit, so
this settles whether the axes are genuinely separable.

We also report LEAVE-ONE-OUT decodability (top-k PC regression, cross-validated) -- the honest version of
the R^2=1.0 number, which was overfit (12 points in 4096 dims).

Env: GEN_MODEL(Llama) LIGHT HUE NSHOT(2) RANK(2) OUTDIR DEVICE
Out: <OUTDIR>/color_grid_ablate_<model>.json
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
LIGHT = os.environ.get("LIGHT", "dark,medium,light").split(",")
HUE = os.environ.get("HUE", "red,orange,yellow,green").split(",")
NSHOT = int(os.environ.get("NSHOT", "2")); RANK = int(os.environ.get("RANK", "2")); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/6_geometry")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def loo_r2(H, y, k=6):
    """leave-one-out R^2 decoding y from the top-k PCs of H (honest, cross-validated)."""
    n = len(y); Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False); Z = U[:, :k] * S[:k]
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        w = np.linalg.lstsq(np.c_[Z[tr], np.ones(len(tr))], y[tr], rcond=None)[0]
        pred[i] = np.r_[Z[i], 1] @ w
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / (ss + 1e-12))


@torch.no_grad()
def extract(model, tok, blocks, nL, items, dev, carrier="The colour {item}"):
    H = {L: [] for L in range(nL)}
    for it in items:
        text = carrier.format(item=it)
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


def axis_Q(H, y, nL, dev, rank):
    """per-layer orthonormal basis of the top-`rank` directions most correlated with coordinate y."""
    Q = {}
    for L in range(nL):
        Hc = H[L] - H[L].mean(0)
        # rank directions: readout dir Hc^T y, plus next PCs of the y-weighted residual
        r = Hc.T @ (y - y.mean())
        cols = [r]
        if rank > 1:
            proj = Hc - np.outer(Hc @ (r / (np.linalg.norm(r) + 1e-9)), r / (np.linalg.norm(r) + 1e-9))
            U, S, Vt = np.linalg.svd(proj * (y - y.mean())[:, None], full_matrices=False)
            for j in range(rank - 1): cols.append(Vt[j])
        Mt = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q[L] = q
    return Q


@torch.no_grad()
def score_task(model, tok, blocks, dev, pairs, preamble, question, proj_Q=None):
    """pairs = [(optA, optB, correct_idx)]; accuracy of picking the correct option by continuation logprob."""
    hooks = []
    if proj_Q is not None:
        def mkp(L):
            Q = proj_Q.get(L)
            def hh(_m, _i, out):
                if Q is None: return out
                h = out[0] if isinstance(out, tuple) else out
                hf = h.float(); hf = hf - (hf @ Q) @ Q.T; h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hh
        hooks = [blocks[L].register_forward_hook(mkp(L)) for L in range(len(blocks))]
    correct = 0
    try:
        for a, b, ci in pairs:
            prompt = preamble + question.format(A=a, B=b)
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
            opts = [tok(" " + a, add_special_tokens=False)["input_ids"], tok(" " + b, add_special_tokens=False)["input_ids"]]
            seqs = [p_ids + o for o in opts]; ml = max(len(s) for s in seqs)
            batch = torch.full((2, ml), tok.pad_token_id or 0, device=dev, dtype=torch.long)
            attn = torch.zeros((2, ml), device=dev, dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, :len(s)] = torch.tensor(s, device=dev); attn[i, :len(s)] = 1
            lg = model(input_ids=batch, attention_mask=attn).logits.float(); lp = torch.log_softmax(lg, -1)
            sc = []
            for i, o in enumerate(opts):
                s = sum(lp[i, len(p_ids) - 1 + t, tokid].item() for t, tokid in enumerate(o)) / len(o); sc.append(s)
            if int(np.argmax(sc)) == ci: correct += 1
    finally:
        for h in hooks: h.remove()
    return correct / max(len(pairs), 1)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    items, coords = [], []
    for li, l in enumerate(LIGHT):
        for hi, h in enumerate(HUE): items.append(f"{l} {h}"); coords.append((li, hi))
    coords = np.array(coords)
    def it(li, hi): return items[li * len(HUE) + hi]
    # lightness task: same hue, different lightness -> darker = lower lightness idx
    lt_pairs = []
    for hi in range(len(HUE)):
        for li in range(len(LIGHT)):
            for lj in range(li + 1, len(LIGHT)):
                lt_pairs.append((it(li, hi), it(lj, hi), 0))          # darker (li) is correct = index 0
    # hue task: same lightness, different hue -> redder = lower hue idx
    hue_pairs = []
    for li in range(len(LIGHT)):
        for hi in range(len(HUE)):
            for hj in range(hi + 1, len(HUE)):
                hue_pairs.append((it(li, hi), it(li, hj), 0))         # redder (hi) is correct = index 0
    LT_Q = "Between {A} and {B}, the darker colour is"
    HUE_Q = "Between {A} and {B}, the colour closer to red is"
    lt_pre = "Between light gray and dark gray, the darker colour is dark gray.\nBetween dark blue and light blue, the darker colour is dark blue.\n"
    hue_pre = "Between green and orange, the colour closer to red is orange.\nBetween yellow and red, the colour closer to red is red.\n"

    model, tok = load_with_fallback(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    H = extract(model, tok, blocks, nL, items, dev)

    # honest decodability (LOO) at the layer that best fits the 2D grid earlier (scan, report best)
    loo = {"lightness": max(loo_r2(H[L], coords[:, 0].astype(float)) for L in range(nL)),
           "hue": max(loo_r2(H[L], coords[:, 1].astype(float)) for L in range(nL))}
    print(f"[{tag}] LOO decode R2 (honest): lightness={loo['lightness']:.2f} hue={loo['hue']:.2f}", flush=True)

    Q_light = axis_Q(H, coords[:, 0].astype(float), nL, dev, RANK)
    Q_hue = axis_Q(H, coords[:, 1].astype(float), nL, dev, RANK)
    Q_rand = {}
    for L in range(nL):
        Mt = torch.tensor(rng.standard_normal((cm.hidden_size, RANK)).astype("float32"), device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q_rand[L] = q

    conds = {"baseline": None, "ablate_lightness": Q_light, "ablate_hue": Q_hue, "ablate_random": Q_rand}
    res = {}
    for name, Q in conds.items():
        a_lt = score_task(model, tok, blocks, dev, lt_pairs, lt_pre, LT_Q, proj_Q=Q)
        a_hue = score_task(model, tok, blocks, dev, hue_pairs, hue_pre, HUE_Q, proj_Q=Q)
        res[name] = {"lightness_task": a_lt, "hue_task": a_hue}
        print(f"[{tag}] {name:16}: lightness_task={a_lt:.2f}  hue_task={a_hue:.2f}", flush=True)

    out = {"model": tag, "light": LIGHT, "hue": HUE, "rank": RANK, "loo_decode": loo,
           "n_lightness_pairs": len(lt_pairs), "n_hue_pairs": len(hue_pairs), "results": res}
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/color_grid_ablate_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
