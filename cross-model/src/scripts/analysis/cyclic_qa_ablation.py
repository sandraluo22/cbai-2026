"""Does the model USE a cyclic-position eigenmode to do modular arithmetic on cycles it learned in
pretraining? We test real cycles -- months (12-cycle) and days-of-week (7-cycle) -- with questions like
"N months after February is ___" / "N days after Monday is ___".

(1) baseline accuracy of the modular arithmetic (full-continuation scoring over the cycle's items).
(2) build the cycle's item representations from a neutral enumeration, per layer, and take the CYCLE
    Laplacian eigenmodes (Fourier modes: k=1 fundamental = circular position, up to Nyquist). Identify the
    most PROMINENT mode by represented power.
(3) ablate a mode by projecting the residual stream onto the orthogonal complement of that mode's per-layer
    readout direction (Hc_L^T u_k) at every layer, and re-measure arithmetic accuracy. Conditions:
    fundamental (position) vs a high-frequency mode vs a random direction of equal rank.
If ablating the fundamental circular-position mode collapses accuracy while high-freq/random do not, the
model performs the modular arithmetic THROUGH that eigenmode.

Env: GEN_MODEL(Llama) CYCLE(months|days) NSHOT(3) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/cyclic_qa_ablation_<model>_<cycle>.json
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
CYCLE = os.environ.get("CYCLE", "months")
NSHOT = int(os.environ.get("NSHOT", "3")); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SPEC = {"months": (MONTHS, "months"), "days": (DAYS, "days")}


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def cycle_eigenmodes(nn):
    A = np.zeros((nn, nn))
    for i in range(nn):
        A[i, (i + 1) % nn] = A[i, (i - 1) % nn] = 1.0
    L = np.diag(A.sum(1)) - A
    w, U = np.linalg.eigh(L)                          # ascending: const, then cos/sin pairs (k=1 fundamental)
    return w, U


@torch.no_grad()
def item_reps(model, tok, blocks, nL, items, dev):
    """per-layer residual at each item's last token, from a neutral enumeration."""
    text = ", ".join(items) + "."
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids = torch.tensor([enc["input_ids"]], device=dev); offs = enc["offset_mapping"]
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    model(input_ids=ids)
    for h in hs: h.remove()
    # char span of each item in text
    spans = []; pos = 0
    for it in items:
        s = text.index(it, pos); spans.append((s, s + len(it))); pos = s + len(it)
    H = {L: np.zeros((len(items), grabbed[0].shape[-1])) for L in range(nL)}
    for i, (a, b) in enumerate(spans):
        toks = [t for t, (o0, o1) in enumerate(offs) if o0 is not None and o1 > o0 and o0 < b and o1 > a]  # overlap
        last = toks[-1]
        for L in range(nL): H[L][i] = grabbed[L][0, last].float().cpu().numpy()
    return H


def build_queries(items, unit, nn):
    q = []
    for si in range(nn):
        for k in range(1, nn):
            q.append((si, k, (si + k) % nn))
    return q


def few_shot(items, unit, nn, rng, nshot, exclude):
    ex = []
    while len(ex) < nshot:
        si = int(rng.integers(nn)); k = int(rng.integers(1, nn))
        if (si, k) in exclude or (si, k) in [(e[0], e[1]) for e in ex]: continue
        ex.append((si, k, (si + k) % nn))
    return "".join(f"{k} {unit} after {items[si]} is {items[ans]}.\n" for si, k, ans in ex)


@torch.no_grad()
def score_queries(model, tok, blocks, dev, items, unit, nn, queries, preamble, proj_Q=None):
    """accuracy of modular arithmetic via full-continuation scoring over the nn items."""
    item_ids = [tok(" " + it, add_special_tokens=False)["input_ids"] for it in items]
    hooks = []
    if proj_Q is not None:
        def mkp(L):
            Q = proj_Q.get(L)
            def hh(_m, _i, out):
                if Q is None: return out
                h = out[0] if isinstance(out, tuple) else out
                hf = h.float(); hf = hf - (hf @ Q) @ Q.T
                h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hh
        hooks = [blocks[L].register_forward_hook(mkp(L)) for L in range(len(blocks))]
    correct = 0
    try:
        for si, k, ans in queries:
            prompt = preamble + f"{k} {unit} after {items[si]} is"
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
            seqs = [p_ids + ic for ic in item_ids]; ml = max(len(s) for s in seqs)
            batch = torch.full((nn, ml), tok.pad_token_id or 0, device=dev, dtype=torch.long)
            attn = torch.zeros((nn, ml), device=dev, dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, :len(s)] = torch.tensor(s, device=dev); attn[i, :len(s)] = 1
            logits = model(input_ids=batch, attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits, -1); Lp = len(p_ids); scores = []
            for i, ic in enumerate(item_ids):
                s = 0.0
                for t, tokid in enumerate(ic):
                    s += lp[i, Lp - 1 + t, tokid].item()
                scores.append(s / len(ic))
            if int(np.argmax(scores)) == ans: correct += 1
    finally:
        for h in hooks: h.remove()
    return correct / max(len(queries), 1)


def build_Q(H, U, modes, nL, dev):
    """orthonormal per-layer basis of the readout directions r_{k,L}=Hc_L^T u_k for the selected modes."""
    Q = {}
    for L in range(nL):
        Hc = H[L] - H[L].mean(0)
        cols = [Hc.T @ U[:, k] for k in modes]
        cols = [c for c in cols if np.linalg.norm(c) > 1e-8]
        if not cols: Q[L] = None; continue
        Mt = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q[L] = q
    return Q


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED); tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    items, unit = SPEC[CYCLE]; nn = len(items)
    cfg = replace(get_config("gemma_qwen"), device=dev)
    model, tok = load_with_fallback(hf, mirror, cfg)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    print(f"[{tag}] cycle={CYCLE} nn={nn}", flush=True)

    queries = build_queries(items, unit, nn)
    preamble = few_shot(items, unit, nn, rng, NSHOT, exclude=set())
    H = item_reps(model, tok, blocks, nL, items, dev)
    w, U = cycle_eigenmodes(nn)

    # represented power per mode, averaged over layers -> prominent mode
    powL = []
    for L in range(nL):
        Hc = H[L] - H[L].mean(0); c = U.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p = p / (p.sum() + 1e-12); powL.append(p)
    power = np.mean(powL, 0)
    prominent = int(np.argmax(power))
    fundamental = [1, 2]                                     # k=1 cos/sin pair (lowest non-trivial)
    highfreq = [nn - 1, nn - 2]                              # top (Nyquist) modes
    print(f"[{tag}] power by mode: " + " ".join(f"m{k}={power[k]:.2f}" for k in range(1, min(nn, 7))) +
          f" | prominent=m{prominent}", flush=True)

    base_acc = score_queries(model, tok, blocks, dev, items, unit, nn, queries, preamble)
    print(f"[{tag}] baseline accuracy = {base_acc:.3f} ({len(queries)} queries)", flush=True)

    conds = {"fundamental(pos)": fundamental, "high_freq": highfreq}
    res = {"baseline": base_acc, "power_by_mode": power.tolist(), "prominent_mode": prominent, "ablate": {}}
    for name, modes in conds.items():
        Q = build_Q(H, U, modes, nL, dev)
        acc = score_queries(model, tok, blocks, dev, items, unit, nn, queries, preamble, proj_Q=Q)
        res["ablate"][name] = {"modes": modes, "accuracy": acc, "drop": base_acc - acc}
        print(f"[{tag}] ablate {name:16} modes={modes}: acc={acc:.3f} (drop {base_acc-acc:+.3f})", flush=True)
    # random control: same rank as fundamental (2), random per-layer directions
    rand_accs = []
    for _ in range(2):
        Qr = {}
        for L in range(nL):
            Mt = torch.tensor(rng.standard_normal((cm.hidden_size, len(fundamental))).astype("float32"), device=dev)
            q, _ = torch.linalg.qr(Mt, mode="reduced"); Qr[L] = q
        rand_accs.append(score_queries(model, tok, blocks, dev, items, unit, nn, queries, preamble, proj_Q=Qr))
    res["ablate"]["random"] = {"modes": None, "accuracy": float(np.mean(rand_accs)), "drop": base_acc - float(np.mean(rand_accs))}
    print(f"[{tag}] ablate random (rank {len(fundamental)}): acc={np.mean(rand_accs):.3f} (drop {base_acc-np.mean(rand_accs):+.3f})", flush=True)

    out = {"model": tag, "cycle": CYCLE, "n": nn, "n_layers": nL, **res}
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/cyclic_qa_ablation_{tag}_{CYCLE}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
