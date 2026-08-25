"""Stage 4b: the Steer2Edit NULL geometry, in closed form.

Steer2Edit (arXiv:2602.09870) sets, per editable component i,

    g_i  = cos(v_i, W_i mu_i),        mu_i = E[h_i]
    k_i  = W_i^T v_i / ||W_i^T v_i||
    dW_i = sign(g_i) max(|g_i| - rho*alpha, 0)/(rho(1-alpha)) * v_hat_i k_i^T

Every factor is a function of v and the FROZEN base weights, so the pairwise
weight geometry never needs a dW to be materialised:

    <dW_a, dW_b>_F = sum_i c_i^a c_i^b (v_hat^a . v_hat^b)(k_hat^a . k_hat^b)

verified against explicit construction to ~1e-17. That makes this the exact null
for stage 5: the weight geometry you get when the edit carries no information
beyond the vector. It also means running Steer2Edit as a real arm of an
activation-vs-weight study would be circular -- there is only one information
source.

Components, as in the paper: each attention head's output projection (column
blocks of o_proj) and each column of down_proj.

Output: out/s2e_null.npz {names, cos}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import chat, layer_grid, load_base, out_path  # noqa: E402

RHO = float(os.environ.get("RHO", 1.0))
ALPHA = float(os.environ.get("S2E_ALPHA", 0.02))


@torch.no_grad()
def mean_inputs(model, tok, texts, layers):
    """mu_i = mean input activation to o_proj and down_proj at each layer."""
    acc, hooks = {}, []

    def mk(key):
        def f(mod, inp, out):
            h = inp[0].detach().float()
            h = h.reshape(-1, h.shape[-1])
            s, n = acc.get(key, (0.0, 0))
            acc[key] = (s + h.sum(0).cpu().numpy(), n + h.shape[0])
        return f

    for l in layers:
        blk = model.model.layers[l]
        hooks.append(blk.self_attn.o_proj.register_forward_hook(mk((l, "o_proj"))))
        hooks.append(blk.mlp.down_proj.register_forward_hook(mk((l, "down_proj"))))
    try:
        for t in texts:
            enc = tok(t, return_tensors="pt").to(model.device)
            model(**enc)
    finally:
        for h in hooks:
            h.remove()
    return {k: s / max(n, 1) for k, (s, n) in acc.items()}


def components(model, layer, n_head, head_dim):
    """[(W_i, slice)] -- attention heads (o_proj column blocks) and down_proj columns."""
    blk = model.model.layers[layer]
    Wo = blk.self_attn.o_proj.weight.detach().float().cpu().numpy()   # (d, n_head*head_dim)
    Wd = blk.mlp.down_proj.weight.detach().float().cpu().numpy()      # (d, d_ff)
    out = []
    for h in range(n_head):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        out.append(("o_proj", sl, Wo[:, sl]))
    out.append(("down_proj", slice(None), Wd))
    return out


def main():
    names = C.concept_set()
    failed = set(json.load(open(out_path("gen_failed.json")))) if \
        os.path.exists(out_path("gen_failed.json")) else set()
    names = [n for n in names if n not in failed]
    kind = os.environ.get("VKIND", "desc")

    model, tok = load_base()
    layers = layer_grid(model)
    L = int(os.environ.get("LAYER", layers[len(layers) // 2]))
    cfg = model.config
    n_head = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_head)

    texts = [chat(tok, C.NEUTRAL, p) for p in C.PROBE_PROMPTS]
    mus = mean_inputs(model, tok, texts, [L])
    V = np.load(out_path("vecs.npz"))
    vs = {n: V[f"{n}|{kind}|{L}"].astype(np.float64) for n in names}

    comps = components(model, L, n_head, head_dim)
    print(f"[s2e] L{L}, {len(comps)} components (heads + down_proj), {len(names)} concepts",
          flush=True)

    # per component: c_i and k_hat_i for every concept
    Cc = np.zeros((len(names), len(comps)))
    Ks = []
    for ci, (kindname, sl, Wi) in enumerate(comps):
        mu = mus[(L, kindname)]
        mu = mu[sl] if kindname == "o_proj" else mu
        Wmu = Wi @ mu
        nWmu = np.linalg.norm(Wmu) + 1e-12
        Kc = np.zeros((len(names), Wi.shape[1]))
        for ni, n in enumerate(names):
            v = vs[n]
            g = float(v @ Wmu / (np.linalg.norm(v) * nWmu))
            Cc[ni, ci] = np.sign(g) * max(abs(g) - RHO * ALPHA, 0) / (RHO * (1 - ALPHA))
            k = Wi.T @ v
            Kc[ni] = k / (np.linalg.norm(k) + 1e-12)
        Ks.append(Kc)
        if ci % 8 == 0:
            print(f"    component {ci + 1}/{len(comps)}", flush=True)

    vhat = np.stack([vs[n] / np.linalg.norm(vs[n]) for n in names])
    Vc = vhat @ vhat.T                                   # (N,N) cos(v_a, v_b)

    num = np.zeros((len(names), len(names)))
    for ci in range(len(comps)):
        num += np.outer(Cc[:, ci], Cc[:, ci]) * Vc * (Ks[ci] @ Ks[ci].T)
    nrm = np.sqrt((Cc ** 2).sum(1))
    Cn = num / (np.outer(nrm, nrm) + 1e-12)

    np.savez(out_path("s2e_null.npz"), names=np.array(names), cos=Cn, act_cos=Vc)
    off = [(Cn[i, j], Vc[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    o = np.array(off)
    print(f"\n[s2e] mean null cos {o[:,0].mean():+.4f} | mean activation cos {o[:,1].mean():+.4f}"
          f" | corr {np.corrcoef(o[:,0], o[:,1])[0,1]:+.4f}")
    print("S2E_DONE")


if __name__ == "__main__":
    main()
