"""FULL head-to-head composition map: which later heads USE which earlier heads' outputs?

For every early head e (layers 0..EARLY_MAX) we mean-ablate its output and measure, for every later
head l, three distinct kinds of dependence:

  OV / direct   ||delta z_l|| / ||z_l||        — did l's OUTPUT change? (l reads e's write through its
                                                 value path, or through anything upstream of it)
  QK / pattern  0.5 * ||delta attn_l||_1       — did l's ATTENTION PATTERN change? (e's output feeds l's
                                                 query/key computation: l attends elsewhere without e)
  parity-write  |delta (z_l . w_l)| / sd       — did l's WRITE TO THE PARITY DIRECTION change? (w_l =
                                                 W_O[l,h]^T v; connects the map to this project's variable)

One forward pass per early head yields a full row of each matrix, so the whole [n_early x n_late] map
costs n_early passes. Measured on in-context grid walks at readout positions, where parity is real.

Interpretation: OV-effect without QK-effect = l consumes e's output as content. QK-effect = e determines
where l looks. Large parity-write effect = e is upstream of l's contribution to the parity variable —
this is what shows where an early writer like L2H26 (known parity writer, but inert as a steering site)
actually sends its influence.

Env: GEN_MODEL(Llama) EARLY_MAX(15) K(4) NWALKS(3) WLEN(420) CTXLO(300) NQ(40)
     PAR_NPY(runs/axes/4_circuits/parity/seed_stable_r1_<model>.npy) TOPK(40) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/head_composition_map<OUTTAG>_<model>.json  (+ .npz with the full matrices)
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
EARLY_MAX = int(os.environ.get("EARLY_MAX", "15"))
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "420")); CTXLO = int(os.environ.get("CTXLO", "300"))
NQ = int(os.environ.get("NQ", "40")); TOPK = int(os.environ.get("TOPK", "40"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
PAR_NPY = os.environ.get("PAR_NPY", f"{P}/seed_stable_r1_{GEN_MODEL}.npy")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH
    rng = np.random.default_rng(SEED)

    v = None
    if os.path.exists(PAR_NPY):
        v = np.load(PAR_NPY).astype(np.float32); v = v / np.linalg.norm(v)
        vt = torch.tensor(v, device=dev)
        Wv = torch.stack([ (attn_proj(blocks[l], cm)[0].weight.detach().float().t() @ vt).view(nH, hd)
                           for l in range(nL) ])                       # [nL, nH, hd]
        print(f"[{tag}] parity direction loaded from {PAR_NPY}", flush=True)
    else:
        Wv = None; print(f"[{tag}] no parity direction at {PAR_NPY}; skipping parity-write matrix", flush=True)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); walks = G.generate_walks(graph, cfg)

    caps = {}
    hooks = []
    for l in range(nL):
        proj = attn_proj(blocks[l], cm)[0]
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mk(l)))

    abl = {"l": None, "h": None}
    def patch(l):
        def hh(_m, args):
            if abl["l"] != l: return
            x = args[0].clone(); s = slice(abl["h"] * hd, (abl["h"] + 1) * hd)
            x[0, :, s] = x[0, :, s].mean(0, keepdim=True)              # mean-ablate this head's output
            return (x,) + tuple(args[1:])
        return hh
    for l in range(EARLY_MAX + 1):
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(patch(l)))

    nE = (EARLY_MAX + 1) * nH
    OV = np.zeros((nE, nL * nH)); QK = np.zeros((nE, nL * nH)); PW = np.zeros((nE, nL * nH))
    cnt = np.zeros(nE)

    for wi, wk in enumerate(walks):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk)
        read = [spans[s][-1] for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if not read: continue
        rt = torch.tensor(read, device=dev)
        qsel = torch.tensor(sorted(rng.choice(len(read), min(NQ, len(read)), replace=False)), device=dev)
        qpos = rt[qsel]
        abl["l"] = None; caps.clear()
        o0 = model(input_ids=ids, output_attentions=True)
        Z0 = torch.stack([caps[l][0, rt].float().view(len(read), nH, hd) for l in range(nL)])      # [nL,T,nH,hd]
        A0 = torch.stack([o0.attentions[l][0][:, qpos, :].float() for l in range(nL)])     # [nL,nH,NQ,T]
        z0n = Z0.norm(dim=-1).mean(1)                                                      # [nL,nH]
        if Wv is not None:
            p0 = (Z0 * Wv[:, None, :, :]).sum(-1)                                          # [nL,T,nH]
            psd = p0.std(1) + 1e-6                                                         # [nL,nH]
        del o0
        for le in range(EARLY_MAX + 1):
            for he in range(nH):
                abl["l"] = le; abl["h"] = he; caps.clear()
                o1 = model(input_ids=ids, output_attentions=True)
                Z1 = torch.stack([caps[l][0, rt].float().view(len(read), nH, hd) for l in range(nL)])
                ov = ((Z1 - Z0).norm(dim=-1).mean(1) / (z0n + 1e-6))                       # [nL,nH]
                qk = torch.zeros(nL, nH, device=dev)
                for l in range(le + 1, nL):
                    a1 = o1.attentions[l][0][:, qpos, :].float()
                    qk[l] = 0.5 * (a1 - A0[l]).abs().sum(-1).mean(-1)
                pw = torch.zeros(nL, nH, device=dev)
                if Wv is not None:
                    p1 = (Z1 * Wv[:, None, :, :]).sum(-1)
                    pw = ((p1 - p0).abs().mean(1) / psd)
                ei = le * nH + he
                OV[ei] += ov.flatten().cpu().numpy(); QK[ei] += qk.flatten().cpu().numpy()
                PW[ei] += pw.flatten().cpu().numpy(); cnt[ei] += 1
                del o1, Z1
            print(f"  [walk {wi}] early layer {le} done", flush=True)
        del Z0, A0
        gc.collect(); torch.cuda.empty_cache()
    for h in hooks: h.remove()
    abl["l"] = None

    c = np.maximum(cnt, 1)[:, None]
    OV /= c; QK /= c; PW /= c
    # mask out non-causal (late <= early) entries
    ei_l = np.repeat(np.arange(EARLY_MAX + 1), nH)
    li_l = np.repeat(np.arange(nL), nH)
    mask = li_l[None, :] > ei_l[:, None]
    OV *= mask; QK *= mask; PW *= mask
    nm_e = [f"L{i//nH}H{i%nH}" for i in range(nE)]
    nm_l = [f"L{i//nH}H{i%nH}" for i in range(nL * nH)]

    def top(Mx, k=TOPK):
        idx = np.argsort(Mx, axis=None)[::-1][:k]
        return [{"early": nm_e[i // (nL * nH)], "late": nm_l[i % (nL * nH)],
                 "val": round(float(Mx.flat[i]), 4)} for i in idx]
    out = {"model": tag, "early_max": EARLY_MAX, "n_heads": nH, "n_layers": nL,
           "walks": NWALKS, "wlen": WLEN, "ctxlo": CTXLO, "nq": NQ,
           "top_OV": top(OV), "top_QK": top(QK), "top_parity_write": top(PW) if Wv is not None else None,
           "hub_early_by_OV": [{"head": nm_e[i], "total": round(float(OV[i].sum()), 3)}
                               for i in np.argsort(OV.sum(1))[::-1][:20]],
           "hub_early_by_QK": [{"head": nm_e[i], "total": round(float(QK[i].sum()), 3)}
                               for i in np.argsort(QK.sum(1))[::-1][:20]],
           "hub_late_by_OV": [{"head": nm_l[i], "total": round(float(OV[:, i].sum()), 3)}
                              for i in np.argsort(OV.sum(0))[::-1][:20]]}
    if Wv is not None:
        out["hub_early_by_parity_write"] = [{"head": nm_e[i], "total": round(float(PW[i].sum()), 3)}
                                            for i in np.argsort(PW.sum(1))[::-1][:20]]
    p = f"{OUTDIR}/head_composition_map{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    np.savez_compressed(p.replace(".json", ".npz"), OV=OV.astype("float32"), QK=QK.astype("float32"),
                        PW=PW.astype("float32"), early=np.array(nm_e), late=np.array(nm_l))
    print("\nTOP OV (late head's output changes when early head ablated):", flush=True)
    for d in out["top_OV"][:12]: print(f"   {d['early']:8} -> {d['late']:8} {d['val']:.3f}", flush=True)
    print("TOP QK (late head's ATTENTION changes):", flush=True)
    for d in out["top_QK"][:12]: print(f"   {d['early']:8} -> {d['late']:8} {d['val']:.3f}", flush=True)
    if Wv is not None:
        print("TOP parity-write influence:", flush=True)
        for d in out["top_parity_write"][:12]: print(f"   {d['early']:8} -> {d['late']:8} {d['val']:.3f}", flush=True)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
