"""Do the DAS coordinate axes SURVIVE ablating a head set? (heads -> axes, causally)

Offline cos^2 attribution says which heads BUILD a layer's axes, but only within that layer: ablating a
head at layer L' < L also changes every later layer's input, so the cross-layer question needs the model
re-run. Here a head set is mean-ablated and the per-node attention-output cloud is RE-CAPTURED at every
layer, then compared to the intact cloud through the DAS rotation R that was fitted on the intact model:

    Y = (znode - mean) R^T          [n, r] flattened
    retained = ||Y_ablated||^2 / ||Y_intact||^2      how much axis energy is left
    aligned  = cos^2(Y_ablated, Y_intact)            whether what is left points the same way

Both matter and they come apart: a set can leave the axes pointing the right way but shrunken, or leave
the energy while rotating it. Neighbour mass/accuracy is read off the SAME forward passes so the axes
and the behaviour cannot drift apart for incidental reasons. Controls are drawn at the SAME LAYERS as
the set they replace (RANDMODE=layer), since these sets are depth-structured by construction.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAZY(0) SETS("i6=L15H30+...,d3=L14H19+...") RNPZ(path to the
     layer-sweep R npz) LAYERS("1,10,14,15,16,20,21,25,31") NWALKS(8) SPN(300) CTXLO(400)
     WLEN_CAP(1600) NRAND(5) RANDMODE(layer|all) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/axes_under_ablation<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama"); K = int(os.environ.get("K", "16"))
GRAPH = os.environ.get("GRAPH", "ring"); LAZY = float(os.environ.get("LAZY", "0"))
SETS = {s.split("=")[0]: s.split("=")[1].split("+") for s in os.environ.get("SETS", "").split(",") if "=" in s}
RNPZ = os.environ.get("RNPZ", "")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "1,10,14,15,16,20,21,25,31").split(",")]
NWALKS = int(os.environ.get("NWALKS", "8")); SPN = int(os.environ.get("SPN", "300"))
CTXLO = int(os.environ.get("CTXLO", "400")); WLEN_CAP = int(os.environ.get("WLEN_CAP", "1600"))
NRAND = int(os.environ.get("NRAND", "5")); RANDMODE = os.environ.get("RANDMODE", "layer")
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)
    Rz = np.load(RNPZ)

    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    lr_ = np.random.default_rng(SEED); seqs = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(wl - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        seqs.append((torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev), nodes))

    cap, st = {}, {"abl": None, "means": None}
    def mk(L):
        def pre(_m, args):
            x = args[0]
            if st["abl"] is not None:
                hs = [h for (l, h) in st["abl"] if l == L]
                if hs:
                    x = x.clone()
                    for h in hs:
                        sl = slice(h * hd, (h + 1) * hd)
                        x[0, :, sl] = st["means"][L][sl].to(x.dtype)
            cap[L] = x.detach()
            return (x,) + tuple(args[1:])
        return pre
    hooks = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)) for L in range(nL)]

    def sweep():
        """returns znode per layer + neighbour mass/acc, from one pass over the walks"""
        zs = {L: torch.zeros(n, nH * hd, device=dev) for L in range(nL)}
        zc = torch.zeros(n, device=dev); ok = ms = tot = 0.0
        for ids, nodes in seqs:
            cap.clear(); lg = model(input_ids=ids).logits[0]
            idx = torch.tensor([t + 1 for t in range(len(nodes))], device=dev)
            oh = torch.zeros(len(nodes), n, device=dev)
            oh[torch.arange(len(nodes)), torch.tensor(nodes, device=dev)] = 1.0
            for L in range(nL): zs[L] += oh.t() @ cap[L][0, idx].float()
            zc += oh.sum(0)
            rp = [s for s in range(len(nodes) - 1) if s + 1 >= CTXLO]
            p = torch.softmax(lg[torch.tensor([s + 1 for s in rp], device=dev)][:, cand_t].float(), 1)
            for j, s in enumerate(rp):
                nb = list(graph.adjacency[nodes[s]])
                ms += float(p[j, nb].sum()); ok += int(int(p[j].argmax()) in nb); tot += 1
        return ({L: (zs[L] / zc.clamp(min=1)[:, None]).cpu().numpy() for L in range(nL)},
                {"nbr_mass": round(ms / tot, 4), "nbr_acc": round(ok / tot, 4)})

    # per-layer o_proj-input means for mean-ablation, from the intact pass
    st["abl"] = None
    zint, bint = sweep()
    st["means"] = {L: cap[L][0].float().mean(0) for L in range(nL)}
    print(f"[{tag}] {GRAPH}{n} lazy={LAZY} intact {bint}", flush=True)

    def axes(z):
        out = {}
        for L in LAYERS:
            Zi = zint[L] - zint[L].mean(0, keepdims=True); Za = z[L] - z[L].mean(0, keepdims=True)
            R = Rz[f"L{L}"]
            Yi = (Zi @ R.T).ravel(); Ya = (Za @ R.T).ravel()
            ni = float(Yi @ Yi)
            out[L] = {"retained": round(float(Ya @ Ya) / ni, 4) if ni > 0 else None,
                      "aligned": round(float((Ya @ Yi) ** 2 / max(float(Ya @ Ya) * ni, 1e-12)), 4)}
        return out

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]
    res = {"model": tag, "intact": bint, "rnpz": RNPZ, "sets": SETS, "rows": {}}
    for nm, hl in SETS.items():
        hs = parse(hl); st["abl"] = hs
        z, b = sweep(); a = axes(z)
        rnd = []
        for _ in range(NRAND):
            st["abl"] = [(l, int(rng.choice([x for x in range(nH) if x != h]))) for l, h in hs] \
                if RANDMODE == "layer" else \
                [(int(rng.integers(nL)), int(rng.integers(nH))) for _ in hs]
            zr, br = sweep(); rnd.append((axes(zr), br))
        st["abl"] = None
        res["rows"][nm] = {"heads": hl, "behav": b, "axes": a,
                           "rand_behav": {k: round(float(np.mean([r[1][k] for r in rnd])), 4)
                                          for k in b},
                           "rand_axes": {str(L): {m: round(float(np.mean([r[0][L][m] for r in rnd])), 4)
                                                  for m in ("retained", "aligned")} for L in LAYERS}}
        print(f"\nablate {nm} (k={len(hs)})  nbr_mass {b['nbr_mass']:.4f} acc {b['nbr_acc']:.4f}   "
              f"(matched random: {res['rows'][nm]['rand_behav']})", flush=True)
        print(f"{'L':>4}{'retained':>10}{'rand':>8}{'aligned':>10}{'rand':>8}")
        for L in LAYERS:
            rr = res["rows"][nm]["rand_axes"][str(L)]
            print(f"{L:4}{a[L]['retained']:10.3f}{rr['retained']:8.3f}"
                  f"{a[L]['aligned']:10.3f}{rr['aligned']:8.3f}", flush=True)
        json.dump(res, open(f"{OUTDIR}/axes_under_ablation{OUTTAG}_{tag}.json", "w"), indent=2)
    for h in hooks: h.remove()
    print(f"\nDONE -> {OUTDIR}/axes_under_ablation{OUTTAG}_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
