"""What do the individual dimensions of the DAS rank-16 parity subspace DO? Steer along each basis vector
v_i of the trained rotation (from SAVE_R npz): add ±alpha·v_i to L14H26's output at every node token during
a grid walk, and measure at readout positions (1) the shift of predicted-token probability mass between the
two parity colour classes (which sublattice the dimension pins) and (2) the change in next-token neighbour
validity (how much the dimension is load-bearing for behaviour). Random unit vectors in the head's output
space are the control.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) K(4) NWALKS(8) SAMPLES_PER_NODE(80) CTXLO(200) WLEN_CAP(900)
     R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_<model>.npz) RKEY(4x4_R16)
     AMULT(4) NRAND(8) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/das_basis_steer<OUTTAG>_<model>.json
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
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj
from das_parity_scale import capture_znode

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "8"))
SPN = int(os.environ.get("SAMPLES_PER_NODE", "80")); CTXLO = int(os.environ.get("CTXLO", "200"))
WLEN_CAP = int(os.environ.get("WLEN_CAP", "900"))
R_NPZ = os.environ.get("R_NPZ", f"runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_{GEN_MODEL}.npz")
RKEY = os.environ.get("RKEY", "4x4_R16")
AMULT = float(os.environ.get("AMULT", "4")); NRAND = int(os.environ.get("NRAND", "8"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K, n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    wdata, znode = capture_znode(model, tok, blocks, cm, graph, cfg, dev, csl, hd, n, CTXLO, NWALKS)

    R = np.load(R_NPZ)[RKEY]                       # r x hd, orthonormal rows
    rand = rng.standard_normal((NRAND, hd)); rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    basis = np.concatenate([R, rand]); names = [f"das{i}" for i in range(R.shape[0])] + [f"rand{i}" for i in range(NRAND)]
    # per-direction magnitude: AMULT x the std of node-mean projections onto that direction
    stds = np.abs(znode @ basis.T).std(0); stds = np.maximum(stds, np.median(stds) * 0.25)
    alphas = AMULT * stds

    state = {"delta": None}
    def patch_pre(_m, args):
        if state["delta"] is not None:
            x = args[0].clone(); x[0, :, csl] = x[0, :, csl] + state["delta"].to(x.dtype)
            return (x,) + tuple(args[1:])
    ph = proj.register_forward_pre_hook(patch_pre)

    def eval_all(vec_alpha):
        """vec_alpha: hd vector (already scaled) or None. Returns (nbr_validity, parity_mass=lse_pos-lse_neg)."""
        vals = []; pmass = []
        for w in wdata:
            if vec_alpha is None: state["delta"] = None
            else:
                Dl = torch.zeros(w["seqlen"], hd, device=dev)
                for t, _nd in w["ntok"]: Dl[t] = vec_alpha
                state["delta"] = Dl
            logits = model(input_ids=w["ids"]).logits[0][torch.tensor(w["readpos"], device=dev)][:, cand_t].float()
            state["delta"] = None
            lsm = torch.log_softmax(logits, 1)
            pmass.append((torch.logsumexp(lsm[:, pos_idx], 1) - torch.logsumexp(lsm[:, neg_idx], 1)).mean().item())
            top = cand_t[logits.argmax(1)].cpu().numpy()
            w2n = {int(cand_t[i]): i for i in range(n)}
            ok = [w2n[int(tt)] in nbr[nd] for tt, nd in zip(top, w["readnode"])]
            vals.append(float(np.mean(ok)))
        return float(np.mean(vals)), float(np.mean(pmass))

    base_val, base_pm = eval_all(None)
    print(f"[{tag}] {K}x{K} base validity={base_val:.3f} parity_mass={base_pm:+.3f} wl={wl}", flush=True)
    vt = torch.tensor(basis, dtype=torch.float32, device=dev)
    rows = []
    for i, nm in enumerate(names):
        vp, pp = eval_all(vt[i] * float(alphas[i]))
        vm, pm = eval_all(-vt[i] * float(alphas[i]))
        rows.append({"dim": nm, "alpha": round(float(alphas[i]), 3),
                     "d_validity_plus": round(vp - base_val, 4), "d_validity_minus": round(vm - base_val, 4),
                     "parity_shift_plus": round(pp - base_pm, 4), "parity_shift_minus": round(pm - base_pm, 4)})
        print(f"  {nm:7} a={alphas[i]:.2f}  dval {vp-base_val:+.3f}/{vm-base_val:+.3f}  pshift {pp-base_pm:+.3f}/{pm-base_pm:+.3f}", flush=True)
    ph.remove()
    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "k": K, "rkey": RKEY, "amult": AMULT,
           "base_validity": round(base_val, 4), "base_parity_mass": round(base_pm, 4), "dims": rows}
    p = f"{OUTDIR}/das_basis_steer{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
