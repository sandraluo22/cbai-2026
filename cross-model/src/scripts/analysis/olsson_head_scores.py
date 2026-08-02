"""Bridge our task-specific head identification to the STANDARD (Olsson et al. 2022) scores, so our
PLUS/MINUS sign families can be compared directly with Arditi's induction / previous-token head lists.

Standard scores, measured on REPEATED RANDOM TOKEN SEQUENCES (no graph, no task):
  prev_token_score  mean attention from position t to position t-1
  induction_score   mean attention from t to (previous occurrence of token t) + 1, i.e. the token that
                    FOLLOWED the current token last time — the prefix-matching motif

Both are task-independent by construction, which is exactly the methodological contrast: our carrier /
necessity / write-mechanism scores are measured ON the grid task and are specific to the parity variable.

Then we correlate the two families of scores over all heads and report where our circuit heads land.

Env: GEN_MODEL(Llama) SEQLEN(60) NSEQ(12) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/olsson_head_scores_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
SEQLEN = int(os.environ.get("SEQLEN", "60")); NSEQ = int(os.environ.get("NSEQ", "12"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
OUTDIR = os.environ.get("OUTDIR", P)

PLUS = ["L21H2","L14H19","L14H17","L10H2","L2H22","L7H25","L8H11","L1H20","L4H16","L13H18"]
MINUS = ["L16H1","L16H20","L14H26","L15H30","L4H12","L9H11","L1H21","L21H10","L3H17"]
NEUT = ["L2H26","L25H7"]


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    rng = np.random.default_rng(SEED)
    V = model.get_input_embeddings().weight.shape[0]

    prev = np.zeros((nL, nH)); ind = np.zeros((nL, nH)); cnt = 0
    for _ in range(NSEQ):
        toks = rng.integers(1000, min(V, 30000), size=SEQLEN).tolist()
        ids = torch.tensor([[tok.bos_token_id or toks[0]] + toks + toks], device=dev)
        o = model(input_ids=ids, output_attentions=True)
        T = ids.shape[1]
        # second copy occupies positions 1+SEQLEN .. 1+2*SEQLEN-1
        q = np.arange(1 + SEQLEN, 1 + 2 * SEQLEN - 1)          # queries in the repeat
        prev_k = q - 1                                          # previous token
        ind_k = q - SEQLEN + 1                                  # token AFTER the earlier occurrence
        for l in range(nL):
            A = o.attentions[l][0].float().cpu().numpy()        # [nH, T, T]
            prev[l] += A[:, q, prev_k].mean(1)
            ind[l] += A[:, q, ind_k].mean(1)
        cnt += 1
        del o
    prev /= cnt; ind /= cnt

    names = [f"L{l}H{h}" for l in range(nL) for h in range(nH)]
    pf, inf = prev.flatten(), ind.flatten()
    zp = (pf - pf.mean()) / pf.std(); zi = (inf - inf.mean()) / inf.std()
    order_p = np.argsort(-pf); order_i = np.argsort(-inf)
    print(f"[{tag}] top-10 PREVIOUS-TOKEN heads: " + ", ".join(f"{names[i]}({pf[i]:.2f})" for i in order_p[:10]), flush=True)
    print(f"[{tag}] top-10 INDUCTION heads:      " + ", ".join(f"{names[i]}({inf[i]:.2f})" for i in order_i[:10]), flush=True)

    def fam(lst, label):
        idx = [names.index(x) for x in lst if x in names]
        print(f"\n  {label} (n={len(idx)}): prev_z mean={zp[idx].mean():+.2f}  ind_z mean={zi[idx].mean():+.2f}")
        for x in lst:
            i = names.index(x)
            rp = int((pf > pf[i]).sum()) + 1; ri = int((inf > inf[i]).sum()) + 1
            print(f"     {x:8} prev={pf[i]:.3f}(z{zp[i]:+.1f}, rank {rp:4})   ind={inf[i]:.3f}(z{zi[i]:+.1f}, rank {ri:4})")
        return {"prev_z_mean": round(float(zp[idx].mean()), 3), "ind_z_mean": round(float(zi[idx].mean()), 3)}
    out = {"model": tag, "seqlen": SEQLEN, "nseq": NSEQ,
           "top_prev_token": [{"head": names[i], "score": round(float(pf[i]), 4)} for i in order_p[:20]],
           "top_induction": [{"head": names[i], "score": round(float(inf[i]), 4)} for i in order_i[:20]],
           "families": {}}
    out["families"]["PLUS(history-agg)"] = fam(PLUS, "PLUS  (broad, same-node history, +coupling)")
    out["families"]["MINUS(local-flip)"] = fam(MINUS, "MINUS (peaked, previous token, -coupling)")
    out["families"]["NEUTRAL"] = fam(NEUT, "NEUTRAL (scaffold / reader)")
    out["per_head"] = {names[i]: {"prev": round(float(pf[i]), 4), "ind": round(float(inf[i]), 4)}
                       for i in range(len(names))}
    # do our carrier scores correlate with the standard scores?
    f = f"{P}/head_interchange_roles_rot90_Llama.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        carrier = -z["patch"].flatten()
        out["corr_carrier_vs_prev"] = round(float(np.corrcoef(carrier, pf)[0, 1]), 3)
        out["corr_carrier_vs_induction"] = round(float(np.corrcoef(carrier, inf)[0, 1]), 3)
        print(f"\ncorr(our parity-carrier score, prev_token_score) = {out['corr_carrier_vs_prev']:+.3f}")
        print(f"corr(our parity-carrier score, induction_score)  = {out['corr_carrier_vs_induction']:+.3f}")
    p = f"{OUTDIR}/olsson_head_scores_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
