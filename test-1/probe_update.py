"""Mechanistic probe of the in-context 'posterior update' in Llama-3.1-8B.

Part 1 -- BEHAVIORAL UPDATE KERNEL. On plain grid random walks, regress the model's
next-node log-odds on the AGES of past evidence:

    logp_s(j) - mean_j logp_s(j)  ~  sum_B w_row(B)  * #[a_s -> j transitions, age in B]
                                   + sum_B w_col(B)  * #[c != a_s -> j transitions, age in B]

w_row(B): weight of a matched-row observation (a->j raises p(j|a)) by age bin -- the
empirical forgetting kernel of the update. w_col(B): row-MISmatched leakage (does seeing
c->j raise p(j|a)?) -- distinguishes a true conditional/transition update from a unigram
popularity update. Log-linearity in counts is the conjugate-count signature; compare
w_row to the Bayesian surrogate's fitted gamma^age.

Part 2 -- MECHANISM LOCATION. Capture attention patterns; for each occurrence of node a,
measure per-head attention mass on the SUCCESSOR SLOTS of earlier occurrences of a
(positions u+1 with x_u = a; the induction-copy pattern). Rank heads, and report the
age profile of the top heads' successor-slot attention -- the implementation of w_row.

Env: NWALKS(24) WLEN(400) ATTN_WALKS(8) SMIN(50) OUTDIR(out_probe) WORDS16 DEVICE
Out: <OUTDIR>/update_kernel.json
"""
from __future__ import annotations
import os, sys, json
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import graph as G

NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "400"))
ATTN_WALKS = int(os.environ.get("ATTN_WALKS", "8"))
SMIN = int(os.environ.get("SMIN", "50"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(_here, "out_probe"))
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL = "NousResearch/Meta-Llama-3.1-8B"
BINS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 10**9)]
if os.environ.get("EXTBINS") == "1":
    BINS = BINS[:-1] + [(321, 640), (641, 1280), (1281, 10**9)]


def binof(age):
    for bi, (lo, hi) in enumerate(BINS):
        if lo <= age <= hi:
            return bi
    return None


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    model.to(DEVICE).eval()
    nL = model.config.num_hidden_layers
    nH = model.config.num_attention_heads

    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4,
                  n_walks=NWALKS, walk_length=WLEN, seed=11)
    grid = G.build_graph(cfg)
    if os.environ.get("WORDS16"):
        grid.words = os.environ["WORDS16"].split(",")
    words = grid.words
    n = 16
    walks = G.generate_walks(grid, cfg)
    cand = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1
        cand.append(ids[0])
    cand_t = torch.tensor(cand, device=DEVICE)
    bos = tok.bos_token_id
    nB = len(BINS)

    # ---- part 1: kernel regression ----------------------------------------
    Xr, Xc, Yv = [], [], []
    for wi, wk in enumerate(walks):
        ids = torch.tensor([[bos] + [cand[x] for x in wk.nodes]], device=DEVICE)
        lg = model(input_ids=ids).logits[0].float()
        nodes = wk.nodes
        for s in range(SMIN, len(nodes) - 1):
            a = nodes[s]
            lp = torch.log_softmax(lg[1 + s][cand_t], 0).cpu().numpy()
            y = lp - lp.mean()
            fr = np.zeros((n, nB))
            fc = np.zeros((n, nB))
            for u in range(s):
                b = binof(s - u)
                if b is None:
                    continue
                j = nodes[u + 1]
                if nodes[u] == a:
                    fr[j, b] += 1
                else:
                    fc[j, b] += 1
            Yv.append(y)
            Xr.append(fr - fr.mean(0))      # center across candidates (per-position const)
            Xc.append(fc - fc.mean(0))
        if (wi + 1) % 8 == 0:
            print(f"kernel pass walk {wi+1}/{NWALKS}", flush=True)
    Y = np.concatenate(Yv)                                    # [S*16]
    X = np.concatenate([np.concatenate(Xr, 0), np.concatenate(Xc, 0)], 1)  # [S*16, 2nB]
    lam = 1e-3
    W = np.linalg.solve(X.T @ X + lam * np.eye(2 * nB), X.T @ Y)
    resid = Y - X @ W
    r2 = 1 - (resid ** 2).sum() / (Y ** 2).sum()
    w_row, w_col = W[:nB], W[nB:]
    print("w_row (matched-row kernel by age bin):", np.round(w_row, 3), flush=True)
    print("w_col (row-mismatch leakage):        ", np.round(w_col, 3), flush=True)

    # exponential fit to w_row at bin centers
    centers = np.array([np.sqrt(lo * min(hi, WLEN)) for lo, hi in BINS])
    pos = w_row > 0
    if pos.sum() >= 3:
        A = np.vstack([np.ones(pos.sum()), centers[pos]]).T
        coef, *_ = np.linalg.lstsq(A, np.log(w_row[pos]), rcond=None)
        gamma_hat = float(np.exp(coef[1]))
    else:
        gamma_hat = None

    # ---- part 2: successor-slot attention profiles -------------------------
    prof = np.zeros((nL, nH, nB))
    prof_cnt = np.zeros(nB)
    self_attn = np.zeros((nL, nH))
    occ_used = 0
    for wi, wk in enumerate(walks[:ATTN_WALKS] if ATTN_WALKS > 0 else []):
        ids = torch.tensor([[bos] + [cand[x] for x in wk.nodes]], device=DEVICE)
        out = model(input_ids=ids, output_attentions=True)
        att = [a[0].float() for a in out.attentions]          # nL x [H, T, T]
        nodes = wk.nodes
        for s in range(SMIN, len(nodes) - 1):
            a = nodes[s]
            q = 1 + s
            slots = {}
            for u in range(s):
                if nodes[u] == a:
                    b = binof(s - u)
                    if b is not None:
                        slots.setdefault(b, []).append(1 + u + 1)
            if not slots:
                continue
            occ_used += 1
            for b, ps in slots.items():
                prof_cnt[b] += 1
                pt = torch.tensor(ps, device=DEVICE)
                for L in range(nL):
                    prof[L, :, b] += att[L][:, q, pt].sum(-1).cpu().numpy()
        del att, out
        print(f"attn pass walk {wi+1}/{ATTN_WALKS}", flush=True)
    prof = prof / np.maximum(prof_cnt, 1)[None, None, :]

    # rank heads by short-age successor-slot attention (bins 0-2)
    score = prof[:, :, :3].mean(-1)
    order = np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1], score.shape))[0]
    top = [(int(L), int(h), float(score[L, h])) for L, h in order[:12]]
    top_profile = np.mean([prof[L, h] for L, h, _ in top[:6]], 0)
    print("top successor-slot heads (L,h,score):", top[:8], flush=True)

    json.dump({
        "bins": BINS, "bin_centers": centers.tolist(),
        "w_row": w_row.tolist(), "w_col": w_col.tolist(), "regression_r2": float(r2),
        "gamma_hat_from_kernel": gamma_hat,
        "top_heads": top, "top6_head_age_profile": top_profile.tolist(),
        "n_occurrences_regression": int(len(Yv)), "n_occ_attention": occ_used,
    }, open(os.path.join(OUTDIR, "update_kernel.json"), "w"), indent=2)
    print("DONE ->", os.path.join(OUTDIR, "update_kernel.json"), flush=True)


if __name__ == "__main__":
    main()
