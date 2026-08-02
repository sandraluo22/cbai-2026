"""Does the parity direction have ONE consistent output effect that averaging destroyed, or is its effect
context-dependent, or is it tracking a CONTEXT-GLOBAL quantity rather than any token feature?

Three analyses on one capture pass, at the residual site (LAYER):

(A) LOGIT-DIFF SPECTRUM. For every position, D_t = logits(ablated) - logits(clean) over the whole
    vocabulary. Instead of averaging D_t over positions (which cancels opposing effects — the flaw in the
    earlier readout), we take the SPECTRUM of the position x vocab matrix via a random sketch
    (Johnson-Lindenstrauss, SKETCH dims) plus an exact cross-covariance so the top components can be
    recovered in real vocabulary space. Rank-1 dominance => one consistent effect, readable directly.
    Flat spectrum => genuinely context-dependent, and we then cluster positions by their component
    scores and interpret each cluster. Random directions get the same treatment for comparison.

(B) COEFFICIENT REGRESSION. Regress the direction's coefficient h.v against a battery that includes
    CONTEXT-GLOBAL quantities (next-token entropy, running mean surprisal, position, distance since the
    token last occurred, in-context vs unigram predictability) alongside the token-local features that
    already failed. Motivated by the grid result that parity is ACCUMULATED over context, not local:
    if the coefficient tracks uncertainty / context-reliance rather than any lexical category, that
    explains the diffuse damage and the absent token profile at once.

(C) SEED STABILITY. Principal angles between residual DAS subspaces trained with different seeds
    (loaded from *_seed*.npz), and the seed-stable common subspace. These subspaces are known to be
    optimizer-unstable (r1 flipped 21% in one run, 73% in another), so this bounds how much
    interpretive weight any single one can carry.

Env: GEN_MODEL(Llama) LAYER(14) NDOCS(200) MAXTOK(320) SKETCH(256) NPOS_SVD(20000) NCOMP(6) NCLUST(4)
     SEED_GLOB(das_multihead_resid_L14_save_seed*_Llama.npz) PAR_NPZ OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_effect_spectrum<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, glob
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
LAYER = int(os.environ.get("LAYER", "14"))
NDOCS = int(os.environ.get("NDOCS", "200")); MAXTOK = int(os.environ.get("MAXTOK", "320"))
SKETCH = int(os.environ.get("SKETCH", "256")); NPOS_SVD = int(os.environ.get("NPOS_SVD", "20000"))
NCOMP = int(os.environ.get("NCOMP", "6")); NCLUST = int(os.environ.get("NCLUST", "4"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
PAR_NPZ = os.environ.get("PAR_NPZ", f"{P}/das_multihead_resid_L{LAYER}_save_{GEN_MODEL}.npz")
SEED_GLOB = os.environ.get("SEED_GLOB", f"{P}/das_multihead_resid_L{LAYER}_save_seed*_{GEN_MODEL}.npz")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


def principal_angles(A, B):
    """A,B: [r, d] orthonormal rows. Returns cosines of principal angles, descending."""
    return np.linalg.svd(A @ B.T, compute_uv=False)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); dm = cm.hidden_size
    rng = np.random.default_rng(0)

    # ---------- (C) seed stability ----------
    seedfiles = sorted(glob.glob(SEED_GLOB))
    stab = {"files": [os.path.basename(f) for f in seedfiles]}
    if len(seedfiles) >= 2:
        for r in (1, 8):
            mats = []
            for f in seedfiles:
                z = np.load(f); k = f"4x4_r{r}"
                if k in z.files:
                    q, _ = np.linalg.qr(z[k].astype(np.float64).T); mats.append(q.T[:r])
            if len(mats) < 2: continue
            pw = []
            for i in range(len(mats)):
                for j in range(i + 1, len(mats)):
                    ca = principal_angles(mats[i], mats[j])
                    pw.append({"pair": [i, j], "mean_cos2": round(float((ca ** 2).mean()), 4),
                               "cosines": [round(float(x), 3) for x in ca[:8]]})
            rnd = []
            for _ in range(50):
                a = np.linalg.qr(rng.standard_normal((dm, r)))[0].T
                b = np.linalg.qr(rng.standard_normal((dm, r)))[0].T
                rnd.append(float((principal_angles(a, b) ** 2).mean()))
            stab[f"r{r}"] = {"n_seeds": len(mats), "pairs": pw,
                             "mean_cos2_across_seeds": round(float(np.mean([p["mean_cos2"] for p in pw])), 4),
                             "random_baseline_mean_cos2": round(float(np.mean(rnd)), 4),
                             "expected_random": round(r / dm, 5)}
            print(f"[stability r{r}] across-seed mean cos^2 = {stab[f'r{r}']['mean_cos2_across_seeds']}"
                  f"  vs random {stab[f'r{r}']['random_baseline_mean_cos2']}", flush=True)
            # seed-stable common subspace (top eigenvectors of the mean projector)
            Pm = np.mean([m.T @ m for m in mats], 0)
            ev, evec = np.linalg.eigh(Pm)
            stab[f"r{r}"]["projector_eigs_top8"] = [round(float(x), 3) for x in ev[::-1][:8]]
            if r == 1:
                np.save(f"{OUTDIR}/seed_stable_r1_{tag}.npy", evec[:, -1])
                stab["stable_r1_saved"] = True
    else:
        print("[stability] fewer than 2 seed files; skipping", flush=True)

    # ---------- directions to probe ----------
    z = np.load(PAR_NPZ)
    subs = {}
    for r in (1, 8):
        k = f"4x4_r{r}"
        if k in z.files:
            q, _ = np.linalg.qr(z[k].astype(np.float64).T); subs[f"par_r{r}"] = q.T[:r]
    if os.path.exists(f"{OUTDIR}/seed_stable_r1_{tag}.npy"):
        v = np.load(f"{OUTDIR}/seed_stable_r1_{tag}.npy"); subs["par_stable_r1"] = (v / np.linalg.norm(v))[None, :]
    subs["rand_r1"] = np.linalg.qr(rng.standard_normal((dm, 1)))[0].T
    subs["rand_r8"] = np.linalg.qr(rng.standard_normal((dm, 8)))[0].T
    names = list(subs)
    Q = {n: torch.tensor(v, dtype=torch.float32, device=dev) for n, v in subs.items()}
    print(f"[{tag}] probing: {names}", flush=True)

    V = model.get_output_embeddings().weight.shape[0]
    R = torch.tensor(rng.standard_normal((V, SKETCH)) / np.sqrt(SKETCH), dtype=torch.float32, device=dev)

    state = {"Q": None}
    def rh(_m, _i, out):
        if state["Q"] is None: return out
        h = out[0] if isinstance(out, tuple) else out
        x = h[0].float(); q = state["Q"]
        h = h.clone(); h[0] = (x - (x @ q.t()) @ q).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hook = blocks[LAYER].register_forward_hook(rh)

    import string as _string
    def catc(p):
        wi = p.startswith("Ġ") or p.startswith("▁") or p.startswith(" ")
        core = p.lstrip("Ġ▁ ")
        if core and all(c in _string.punctuation for c in core): return "punct"
        if core and all(c.isdigit() for c in core): return "digit"
        return "word_initial" if wi else "continuation"

    from datasets import load_dataset
    docs = []
    for ex in load_dataset(DATASET, split="train", streaming=True):
        if len(docs) >= NDOCS: break
        t = ex["text"]
        if t and len(t) >= 40: docs.append(t)

    sk = {n: [] for n in names}
    CC = {n: torch.zeros(V, SKETCH, device=dev) for n in names}
    feats = {k: [] for k in ("entropy", "surprisal", "runsurp", "pos", "is_rep", "dist_prev", "cat")}
    coefs = {n: [] for n in names}; effnorm = {n: [] for n in names}; kls = {n: [] for n in names}
    for di, text in enumerate(docs):
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] < 16: continue
        idl = ids[0].tolist(); pieces = tok.convert_ids_to_tokens(idl)
        state["Q"] = None
        o0 = model(input_ids=ids, output_hidden_states=True)
        h = o0.hidden_states[LAYER + 1][0].float()
        L0 = o0.logits[0].float()
        lsm0 = torch.log_softmax(L0, -1)
        ent = (-(lsm0.exp() * lsm0).sum(-1)).cpu().numpy()
        sur = np.array([-float(lsm0[t, idl[t + 1]]) for t in range(len(idl) - 1)] + [0.0])
        keep = list(range(3, len(idl) - 1))
        last = {}; dist = []; isrep = []
        for t in range(len(idl)):
            nxt = idl[t + 1] if t + 1 < len(idl) else -1
            dist.append(t - last[nxt] if nxt in last else -1); isrep.append(1 if nxt in last else 0)
            last[idl[t]] = t
        run = np.cumsum(sur) / np.maximum(np.arange(1, len(sur) + 1), 1)
        feats["entropy"] += [float(ent[t]) for t in keep]
        feats["surprisal"] += [float(sur[t]) for t in keep]
        feats["runsurp"] += [float(run[t]) for t in keep]
        feats["pos"] += [t for t in keep]
        feats["is_rep"] += [isrep[t] for t in keep]
        feats["dist_prev"] += [dist[t] for t in keep]
        feats["cat"] += [catc(pieces[t + 1]) for t in keep]
        kt = torch.tensor(keep, device=dev)
        for n in names:
            q = Q[n]
            c = (h[kt] @ q.t()).pow(2).sum(1).sqrt() if q.shape[0] > 1 else (h[kt] @ q[0])
            coefs[n] += c.cpu().numpy().tolist()
            state["Q"] = q
            L1 = model(input_ids=ids).logits[0].float()
            state["Q"] = None
            D = (L1 - L0)[kt]
            sk[n].append((D @ R).cpu().numpy())
            CC[n] += D.t() @ (D @ R)
            effnorm[n] += D.norm(dim=1).cpu().numpy().tolist()
            lsm1 = torch.log_softmax(L1[kt], -1)
            kls[n] += (lsm0[kt].exp() * (lsm0[kt] - lsm1)).sum(-1).cpu().numpy().tolist()
        if (di + 1) % 40 == 0: print(f"[{tag}] {di+1}/{len(docs)} docs", flush=True)
    hook.remove()

    for k in feats:
        if k != "cat": feats[k] = np.array(feats[k], dtype=float)
    cat = np.array(feats["cat"]); npos = len(feats["entropy"])
    print(f"\n{npos} positions captured\n", flush=True)

    # ---------- (A) spectrum ----------
    spec = {}
    idx = rng.choice(npos, min(NPOS_SVD, npos), replace=False)
    for n in names:
        S = np.concatenate(sk[n], 0)[idx]
        S = S - S.mean(0, keepdims=True)
        sv = np.linalg.svd(S, compute_uv=False)
        ev = (sv ** 2) / (sv ** 2).sum()
        U, sg, Vt = np.linalg.svd(S, full_matrices=False)
        comp = []
        Cn = CC[n].cpu().numpy()
        for j in range(min(NCOMP, len(sg))):
            vv = Cn @ Vt[j]
            vv = vv / (np.linalg.norm(vv) + 1e-9)
            top = np.argsort(vv)[::-1][:12]; bot = np.argsort(vv)[:12]
            comp.append({"evr": round(float(ev[j]), 4),
                         "promoted": [tok.decode([int(i)]) for i in top],
                         "suppressed": [tok.decode([int(i)]) for i in bot]})
        spec[n] = {"evr_top10": [round(float(x), 4) for x in ev[:10]],
                   "participation_ratio": round(float((ev.sum() ** 2) / (ev ** 2).sum()), 2),
                   "components": comp}
        print(f"  [spectrum] {n:14} EVR1={ev[0]:.3f} EVR2={ev[1]:.3f} EVR3={ev[2]:.3f}  "
              f"participation_ratio={spec[n]['participation_ratio']}", flush=True)
        print(f"      comp1 +: {' '.join(repr(x) for x in comp[0]['promoted'][:7])}", flush=True)
        print(f"      comp1 -: {' '.join(repr(x) for x in comp[0]['suppressed'][:7])}", flush=True)

    # ---------- (B) coefficient regression ----------
    def design():
        X = [feats["entropy"], feats["surprisal"], feats["runsurp"], np.log1p(feats["pos"]),
             feats["is_rep"], np.log1p(np.maximum(feats["dist_prev"], 0))]
        nm = ["entropy", "surprisal", "run_surprisal", "log_pos", "is_repeat", "log_dist_prev"]
        for c in ("word_initial", "continuation", "punct", "digit"):
            X.append((cat == c).astype(float)); nm.append(f"cat_{c}")
        A = np.stack(X, 1)
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        return np.concatenate([A, np.ones((len(A), 1))], 1), nm
    Xd, xn = design()
    reg = {}
    for n in names:
        for lbl, y in (("coef", np.array(coefs[n])), ("effect_norm", np.array(effnorm[n])), ("kl", np.array(kls[n]))):
            yy = (y - y.mean()) / (y.std() + 1e-9)
            beta, *_ = np.linalg.lstsq(Xd, yy, rcond=None)
            pred = Xd @ beta
            r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            uni = {xn[i]: round(float(np.corrcoef(Xd[:, i], yy)[0, 1]), 3) for i in range(len(xn))}
            reg[f"{n}|{lbl}"] = {"R2": round(float(r2), 4),
                                 "betas": {xn[i]: round(float(beta[i]), 3) for i in range(len(xn))},
                                 "univariate_r": uni}
        r = reg[f"{n}|coef"]
        top = sorted(r["univariate_r"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  [regress] {n:14} coef R2={r['R2']:.3f}  strongest: " +
              " ".join(f"{k}={v:+.2f}" for k, v in top), flush=True)

    out = {"model": tag, "layer": LAYER, "ndocs": len(docs), "npos": int(npos),
           "seed_stability": stab, "spectrum": spec, "regression": reg}
    p = f"{OUTDIR}/pile_effect_spectrum{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
