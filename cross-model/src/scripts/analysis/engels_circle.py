"""Engels et al. (arXiv 2405.14860) setup: is the natural day/month CIRCLE the same object as the
toy-task ring geometry, and is it causally used?

Their prompt format, not ours — spelled-out offset, "from" not "after", no Q/A scaffolding:
    "Let's do some day of the week math. Two days from Monday is" -> " Wednesday"
    "Let's do some calendar math. Four months from January is"    -> " May"

Three measurements, per layer:

  (1) GEOMETRY   PCA over per-concept mean hidden states at the CONCEPT token. Reports variance in the
                 top-2 plane and whether the concepts are angularly ORDERED in it (a real circle puts
                 day i at angle ~2*pi*i/7). `circ_order_r` = circular-circular correlation between the
                 PCA angle and the true index angle. This is the detectability measure.

  (2) CAUSAL     project the top-k plane OUT at that layer and re-measure task accuracy, against NRAND
                 random rank-k subspaces. This is the usage measure. Engels report interventions with
                 pca_k=5, so k is configurable.

  (3) BRIDGE     principal angles between the natural day/month circular plane and the TOY-RING DAS
                 subspace (ring16_r8 / ring32_r16). If the toy ring geometry and the natural circle are
                 the same object, these should share directions. This is the measurement that connects
                 our negative ring result to their positive one.

Relevant context: arXiv 2605.01148 finds concept+offset combine at L18 while circular structure only
emerges at L22-25, i.e. the geometry is downstream of the computation. So (1) and (2) are expected to
PEAK AT DIFFERENT LAYERS, and that dissociation is the point of sweeping layers here.

Env: GEN_MODEL(Llama) TASK(days|months) LAYERS("10,14,18,20,22,24,26") PCA_K(5) NRAND(5)
     RING_NPZ RING_KEY(ring16_r8) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/engels_circle<OUTTAG>_<TASK>_<model>.json
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
TASK = os.environ.get("TASK", "days")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "10,14,18,20,22,24,26").split(",")]
PCA_K = int(os.environ.get("PCA_K", "5"))
NRAND = int(os.environ.get("NRAND", "5"))
RING_NPZ = os.environ.get("RING_NPZ", ""); RING_KEY = os.environ.get("RING_KEY", "ring16_r8")
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_transfer")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
WORD = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "Eleven", "Twelve"]


def spec():
    if TASK == "days":
        return DAYS, 7, "Let's do some day of the week math.", "days"
    return MONTHS, 12, "Let's do some calendar math.", "months"


def build():
    """(prompt, concept_index, offset, correct_index). Engels format verbatim."""
    names, n, frame, unit = spec()
    out = []
    for i, nm in enumerate(names):
        for k in range(0, min(len(WORD) - 1, n) + 1):
            u = unit if k > 1 else unit[:-1]        # "One day from", not "One days from"
            out.append((f"{frame} {WORD[k]} {u} from {nm} is", i, k, (i + k) % n))
    return out


def orth_rows(R):
    q, _ = np.linalg.qr(np.asarray(R, np.float64).T)
    return q[:, :R.shape[0]]                       # [d, r] columns orthonormal


def circ_rsa(ang, n):
    """Rotation- and reflection-INVARIANT circularity. Pairwise angular separation in the PCA plane vs
    true cyclic distance; a perfect circle gives dtheta = (2pi/n)*dcyc so Pearson r = 1 for ANY basis.
    Validated: 1.000 for a perfect circle at every rotation and under reflection; random null mean ~0,
    p95 = 0.389 (n=7) / 0.241 (n=12).
    (The naive version — correlating PCA angle against true angle — is rotation-DEPENDENT and useless:
    a perfect n=7 circle scored 0.431 while the random null averaged 0.350 and reached 0.766.)"""
    iu = np.triu_indices(n, 1)
    d = np.abs(ang[:, None] - ang[None, :]); d = np.minimum(d, 2 * np.pi - d)[iu]
    c = np.array([[min((i - j) % n, (j - i) % n) for j in range(n)] for i in range(n)], float)[iu]
    if d.std() < 1e-12: return 0.0
    return float(np.corrcoef(d, c)[0, 1])


def order_ok(ang, n):
    """discrete check: does sorting by PCA angle recover the cyclic sequence up to rotation/reflection?
    Fires at 0.3% (n=7) / 0.0% (n=12) by chance."""
    o = list(np.argsort(ang))
    for seq in (o, o[::-1]):
        dbl = seq + seq
        for st in range(n):
            w = dbl[st:st + n]
            if w == list(range(n)) or w == list(range(n))[::-1]: return 1
    return 0


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    names, n, _, _ = spec()
    items = build()
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in names]
    assert len(set(cand)) == n, "candidate first tokens must be distinct"
    cand_t = torch.tensor(cand, device=dev)
    rng = np.random.default_rng(SEED)

    # ---- concept-token position: last token of the concept name inside the prompt ----
    pre = {}
    for p, ci, k, ans in items:
        ids = tok(p, return_tensors="pt")["input_ids"]
        # find the concept name's final token by matching the tokenisation of " <name> is"
        tail = tok(" " + names[ci] + " is", add_special_tokens=False)["input_ids"]
        pos = ids.shape[1] - len(tail)                    # first token of " <name>"
        pre[p] = (ids.to(dev), pos + len(tok(" " + names[ci], add_special_tokens=False)["input_ids"]) - 1)

    state = {"proj": None, "layer": None}
    hooks = []
    for l in range(cm.num_hidden_layers):
        def mk(l):
            def rh(_m, _i, out):
                if state["proj"] is None or state["layer"] != l: return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); P = state["proj"]
                h[0] = (h[0].float() - (h[0].float() @ P.t()) @ P).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return rh
        hooks.append(blocks[l].register_forward_hook(mk(l)))

    def concept_decodable(L):
        """leave-one-prompt-out nearest-centroid accuracy for WHICH concept, from the concept-token
        hidden state at layer L, under whatever projection is currently installed. If projecting the
        circular plane out drops this to chance, the intervention deleted the INPUT, and the large
        early-layer behavioural drop says nothing about circular computation."""
        X, Y = [], []
        for p, ci, k, ans in items:
            ids, cpos = pre[p]
            o = model(input_ids=ids, output_hidden_states=True)
            X.append(o.hidden_states[L + 1][0, cpos].float().cpu().numpy()); Y.append(ci)
        X = np.stack(X); Y = np.array(Y); ok = 0
        for i in range(len(X)):
            m = np.ones(len(X), bool); m[i] = False
            cen = np.stack([X[m & (Y == c)].mean(0) for c in range(n)])
            ok += int(int(np.argmin(np.linalg.norm(cen - X[i], axis=1))) == Y[i])
        return ok / len(X)

    def accuracy():
        """returns (cyclic_acc over offset>0, identity_acc over offset==0)"""
        ok = m = iok = im = 0
        for p, ci, k, ans in items:
            ids, _ = pre[p]
            lg = model(input_ids=ids).logits[0, -1].float()
            c = int(int(lg[cand_t].argmax()) == ans)
            if k == 0: iok += c; im += 1
            else: ok += c; m += 1
        return ok / max(m, 1), iok / max(im, 1)

    base_acc, base_id = accuracy()
    print(f"[{tag}/{TASK}] Engels-format clean: cyclic acc = {base_acc:.3f}, "
          f"IDENTITY (offset 0) acc = {base_id:.3f}  (n={len(items)} prompts)", flush=True)
    print(f"  example: {items[1][0]!r} -> {' ' + names[items[1][3]]!r}", flush=True)

    ringR = None
    if RING_NPZ and os.path.exists(RING_NPZ):
        z = np.load(RING_NPZ)
        if RING_KEY in z.files: ringR = orth_rows(z[RING_KEY])

    res = {"model": tag, "task": TASK, "clean_acc": round(base_acc, 4), "pca_k": PCA_K,
           "n_prompts": len(items), "ring_npz": RING_NPZ, "ring_key": RING_KEY, "layers": {}}
    print(f"\n{'layer':>5} {'top2_var':>9} {'crcRSA':>8} {'ord':>4} {'acc_proj':>9} {'acc_rand':>9} "
          f"{'excess':>8} {'dec_cln':>8} {'dec_prj':>8} {'ringcos':>8} {'null':>7}")
    for L in LAYERS:
        # (1) geometry: per-concept mean hidden state at the concept token
        acc_by = {i: [] for i in range(n)}
        for p, ci, k, ans in items:
            ids, cpos = pre[p]
            o = model(input_ids=ids, output_hidden_states=True)
            acc_by[ci].append(o.hidden_states[L + 1][0, cpos].float().cpu().numpy())
        Mn = np.stack([np.mean(acc_by[i], 0) for i in range(n)])
        Mn = Mn - Mn.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Mn, full_matrices=False)
        var = S ** 2
        top2 = float(var[:2].sum() / var.sum())
        ang = np.arctan2(U[:, 1] * S[1], U[:, 0] * S[0])
        circ_r = circ_rsa(ang, n); oo = order_ok(ang, n)

        # (2) causal: project out the top-k plane at this layer
        Pk = torch.tensor(Vt[:PCA_K], dtype=torch.float32, device=dev)
        state["layer"] = L; state["proj"] = Pk
        a_proj, i_proj = accuracy()
        rs = []; ris = []
        for _ in range(NRAND):
            Rr = np.linalg.qr(rng.standard_normal((cm.hidden_size, PCA_K)))[0].T.astype(np.float32)
            state["proj"] = torch.tensor(Rr, device=dev)
            _a, _i = accuracy(); rs.append(_a); ris.append(_i)
        state["proj"] = None; state["layer"] = None
        a_rand = float(np.mean(rs)); i_rand = float(np.mean(ris))
        dec_clean = concept_decodable(L)
        state["layer"] = L; state["proj"] = Pk
        dec_proj = concept_decodable(L)
        state["proj"] = None; state["layer"] = None

        # (3) bridge: principal angles vs the toy-ring DAS subspace
        rc = rc0 = float("nan")
        if ringR is not None:
            P2 = orth_rows(Vt[:2])
            rc = float(np.linalg.svd(P2.T @ ringR, compute_uv=False).max())
            nulls = []
            for _ in range(20):
                Q = np.linalg.qr(rng.standard_normal((cm.hidden_size, ringR.shape[1])))[0]
                nulls.append(float(np.linalg.svd(P2.T @ Q, compute_uv=False).max()))
            rc0 = float(np.mean(nulls))
        res["layers"][str(L)] = {"top2_var": round(top2, 4), "circ_rsa": round(circ_r, 4),
                                 "order_ok": oo,
                                 "acc_proj": round(a_proj, 4), "acc_rand": round(a_rand, 4),
                                 "excess": round(a_proj - a_rand, 4),
                                 "id_proj": round(i_proj, 4), "id_rand": round(i_rand, 4),
                                 "id_excess": round(i_proj - i_rand, 4),
                                 "concept_decode_clean": round(dec_clean, 4),
                                 "concept_decode_proj": round(dec_proj, 4),
                                 "ring_max_cos": None if np.isnan(rc) else round(rc, 4),
                                 "ring_max_cos_null": None if np.isnan(rc0) else round(rc0, 4)}
        print(f"{L:5} {top2:9.3f} {circ_r:8.3f} {oo:4} {a_proj:9.3f} {a_rand:9.3f} "
              f"{(a_proj - a_rand):+8.3f} {dec_clean:8.3f} {dec_proj:8.3f} {rc:8.3f} {rc0:7.3f}",
              flush=True)
    for h in hooks: h.remove()
    p = f"{OUTDIR}/engels_circle{OUTTAG}_{TASK}_{tag}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
