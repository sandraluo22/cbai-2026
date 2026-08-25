"""Stage 5: does the WEIGHT geometry carry anything the ACTIVATION geometry does not?

Three geometries over the same concepts:

  A   cos(v_a, v_b)                activation space -- the reference
  N   cos(dW_a, dW_b) Steer2Edit   the NULL. Steer2Edit builds dW = c*v_hat k_hat^T
                                   with k_hat = W^T v / ||W^T v||, so
                                       cos(dW_a,dW_b) = sign(c_a c_b)
                                                        * cos(v_a,v_b)
                                                        * cos_{WW^T}(v_a,v_b)
                                   -- a deterministic function of the vectors and
                                   the frozen base weights, verified to 1e-17.
                                   It is exactly "what the weight geometry looks
                                   like if the edit carries NO information beyond
                                   v", and costs no GPU.
  D   cos(dW_a, dW_b) distilled    the measurement. dW came from v -> teacher
                                   behaviour -> sampled numbers -> SGD, a lossy
                                   channel with no closed form.

The question with content is whether D departs from N. If D == N, the channel
transmitted the vector and nothing else, and weight geometry is a reparameterised
activation geometry. If D != N, distillation contributed structure -- from the
base model's inductive biases or from what the number channel preserves -- and
that residual is the finding.

Because v_desc never touches training, A vs D is an honest independent comparison,
unlike Steer2Edit (fully circular) or u-vs-dW (two views of one adapter).

Output: out/compare.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import out_path, parse_item  # noqa: E402

KIND = os.environ.get("VKIND", "desc")
BLOCK = os.environ.get("BLOCK", "0")


def cosmat(X):
    Z = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Z @ Z.T


def offdiag(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def main():
    W = np.load(out_path("wspace.npz"), allow_pickle=True)
    items = [str(x) for x in W["items"]]
    keep = [i for i, it in enumerate(items) if parse_item(it)[1] == int(BLOCK)]
    items = [items[i] for i in keep]

    # drop adapters whose trait never transmitted -- their dW is training noise
    if os.path.exists(out_path("lora_check.json")):
        chk = json.load(open(out_path("lora_check.json")))
        bad = [it for it in items if it in chk and not chk[it]["passed"]]
        if bad:
            print(f"[cmp] dropping {len(bad)} adapters with no transmission: {bad}")
        items = [it for it in items if it not in bad]
        keep = [i for i, it in enumerate([str(x) for x in W["items"]]) if it in items]

    names = [parse_item(it)[0] for it in items]
    # Fail loudly and legibly rather than with a numpy stack error: an empty set
    # here means the transmission gate rejected everything, which is a RESULT
    # (nothing was installed), not a crash.
    if len(names) < 2:
        print(f"[cmp] STOP: {len(names)} adapter(s) survived the transmission gate.\n"
              f"      Nothing was installed in the weights, so there is no weight\n"
              f"      geometry to compare. This is the gate reporting a null, not a bug.")
        json.dump(dict(n=len(names), aborted="no adapters passed transmission gate"),
                  open(out_path("compare.json"), "w"), indent=1)
        return
    V = np.load(out_path("vecs.npz"))
    layers = sorted({int(k.split("|")[2]) for k in V.files})
    L = int(os.environ.get("LAYER", layers[len(layers) // 2]))

    Xv = np.stack([V[f"{n}|{KIND}|{L}"] for n in names])
    A = cosmat(Xv)

    G = W["gram_flat_signed"][np.ix_(keep, keep)]
    d = np.sqrt(np.diag(G))
    D = G / np.outer(d, d)

    a, dd = offdiag(A), offdiag(D)
    rep = dict(layer=L, kind=KIND, block=int(BLOCK), n=len(names), concepts=names,
               mean_activation_cos=float(a.mean()), mean_weight_cos=float(dd.mean()),
               corr_A_D=float(np.corrcoef(a, dd)[0, 1]))

    print(f"=== {len(names)} concepts, L{L}, v_{KIND}, init block {BLOCK} ===")
    print(f"  mean off-diag cos, activation : {a.mean():+.4f}")
    print(f"  mean off-diag cos, weight     : {dd.mean():+.4f}")
    print(f"  corr(activation, weight)      : {rep['corr_A_D']:+.4f}")

    if os.path.exists(out_path("s2e_null.npz")):
        S = np.load(out_path("s2e_null.npz"), allow_pickle=True)
        sn = [str(x) for x in S["names"]]
        idx = [sn.index(n) for n in names if n in sn]
        if len(idx) == len(names):
            Nm = S["cos"][np.ix_(idx, idx)]
            nn = offdiag(Nm)
            rep.update(corr_A_N=float(np.corrcoef(a, nn)[0, 1]),
                       corr_D_N=float(np.corrcoef(dd, nn)[0, 1]),
                       mean_null_cos=float(nn.mean()))
            print(f"  mean off-diag cos, S2E null   : {nn.mean():+.4f}")
            print(f"  corr(activation, S2E null)    : {rep['corr_A_N']:+.4f}")
            print(f"  corr(weight, S2E null)        : {rep['corr_D_N']:+.4f}")
            # the residual: what the distilled weights carry beyond the null
            b = np.polyfit(nn, dd, 1)
            resid = dd - np.polyval(b, nn)
            rep["resid_sd_beyond_null"] = float(resid.std())
            rep["corr_resid_activation"] = float(np.corrcoef(resid, a)[0, 1])
            print(f"  residual sd of weight beyond null : {resid.std():.4f}")
            print(f"  corr(residual, activation cos)    : {rep['corr_resid_activation']:+.4f}")
            print("    (a large residual means distillation put structure into the\n"
                  "     weights that is NOT a deterministic function of the vector)")

    # ---- RSA done properly: noise ceiling + concept-level permutation ----
    # The bare correlation between two RDMs has no interpretable scale. It is
    # bounded above by the reliability of BOTH matrices, so report it against that
    # ceiling, and get its p-value by permuting CONCEPT LABELS (pairwise entries
    # share concepts, so they are not independent and a naive p-value is wrong).
    rel_w = np.nan
    allit = [str(x) for x in W["items"]]
    reps_by_c = {}
    for i, it in enumerate(allit):
        reps_by_c.setdefault(parse_item(it)[0], []).append(i)
    Gf = W["gram_flat_signed"]
    df = np.sqrt(np.diag(Gf))
    Fw = Gf / np.outer(df, df)
    rr = [Fw[a, b] for c, ix in reps_by_c.items() for a in ix for b in ix if a < b]
    if rr:
        rel_w = float(np.mean(rr))          # same concept, different replicate

    rel_a = np.nan
    if os.path.exists(out_path("vec_stats.json")):
        vs = json.load(open(out_path("vec_stats.json")))
        key = "rel_prompt_sb"
        vals = [vs[n][f"L{L}"][key] for n in names if n in vs and f"L{L}" in vs[n]]
        if vals:
            rel_a = float(np.mean(vals))

    ceiling = float(np.sqrt(max(rel_a, 0) * max(rel_w, 0))) if not (
        np.isnan(rel_a) or np.isnan(rel_w)) else float("nan")

    rng = np.random.default_rng(0)
    def rsa_r(perm=None):
        idx = np.arange(len(names)) if perm is None else perm
        return float(np.corrcoef(offdiag(A[np.ix_(idx, idx)]), dd)[0, 1])
    null = [rsa_r(rng.permutation(len(names))) for _ in range(2000)]
    p = float(np.mean([abs(x) >= abs(rep["corr_A_D"]) for x in null]))

    rep.update(rsa_r=rep["corr_A_D"], rsa_p_concept_perm=p,
               reliability_activation=rel_a, reliability_weight=rel_w,
               rsa_noise_ceiling=ceiling,
               rsa_fraction_of_ceiling=float(rep["corr_A_D"] / ceiling)
               if ceiling and not np.isnan(ceiling) and ceiling > 0 else float("nan"))
    print("\n=== RSA, with the parts that make it interpretable ===")
    print(f"  RSA r (activation RDM vs weight RDM) : {rep['corr_A_D']:+.4f}")
    print(f"  concept-label permutation p          : {p:.4f}   (n_perm=2000)")
    print(f"  reliability, activation side         : {rel_a:.4f}")
    print(f"  reliability, weight side (replicates): {rel_w:.4f}")
    print(f"  noise ceiling sqrt(rel_a*rel_w)      : {ceiling:.4f}")
    print(f"  fraction of achievable structure     : {rep['rsa_fraction_of_ceiling']:.4f}")
    print("  NOTE: a bare RSA r is uninterpretable here -- Steer2Edit's weight geometry")
    print("  is a CLOSED-FORM function of the activation geometry, so some agreement is")
    print("  guaranteed by construction. The residual-vs-null above is the honest number.")

    json.dump(rep, open(out_path("compare.json"), "w"), indent=1)
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
