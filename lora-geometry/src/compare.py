"""Stage 6: does activation space agree with weight space?

Deliberately NOT done by correlating the two similarity matrices. That statistic
is uninterpretable on its own -- it can only say the two agree, never whether
either is right -- and it is the one comparison method Sandra has ruled out. The
question is asked three other ways instead, in increasing order of strength:

  (1) CEILINGS AND FLOORS FIRST. Nothing below means anything until we know how
      well each space reproduces ITSELF.
        activation ceiling   split-half reliability of v_c (stage 2)
        weight ceiling       cos(dW_c^seed_i, dW_c^seed_j), same concept,
                             different training randomness
        data ceiling         the paraphrase twins: same behaviour, different
                             wording, different generated data
        floor                the unrelated-pair distribution
      If the weight ceiling sits near the floor, LoRA solutions are not a
      representation of the concept and every later number is capped at noise.
      This is the gate; it is cheap and it can kill the project in an afternoon.

  (2) ANSWER-KEY RECOVERY. Mean similarity by designed tier (twin / same-pole /
      antonym / same-family / unrelated). Not a matrix correlation -- a test
      against structure we planted.

      The antonym tier was originally billed as the sharp one, on the reasoning
      that a SIGNED representation should put antonyms below unrelated while a
      MAGNITUDE representation should put them ABOVE, since the same machinery
      moves either way. The magnitude half of that is ILL-POSED and the first
      full run proved it the expensive way. ||dW[j,:]|| >= 0 by construction: it
      records THAT a neuron moved, never WHICH WAY, so its cosines are pinned
      positive (observed range across all pairs: 0.932 to 0.998) and "opposite"
      is simply unrepresentable. Any apparent antonym signal in a magnitude
      profile therefore comes from centring, which is relative to whatever
      concept set you happened to include -- and if the profile is not
      unit-normalised first, from edit SIZE (that axis correlated +0.998 with
      ||dW||, and produced a headline that had to be withdrawn).

      So: read the antonym tier only in SIGNED representations, and only from
      the `_cn` (normalised-then-centred) magnitude profiles if at all.

  (3) PREDICTIVE MAPPING, leave-one-concept-out. Fit a linear map from
      activation coordinates to weight coordinates on N-1 concepts, predict the
      held-out one, and ask whether its true weight-space point is the nearest of
      all N candidates. Chance is 1/N. This is the strong form of the question:
      not "do the geometries correlate" but "can you find a concept's adapter
      knowing only its steering vector". Reported with a label-shuffled floor and
      the seed-replicate ceiling.

  Orthogonal-Procrustes residual is reported alongside (3) as the descriptive
  version of the same fit.

Output: out/compare.json, plus printed tables.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import out_path, parse_item, unit  # noqa: E402

K = int(os.environ.get("KDIM", 12))       # coordinate dims for the mapping tests
RIDGE = float(os.environ.get("RIDGE", 1e-2))


# ---------------------------------------------------------------------------
# representations
# ---------------------------------------------------------------------------
def coords_from_gram(G):
    """Exact coordinates from a Gram matrix (G = X X^T), so a rep we only ever
    computed as inner products can still enter the mapping tests."""
    w, V = np.linalg.eigh((G + G.T) / 2)
    w = np.clip(w, 0, None)
    o = np.argsort(w)[::-1]
    return V[:, o] * np.sqrt(w[o])


def cosmat(X):
    Z = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Z @ Z.T


def by_concept(items, X, names):
    """Average per-seed rows into one row per concept, and keep the per-seed rows
    for the seed-ceiling estimate."""
    out, per = [], {}
    for n in names:
        rows = [i for i, it in enumerate(items) if parse_item(it)[0] == n]
        per[n] = [X[i] for i in rows]
        out.append(np.mean([X[i] for i in rows], 0))
    return np.stack(out), per


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------
def tiers(names):
    idx = {n: i for i, n in enumerate(names)}
    keep = lambda ps: [(idx[a], idx[b]) for a, b in ps if a in idx and b in idx]
    return dict(twin=keep(C.twin_pairs()),
                same_pole=keep([p for p in C.synonym_pairs() if p not in
                                [(t, s) for t, s in C.twin_pairs()] and
                                (p[1], p[0]) not in C.twin_pairs()]),
                antonym=keep(C.antonym_pairs()),
                same_family=keep(C.family_pairs()),
                unrelated=keep(C.unrelated_pairs()))


def tier_table(Cm, T):
    out = {}
    for k, pairs in T.items():
        if not pairs:
            continue
        v = np.array([Cm[i, j] for i, j in pairs])
        out[k] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


# ---------------------------------------------------------------------------
# ceilings
# ---------------------------------------------------------------------------
def init_breakdown(items, mats, names):
    """Every pair bucketed by tier x whether the two adapters SHARE a LoRA init.

    The pilot's decisive table, so it is standard output now. It separates two
    things the naive design conflated:
      * across a shared init, dW cosine is a real measurement of the concept
      * across different inits, dW lives in a different random r-dim subspace, so
        the cosine is pinned near zero by geometry no matter what was learned
    A representation that holds up in the "diff init" rows is basis-robust and
    can be trusted without controlling init; one that only works in "SAME init"
    rows is a within-block measurement and must be reported as such.
    """
    idx = {n: i for i, n in enumerate(names)}
    tier_of = {}
    for a, b in C.twin_pairs():
        tier_of[frozenset((a, b))] = "twin"
    for a, b in C.synonym_pairs():
        tier_of.setdefault(frozenset((a, b)), "same_pole")
    for a, b in C.antonym_pairs():
        tier_of.setdefault(frozenset((a, b)), "antonym")
    for a, b in C.family_pairs():
        tier_of.setdefault(frozenset((a, b)), "same_family")

    rows = {}
    for i, ia in enumerate(items):
        na, ba, _ = parse_item(ia)
        if na not in idx:
            continue
        for j, ib in enumerate(items[i + 1:], i + 1):
            nb, bb, _ = parse_item(ib)
            if nb not in idx:
                continue
            t = ("same_concept" if na == nb
                 else tier_of.get(frozenset((na, nb)), "unrelated"))
            rows.setdefault(f"{t}, {'SAME' if ba == bb else 'diff'} init", []).append((i, j))
    return rows


def seed_ceiling(per):
    """cos between two different seeds of the SAME concept -- how well weight
    space reproduces itself under training randomness alone."""
    vals = []
    for n, rows in per.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = unit(rows[i]), unit(rows[j])
                vals.append(float(a @ b))
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            len(vals))


# ---------------------------------------------------------------------------
# predictive mapping
# ---------------------------------------------------------------------------
def loco_retrieval(XA, XB, group=None, shuffle_seed=None):
    """Leave-one-concept-out: fit A -> B on the rest, predict the held-out
    concept's B-coordinates, and rank all N candidates by cosine to the
    prediction.

    Three numbers, because strict top-1 is unfair to this concept set BY DESIGN:
    the paraphrase twins and the same-pole pairs are built to be near-duplicates,
    so retrieving `terse_b` when the target was `terse` is a success of the
    method being scored as a failure.
        top1       the true concept is rank 1                (strict)
        top1_grp   the rank-1 concept shares the target's axis AND pole, i.e. it
                   is the target or one of its designed near-duplicates
        mrr        mean reciprocal rank of the true concept   (graded, no
                   duplicate penalty cliff)
    Report all three; top1_grp is the one to lead with, with its own chance rate
    (group size / N), which is printed rather than assumed.
    """
    N = len(XA)
    perm = np.arange(N)
    if shuffle_seed is not None:
        perm = np.random.default_rng(shuffle_seed).permutation(N)
    XB = XB[perm]
    A = XA - XA.mean(0)
    B = XB - XB.mean(0)
    hits, ghits, rr = [], [], []
    for h in range(N):
        tr = [i for i in range(N) if i != h]
        At, Bt = A[tr], B[tr]
        Wm = np.linalg.solve(At.T @ At + RIDGE * np.eye(At.shape[1]), At.T @ Bt)
        pred = A[h] @ Wm
        sims = np.array([float(unit(pred) @ unit(B[i])) for i in range(N)])
        order = np.argsort(sims)[::-1]
        rank = int(np.where(order == h)[0][0]) + 1
        hits.append(rank == 1)
        rr.append(1.0 / rank)
        ghits.append(group is not None and group[order[0]] == group[h])
    return float(np.mean(hits)), float(np.mean(ghits)), float(np.mean(rr))


def procrustes_resid(XA, XB):
    """Orthogonal Procrustes: best rotation of standardised A onto standardised
    B. Returns normalised residual in [0, 1]; 0 = perfect alignment."""
    A = XA - XA.mean(0); B = XB - XB.mean(0)
    A = A / np.linalg.norm(A); B = B / np.linalg.norm(B)
    k = min(A.shape[1], B.shape[1])
    A, B = A[:, :k], B[:, :k]
    U, s, Vt = np.linalg.svd(A.T @ B)
    return float(1.0 - s.sum() ** 2 / (np.linalg.norm(A) ** 2 * np.linalg.norm(B) ** 2))


# ---------------------------------------------------------------------------
def main():
    W = np.load(out_path("wspace.npz"), allow_pickle=True)
    items = [str(x) for x in W["items"]]
    order = [n for n in C.NAMES if any(parse_item(it)[0] == n for it in items)]
    # ENFORCE the steering-validity gate. A concept whose steering vector never beat
    # a matched-norm random direction contributes an unvalidated direction to every
    # activation-space number, so it is excluded unless explicitly overridden.
    if os.environ.get("STEER_GATE", "1") != "0" and os.path.exists(out_path("steer_failed.json")):
        failed = set(json.load(open(out_path("steer_failed.json"))))
        drop = [n for n in order if n in failed]
        order = [n for n in order if n not in failed]
        if drop:
            print(f"[cmp] steering gate: dropped {len(drop)} unvalidated concepts: {drop}")
    N = len(order)
    L = os.environ.get("LAYER", "")
    POS = os.environ.get("POS", "response")
    # Tier and mapping tables run WITHIN one init block: averaging dW across
    # blocks averages coordinates expressed in different random bases. The
    # init-breakdown table above is the one that spans blocks, on purpose.
    BLOCK = os.environ.get("BLOCK", "")

    reps = {}

    # --- weight space -------------------------------------------------------
    Xg = coords_from_gram(W["gram_flat_signed"])
    reps["W:flat_signed"] = (items, Xg)
    for key in ("prof_neuron_mlp", "prof_neuron_resid",
                "prof_neuron_mlp_c", "prof_neuron_resid_c",
                "prof_neuron_mlp_cn", "prof_neuron_resid_cn"):
        if key in W.files:
            reps["W:" + key.replace("prof_", "")] = (items, np.asarray(W[key], dtype=np.float64))
    # The init breakdown is the one table that must SPAN blocks, so it is built
    # from the unfiltered matrices before BLOCK is applied below.
    Gf = W["gram_flat_signed"]
    df = np.sqrt(np.diag(Gf))
    wmats_full = {"W:flat_signed": Gf / np.outer(df, df)}
    for rn, (_, X) in reps.items():
        if rn != "W:flat_signed":
            wmats_full[rn] = cosmat(X)

    if BLOCK != "":
        keep = [i for i, it in enumerate(items) if parse_item(it)[1] == int(BLOCK)]
        reps = {k: ([items[i] for i in keep], X[keep]) for k, (its, X) in reps.items()}
        print(f"[cmp] restricted to init block {BLOCK}: {len(keep)} adapters")

    # --- activation space ---------------------------------------------------
    V = np.load(out_path("vecs.npz"))
    layer = L or sorted({k.split("|")[2] for k in V.files if len(k.split("|")) == 3})[0]
    reps["A:steer_vec"] = ([f"{n}__b0_d0" for n in order],
                           np.stack([V[f"{n}|{POS}|{layer}"] for n in order]))
    if os.path.exists(out_path("induced.npz")):
        U = np.load(out_path("induced.npz"))
        ks = [it for it in items if f"{it}|{POS}|{layer}" in U.files]
        if ks:
            reps["A:lora_induced"] = (ks, np.stack([U[f"{it}|{POS}|{layer}"] for it in ks]))

    T = tiers(order)
    report = dict(layer=layer, pos=POS, n_concepts=N, tiers={k: len(v) for k, v in T.items()})

    # --- (0) init-block breakdown, at the ADAPTER level (no seed averaging) ---
    wnames = list(wmats_full)
    wmats = wmats_full
    rows = init_breakdown(items, wmats, order)
    ORDER = ["same_concept, SAME init", "same_concept, diff init",
             "twin, SAME init", "twin, diff init",
             "same_pole, SAME init", "same_pole, diff init",
             "antonym, SAME init", "antonym, diff init",
             "same_family, SAME init", "same_family, diff init",
             "unrelated, SAME init", "unrelated, diff init"]
    print("\n=== weight space by tier x shared-LoRA-init (adapter level, no averaging) ===")
    print(f"{'bucket':<28}" + "".join(f"{r.replace('W:', ''):>16}" for r in wnames) + f"{'n':>6}")
    for k in ORDER:
        if k not in rows:
            continue
        print(f"{k:<28}" + "".join(
            f"{np.mean([wmats[r][i, j] for i, j in rows[k]]):>16.3f}" for r in wnames)
            + f"{len(rows[k]):>6}")
        report.setdefault("init_breakdown", {})[k] = {
            r: float(np.mean([wmats[r][i, j] for i, j in rows[k]])) for r in wnames}
        report["init_breakdown"][k]["n"] = len(rows[k])

    # --- (0b) the bridge, WITH its off-diagonal floor -----------------------
    # cos(v_c, u_c) on its own is not interpretable: u is a residual-stream shift
    # and v is a residual-stream contrast on overlapping text, so some alignment
    # is expected for any pair. The quantity that means something is matched
    # (same concept) against MISMATCHED (v of one concept vs u of another).
    if "A:lora_induced" in reps:
        vits, Xv = reps["A:steer_vec"]
        uits, Xu = reps["A:lora_induced"]
        vof = {parse_item(it)[0]: Xv[i] for i, it in enumerate(vits)}
        Zv = {n: unit(v) for n, v in vof.items()}
        match, mismatch, per_c = [], [], {}
        for i, it in enumerate(uits):
            n = parse_item(it)[0]
            u = unit(Xu[i])
            for m, zv in Zv.items():
                c = float(zv @ u)
                (match if m == n else mismatch).append(c)
                if m == n:
                    per_c.setdefault(n, []).append(c)
        print(f"\n=== bridge cos(v, u) at L{layer}, {POS} read ===")
        print(f"  matched (same concept)   {np.mean(match):+.3f}  "
              f"sd {np.std(match, ddof=1):.3f}  n={len(match)}")
        print(f"  MISMATCHED (floor)       {np.mean(mismatch):+.3f}  "
              f"sd {np.std(mismatch, ddof=1):.3f}  n={len(mismatch)}")
        print(f"  separation               {np.mean(match) - np.mean(mismatch):+.3f}")
        # per concept, so one strong concept cannot carry the mean
        print("  per concept (matched vs that concept's own mismatched floor):")
        for n in order:
            if n not in per_c:
                continue
            fl = [float(Zv[m] @ unit(Xu[i])) for i, it in enumerate(uits)
                  for m in [n] if parse_item(it)[0] != n]
            print(f"    {n:<22} {np.mean(per_c[n]):+.3f}   floor {np.mean(fl):+.3f}")
        report["bridge"] = dict(matched=float(np.mean(match)),
                                mismatched=float(np.mean(mismatch)),
                                sep=float(np.mean(match) - np.mean(mismatch)),
                                per_concept={n: float(np.mean(v)) for n, v in per_c.items()})

    # --- (1) ceilings and floors, (2) answer-key recovery -------------------
    per_rep, cmats = {}, {}
    print(f"\n=== tier means (layer {layer}, {POS} read, {N} concepts) ===")
    hdr = f"{'representation':<24}" + "".join(f"{k:>13}" for k in
                                              ("twin", "same_pole", "antonym",
                                               "same_family", "unrelated")) + f"{'seed-ceil':>12}"
    print(hdr)
    for rname, (its, X) in reps.items():
        Xc, per = by_concept(its, X, order)
        per_rep[rname] = (Xc, per)
        Cm = cosmat(Xc)
        cmats[rname] = Cm
        tt = tier_table(Cm, T)
        sc = seed_ceiling(per) if max(len(v) for v in per.values()) > 1 else (float("nan"), 0, 0)
        report.setdefault("tiers_by_rep", {})[rname] = dict(tiers=tt, seed_ceiling=sc)
        row = f"{rname:<24}" + "".join(
            f"{tt[k]['mean']:>13.3f}" if k in tt else f"{'-':>13}"
            for k in ("twin", "same_pole", "antonym", "same_family", "unrelated"))
        print(row + f"{sc[0]:>12.3f}")

    # the sharp test, called out explicitly
    print("\n=== antonym test: signed reps should go BELOW unrelated, "
          "magnitude reps ABOVE ===")
    for rname, Cm in cmats.items():
        tt = tier_table(Cm, T)
        if "antonym" not in tt or "unrelated" not in tt:
            continue
        d = tt["antonym"]["mean"] - tt["unrelated"]["mean"]
        # Centring removes the common mode but does not make a magnitude profile
        # signed: an antonym pair still moves the SAME neurons, so every
        # per-neuron rep is expected positive here, centred or not.
        kind = "magnitude, expect +" if "neuron" in rname else "signed, expect -"
        print(f"  {rname:<24} antonym - unrelated = {d:+.3f}   ({kind})")
        report["tiers_by_rep"][rname]["antonym_minus_unrelated"] = float(d)

    # --- (3) predictive mapping A -> W --------------------------------------
    group = [f"{C.AXIS[n]}|{C.POLE[n]}" for n in order]
    gsz = {g: group.count(g) for g in set(group)}
    chance_grp = float(np.mean([gsz[g] / N for g in group]))
    print(f"\n=== leave-one-concept-out retrieval "
          f"(chance top1 = {1 / N:.3f}, chance top1_grp = {chance_grp:.3f}) ===")
    print(f"{'activation rep':<18}{'-> weight rep':<24}{'top1':>7}{'top1grp':>9}"
          f"{'MRR':>7}{'shufgrp':>9}{'procrustes':>12}")
    for aname in [r for r in reps if r.startswith("A:")]:
        XA = per_rep[aname][0]
        XAk = coords_from_gram(XA @ XA.T)[:, :K]
        for wname in [r for r in reps if r.startswith("W:")]:
            XB = per_rep[wname][0]
            XBk = coords_from_gram(XB @ XB.T)[:, :K]
            t1, g1, mrr = loco_retrieval(XAk, XBk, group)
            sh = float(np.mean([loco_retrieval(XAk, XBk, group, shuffle_seed=s)[1]
                                for s in range(5)]))
            pr = procrustes_resid(XAk, XBk)
            print(f"{aname:<18}{wname:<24}{t1:>7.3f}{g1:>9.3f}{mrr:>7.3f}"
                  f"{sh:>9.3f}{pr:>12.3f}")
            report.setdefault("mapping", {})[f"{aname} -> {wname}"] = dict(
                top1=t1, top1_grp=g1, mrr=mrr, shuffled_top1_grp=sh,
                procrustes_resid=pr, chance=1.0 / N, chance_grp=chance_grp)

    json.dump(report, open(out_path("compare.json"), "w"), indent=1)
    print("\nCOMPARE_DONE")


if __name__ == "__main__":
    main()
