"""16-node 4-regular graph families for the degree-controlled sweep.

Everything here is 16 nodes, 4-regular, 32 edges, connected => a simple random walk has
the SAME uniform stationary distribution and per-step branching on every graph, so node
count, degree, edge count and stationarity are controlled; only wiring varies.

Families:
  torus            C4 x C4 (G0)
  circulant(k)     C16(+-1, +-k), k in 2..7 (G1 = circulant(3))
  interp(alpha)    degree-preserving interpolation G0 -> G1 via targeted double-edge
                   swaps; alpha = normalized edge overlap with G1 (0 = torus, 1 = C16(1,3))
  smallworld(s)    C16(+-1,+-2) after s random degree-preserving double-edge swaps
  sbm(strong|weak) two 8-node communities; strong: 3 intra + 1 inter per node;
                   weak: 2 intra + 2 inter. Both 4-regular.
  expander(lo|hi)  random connected 4-regular graphs selected for smallest / largest
                   normalized-Laplacian spectral gap among 3000 samples.

Also: pair metrics (edge edit distance, spectral distance) and the sweep spec
generator (writes runs/sweep_spec.json consumed by run_sweep.py).

Run: python3 graphs16.py   -> prints family table, writes runs/sweep_spec.json
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 16
rng = np.random.default_rng(0)


def edges_to_adj(edges):
    A = np.zeros((N, N), int)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    return A


def adj_to_edges(A):
    return sorted((a, b) for a in range(N) for b in range(a + 1, N) if A[a, b])


def connected(A):
    seen = {0}
    st = [0]
    while st:
        u = st.pop()
        for v in np.where(A[u])[0]:
            if v not in seen:
                seen.add(int(v))
                st.append(int(v))
    return len(seen) == N


def torus():
    E = []
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            E.append(tuple(sorted((i, 4 * r + (c + 1) % 4))))
            E.append(tuple(sorted((i, 4 * ((r + 1) % 4) + c))))
    return edges_to_adj(sorted(set(E)))


def circulant(k):
    E = set()
    for i in range(N):
        E.add(tuple(sorted((i, (i + 1) % N))))
        E.add(tuple(sorted((i, (i + k) % N))))
    return edges_to_adj(sorted(E))


def double_swap(A, rng, target=None):
    """One random degree-preserving double-edge swap. If target given, only accept
    swaps that strictly increase edge overlap with target and keep connectivity."""
    edges = adj_to_edges(A)
    for _ in range(400):
        (a, b), (c, d) = [edges[i] for i in rng.choice(len(edges), 2, replace=False)]
        if len({a, b, c, d}) < 4:
            continue
        for (p, q), (r, s) in (((a, c), (b, d)), ((a, d), (b, c))):
            if A[p, q] or A[r, s]:
                continue
            if target is not None:
                gain = (target[p, q] + target[r, s]) - (target[a, b] + target[c, d])
                if gain <= 0:
                    continue
            B = A.copy()
            B[a, b] = B[b, a] = B[c, d] = B[d, c] = 0
            B[p, q] = B[q, p] = B[r, s] = B[s, r] = 1
            if connected(B):
                return B
    return None


def interpolants(A0, A1, levels):
    """Swap A0 toward A1; snapshot at overlap fractions `levels` (0..1 over the
    non-shared edges)."""
    base = int((A0 * A1).sum() // 2)
    total = int(A1.sum() // 2)
    out = {}
    A = A0.copy()
    for alpha in levels:
        want = base + alpha * (total - base)
        while (A * A1).sum() // 2 < round(want):
            B = double_swap(A, rng, target=A1)
            if B is None:
                break
            A = B
        out[alpha] = A.copy()
    return out


def smallworld(base_k, n_swaps, seed):
    r = np.random.default_rng(seed)
    A = circulant(base_k)
    done = 0
    while done < n_swaps:
        B = double_swap(A, r)
        if B is None:
            break
        A = B
        done += 1
    return A


def sbm(strong=True):
    E = set()
    for c0 in (0, 8):
        for i in range(8):
            E.add(tuple(sorted((c0 + i, c0 + (i + 1) % 8))))          # intra ring
    if strong:
        for c0 in (0, 8):
            for i in range(4):
                E.add(tuple(sorted((c0 + i, c0 + i + 4))))            # intra chords
        for i in range(8):
            E.add(tuple(sorted((i, 8 + i))))                          # 1 inter/node
    else:
        for i in range(8):
            E.add(tuple(sorted((i, 8 + i))))
            E.add(tuple(sorted((i, 8 + (i + 1) % 8))))                # 2 inter/node
    A = edges_to_adj(sorted(E))
    deg = A.sum(1)
    assert (deg == 4).all(), deg
    return A


def rand_regular(seed):
    r = np.random.default_rng(seed)
    for _ in range(200):
        stubs = np.repeat(np.arange(N), 4)
        r.shuffle(stubs)
        E = set()
        ok = True
        for a, b in stubs.reshape(-1, 2):
            a, b = int(a), int(b)
            if a == b or tuple(sorted((a, b))) in E:
                ok = False
                break
            E.add(tuple(sorted((a, b))))
        if ok:
            A = edges_to_adj(sorted(E))
            if connected(A):
                return A
    return None


def lap_eigs(A):
    d = A.sum(1).astype(float)
    di = 1 / np.sqrt(d)
    L = np.eye(N) - di[:, None] * A * di[None, :]
    return np.linalg.eigvalsh(L)


def pair_metrics(A, B):
    edit = int(np.abs(A - B).sum() // 2)           # edges to change (symmetric diff)
    spec = float(np.linalg.norm(lap_eigs(A) - lap_eigs(B)))
    shared = int((A * B).sum() // 2)
    return {"edit": edit, "spectral": round(spec, 3), "shared_edges": shared}


def build_all():
    fams = {"torus": torus(), "circ3": circulant(3)}
    for k in (2, 5, 6, 7):
        fams[f"circ{k}"] = circulant(k)
    for alpha, A in interpolants(fams["torus"], fams["circ3"],
                                 [0.25, 0.5, 0.75]).items():
        fams[f"interp{alpha}"] = A
    fams["sw4"] = smallworld(2, 4, seed=1)
    fams["sw12"] = smallworld(2, 12, seed=2)
    fams["sbm_strong"] = sbm(True)
    fams["sbm_weak"] = sbm(False)
    gaps = []
    for s in range(3000):
        A = rand_regular(s)
        if A is not None:
            gaps.append((float(lap_eigs(A)[1]), s, A))
    gaps.sort(key=lambda x: x[0])
    fams["expander_lo"] = gaps[0][2]
    fams["expander_hi"] = gaps[-1][2]
    print(f"expander gap range: {gaps[0][0]:.3f} .. {gaps[-1][0]:.3f}")
    for nm, A in fams.items():
        assert (A.sum(1) == 4).all() and connected(A), nm
    return fams


def main():
    fams = build_all()
    words = "clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,chair,candle,bread,mirror,apple,river"
    cells = []

    def cell(name, ga, gb, **kw):
        p = dict(name=name, ga=ga, gb=gb, ctx_a=600, ctx_b=600, tgen=400, npairs=6,
                 temp=1.0, topk=4, qcomm=1.0, words=words, seed=0)
        p.update(kw)
        p["metrics"] = pair_metrics(fams[ga], fams[gb])
        cells.append(p)

    # axis 1: prior disagreement (torus -> circ3 interpolation)
    for gb, al in (("torus", 0.0), ("interp0.25", 0.25), ("interp0.5", 0.5),
                   ("interp0.75", 0.75), ("circ3", 1.0)):
        cell(f"alpha{al}", "torus", gb, alpha=al)
    # axis 2: prior strength (at max disagreement)
    for ca, cb in ((1000, 100), (1000, 300), (1000, 1000), (300, 300), (100, 100)):
        cell(f"prior_{ca}v{cb}", "torus", "circ3", ctx_a=ca, ctx_b=cb, alpha=1.0)
    # axis 3: evidence quality
    for nm, kw in (("temp0.7", dict(temp=0.7)), ("temp1.3", dict(temp=1.3)),
                   ("free", dict(topk=0)), ("q0.5", dict(qcomm=0.5)),
                   ("q0.25", dict(qcomm=0.25))):
        cell(f"ev_{nm}", "torus", "circ3", alpha=1.0, **kw)
    # secondary families vs torus
    for gb in ("circ2", "circ5", "circ6", "circ7", "sbm_strong", "sbm_weak",
               "expander_lo", "expander_hi"):
        cell(f"fam_{gb}", "torus", gb)
    # small-world axis: base circulant(2) vs its rewirings
    for gb in ("sw4", "sw12"):
        cell(f"fam_{gb}", "circ2", gb)

    spec = {"graphs": {nm: adj_to_edges(A) for nm, A in fams.items()},
            "lambda2": {nm: round(float(lap_eigs(A)[1]), 3) for nm, A in fams.items()},
            "cells": cells}
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", "sweep_spec.json")
    json.dump(spec, open(out, "w"), indent=1)
    print(f"{len(cells)} cells -> {out}")
    for c in cells:
        print(f"  {c['name']:16s} {c['ga']:>10s} vs {c['gb']:<11s} {c['metrics']} "
              f"ctx={c['ctx_a']}/{c['ctx_b']} T={c['temp']} k={c['topk']} q={c['qcomm']}")


if __name__ == "__main__":
    main()
