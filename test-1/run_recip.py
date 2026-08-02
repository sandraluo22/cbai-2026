"""Reciprocity-gain experiment: does bidirectional closure contract beliefs beyond the
additive one-way effects?

All conditions share: grid+ring priors (fixed vocab), CTX=600, TGEN=600 alternating
slots (ring/B on even t, grid/A on odd t), k=4 (grid) / 2 (ring), NPAIRS=8. Conditions
differ ONLY in how each slot's token is produced; every stream is then evaluated by the
same matched pair of passive readers (grid-primed, ring-primed) reading the full
stream -- so token volume, speaking share, context length and reader exposure are
identical across conditions.

  bidir   : both slots by live agents conditioned on the full stream (full recursion)
  owAB    : A-slots by a SOLO-live A (hears only its own slots); B-slots by live B
  owBA    : mirror
  frozen  : A-slots by a belief-FROZEN A (conditions on prefix + previous stream token
            only; never accumulates joint evidence); B-slots by live B
  indep   : A-slots by solo-live A, B-slots by solo-live B (merged, never reciprocal)

(replay of the bidir transcript == the reader evaluation of bidir, by construction.)

PERTURBATION: for bidir and owAB, a control and a perturbed branch with COMMON RANDOM
NUMBERS; at t0=300 (an A-slot) the perturbed branch forces A's least-likely token.
Both agents' 16-dim predictives are logged every step in both branches -> the echo
JS(control, perturbed) per agent per step traces A_t -> B_{t+1} -> A_{t+2}.

Out: out_recip/recip.json, out_recip/readers.npz, out_recip/echo.npz
"""
from __future__ import annotations
import os, sys, json, time
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
import models as M

DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WORDS = "clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,chair,candle,bread,mirror,apple,river".split(",")
N, P, CTX, T = 16, 8, 600, 600
KA, KB = 4, 2
T0, TE = 301, 150                      # perturbation step and echo horizon
OUT = os.environ.get("OUTDIR", "/root/test-1/out_recip")
DEEP = list(range(24, 32))


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=P, walk_length=CTX, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    cm = model.config
    blocks = M._decoder_blocks(model)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)
    gpre = [[bos] + [cand[x] for x in w.nodes] for w in gw]
    rpre = [[bos] + [cand[x] for x in w.nodes] for w in rw]

    def prefill(rows):
        ids = torch.tensor(rows, device=DEVICE)
        try:
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=ids, use_cache=True)
        return {"past": o.past_key_values, "logits": o.logits[:, -1, :]}

    def step(st, toks):
        inp = torch.tensor(toks, device=DEVICE)[:, None]
        o = model(input_ids=inp, past_key_values=st["past"], use_cache=True)
        st["past"], st["logits"] = o.past_key_values, o.logits[:, -1, :]

    def probs_of(st):
        return torch.softmax(st["logits"][:, cand_t].float(), -1).cpu().numpy()

    def sample_crn(pp, k, u):
        q = pp.copy()
        if k > 0:
            q[np.argsort(q)[:-k]] = 0.0
        q = q / q.sum()
        return int(np.searchsorted(np.cumsum(q), u * 0.999999))

    U = np.random.default_rng(42).random((P, T))         # common random numbers

    def frozen_probs(prev_tokens):
        rows = [gpre[p] + [cand[prev_tokens[p]]] for p in range(P)]
        ids = torch.tensor(rows, device=DEVICE)
        try:
            o = model(input_ids=ids, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=ids)
        return torch.softmax(o.logits[:, -1, cand_t].float(), -1).cpu().numpy()

    def generate(cond, forced=None, record=False):
        """Returns streams [P][T] (+ per-step predictives of the two slot-generators if
        record). forced: {(p): token} applied at t=T0 on A's slot."""
        live_full = {}
        solo = {}
        if cond in ("bidir", "owBA"):
            live_full["A"] = prefill(gpre)
        if cond in ("bidir", "owAB", "frozen"):
            live_full["B"] = prefill(rpre)
        if cond in ("owAB", "indep"):
            solo["A"] = prefill(gpre)
        if cond in ("owBA", "indep"):
            solo["B"] = prefill(rpre)
        streams = [[] for _ in range(P)]
        rec = {"A": np.zeros((P, T, N), np.float16), "B": np.zeros((P, T, N), np.float16)} \
            if record else None
        for t in range(T):
            side = "B" if t % 2 == 0 else "A"
            k = KB if side == "B" else KA
            if side in live_full:
                pp = probs_of(live_full[side])
            elif side in solo:
                pp = probs_of(solo[side])
            else:                                          # frozen A
                prev_toks = [streams[p][-1] if streams[p] else gw[p].nodes[-1]
                             for p in range(P)]
                pp = frozen_probs(prev_toks)
            if record:
                for s2 in ("A", "B"):
                    if s2 in live_full:
                        rec[s2][:, t] = probs_of(live_full[s2]).astype(np.float16)
                    elif s2 in solo:
                        rec[s2][:, t] = probs_of(solo[s2]).astype(np.float16)
            toks = np.zeros(P, np.int64)
            for p in range(P):
                if forced is not None and t == T0 and side == "A":
                    node = forced[p]
                else:
                    node = sample_crn(pp[p], k, U[p, t])
                streams[p].append(node)
                toks[p] = cand[node]
            for st in live_full.values():
                step(st, toks)
            if side in solo:
                step(solo[side], toks)                     # solo hears ONLY its slots
        for st in list(live_full.values()) + list(solo.values()):
            del st["past"]
        torch.cuda.empty_cache()
        return streams, rec

    A_g, A_r = adjacency(grid), adjacency(ring)
    results = {}
    reader_nm = {}
    COND_LIST = [] if os.environ.get("ECHO_ONLY") == "1" else \
        ["bidir", "owAB", "owBA", "frozen", "indep"]
    for cond in COND_LIST:
        t0 = time.time()
        streams, _ = generate(cond)
        # matched passive readers
        jsr = np.zeros(T)
        NMr = np.zeros((2, len(DEEP), N, cm.hidden_size))
        cntr = np.zeros((2, N))
        grabbed = {}
        def mk(L):
            def hh(_m, _i, o2):
                grabbed[L] = (o2[0] if isinstance(o2, tuple) else o2).detach()
            return hh
        handles = [blocks[L].register_forward_hook(mk(L)) for L in DEEP]
        try:
            for p in range(P):
                prs = {}
                for ci, pre in ((0, gpre[p]), (1, rpre[p])):
                    full = pre + [cand[x] for x in streams[p]]
                    ids = torch.tensor([full], device=DEVICE)
                    grabbed.clear()
                    o = model(input_ids=ids)
                    prs[ci] = torch.softmax(o.logits[0][:, cand_t].float(), -1).cpu().numpy()
                    pos = list(range(len(pre) + 300, len(pre) + T))
                    nds = streams[p][300:]
                    for li, L in enumerate(DEEP):
                        hh2 = grabbed[L][0][pos].float().cpu().numpy()
                        np.add.at(NMr[ci, li], nds, hh2)
                    np.add.at(cntr[ci], nds, 1)
                for t in range(T):
                    a = prs[0][len(gpre[p]) - 1 + t]
                    b = prs[1][len(rpre[p]) - 1 + t]
                    m = 0.5 * (a + b)
                    def kl(x, y):
                        mk2 = x > 0
                        return (x[mk2] * np.log(x[mk2] / np.maximum(y[mk2], 1e-12))).sum()
                    jsr[t] += (0.5 * kl(a, m) + 0.5 * kl(b, m)) / P
        finally:
            for h in handles:
                h.remove()
        NMr = NMr / np.maximum(cntr, 1)[:, None, :, None]
        reader_nm[cond] = NMr.astype(np.float16)
        # stream stats
        gen_valid = {"grid": float(np.mean([[A_g[s[t-1], s[t]] for t in range(1, T)]
                                            for s in streams])),
                     "ring": float(np.mean([[A_r[s[t-1], s[t]] for t in range(1, T)]
                                            for s in streams]))}
        results[cond] = {"reader_js": np.round(jsr, 4).tolist(),
                         "streams": [list(map(int, s)) for s in streams],
                         "validity": gen_valid}
        print(f"{cond}: final reader-JS={jsr[-50:].mean():.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    if reader_nm:
        np.savez_compressed(os.path.join(OUT, "readers.npz"),
                            **{f"nm_{c}": v for c, v in reader_nm.items()})

    # ---- perturbation impulse response ------------------------------------
    echo = {}
    for cond in ("bidir", "owAB"):
        sc, rc = generate(cond, record=True)
        forced = {}
        for p in range(P):
            # least-likely token under the A-generator at T0 in the control branch
            pa = rc["A"][p, T0] if rc["A"][p, T0].sum() > 0 else np.ones(N) / N
            forced[p] = int(np.argmin(pa))
        sp, rp = generate(cond, forced=forced, record=True)
        e = {}
        for side in ("A", "B"):
            js = np.zeros(TE)
            for p in range(P):
                for i in range(TE):
                    a = rc[side][p, T0 + i].astype(np.float64)
                    b = rp[side][p, T0 + i].astype(np.float64)
                    if a.sum() == 0 or b.sum() == 0:
                        continue
                    a /= a.sum(); b /= b.sum()
                    m = 0.5 * (a + b)
                    def kl(x, y):
                        mk2 = x > 0
                        return (x[mk2] * np.log(x[mk2] / np.maximum(y[mk2], 1e-12))).sum()
                    js[i] += (0.5 * kl(a, m) + 0.5 * kl(b, m)) / P
            e[side] = np.round(js, 5).tolist()
        echo[cond] = e
        print(f"echo {cond} done", flush=True)
    prev = {}
    pj = os.path.join(OUT, "recip.json")
    if os.path.isfile(pj) and os.environ.get("ECHO_ONLY") == "1":
        prev = json.load(open(pj))
        results = {c: dict(v, streams=prev["streams"][c])
                   for c, v in prev["results"].items()}
    json.dump({"results": {c: {k2: v for k2, v in r.items() if k2 != "streams"}
                           for c, r in results.items()},
               "streams": {c: results[c]["streams"] for c in results},
               "echo": echo},
              open(os.path.join(OUT, "recip.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
