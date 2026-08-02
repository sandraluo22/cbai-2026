"""Basin-stability experiment: R rollouts of ONE fixed condition (identical grid/ring
prefix walks, pair 0 of the k2_fix setup; only the generation sampling differs), then
perturbation arms from each rollout's endpoint:

  C (control) : continue 100 steps, top-k=2 throughout
  F (force)   : inject ONE off-trap token (random node outside the rollout's dominant
                late pair, appended to both contexts), then continue 100 steps k=2
  T (untrap)  : continue with top-k REMOVED for 20 steps, then k=2 for 80

Per rollout: full streams (main + arms), late occupancy, transition counts, dominant
pair; per-rollout LATE node-means (deep layers 24-31, both contexts) for per-run
geometry. Verdict logic downstream: if perturbed runs return to the SAME pair ->
stable basin; if they disperse -> stochastic lock-in.

Env: R(40) DEVICE. Out: out_basin/basin.json + nodemeans_basin.npz
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

R = int(os.environ.get("R", "40"))
CTX, T, TP = 1000, 600, 100
K = int(os.environ.get("K", "2"))
BAN = os.environ.get("BAN", "none")          # none | backtrack (bans self + 2-cycles)
PERM = int(os.environ.get("PERM", "-1"))     # >=0: shuffle word->node assignment
GRADE = os.environ.get("GRADE", "0") == "1"  # graded-evidence escape arms
GRADE2 = os.environ.get("GRADE2", "0") == "1"  # factorial: n x near/far, fixed segments
SOLO = os.environ.get("SOLO", "0") == "1"      # free generation: no coupling
CAPTURE = os.environ.get("CAPTURE", "1") == "1"
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WORDS = "clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,chair,candle,bread,mirror,apple,river".split(",")
if PERM >= 0:
    _pr = np.random.default_rng(PERM)
    WORDS = list(np.array(WORDS)[_pr.permutation(16)])
N = 16
DEEP = list(range(24, 32))
OUT = os.environ.get("OUTDIR", "/root/test-1/out_basin")


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=1, walk_length=CTX, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    gwalk = G.generate_walks(grid, replace(cfg, graph_type="grid"))[0].nodes
    rwalk = G.generate_walks(ring, replace(cfg, graph_type="ring"))[0].nodes

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

    grow = [bos] + [cand[x] for x in gwalk]
    rrow = [bos] + [cand[x] for x in rwalk]
    # batch rows: 0..R-1 grid contexts, R..2R-1 ring contexts (one rollout each)
    rows = [list(grow) for _ in range(R)] + [list(rrow) for _ in range(R)]

    def prefill(token_rows):
        ids = torch.tensor(token_rows, device=DEVICE)
        try:
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=ids, use_cache=True)
        return o.past_key_values, o.logits[:, -1, :]

    def gen_phase(past, logits, n_steps, t0, streams, rng, k_sched, forced=None):
        """Alternating generation (ring on even global t). k_sched(t)->k. forced:
        {rollout: node} applied at the FIRST step instead of sampling."""
        for i in range(n_steps):
            t = t0 + i
            gen_rows = (list(range(R, 2 * R)) if t % 2 == 0 else list(range(R)))
            lg = logits[gen_rows][:, cand_t].float()
            probs = torch.softmax(lg, -1).cpu().numpy()
            toks = np.zeros(R, np.int64)
            for r in range(R):
                if forced is not None and i == 0:
                    node = forced[r]
                else:
                    pp = probs[r].copy()
                    if BAN == "backtrack":
                        s_hist = streams[r]
                        if len(s_hist) >= 1:
                            pp[s_hist[-1]] = 0.0          # no self-loop
                        if len(s_hist) >= 2:
                            pp[s_hist[-2]] = 0.0          # no 2-cycle return
                    k = k_sched(t)
                    if k > 0 and pp.sum() > 0:
                        kept = np.argsort(pp)[-k:]
                        mask = np.zeros(N); mask[kept] = 1
                        pp = pp * mask
                    if pp.sum() <= 0:
                        pp = probs[r].copy()
                    node = int(rng.choice(N, p=pp / pp.sum()))
                streams[r].append(node)
                toks[r] = cand[node]
            inp = torch.tensor(np.concatenate([toks, toks]), device=DEVICE)[:, None]
            o = model(input_ids=inp, past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1, :]
        return past, logits

    # ---- main phase --------------------------------------------------------
    rng = np.random.default_rng(123)
    t0 = time.time()
    past, logits = prefill(rows)
    streams = [[] for _ in range(R)]
    if SOLO:                                     # each context samples its OWN token
        streams2 = [[] for _ in range(R)]        # ring-context solo streams
        for t in range(T):
            lga = torch.softmax(logits[:, cand_t].float(), -1).cpu().numpy()
            toks = np.zeros(2 * R, np.int64)
            for row in range(2 * R):
                pp = lga[row].copy()
                if K > 0:
                    pp[np.argsort(pp)[:-K]] = 0.0
                node = int(rng.choice(N, p=pp / pp.sum()))
                (streams if row < R else streams2)[row % R].append(node)
                toks[row] = cand[node]
            o = model(input_ids=torch.tensor(toks, device=DEVICE)[:, None],
                      past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1, :]
    else:
        past, logits = gen_phase(past, logits, T, 0, streams, rng, lambda t: K)
    del past
    torch.cuda.empty_cache()
    print(f"main phase done ({time.time()-t0:.0f}s)", flush=True)

    def top_pair(seq, lo, hi):
        cnt = {}
        for a, b in zip(seq[lo:hi - 1], seq[lo + 1:hi]):
            kk = tuple(sorted((a, b)))
            cnt[kk] = cnt.get(kk, 0) + 1
        (a, b), c = max(cnt.items(), key=lambda kv: kv[1])
        return [int(a), int(b)], c / max(hi - lo - 1, 1)

    main_traps = [top_pair(s, 300, T) for s in streams]
    solo_extra = {}
    if SOLO:
        solo_extra = {"ring_traps": [top_pair(s, 300, T) for s in streams2],
                      "ring_streams": [list(map(int, s)) for s in streams2],
                      "ring_occup": [[int((np.array(s[300:]) == i).sum()) for i in range(N)]
                                     for s in streams2]}
    occup = [[int((np.array(s[300:]) == i).sum()) for i in range(N)] for s in streams]
    trans = []
    for s in streams:
        Cm = np.zeros((N, N), int)
        for a, b in zip(s[300:-1], s[301:]):
            Cm[a, b] += 1
        trans.append(Cm.tolist())

    # ---- per-rollout late geometry (deep layers, both contexts) ------------
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o2): grabbed[L] = (o2[0] if isinstance(o2, tuple) else o2).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in DEEP] if CAPTURE else []
    NM = np.zeros((R, 2, len(DEEP), N, cm.hidden_size), np.float16)
    try:
        for r in (range(R) if CAPTURE else []):
            for ci, base in enumerate((grow, rrow)):
                full = base + [cand[x] for x in streams[r]]
                nodes_all = (gwalk if ci == 0 else rwalk) + streams[r]
                fids = torch.tensor([full], device=DEVICE)
                grabbed.clear()
                try:
                    model(input_ids=fids, logits_to_keep=1)
                except TypeError:
                    model(input_ids=fids)
                pos = list(range(1 + CTX + 300, 1 + CTX + T))
                nds = nodes_all[CTX + 300:CTX + T]
                for li, L in enumerate(DEEP):
                    hh = grabbed[L][0][pos].float().cpu().numpy()
                    sums = np.zeros((N, cm.hidden_size))
                    cnts = np.zeros(N)
                    np.add.at(sums, nds, hh)
                    np.add.at(cnts, nds, 1)
                    NM[r, ci, li] = (sums / np.maximum(cnts, 1)[:, None]).astype(np.float16)
            if (r + 1) % 10 == 0:
                print(f"capture rollout {r+1}/{R} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()
    if CAPTURE:
        np.savez_compressed(os.path.join(OUT, "nodemeans_basin.npz"), nm=NM,
                            deep_layers=np.array(DEEP), words=np.array(WORDS))

    # ---- perturbation arms -------------------------------------------------
    arms = {}
    if SOLO:
        ARMLIST = []
    elif GRADE2:
        ARMLIST = ["C", "n1_near", "n1_far", "n4_near", "n4_far"]
    elif GRADE:
        ARMLIST = ["C"] + [f"G{n}" for n in (1, 2, 4, 8)]
    else:
        ARMLIST = ["C", "F", "T"]
    for arm in ARMLIST:
        rng_a = np.random.default_rng(777)
        branch_rows = []
        forced = None
        if arm == "F":
            forced = {}
            for r in range(R):
                tp = set(main_traps[r][0])
                choices = [n for n in range(N) if n not in tp]
                forced[r] = int(rng_a.choice(choices))
        seg = None
        if GRADE2 and arm != "C":                   # factorial: fixed seeded segments
            nseg = int(arm[1])
            near = arm.endswith("near")
            seg = {}
            gadj = [sorted(grid.adjacency[i]) for i in range(N)]
            # grid BFS distances
            import collections
            D = np.full((N, N), 99)
            for src in range(N):
                D[src, src] = 0
                dq = collections.deque([src])
                while dq:
                    u = dq.popleft()
                    for v in gadj[u]:
                        if D[src, v] > D[src, u] + 1:
                            D[src, v] = D[src, u] + 1
                            dq.append(v)
            rng_seg = np.random.default_rng(31)
            for r in range(R):
                a1, b1 = main_traps[r][0]
                if near:
                    cands_s = sorted(set(gadj[a1] + gadj[b1]) - {a1, b1})
                else:
                    cands_s = [n for n in range(N)
                               if D[a1, n] >= 3 and D[b1, n] >= 3 and n not in (a1, b1)]
                    if not cands_s:
                        cands_s = [n for n in range(N) if n not in (a1, b1)]
                cur = int(cands_s[int(rng_seg.integers(len(cands_s)))])
                walk = [cur]
                for _ in range(nseg - 1):
                    cur = int(rng_seg.choice(gadj[cur]))
                    walk.append(cur)
                seg[r] = walk
        if arm.startswith("G"):                     # graded: n-step valid grid walk
            nseg = int(arm[1:])
            seg = {}
            gadj = [sorted(grid.adjacency[i]) for i in range(N)]
            for r in range(R):
                tp = set(main_traps[r][0])
                cur = int(rng_a.choice([n for n in range(N) if n not in tp]))
                walk = [cur]
                for _ in range(nseg - 1):
                    cur = int(rng_a.choice(gadj[cur]))
                    walk.append(cur)
                seg[r] = walk
        ext = (lambda r: [cand[x] for x in seg[r]]) if seg else (lambda r: [])
        for r in range(R):
            branch_rows.append(grow + [cand[x] for x in streams[r]] + ext(r))
        for r in range(R):
            branch_rows.append(rrow + [cand[x] for x in streams[r]] + ext(r))
        past, logits = prefill(branch_rows)
        bstreams = [[] for _ in range(R)]
        if arm == "T":
            ksched = lambda t: (0 if t < T + 20 else K)
        else:
            ksched = lambda t: K
        past, logits = gen_phase(past, logits, TP, T, bstreams, rng_a, ksched,
                                 forced=forced)
        del past
        torch.cuda.empty_cache()
        arms[arm] = {"streams": [list(map(int, s)) for s in bstreams],
                     "seg": ({r: seg[r] for r in seg} if seg else None),
                     "last50_top": [top_pair(s, TP - 50, TP) for s in bstreams]}
        print(f"arm {arm} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"R": R, "ctx": CTX, "tgen": T, "tp": TP, "k": K, "words": WORDS,
               **solo_extra,
               "main_streams": [list(map(int, s)) for s in streams],
               "main_traps": main_traps, "occupancy": occup, "transitions": trans,
               "arms": arms},
              open(os.path.join(OUT, "basin.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
