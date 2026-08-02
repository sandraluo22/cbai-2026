"""Transcript-FORMATION analysis: does the online coupled model predict the bidir
stream better than static or deaf alternatives?  (Sandra's decisive test.)

Uses the saved bidir streams from out_recip. All parts teacher-forced.

PART A (models #3 + timing #4). For every slot of the actual bidir stream, the actual
token's log-likelihood under three LLM-instantiated policies of the slot's owner:
  static : owner's PRIOR policy given (own prefix, previous stream token) only
  deaf   : owner updated ONLY on its own past slots (prefix + own-slot subsequence)
  coupled: owner updated on the full stream (the live agent)
Reported by owner and by time-bin. Plus: prior-decay curve JS(coupled_t, static_t)
and mutual-validity(t) of the stream.

PART B (counterfactuals #1 + forced continuation #2). At sampled early A-turns t,
compare the PARTNER's (B's) next-step predictive after the actual token vs matched
counterfactuals (owner's 2nd top-k choice, owner's static-policy top choice, random
other node), with the SAME actual continuation teacher-forced for H=8 further steps
-> immediate influence and belief-deflection decay without trajectory forking.
Symmetric B->A test at sampled early B-turns.

PART C (delayed reciprocity #5). Generate new streams where mutual hearing is enabled
only from turn k (before k: each agent hears only its own slots), k in {100, 300};
record stream validity trajectory and late dominant pairs.

Out: out_formation/formation.json
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
H = 8
RECIP = os.environ.get("RECIP", "/root/test-1/out_recip")
OUT = os.environ.get("OUTDIR", "/root/test-1/out_formation")
BINS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 600)]


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def jsd(a, b):
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    def kl(x, y):
        mk = x > 0
        return float((x[mk] * np.log(x[mk] / np.maximum(y[mk], 1e-12))).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    rec = json.load(open(os.path.join(RECIP, "recip.json")))
    streams = rec["streams"]["bidir"]
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=P, walk_length=CTX, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    A_g, A_r = adjacency(grid), adjacency(ring)
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)
    pre = {"A": [[bos] + [cand[x] for x in w.nodes] for w in gw],
           "B": [[bos] + [cand[x] for x in w.nodes] for w in rw]}
    prevlast = {"A": [w.nodes[-1] for w in gw], "B": [w.nodes[-1] for w in rw]}

    def fwd_probs(rows):
        ids = torch.tensor(rows, device=DEVICE)
        o = model(input_ids=ids)
        return torch.softmax(o.logits[:, :, cand_t].float(), -1).cpu().numpy()

    # ---- PART A ------------------------------------------------------------
    t0 = time.time()
    ll = {m2: {"A": [[] for _ in BINS], "B": [[] for _ in BINS]}
          for m2 in ("static", "deaf", "coupled")}
    prior_js = np.zeros(T); prior_cnt = np.zeros(T)
    validity = np.zeros(T)
    for p in range(P):
        seq = streams[p]
        # static lookup: owner's prior policy for each possible previous node
        stat = {}
        for side in ("A", "B"):
            rows = [pre[side][p] + [cand[v]] for v in range(N)]
            pr = fwd_probs(rows)
            stat[side] = pr[:, -1, :]                     # [16 prev, 16 next]
        # coupled: full-stream forward per owner
        coup = {}
        for side in ("A", "B"):
            full = pre[side][p] + [cand[x] for x in seq]
            pr = fwd_probs([full])[0]
            L0 = len(pre[side][p])
            coup[side] = {t: pr[L0 - 1 + t] for t in range(T)}
        # deaf: own-slot subsequence forward per owner
        deaf = {}
        for side in ("A", "B"):
            own_ts = [t for t in range(T) if ("B" if t % 2 == 0 else "A") == side]
            sub = pre[side][p] + [cand[seq[t]] for t in own_ts]
            pr = fwd_probs([sub])[0]
            L0 = len(pre[side][p])
            # predictive BEFORE appending own token #i is at position L0-1+i
            deaf[side] = {own_ts[i]: pr[L0 - 1 + i] for i in range(len(own_ts))}
        for t in range(T):
            side = "B" if t % 2 == 0 else "A"
            prv = seq[t - 1] if t >= 1 else prevlast[side][p]
            x = seq[t]
            bi = next(i for i, (lo, hi) in enumerate(BINS) if lo <= t < hi)
            ll["static"][side][bi].append(np.log(max(stat[side][prv][x], 1e-9)))
            ll["coupled"][side][bi].append(np.log(max(coup[side][t][x], 1e-9)))
            ll["deaf"][side][bi].append(np.log(max(deaf[side][t][x], 1e-9)))
            prior_js[t] += jsd(coup[side][t], stat[side][prv]); prior_cnt[t] += 1
            if t >= 1:
                validity[t] += (A_g[prv, x] or A_r[prv, x]) / P
        print(f"partA pair {p} ({time.time()-t0:.0f}s)", flush=True)
    partA = {"loglik": {m2: {s: [float(np.mean(v)) if v else None for v in ll[m2][s]]
                             for s in ("A", "B")} for m2 in ll},
             "prior_decay_js": (prior_js / np.maximum(prior_cnt, 1)).round(4).tolist(),
             "mutual_validity": validity.round(3).tolist(), "bins": BINS}

    # ---- PART B ------------------------------------------------------------
    partB = {}
    for owner, partner in (("A", "B"), ("B", "A")):
        turns = [t for t in range(4, 100)
                 if ("B" if t % 2 == 0 else "A") == owner][::5][:10]
        imm = {v: [] for v in ("second", "static_top", "random")}
        decay = {v: np.zeros(H) for v in ("second", "static_top", "random")}
        for p in range(P):
            seq = streams[p]
            rows, meta = [], []
            for t in turns:
                base = pre[partner][p] + [cand[x] for x in seq[:t]]
                prv = seq[t - 1] if t >= 1 else prevlast[owner][p]
                # counterfactual choices
                full_own = pre[owner][p] + [cand[x] for x in seq[:t]]
                pown = fwd_probs([full_own])[0][-1]
                order = np.argsort(pown)[::-1]
                second = int(order[1] if order[0] == seq[t] else order[0])
                srows = fwd_probs([pre[owner][p] + [cand[prv]]])[0][-1]
                stat_top = int(np.argmax(srows))
                rng = np.random.default_rng(1000 + t)
                rnd = int(rng.choice([v for v in range(N) if v != seq[t]]))
                for lab, v in (("actual", seq[t]), ("second", second),
                               ("static_top", stat_top), ("random", rnd)):
                    cont = [cand[x] for x in seq[t + 1:t + 1 + H]]
                    rows.append(base + [cand[v]] + cont)
                    meta.append((t, lab))
            maxlen = max(len(r) for r in rows)
            prs = fwd_probs([r + [bos] * (maxlen - len(r)) for r in rows])
            L0s = {t: len(pre[partner][p]) + t for t in turns}
            ref = {}
            for r_i, (t, lab) in enumerate(meta):
                pos0 = L0s[t]                       # predictive right after token t
                if lab == "actual":
                    ref[t] = prs[r_i]
            for r_i, (t, lab) in enumerate(meta):
                if lab == "actual":
                    continue
                imm[lab].append(jsd(prs[r_i][L0s[t]], ref[t][L0s[t]]))
                for h in range(H):
                    decay[lab][h] += jsd(prs[r_i][L0s[t] + 1 + h],
                                         ref[t][L0s[t] + 1 + h]) / (len(turns) * P)
        partB[f"{owner}->{partner}"] = {
            "immediate_js": {k: float(np.mean(v)) for k, v in imm.items()},
            "forced_decay": {k: v.round(4).tolist() for k, v in decay.items()}}
        print(f"partB {owner}->{partner} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- PART C ------------------------------------------------------------
    partC = {}
    for kdelay in (100, 300):
        rng = np.random.default_rng(42)
        st = {s: {"past": None, "logits": None} for s in ("A", "B")}
        for s in ("A", "B"):
            ids = torch.tensor(pre[s], device=DEVICE)
            try:
                o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            except TypeError:
                o = model(input_ids=ids, use_cache=True)
            st[s] = {"past": o.past_key_values, "logits": o.logits[:, -1, :]}
        seqs = [[] for _ in range(P)]
        val = np.zeros(T)
        for t in range(T):
            side = "B" if t % 2 == 0 else "A"
            k = KB if side == "B" else KA
            pp = torch.softmax(st[side]["logits"][:, cand_t].float(), -1).cpu().numpy()
            toks = np.zeros(P, np.int64)
            for p in range(P):
                q = pp[p].copy()
                q[np.argsort(q)[:-k]] = 0.0
                node = int(rng.choice(N, p=q / q.sum()))
                prv = seqs[p][-1] if seqs[p] else prevlast[side][p]
                val[t] += (A_g[prv, node] or A_r[prv, node]) / P
                seqs[p].append(node)
                toks[p] = cand[node]
            inp = torch.tensor(toks, device=DEVICE)[:, None]
            for s in ("A", "B"):
                if t >= kdelay or s == side:          # before kdelay: hear only self
                    o = model(input_ids=inp, past_key_values=st[s]["past"],
                              use_cache=True)
                    st[s] = {"past": o.past_key_values, "logits": o.logits[:, -1, :]}
        for s in ("A", "B"):
            del st[s]["past"]
        torch.cuda.empty_cache()
        def top_pair(sq):
            cnt = {}
            for a, b in zip(sq[300:-1], sq[301:]):
                kk = tuple(sorted((a, b)))
                cnt[kk] = cnt.get(kk, 0) + 1
            (a, b), c = max(cnt.items(), key=lambda kv: kv[1])
            return [int(a), int(b)], c / 299
        partC[f"k{kdelay}"] = {"validity": val.round(3).tolist(),
                               "late_traps": [top_pair(s) for s in seqs]}
        print(f"partC k={kdelay} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"A": partA, "B": partB, "C": partC},
              open(os.path.join(OUT, "formation.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
