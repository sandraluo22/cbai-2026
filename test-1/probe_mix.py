"""Content-based (non-positional) latent-mixture probe.

Stream: two GENUINE independent walks (ring, grid) interleaved by an iid Bernoulli(0.5)
schedule -- no positional cue identifies the source; the threads' current positions are
latent states recoverable only from content. Both primed contexts (ring-prefix,
grid-prefix) read the same stream.

At every late position t we measure the model's predictive mass on:
  M_ring(gap) : ring-neighbours of the most recent RING-thread token (gap = how many
                stream steps back it sits)   -- requires content de-interleaving
  M_grid(gap) : grid-neighbours of the most recent GRID-thread token
  baselines   : ring-/grid-neighbours of the immediately previous token

Baselines replayed on the same streams with frozen (gamma=0.96, alpha0=0.05):
  merged : single count matrix, predictive from prev token
  oracle : knows the true source labels; per-thread chains; predictive =
           0.5 * p_ring(. | last ring tok) + 0.5 * p_grid(. | last grid tok)
lambda_content = mixture weight between merged and oracle best matching the LLM.

Env: DEVICE. Out: runs/out_probe/mix_probe.json
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
import models as M

DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
WORDS = "clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,chair,candle,bread,mirror,apple,river".split(",")
N, P, CTX, T = 16, 8, 1000, 600
GAMMA, A0 = 0.96, 0.05
LATE = 300
GAPS = [1, 2, 3, 4, 5, 6]           # 6 == "6 or more"


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def gapb(g):
    return min(g, 6)


@torch.no_grad()
def main():
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=P, walk_length=CTX + 450, seed=0)
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    grid.words = list(WORDS); ring.words = list(WORDS)
    A_g, A_r = adjacency(grid), adjacency(ring)
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))

    rng = np.random.default_rng(5)
    streams, sources = [], []
    for p in range(P):
        ri, gi = CTX, CTX                       # thread positions (continue past prefix)
        seq, src = [], []
        for t in range(T):
            if rng.random() < 0.5 and ri < len(rw[p].nodes):
                seq.append(rw[p].nodes[ri]); src.append("R"); ri += 1
            else:
                seq.append(gw[p].nodes[gi]); src.append("G"); gi += 1
        streams.append(seq); sources.append(src)

    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)

    def collect(pred_fn, tagname):
        """pred_fn(p, t) -> 16-prob vector before emitting stream[p][t]."""
        Mr = {g: [] for g in GAPS}
        Mg = {g: [] for g in GAPS}
        b_r1, b_g1 = [], []
        for p in range(P):
            seq, src = streams[p], sources[p]
            for t in range(LATE, T):
                pr = pred_fn(p, t)
                if pr is None:
                    continue
                lr = lg = None
                for u in range(t - 1, -1, -1):
                    if src[u] == "R" and lr is None:
                        lr = u
                    if src[u] == "G" and lg is None:
                        lg = u
                    if lr is not None and lg is not None:
                        break
                if lr is not None:
                    Mr[gapb(t - lr)].append(pr[A_r[seq[lr]]].sum())
                if lg is not None:
                    Mg[gapb(t - lg)].append(pr[A_g[seq[lg]]].sum())
                b_r1.append(pr[A_r[seq[t - 1]]].sum())
                b_g1.append(pr[A_g[seq[t - 1]]].sum())
        out = {"M_ring_by_gap": {g: float(np.mean(v)) for g, v in Mr.items() if v},
               "M_grid_by_gap": {g: float(np.mean(v)) for g, v in Mg.items() if v},
               "ring_nbrs_prev": float(np.mean(b_r1)),
               "grid_nbrs_prev": float(np.mean(b_g1))}
        print(tagname, json.dumps(out["M_ring_by_gap"]), flush=True)
        return out

    # ---- LLM pass (both primed contexts) ----------------------------------
    llm_out = {}
    for ctxname, walks in (("ring", rw), ("grid", gw)):
        probs_cache = {}
        for p in range(P):
            pref = walks[p].nodes[:CTX]
            full = pref + streams[p]
            ids = torch.tensor([[bos] + [cand[x] for x in full]], device=DEVICE)
            lg2 = model(input_ids=ids).logits[0][:, cand_t].float()
            pr = torch.softmax(lg2, -1).cpu().numpy()
            for t in range(LATE, T):
                probs_cache[(p, t)] = pr[CTX + t]      # position of token t-1's logits
        llm_out[ctxname] = collect(lambda p, t: probs_cache.get((p, t)), f"LLM-{ctxname}")

    # ---- surrogate replays (frozen gamma/alpha) --------------------
    def replay_nodes(kind):
        preds = {}
        for p in range(P):
            pref = rw[p].nodes[:CTX]
            Cm = np.zeros((N, N)); Cr = np.zeros((N, N)); Cg = np.zeros((N, N))
            for a, b in zip(pref, pref[1:]):
                Cm *= GAMMA; Cr *= GAMMA; Cg *= GAMMA
                Cm[a, b] += 1; Cr[a, b] += 1; Cg[a, b] += 1
            prev = pref[-1]
            last_node = {"R": prev, "G": prev}
            seq, src = streams[p], sources[p]
            for t in range(T):
                if t >= LATE:
                    if kind == "merged":
                        row = A0 + Cm[prev]; pr = row / row.sum()
                    else:
                        pr = np.zeros(N)
                        for w, Cx, ln in ((0.5, Cr, last_node["R"]),
                                          (0.5, Cg, last_node["G"])):
                            row = A0 + Cx[ln]
                            pr = pr + w * row / row.sum()
                    preds[(p, t)] = pr
                Cm *= GAMMA; Cr *= GAMMA; Cg *= GAMMA
                Cm[prev, seq[t]] += 1
                if src[t] == "R":
                    Cr[last_node["R"], seq[t]] += 1; last_node["R"] = seq[t]
                else:
                    Cg[last_node["G"], seq[t]] += 1; last_node["G"] = seq[t]
                prev = seq[t]
        return preds

    reps = {}
    for kind in ("merged", "oracle"):
        preds = replay_nodes(kind)
        reps[kind] = collect(lambda p, t: preds.get((p, t)), kind)

    # ---- lambda_content fit ------------------------------------------------
    keys = [("M_ring_by_gap", g) for g in GAPS] + [("M_grid_by_gap", g) for g in GAPS]
    def vec(d):
        return np.array([d[k].get(g, np.nan) for k, g in keys])
    llm_v = np.nanmean([vec(llm_out["ring"]), vec(llm_out["grid"])], 0)
    mv, ov = vec(reps["merged"]), vec(reps["oracle"])
    mask = ~np.isnan(llm_v) & ~np.isnan(mv) & ~np.isnan(ov)
    lams = np.linspace(0, 1, 101)
    errs = [np.mean((lam * ov[mask] + (1 - lam) * mv[mask] - llm_v[mask]) ** 2)
            for lam in lams]
    lam_hat = float(lams[int(np.argmin(errs))])
    print("lambda_content =", lam_hat, flush=True)

    out = {"llm": llm_out, "merged": reps["merged"], "oracle": reps["oracle"],
           "lambda_content": lam_hat, "p_switch": 0.5}
    os.makedirs(os.path.join(_here, "runs", "out_probe"), exist_ok=True)
    json.dump(out, open(os.path.join(_here, "runs", "out_probe", "mix_probe.json"), "w"),
              indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
