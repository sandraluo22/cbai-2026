"""Degree-controlled sweep: coupled dueling contexts over 16-node 4-regular graph pairs
(spec from graphs16.py). All graphs share node count, degree (4), edge count (32) and
the uniform random-walk stationary distribution -- only wiring varies.

Per cell (see runs/sweep_spec.json): agents A and B primed with ctx_a / ctx_b walk
tokens on graphs ga / gb; TGEN alternating generation steps (B on even steps).
Evidence-quality knobs: temperature, top-k (0 = free sampling), and qcomm = probability
the generator's token is delivered to the partner (otherwise the partner appends its
OWN sample, so context lengths stay uniform; q=0 -> two free-runners).

Logged per step: mean Jensen-Shannon divergence between the two agents' 16-way
predictives (consensus trajectory), per-context appended node sequences (win/graph-fit
analysis). Captured: per-node mean residuals, windows base / joint_early / joint_late.

Env: SPEC(path to sweep_spec.json) OUTBASE(/root/test-1/out_sweep) CELLS(optional
     comma filter) CM_SRC DEVICE
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

SPEC = json.load(open(os.environ.get("SPEC", os.path.join(_here, "sweep_spec.json"))))
OUTBASE = os.environ.get("OUTBASE", "/root/test-1/out_sweep")
ONLY = set(os.environ.get("CELLS", "").split(",")) - {""}
DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
N = 16


def make_graph(name, words):
    global N
    N = len(words)
    entry = SPEC["graphs"][name]
    if isinstance(entry, dict) and "directed" in entry:   # directed process
        edges = entry["directed"]
        adj = [[] for _ in range(N)]
        A = np.zeros((N, N))
        for a, b in edges:
            adj[a].append(b)
            A[a, b] = 1
        d = np.maximum(A.sum(1), 1)
        L = np.eye(N) - (1 / np.sqrt(d))[:, None] * ((A + A.T) / 2) * (1 / np.sqrt(d))[None, :]
        _, U = np.linalg.eigh(L)
        coords = [(float(U[i, 1]), float(U[i, 2])) for i in range(N)]
        return G.Graph(n_nodes=N, words=list(words), adjacency=[sorted(x) for x in adj],
                       coords=coords), A.astype(bool)
    edges = entry
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    A = np.zeros((N, N))
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    d = A.sum(1)
    L = np.eye(N) - (1 / np.sqrt(d))[:, None] * A * (1 / np.sqrt(d))[None, :]
    _, U = np.linalg.eigh(L)
    coords = [(float(U[i, 1]), float(U[i, 2])) for i in range(N)]
    return G.Graph(n_nodes=N, words=list(words), adjacency=[sorted(x) for x in adj],
                   coords=coords), A.astype(bool)


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def run_cell(model, tok, blocks, nL, hidden, cell, base_cfg):
    name = cell["name"]
    out = os.path.join(OUTBASE, name)
    os.makedirs(out, exist_ok=True)
    words = cell["words"].split(",")
    P, T = cell["npairs"], cell["tgen"]
    ctx = {"A": cell["ctx_a"], "B": cell["ctx_b"]}
    gA, adjA = make_graph(cell["ga"], words)
    gB, adjB = make_graph(cell["gb"], words)
    graphs = {"A": (gA, adjA), "B": (gB, adjB)}
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    cand_t = torch.tensor(cand, device=DEVICE)
    rng = np.random.default_rng(cell["seed"] + 1)

    mode = cell.get("mode", "gen")               # "gen" | "real" (ground-truth interleave)
    q_ab = cell.get("qcomm_ab", cell["qcomm"])   # delivery prob A -> B
    q_ba = cell.get("qcomm_ba", cell["qcomm"])   # delivery prob B -> A
    walks, rows, state, seq, prev = {}, {}, {}, {}, {}
    for si, m in (("A", 0), ("B", 500)):
        g, _ = graphs[si]
        wl = ctx[si] + (T // 2 if mode == "real" else 0)
        cfgw = replace(base_cfg, n_walks=P, walk_length=wl, seed=cell["seed"] + m)
        walks[si] = G.generate_walks(g, cfgw)
        rows[si] = [[bos] + [cand[x] for x in wk.nodes[:ctx[si]]] for wk in walks[si]]
        ids = torch.tensor(rows[si], device=DEVICE)
        try:
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=ids, use_cache=True)
        state[si] = {"past": o.past_key_values, "logits": o.logits[:, -1, :]}
        seq[si] = np.zeros((P, T), np.int32)     # what THIS context appended
        prev[si] = np.array([wk.nodes[-1] for wk in walks[si]])

    js_series = np.zeros(T)
    mass_series = {(si, g): np.zeros(T) for si in ("A", "B") for g in ("A", "B")}
    gen_nodes = np.zeros((P, T), np.int32)
    gen_side = []
    if mode == "real":                       # no generation: interleave true walk steps
        for si in ("A", "B"):
            del state[si]
        for p in range(P):
            for t in range(T):
                src = "B" if t % 2 == 0 else "A"
                node = walks[src][p].nodes[ctx[src] + t // 2]
                gen_nodes[p, t] = node
                seq["A"][p, t] = node
                seq["B"][p, t] = node
        gen_side = ["B" if t % 2 == 0 else "A" for t in range(T)]
    else:
      for t in range(T):
        if cell.get("p_emit_a") is not None:      # random speaker schedule (no turn-taking)
            who = "A" if rng.random() < cell["p_emit_a"] else "B"
        elif cell.get("block_len"):                # block schedule: B first, swap each block
            who = "B" if (t // cell["block_len"]) % 2 == 0 else "A"
        else:
            who = "B" if t % 2 == 0 else "A"
        oth = "A" if who == "B" else "B"
        probs = {}
        for si in ("A", "B"):
            lg = state[si]["logits"][:, cand_t].float()
            probs[si] = torch.softmax(lg / cell["temp"], -1).cpu().numpy()
        js_series[t] = np.mean([js(probs["A"][p], probs["B"][p]) for p in range(P)])
        for si in ("A", "B"):
            pvs = [int(seq["A"][p, t - 1]) if t > 0 else int(prev["A"][p])
                   for p in range(P)]
            mass_series[(si, "A")][t] = np.mean(
                [probs[si][p][graphs["A"][1][pvs[p]]].sum() for p in range(P)])
            mass_series[(si, "B")][t] = np.mean(
                [probs[si][p][graphs["B"][1][pvs[p]]].sum() for p in range(P)])

        def draw(si, p):
            pp = probs[si][p].copy()
            k = cell.get("topk_a" if si == "A" else "topk_b", cell["topk"])
            if k > 0:
                pp[np.argsort(pp)[:-k]] = 0.0
            return int(rng.choice(N, p=pp / pp.sum()))

        q_del = q_ab if who == "A" else q_ba
        for p in range(P):
            gtok = draw(who, p)
            gen_nodes[p, t] = gtok
            seq[who][p, t] = gtok
            if rng.random() < q_del:
                seq[oth][p, t] = gtok               # delivered
            else:
                seq[oth][p, t] = draw(oth, p)       # dropped: partner appends own sample
        gen_side.append(who)
        for si in ("A", "B"):
            toks = torch.tensor([cand[seq[si][p, t]] for p in range(P)],
                                device=DEVICE)[:, None]
            o = model(input_ids=toks, past_key_values=state[si]["past"], use_cache=True)
            state[si]["past"] = o.past_key_values
            state[si]["logits"] = o.logits[:, -1, :]
    state.clear()
    torch.cuda.empty_cache()

    # capture
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o2): grabbed[L] = (o2[0] if isinstance(o2, tuple) else o2).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    wins = {}
    for si in ("A", "B"):
        b0 = max(0, ctx[si] - 200)
        wins[si] = {"base": ("prefix", b0, ctx[si]),
                    "joint_early": ("joint", 0, T // 4),
                    "joint_late": ("joint", T // 2, T)}
    nsum = {(si, w): {L: np.zeros((N, hidden)) for L in range(nL)}
            for si in ("A", "B") for w in wins[si]}
    ncnt = {(si, w): np.zeros(N) for si in ("A", "B") for w in wins[si]}
    try:
        for si in ("A", "B"):
            for p in range(P):
                full = rows[si][p] + [cand[x] for x in seq[si][p]]
                nodes_all = walks[si][p].nodes + list(seq[si][p])
                fids = torch.tensor([full], device=DEVICE)
                grabbed.clear()
                try:
                    model(input_ids=fids, logits_to_keep=1)
                except TypeError:
                    model(input_ids=fids)
                for w, (segm, lo, hi) in wins[si].items():
                    off = 0 if segm == "prefix" else ctx[si]
                    pos = list(range(1 + off + lo, 1 + off + hi))
                    nds = nodes_all[off + lo:off + hi]
                    for L in range(nL):
                        rr = grabbed[L][0][pos].float().cpu().numpy()
                        np.add.at(nsum[(si, w)][L], nds, rr)
                    np.add.at(ncnt[(si, w)], nds, 1.0)
    finally:
        for h in handles:
            h.remove()

    save = {"n_layers": np.array([nL]), "words": np.array(words),
            "adjacency_A": adjA.astype(np.int8), "adjacency_B": adjB.astype(np.int8),
            "coords_A": np.array(gA.coords), "coords_B": np.array(gB.coords)}
    for (si, w), per in nsum.items():
        cnt = np.maximum(ncnt[(si, w)], 1.0)
        for L in range(nL):
            save[f"{si}_{w}_layer_{L}"] = (per[L] / cnt[:, None]).astype(np.float16)
    np.savez_compressed(os.path.join(out, "nodemeans.npz"), **save)
    json.dump({"cell": cell, "js": np.round(js_series, 4).tolist(),
               "mass": {f"{si}_on_{g}": np.round(v, 4).tolist()
                        for (si, g), v in mass_series.items()},
               "gen_side": gen_side, "gen_nodes": gen_nodes.tolist(),
               "seq_A": seq["A"].tolist(), "seq_B": seq["B"].tolist(),
               "prefix_last": {si: prev[si].tolist() for si in ("A", "B")}},
              open(os.path.join(out, "metrics.json"), "w"))
    print(f"CELL DONE {name}", flush=True)


@torch.no_grad()
def main():
    base_cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, base_cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    cm = model.config
    blocks = M._decoder_blocks(model)
    t0 = time.time()
    for cell in SPEC["cells"]:
        if ONLY and cell["name"] not in ONLY:
            continue
        run_cell(model, tok, blocks, cm.num_hidden_layers, cm.hidden_size, cell, base_cfg)
        print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
