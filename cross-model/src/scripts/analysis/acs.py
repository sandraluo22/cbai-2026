"""Steer each graph CUT (x, y, diagonal, anti-diagonal, parity) and measure how much the predicted
next-node mass moves onto that cut's + side -- for every model. Generalises the parity dose sweep.

For each cut C we fit the per-layer readout v_L = normalise(Hc_L^T u_C), steer +dose*std along it at
all layers, and (teacher-forced, next-token) record: mass on C's + side, mass on C's - side, true-
neighbour mass, validity. A cut is 'steerable' if pushing it drains mass onto its + side. Comparing
cuts across models tests whether a model carves the grid axis-aligned (x/y strong) or on the diagonal
(Gemma: diag/anti-diag expected stronger, x/y weaker -- its coordinate frame is rotated ~45deg).

Next-token only (fast: cuts x doses x walks forwards). Env: PRESET GEN_MODEL GRAPH(square_grid)
XCTX(150) NWALK_TF(12) CTXLO(100) DOSES(0.5,1,2,4) OUTDIR DEVICE
Out: <OUTDIR>/acs_<model>_<graph>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "distilgpt2": ("distilgpt2", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
XCTX = int(os.environ.get("XCTX", "150")); NWALK_TF = int(os.environ.get("NWALK_TF", "12"))
CTXLO = int(os.environ.get("CTXLO", "100")); TEMP = float(os.environ.get("TEMP", "1.0"))
DOSES = [float(x) for x in os.environ.get("DOSES", "0.5,1,2,4").split(",")]
OUTDIR = os.environ.get("OUTDIR", "runs/axes/3_causal/axis_cut_sweep")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    n = graph.n_nodes; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(float)


def unit(v): v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)


def make_cuts(coords, graph, split="median", K=6):
    x, y = coords[:, 0], coords[:, 1]
    raw = {"x": x, "y": y, "diagonal": x + y, "anti-diagonal": x - y}
    par = two_colour(graph)
    if not np.allclose(par, par[0]): raw["parity"] = par
    cuts = {}
    for k, r in raw.items():
        u = unit(r)
        if split == "matched":
            # percentile-matched: equal-size extremal +/- groups (top-K vs bottom-K, drop middle),
            # so |+side| is identical across cuts and |Δ mass| is comparable.
            order = np.argsort(r)
            minus = order[:K].tolist(); plus = order[-K:].tolist()
        else:
            med = np.median(r)
            plus = np.where(r > med)[0].tolist(); minus = np.where(r <= med)[0].tolist()
        cuts[k] = {"u": u, "plus": plus, "minus": minus}
    return cuts


@torch.no_grad()
def clean_node_means(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers; hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def steer_vecs(means, u, dev, nL):
    out = {}
    for L in range(nL):
        H = means[L]; Hc = H - H.mean(0); v = Hc.T @ u; v = v / (np.linalg.norm(v) + 1e-9)
        std = float((Hc @ v).std()); out[L] = torch.tensor(std * v, device=dev, dtype=torch.float32)
    return out


def steer_hooks(blocks, layers_vec):
    handles = []
    for L, vec in layers_vec.items():
        def hk(_m, _i, out, vec=vec):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vec.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handles.append(blocks[L].register_forward_hook(hk))
    return handles


@torch.no_grad()
def measure(model, tok, blocks, cm, graph, cand_t, dev, walks, layers_vec, plus, minus):
    handles = steer_hooks(blocks, layers_vec) if layers_vec else []
    mp = mm = nbr = 0.0; val = cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                nb = graph.neighbors(nodes[s])
                mp += float(p[plus].sum()); mm += float(p[minus].sum()); nbr += float(p[nb].sum())
                val += int(int(p.argmax()) in nb); cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return {"mass_plus": mp / c, "mass_minus": mm / c, "nbr": nbr / c, "val": val / c}


@torch.no_grad()
def measure_splits(model, tok, blocks, cm, graph, cand_t, dev, walks, layers_vec, splits):
    """mass on the + side of MANY splits in one steered pass (for the random baseline)."""
    handles = steer_hooks(blocks, layers_vec) if layers_vec else []
    mp = [0.0] * len(splits); cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                for j, (plus, _m) in enumerate(splits): mp[j] += float(p[plus].sum())
                cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return [x / c for x in mp]


def random_steer_vecs(means, dev, nL, rng):
    """random hidden direction per layer, scaled by the same per-layer std convention as steer_vecs."""
    out = {}
    for L in range(nL):
        Hc = means[L] - means[L].mean(0)
        v = rng.standard_normal(Hc.shape[1]); v = v / (np.linalg.norm(v) + 1e-9)
        std = float((Hc @ v).std()); out[L] = torch.tensor(std * v, device=dev, dtype=torch.float32)
    return out


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NWALK_TF, 24), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    SPLIT = os.environ.get("SPLIT", "median"); MATCHK = int(os.environ.get("MATCHK", "6"))
    cuts = make_cuts(coords, graph, split=SPLIT, K=MATCHK)
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    walks = G.generate_walks(graph, cfg)[:NWALK_TF]
    means = clean_node_means(model, tok, blocks, cm, walks, dev, n)

    out = {"graph": GRAPH, "model": tag, "doses": DOSES, "cuts": {}}
    clean = measure(model, tok, blocks, cm, graph, cand_t, dev, walks, None,
                    np.arange(n)[:1], np.arange(n)[:1])  # placeholder, recomputed per cut below
    for cname, cd in cuts.items():
        sv = steer_vecs(means, cd["u"], dev, nL)
        base = measure(model, tok, blocks, cm, graph, cand_t, dev, walks, None, cd["plus"], cd["minus"])
        rows = {"clean": base}
        for d in DOSES:
            lv = {L: d * sv[L] for L in range(nL)}
            rows[f"{d:g}"] = measure(model, tok, blocks, cm, graph, cand_t, dev, walks, lv, cd["plus"], cd["minus"])
        out["cuts"][cname] = {"plus": cd["plus"], "minus": cd["minus"], "sweep": rows}
        ctl = rows[f"{DOSES[-1]:g}"]["mass_plus"] - base["mass_plus"]
        print(f"[{tag}/{cname:13s}] +side mass clean {base['mass_plus']:.2f} -> dose{DOSES[-1]:g} "
              f"{rows[f'{DOSES[-1]:g}']['mass_plus']:.2f}  (control {ctl:+.2f}) | val {base['val']:.2f}->{rows[f'{DOSES[-1]:g}']['val']:.2f}", flush=True)

    # ---- random-vector baseline: steer a random hidden direction (matched magnitude), measure the
    # spurious |Δ mass on + side| averaged over all cuts' +/- splits and NRAND draws ----
    NRAND = int(os.environ.get("NRAND", "4")); RSEED = int(os.environ.get("RSEED", "0"))
    rng = np.random.default_rng(RSEED)
    splits = [(np.array(cuts[c]["plus"]), np.array(cuts[c]["minus"])) for c in cuts]
    base_plus = [out["cuts"][c]["sweep"]["clean"]["mass_plus"] for c in cuts]
    rand = {"clean": {"abs_dmass_mean": 0.0, "abs_dmass_std": 0.0}}
    for d in DOSES:
        accum = []
        for _ in range(NRAND):
            rv = random_steer_vecs(means, dev, nL, rng)
            lv = {L: d * rv[L] for L in range(nL)}
            mps = measure_splits(model, tok, blocks, cm, graph, cand_t, dev, walks, lv, splits)
            accum.append([abs(mps[j] - base_plus[j]) for j in range(len(splits))])
        a = np.array(accum)
        rand[f"{d:g}"] = {"abs_dmass_mean": float(a.mean()), "abs_dmass_std": float(a.std())}
        print(f"[{tag}/random       ] dose{d:g} |Δ|mass on + side = {a.mean():.3f} ± {a.std():.3f}", flush=True)
    out["random"] = rand

    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/acs_{tag}_{GS}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/acs_{tag}_{GS}.json", flush=True)


if __name__ == "__main__":
    main()
