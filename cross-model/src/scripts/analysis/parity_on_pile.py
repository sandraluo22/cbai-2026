"""What does the grid PARITY direction correspond to in natural text? Derive the parity axis in L14H26's
head-output space from the in-context grid (proto_delta = mean head output of parity-+ nodes minus parity--
nodes), then run Pile text through the model and project L14H26's output at every token onto that unit axis.
Report the max-activating (and min-activating) natural-language contexts, plus how their firing compares to
the grid parity-node reference level. Tells us whether the grid-derived parity feature is a real reusable
feature (fires on some coherent natural pattern) or grid-task-specific (fires on nothing coherent).

v2 additions:
  - DAS directions: if DAS_NPZ is set (default: the das_grid_patch interchange-trained subspace), also project
    onto the learned rank-1 DAS direction ("das1", sign-aligned to proto) and the rank-4 subspace norm
    ("das4norm"), and report proto-vs-das1 agreement (cosine, token-level r, top-K overlap).
  - Nulls: (a) NRAND random unit directions in head-output space; (b) NPART random BALANCED node partitions of
    the same grid znode (axis = mean diff of a random 8/8 colouring — matched null for "a mean-diff axis from
    this head fires somewhere on the Pile"). For each null family we record the distribution of global maxes
    and of the word_initial-minus-continuation category contrast, and where the real axes rank.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NDOCS(300) MAXTOK(512) TOPK(40) WIN(14)
     DATASET(NeelNanda/pile-10k) DAS_NPZ(runs/axes/4_circuits/das/das_grid_patch_<model>_L<l>H<h>.npz)
     NRAND(64) NPART(32) OUTTAG("") OUTDIR DEVICE
Out: <OUTDIR>/parity_on_pile<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, heapq
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
NDOCS = int(os.environ.get("NDOCS", "300")); MAXTOK = int(os.environ.get("MAXTOK", "512"))
TOPK = int(os.environ.get("TOPK", "40")); WIN = int(os.environ.get("WIN", "14"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k"); CTXLO = int(os.environ.get("CTXLO", "100"))
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L{HEAD_LAYER}H{HEAD_IDX}.npz")
NRAND = int(os.environ.get("NRAND", "64")); NPART = int(os.environ.get("NPART", "32"))
OUTTAG = os.environ.get("OUTTAG", "")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


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


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4, n_walks=24, walk_length=260, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    model, tok = lw(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)

    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hk = proj.register_forward_pre_hook(cap)

    # ---- parity axis from grid: per-node L14H26 output mean, proto_delta = mean(+) - mean(-) ----
    walks = G.generate_walks(graph, cfg)
    zsum = np.zeros((n, hd)); zcnt = np.zeros(n)
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
        model(input_ids=ids)
        for s in range(len(nodes) - 1):
            if cl[s] < CTXLO: continue
            zsum[nodes[s]] += zc["z"][0, spans[s][-1], csl].float().cpu().numpy(); zcnt[nodes[s]] += 1
    znode = zsum / np.maximum(zcnt, 1)[:, None]
    pdir = znode[col > 0].mean(0) - znode[col < 0].mean(0)
    pdir = pdir / (np.linalg.norm(pdir) + 1e-12)
    # grid reference: projection of each node's head output onto the axis (per colour)
    gproj = znode @ pdir
    ref = {"grid_pos_mean": float(gproj[col > 0].mean()), "grid_neg_mean": float(gproj[col < 0].mean()),
           "grid_abs_max": float(np.abs(gproj).max())}
    print(f"[{tag}] parity axis ready. grid proj +{ref['grid_pos_mean']:.2f}/{ref['grid_neg_mean']:.2f} (sep={ref['grid_pos_mean']-ref['grid_neg_mean']:.2f})", flush=True)

    # ---- direction bank: real axes + matched nulls ----
    rng = np.random.default_rng(0)
    real_names = ["proto"]; real_dirs = [pdir]
    das_meta = None; R4 = None
    if DAS_NPZ and os.path.exists(DAS_NPZ):
        dz = np.load(DAS_NPZ)
        k1 = "global_R1" if "global_R1" in dz.files else ("R_1" if "R_1" in dz.files else None)
        k4 = "global_R4" if "global_R4" in dz.files else ("R_4" if "R_4" in dz.files else None)
        if k1:
            d1 = dz[k1][0].astype(np.float64); d1 = d1 / (np.linalg.norm(d1) + 1e-12)
            c = float(d1 @ pdir); das_meta = {"npz": DAS_NPZ, "key": k1, "cos_das1_proto": round(abs(c), 3), "sign_flipped": c < 0}
            if c < 0: d1 = -d1
            real_names.append("das1"); real_dirs.append(d1)
        if k4: R4 = dz[k4].astype(np.float32)
        print(f"[{tag}] DAS npz loaded ({k1}/{k4}), |cos(das1,proto)|={das_meta['cos_das1_proto'] if das_meta else 'n/a'}", flush=True)
    elif DAS_NPZ:
        print(f"[{tag}] WARNING: DAS_NPZ not found: {DAS_NPZ} — running proto only", flush=True)
    n_real = len(real_names)
    pos_set = frozenset(np.where(col > 0)[0].tolist())
    part_axes = []
    while len(part_axes) < NPART:                                  # matched null: random balanced 2-colouring
        s = frozenset(rng.choice(n, n // 2, replace=False).tolist())
        if s == pos_set or (frozenset(range(n)) - s) == pos_set: continue
        a = znode[list(s)].mean(0) - znode[[i for i in range(n) if i not in s]].mean(0)
        nm = np.linalg.norm(a)
        if nm < 1e-9: continue
        part_axes.append(a / nm)
    rand_axes = rng.standard_normal((NRAND, hd)); rand_axes /= np.linalg.norm(rand_axes, axis=1, keepdims=True)
    W = np.concatenate([np.stack(real_dirs), np.stack(part_axes), rand_axes]).astype(np.float32)   # [nw, hd]
    W_t = torch.tensor(W, device=dev); nw = W.shape[0]
    R4_t = torch.tensor(R4, device=dev) if R4 is not None else None
    null_slices = {"partition": slice(n_real, n_real + NPART), "random": slice(n_real + NPART, nw)}

    # ---- stream Pile, project L14H26 output at every token onto the whole bank ----
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)
    cnt = 0; push = 0
    real_vals = {m: [] for m in real_names + (["das4norm"] if R4 is not None else [])}
    heaps = {m: ([], []) for m in real_vals}                       # (top_pos, top_neg); das4norm uses pos only
    import string as _string
    catsum = {m: {"word_initial": [0.0, 0], "continuation": [0.0, 0], "punct": [0.0, 0], "digit": [0.0, 0]} for m in real_vals}
    null_max = np.full(nw - n_real, -np.inf)                       # per-null-direction global max of |proj|
    null_wi = np.zeros(nw - n_real); null_wi_n = 0                 # word_initial sums (counts shared)
    null_cont = np.zeros(nw - n_real); null_cont_n = 0
    def categorize(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"
    def add(heap, key, ctx):
        nonlocal push
        push += 1
        if len(heap) < TOPK: heapq.heappush(heap, (key, push, ctx))
        elif key > heap[0][0]: heapq.heapreplace(heap, (key, push, ctx))
    def ctx_str(ids_list, t):
        a = max(0, t - WIN)
        pre = tok.decode(ids_list[a:t]); mid = tok.decode(ids_list[t:t + 1]); post = tok.decode(ids_list[t + 1:t + 4])
        return (pre[-90:] + "⟦" + mid + "⟧" + post).replace("\n", "⏎")
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        idl = ids[0].tolist(); zc.clear(); model(input_ids=ids)
        Z = zc["z"][0, :, csl].float()
        P = (Z @ W_t.t()).cpu().numpy(); P[:3, :] = 0.0            # ignore BOS/first tokens
        cols = {m: P[:, i] for i, m in enumerate(real_names)}
        if R4_t is not None: cols["das4norm"] = torch.linalg.vector_norm(Z @ R4_t.t(), dim=1).cpu().numpy(); cols["das4norm"][:3] = 0.0
        pieces = tok.convert_ids_to_tokens(idl)
        cats = [categorize(pieces[ti]) for ti in range(len(idl))]
        for m, pj in cols.items():
            real_vals[m].append(pj[3:].astype("float32"))
            for ti in range(3, len(idl)):
                catsum[m][cats[ti]][0] += float(pj[ti]); catsum[m][cats[ti]][1] += 1
            for t in np.argsort(pj)[::-1][:8]:
                add(heaps[m][0], float(pj[t]), {"proj": round(float(pj[t]), 3), "ctx": ctx_str(idl, int(t)), "doc": cnt, "t": int(t)})
            if m != "das4norm":
                for t in np.argsort(pj)[:8]:
                    add(heaps[m][1], -float(pj[t]), {"proj": round(float(pj[t]), 3), "ctx": ctx_str(idl, int(t)), "doc": cnt, "t": int(t)})
        Nl = np.abs(P[3:, n_real:])
        if Nl.size: null_max = np.maximum(null_max, Nl.max(0))
        wi_mask = np.array([c == "word_initial" for c in cats[3:]]); cont_mask = np.array([c == "continuation" for c in cats[3:]])
        null_wi += P[3:, n_real:][wi_mask].sum(0); null_wi_n += int(wi_mask.sum())
        null_cont += P[3:, n_real:][cont_mask].sum(0); null_cont_n += int(cont_mask.sum())
        cnt += 1
        if cnt % 50 == 0: print(f"[{tag}] {cnt}/{NDOCS} docs", flush=True)
    hk.remove()

    def dir_stats(m):
        allv = np.concatenate(real_vals[m])
        s = {"n_tokens": int(allv.size), "mean": float(allv.mean()), "std": float(allv.std()),
             "p99": float(np.percentile(allv, 99)), "p1": float(np.percentile(allv, 1)),
             "max": float(allv.max()), "min": float(allv.min())}
        if m == "proto":
            s["pile_max_vs_grid_sep"] = round(float(allv.max()) / (ref["grid_pos_mean"] - ref["grid_neg_mean"] + 1e-9), 3)
        return s
    stats = {m: dir_stats(m) for m in real_vals}
    catmean = {m: {k: {"mean_proj": round(v[0] / max(v[1], 1), 4), "n": v[1]} for k, v in catsum[m].items()} for m in real_vals}

    # null summaries: where does each real axis's global |max| and word-boundary contrast rank vs the nulls?
    null_contrast = null_wi / max(null_wi_n, 1) - null_cont / max(null_cont_n, 1)
    def null_summary(fam):
        sl = null_slices[fam]; nm = null_max[sl.start - n_real: sl.stop - n_real]; nc = null_contrast[sl.start - n_real: sl.stop - n_real]
        out = {"n_dirs": int(sl.stop - sl.start),
               "max_mean": round(float(nm.mean()), 3), "max_std": round(float(nm.std()), 3), "max_p95": round(float(np.percentile(nm, 95)), 3),
               "contrast_absmean": round(float(np.abs(nc).mean()), 4), "contrast_absp95": round(float(np.percentile(np.abs(nc), 95)), 4)}
        for m in real_names:
            rmax = max(abs(stats[m]["max"]), abs(stats[m]["min"]))
            rcon = catmean[m]["word_initial"]["mean_proj"] - catmean[m]["continuation"]["mean_proj"]
            out[m] = {"real_absmax": round(rmax, 3), "max_z": round((rmax - nm.mean()) / (nm.std() + 1e-9), 2),
                      "max_rank": int((nm >= rmax).sum()),
                      "real_contrast": round(rcon, 4), "contrast_z": round((abs(rcon) - np.abs(nc).mean()) / (np.abs(nc).std() + 1e-9), 2),
                      "contrast_rank": int((np.abs(nc) >= abs(rcon)).sum())}
        return out
    nulls = {fam: null_summary(fam) for fam in null_slices}

    agreement = None
    if "das1" in real_vals:
        a = np.concatenate(real_vals["proto"]); b = np.concatenate(real_vals["das1"])
        keyset = lambda m, hi: {(c["doc"], c["t"]) for _, _, c in heaps[m][hi]}
        agreement = dict(das_meta, token_pearson_r=round(float(np.corrcoef(a, b)[0, 1]), 3),
                         topk_pos_overlap=round(len(keyset("proto", 0) & keyset("das1", 0)) / max(len(heaps["proto"][0]), 1), 3),
                         topk_neg_overlap=round(len(keyset("proto", 1) & keyset("das1", 1)) / max(len(heaps["proto"][1]), 1), 3))

    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "dataset": DATASET, "ndocs": cnt, "grid_ref": ref,
           "pile_stats": stats, "by_token_category": catmean, "das_agreement": agreement, "null_summary": nulls,
           "top_positive": {m: [c for _, _, c in sorted(heaps[m][0], reverse=True)] for m in real_vals},
           "top_negative": {m: [c for _, _, c in sorted(heaps[m][1], reverse=True)] for m in real_vals if m != "das4norm"}}
    p = f"{OUTDIR}/parity_on_pile{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)
    for m in real_vals:
        s = stats[m]
        print(f"[{m}] pile p99={s['p99']:.2f} max={s['max']:.2f}", flush=True)
        print(f"  BY CATEGORY: " + "  ".join(f"{k}={v['mean_proj']:+.3f}" for k, v in sorted(catmean[m].items(), key=lambda kv: kv[1]['mean_proj'])), flush=True)
        for c in out["top_positive"][m][:8]: print(f"  +{c['proj']:.2f}  {c['ctx']}", flush=True)
    if agreement: print("DAS-vs-proto agreement:", json.dumps(agreement), flush=True)
    print("NULLS:", json.dumps(nulls, indent=1), flush=True)


if __name__ == "__main__":
    main()
