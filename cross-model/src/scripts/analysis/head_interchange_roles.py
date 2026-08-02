"""Per-head INTERCHANGE PATCHING (not ablation): which heads actually CARRY the variable?

Ablation asks "is this head necessary" (it breaks things, but redundancy hides real carriers). Patching
asks the sharper question: if we transplant this head's output from a counterfactual run in which the
variable has the OPPOSITE value, does the model's behaviour follow? That is sufficiency + specificity.

Construction: one walk, two token sequences. The clean sequence emits word[X_t] at step t. The
counterfactual emits word[pi(X_t)], i.e. the same node sequence relabelled by a graph automorphism —
rot90 inverts every node's parity, rot180 reverses both coordinates (parity preserved). All words are
single-token, so the two sequences align position-for-position. For each head we splice its output from
the counterfactual run into the clean run at every node position and read the behavioural margin.

  PERM=rot90   margin = logmass(opposite colour class) - logmass(same)      -> parity carriers
  PERM=rot180  margin = logmass(nbrs of pi(X)) - logmass(nbrs of X)         -> coordinate carriers

Reported per head, alongside the ABLATION score (mean-ablate the same head, same metric) so carrying and
necessity can be compared directly. SHUFFLE=1 repeats everything on order-shuffled walks (same token
multiset, no graph structure) as the control for "is this about the task at all".

Env: GEN_MODEL(Llama) PERM(rot90) K(4) NWALKS(3) WLEN(1200) CTXLO(800) SHUFFLE(0)
     LAYERS(all) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/head_interchange_roles_<PERM><OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
from graph import Walk
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
PERM = os.environ.get("PERM", "rot90")
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
SHUFFLE = os.environ.get("SHUFFLE", "0") == "1"; SEED = int(os.environ.get("SEED", "0"))
LAZY = float(os.environ.get("LAZY", "0"))   # self-loop probability: destroys parity persistence, leaves coords
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def build_perm(coords, k, col):
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    if PERM == "rot90":
        pi = np.array([idx[(int(c), int(k - 1 - r))] for (r, c) in coords], int)
        assert all(col[pi[i]] == -col[i] for i in range(len(pi))), "rot90 must invert parity"
    elif PERM == "rot180":
        pi = np.array([idx[(int(k - 1 - r), int(k - 1 - c))] for (r, c) in coords], int)
        assert all(col[pi[i]] == col[i] for i in range(len(pi))), "rot180 must preserve parity"
    else:
        raise ValueError(PERM)
    return pi


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH
    rng = np.random.default_rng(SEED)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, int)
    pi = build_perm(coords, K, col)
    words = list(graph.words)
    for w in words:
        assert len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1, f"word {w!r} is not single-token"
    cf_words = [words[pi[i]] for i in range(n)]
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev); neg_i = torch.tensor(np.where(col < 0)[0], device=dev)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    tgt_src = []
    for u in range(n):
        if pi[u] == u: tgt_src.append(None); continue
        T = sorted(nbr[pi[u]] - nbr[u]); S = sorted(nbr[u] - nbr[pi[u]])
        tgt_src.append(None if (not T or not S) else
                       (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))

    # Build token ids DIRECTLY from the single-token words instead of tokenizing the joined text: BPE can
    # merge differently across space boundaries for different word sequences, which would misalign the
    # clean and counterfactual runs. With ids = [BOS] + one token per step, step s sits at position s+1.
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

    walks = G.generate_walks(graph, cfg)
    if LAZY > 0:
        lr = np.random.default_rng(SEED); lazyw = []
        for w in range(NWALKS):
            cur = w % n; nodes = [cur]
            for _ in range(WLEN - 1):
                if lr.random() >= LAZY: cur = int(lr.choice(graph.neighbors(cur)))
                nodes.append(cur)
            lazyw.append(Walk(walk_id=w, nodes=nodes, words=[words[x] for x in nodes]))
        walks = lazyw
    if SHUFFLE:
        sw = []
        for wk in walks:
            nodes = list(wk.nodes); rng.shuffle(nodes)
            sw.append(Walk(walk_id=wk.walk_id, nodes=nodes, words=[words[x] for x in nodes]))
        walks = sw

    caps = {}
    hooks = []
    for l in range(nL):
        proj = attn_proj(blocks[l], cm)[0]
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mk(l)))

    mode = {"l": None, "h": None, "src": None, "kind": None}
    def patcher(l):
        def hh(_m, args):
            if mode["l"] != l: return
            x = args[0].clone(); s = slice(mode["h"] * hd, (mode["h"] + 1) * hd)
            if mode["kind"] == "patch": x[0, :, s] = mode["src"][0, :, s]
            else: x[0, :, s] = x[0, :, s].mean(0, keepdim=True)
            return (x,) + tuple(args[1:])
        return hh
    for l in range(nL):
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(patcher(l)))

    def margin(ids, readpos, readnode):
        lg = model(input_ids=ids).logits[0][torch.tensor(readpos, device=dev)][:, cand_t].float()
        lsm = torch.log_softmax(lg, 1)
        if PERM == "rot90":
            vals = []
            for j, nd in enumerate(readnode):
                same = lsm[j, pos_i] if col[nd] > 0 else lsm[j, neg_i]
                opp = lsm[j, neg_i] if col[nd] > 0 else lsm[j, pos_i]
                vals.append(float(torch.logsumexp(opp, 0) - torch.logsumexp(same, 0)))
            return float(np.mean(vals))
        vals = []
        for j, nd in enumerate(readnode):
            ts = tgt_src[nd]
            if ts is None: continue
            T, S = ts
            vals.append(float(torch.logsumexp(lsm[j, T], 0) - torch.logsumexp(lsm[j, S], 0)))
        return float(np.mean(vals)) if vals else 0.0

    patch_score = np.zeros((nL, nH)); abl_score = np.zeros((nL, nH)); base_all = []
    for wi, wk in enumerate(walks):
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        ids_cf = torch.tensor([[bos] + [wid[pi[x]] for x in wk.nodes]], device=dev)
        assert ids.shape == ids_cf.shape, "clean/cf sequences must align"
        readpos = [s + 1 for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        readnode = [wk.nodes[s] for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if not readpos: continue
        mode["l"] = None; caps.clear(); model(input_ids=ids_cf)
        src = {l: caps[l].clone() for l in range(nL)}
        mode["l"] = None
        base = margin(ids, readpos, readnode); base_all.append(base)
        print(f"[walk {wi}] baseline margin = {base:+.3f}", flush=True)
        for l in range(nL):
            for h in range(nH):
                mode["l"] = l; mode["h"] = h; mode["src"] = src[l]; mode["kind"] = "patch"
                patch_score[l, h] += margin(ids, readpos, readnode) - base
                mode["kind"] = "ablate"
                abl_score[l, h] += margin(ids, readpos, readnode) - base
                mode["l"] = None
            print(f"  layer {l} done", flush=True)
        del src; gc.collect(); torch.cuda.empty_cache()
    for h in hooks: h.remove()
    nw = max(len(base_all), 1)
    patch_score /= nw; abl_score /= nw

    names = [f"L{l}H{h}" for l in range(nL) for h in range(nH)]
    ps, as_ = patch_score.flatten(), abl_score.flatten()
    order = np.argsort(ps)[::-1]
    out = {"model": tag, "perm": PERM, "shuffle": SHUFFLE, "lazy": LAZY, "k": K, "ctxlo": CTXLO, "wlen": WLEN,
           "baseline_margin": round(float(np.mean(base_all)), 4),
           "patch_sd_over_heads": round(float(ps.std()), 4),
           "top_carriers_by_patch": [{"head": names[i], "patch": round(float(ps[i]), 4),
                                      "ablate": round(float(as_[i]), 4),
                                      "z": round(float((ps[i] - ps.mean()) / (ps.std() + 1e-9)), 2)}
                                     for i in order[:25]],
           "top_by_ablation": [{"head": names[i], "ablate": round(float(as_[i]), 4),
                                "patch": round(float(ps[i]), 4)}
                               for i in np.argsort(np.abs(as_))[::-1][:25]],
           "corr_patch_vs_ablate": round(float(np.corrcoef(ps, np.abs(as_))[0, 1]), 3)}
    p = f"{OUTDIR}/head_interchange_roles_{PERM}{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    np.savez_compressed(p.replace(".json", ".npz"), patch=patch_score, ablate=abl_score,
                        names=np.array(names))
    print(f"\nbaseline margin {out['baseline_margin']:+.3f}   (patch sd over heads {out['patch_sd_over_heads']:.4f})")
    print("TOP CARRIERS (interchange patch moves the margin toward the counterfactual):")
    for d in out["top_carriers_by_patch"][:12]:
        print(f"   {d['head']:8} patch={d['patch']:+.3f} (z={d['z']:+.1f})  ablate={d['ablate']:+.3f}")
    print(f"corr(patch, |ablate|) = {out['corr_patch_vs_ablate']}")
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
