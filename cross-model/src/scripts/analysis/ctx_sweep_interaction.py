"""Does the superadditive interaction emerge at a stage, or only once accuracy nears ceiling?

The threat this addresses: keep-set recovery is measured against a full-vs-floor range, and near ceiling
that range compresses. An "interaction" that appears only where accuracy is already ~1.0 is more likely
metric nonlinearity than a mechanism switching on. A real mechanism should emerge at a specific point in
the in-context phase transition, alongside the head attention and the geometry.

At each context length we measure, on the SAME forward passes:
    acc              neighbour validity (the phase transition itself)
    interaction      recovery(all21) - [recovery(induction12) + recovery(duplicate9)]
                     on BOTH the parity margin and neighbour validity, since the two saturate differently
    ind_attn         induction-motif attention mass, on-task (successor of a previous visit)
    dup_attn         same-token attention mass, on-task (the previous visit itself)
    dirichlet        graph smoothness of the node-mean representation,
                     sum_{(u,v) in E} ||x_u - x_v||^2 / (deg-normalised) / sum_u ||x_u||^2
                     LOW = smooth over the graph = geometry present. Fit-free, no PCA, no probe.

If interaction, attention motifs and Dirichlet energy all move together at one context length, that is a
mechanism. If interaction only departs from zero where acc is already >0.95, suspect the metric.

Env: GEN_MODEL(Llama) K(4) CTXS("50,100,200,400,800,1150") LAYER(14) NWALKS(3) WLEN(1200)
     NRAND(2) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/ctx_sweep_interaction<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4"))
CTXS = [int(x) for x in os.environ.get("CTXS", "50,100,200,400,800,1150").split(",")]
LAYER = int(os.environ.get("LAYER", "14"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
NRAND = int(os.environ.get("NRAND", "2")); SEED = int(os.environ.get("SEED", "0"))
# WINDOW: readouts are taken from [ctx, ctx+WINDOW). This MUST be narrow or the "context length" label is
# meaningless — with the original 150 the ctx=3 row averaged contexts 3..153 and read 0.89 accuracy, which
# is impossible from 3 tokens (chance neighbour validity on a 4x4 grid is ~0.156). Narrow window + more
# walks keeps the sample count up without smearing the phase transition.
WINDOW = int(os.environ.get("WINDOW", "8"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

INDUCTION12 = ["L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7",
               "L14H26", "L9H11", "L1H20", "L21H10", "L4H12", "L3H17"]
DUPLICATE9 = ["L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11", "L4H16", "L1H21", "L2H26"]
ALL21 = INDUCTION12 + DUPLICATE9


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev)
    neg_i = torch.tensor(np.where(col < 0)[0], device=dev)
    walks = list(G.generate_walks(graph, cfg))
    ids_all = [torch.tensor([[bos] + [wid[x] for x in w.nodes]], device=dev) for w in walks]

    st = {"heads": None, "keep": None}
    hooks = []
    for l in range(nL):
        def mk(l):
            def ph(_m, args):
                hs, kp = st["heads"], st["keep"]
                x = None
                if hs:
                    sel = [h for (ll, h) in hs if ll == l]
                elif kp is not None:
                    sel = [h for h in range(nH) if (l, h) not in kp]
                else:
                    return
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    def metrics(lo, hi):
        """neighbour validity + parity margin over readouts in [lo, hi)"""
        ok = tot = 0; pm = 0.0
        for wi, w in enumerate(walks):
            steps = [s for s in range(len(w.nodes) - 1) if lo <= s + 1 < hi]
            if not steps: continue
            rp = torch.tensor([s + 1 for s in steps], device=dev)
            lsm = torch.log_softmax(model(input_ids=ids_all[wi]).logits[0][rp][:, cand_t].float(), 1)
            top = lsm.argmax(1).tolist()
            cur = torch.tensor([col[w.nodes[s]] for s in steps], device=dev)
            opp = torch.where((cur > 0)[:, None], lsm[:, neg_i], lsm[:, pos_i])
            sam = torch.where((cur > 0)[:, None], lsm[:, pos_i], lsm[:, neg_i])
            pm += float((torch.logsumexp(opp, 1) - torch.logsumexp(sam, 1)).sum())
            for t_, s in zip(top, steps):
                ok += int(t_ in graph.adjacency[w.nodes[s]]); tot += 1
        return ok / max(tot, 1), pm / max(tot, 1)

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]

    def motifs_and_dirichlet(lo, hi):
        """on-task induction / same-token attention mass, and Dirichlet energy of node means"""
        ind = dup = 0.0; nq = 0
        S = torch.zeros(n, cm.hidden_size, device=dev); C = torch.zeros(n, device=dev)
        for wi, w in enumerate(walks):
            nodes = w.nodes
            qs = [t for t in range(lo, min(hi, len(nodes) - 1))]
            if not qs: continue
            o = model(input_ids=ids_all[wi], output_attentions=True, output_hidden_states=True)
            A = torch.stack([o.attentions[l][0] for l in range(nL)]).mean(0)     # [nH, T, T] layer-mean
            H = o.hidden_states[LAYER + 1][0]
            for t in qs:
                u = nodes[t]; S[u] += H[t + 1].float(); C[u] += 1
                prev = [s for s in range(t) if nodes[s] == u]
                if prev:
                    si = torch.tensor([s + 1 for s in prev], device=dev)
                    ii = torch.tensor([s + 2 for s in prev if s + 1 < t], device=dev)
                    dup += float(A[:, t + 1, si].sum(-1).mean())
                    if len(ii): ind += float(A[:, t + 1, ii].sum(-1).mean())
                    nq += 1
            del o, A
        Mn = (S / C.clamp(min=1)[:, None]); Mn = Mn - Mn.mean(0, keepdim=True)
        Mp = Mn.cpu().numpy()
        num = sum(float(((Mp[u] - Mp[v]) ** 2).sum()) for u in range(n) for v in graph.adjacency[u]) / 2
        den = float((Mp ** 2).sum())
        return ind / max(nq, 1), dup / max(nq, 1), num / max(den, 1e-9)

    res = {"model": tag, "layer": LAYER, "ctxs": CTXS, "window": WINDOW, "rows": {}}
    print(f"{'ctx':>6} {'acc':>7} {'par_marg':>9} {'inter_acc':>10} {'inter_par':>10} "
          f"{'ind_attn':>9} {'dup_attn':>9} {'dirichlet':>10}")
    for c in CTXS:
        lo, hi = c, min(c + WINDOW, WLEN - 1)
        st["heads"] = st["keep"] = None
        acc_f, par_f = metrics(lo, hi)
        ia, da, dr = motifs_and_dirichlet(lo, hi)
        st["keep"] = set(parse(ALL21)); acc_a, par_a = metrics(lo, hi)
        st["keep"] = set(parse(INDUCTION12)); acc_i, par_i = metrics(lo, hi)
        st["keep"] = set(parse(DUPLICATE9)); acc_d, par_d = metrics(lo, hi)
        st["keep"] = set(); acc_0, par_0 = metrics(lo, hi)
        st["keep"] = None
        def rec(x, f, z): return (x - z) / (f - z) if abs(f - z) > 1e-9 else 0.0
        ii_acc = rec(acc_a, acc_f, acc_0) - (rec(acc_i, acc_f, acc_0) + rec(acc_d, acc_f, acc_0))
        ii_par = rec(par_a, par_f, par_0) - (rec(par_i, par_f, par_0) + rec(par_d, par_f, par_0))
        res["rows"][str(c)] = {"acc": round(acc_f, 4), "par_margin": round(par_f, 4),
                               "acc_all21": round(acc_a, 4), "acc_ind12": round(acc_i, 4),
                               "acc_dup9": round(acc_d, 4), "acc_floor": round(acc_0, 4),
                               "interaction_acc": round(ii_acc, 4), "interaction_par": round(ii_par, 4),
                               "ind_attn": round(ia, 5), "dup_attn": round(da, 5),
                               "dirichlet": round(dr, 5)}
        print(f"{c:6} {acc_f:7.4f} {par_f:9.4f} {ii_acc:10.4f} {ii_par:10.4f} "
              f"{ia:9.5f} {da:9.5f} {dr:10.5f}", flush=True)
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/ctx_sweep_interaction{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
