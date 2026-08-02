"""HOW does a head write the variable into the residual stream? Attention-driven or value-driven?

A head's write to a direction u decomposes EXACTLY:

    write(t) = z_t . (W_O_h^T u) = sum_s  a_ts * val(s),     val(s) = v_s . (W_O_h^T u)

where v_s is the head's value vector at source position s and a_ts its attention. So the write at a
readout position is an attention-weighted average of a per-source-position scalar "value score". Two
qualitatively different mechanisms produce a parity/coordinate write:

  VALUE-driven     val(s) itself tracks the variable (e.g. val(s) is +a for even-parity source nodes and
                   -a for odd), and the head can attend broadly — the write reads out whatever it looks at.
  ATTENTION-driven val(s) is roughly constant across nodes, but the head selectively attends to sources
                   with the needed property (e.g. only to same-parity positions), so the write is created
                   by WHERE it looks, not by what the values say.

We measure both, plus a variance decomposition that freezes one factor at its mean and re-computes the
write, attributing write variance to attention vs values. Also reports how attention mass splits by
source-node parity / coordinate distance, and verifies the decomposition reproduces the true write.

Env: GEN_MODEL(Llama) HEADS("14:26,14:19,2:26,21:10,9:23") DIR(parity|coord)
     PAR_NPY(...seed_stable_r1_<model>.npy) K(4) NWALKS(2) WLEN(1200) CTXLO(800) NQ(60)
     SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/write_mechanism_<DIR><OUTTAG>_<model>.json
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
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEADS = [tuple(int(x) for x in h.split(":")) for h in
         os.environ.get("HEADS", "14:26,14:19,2:26,21:10,9:23").split(",")]
DIR = os.environ.get("DIR", "parity")
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "2"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
NQ = int(os.environ.get("NQ", "60")); SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
PAR_NPY = os.environ.get("PAR_NPY", f"{P}/seed_stable_r1_{GEN_MODEL}.npy")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    try: model.set_attn_implementation("eager")
    except Exception: model.config._attn_implementation = "eager"
    cm = model.config; blocks = M._decoder_blocks(model)
    nH = cm.num_attention_heads; dm = cm.hidden_size
    hd = getattr(cm, "head_dim", None) or dm // nH
    nkv = getattr(cm, "num_key_value_heads", nH); rep = nH // nkv
    rng = np.random.default_rng(SEED)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, float)

    if DIR == "parity":
        u = np.load(PAR_NPY).astype(np.float32); u = u / np.linalg.norm(u)
        node_label = col.copy()
    else:
        # coordinate direction must come from an INDEPENDENT source (the rot180 interchange DAS), not be
        # fit against the same labels we then correlate against — that was circular and inflated val~label.
        cz = np.load(os.environ.get("COORD_NPZ", f"{P}/das_multihead_resid_rot180_L14_save_{GEN_MODEL}.npz"))
        u = cz["4x4_r1"][0].astype(np.float32); u = u / np.linalg.norm(u)
        node_label = coords[:, 0] - coords[:, 0].mean()
        print(f"[{tag}] coord direction from rot180 residual DAS (independent of the label)", flush=True)
    ut = torch.tensor(u, device=dev) if u is not None else None

    caps = {}
    hooks = []
    for (l, h) in HEADS:
        att = blocks[l].self_attn
        def mkv(l):
            def hh(_m, _i, out): caps[("v", l)] = out.detach()
            return hh
        hooks.append(att.v_proj.register_forward_hook(mkv(l)))
        def mkz(l):
            def hh(_m, args): caps[("z", l)] = args[0].detach()
            return hh
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mkz(l)))

    res = {}
    for (l, h) in HEADS:
        res[f"L{l}H{h}"] = {"n": 0, "val_parity_corr": [], "attn_same_frac": [], "recon_corr": [],
                            "var_full": [], "var_attn_frozen": [], "var_val_frozen": [],
                            "val_sd_across_nodes": [], "attn_entropy": []}

    walks = G.generate_walks(graph, cfg)
    for wi, wk in enumerate(walks):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk)
        tok_of_step = [sp[-1] for sp in spans]
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        qsteps = sorted(rng.choice(steps, min(NQ, len(steps)), replace=False))
        caps.clear()
        o = model(input_ids=ids, output_attentions=True)
        src_steps = list(range(len(wk.nodes)))
        src_pos = torch.tensor([tok_of_step[s] for s in src_steps], device=dev)
        src_node = np.array([wk.nodes[s] for s in src_steps])
        for (l, hh_) in HEADS:
            key = f"L{l}H{hh_}"
            W = attn_proj(blocks[l], cm)[0].weight.detach().float()
            w_h = (W.t() @ ut).view(nH, hd)[hh_] if ut is not None else None
            if w_h is None:
                z_all = caps[("z", l)][0].float().view(-1, nH, hd)[:, hh_]
                zc = z_all[src_pos] - z_all[src_pos].mean(0)
                lab = torch.tensor(node_label[src_node], dtype=torch.float32, device=dev)
                w_h = (zc * (lab - lab.mean())[:, None]).mean(0)
                w_h = w_h / (w_h.norm() + 1e-9)
            V = caps[("v", l)][0].float().view(-1, nkv, hd)[:, hh_ // rep]     # [T, hd]
            val = V @ w_h                                                       # [T]
            A = o.attentions[l][0, hh_].float()                                 # [T, T]
            qpos = torch.tensor([tok_of_step[s] for s in qsteps], device=dev)
            a_q = A[qpos]                                                       # [NQ, T]
            write = (a_q * val[None, :]).sum(1)                                 # [NQ]
            z_true = caps[("z", l)][0].float().view(-1, nH, hd)[qpos, hh_] @ w_h
            r = float(np.corrcoef(write.cpu().numpy(), z_true.cpu().numpy())[0, 1])
            vs = val[src_pos].cpu().numpy(); lab = node_label[src_node]
            res[key]["val_parity_corr"].append(float(np.corrcoef(vs, lab)[0, 1]))
            per_node = np.array([vs[src_node == q].mean() if (src_node == q).sum() else 0.0 for q in range(n)])
            res[key]["val_sd_across_nodes"].append(float(per_node.std() / (vs.std() + 1e-9)))
            a_src = a_q[:, src_pos].cpu().numpy()
            qn = np.array([wk.nodes[s] for s in qsteps])
            same = np.array([[1.0 if col[src_node[j]] == col[qn[i]] else 0.0 for j in range(len(src_node))]
                             for i in range(len(qn))])
            m = a_src.sum(1) + 1e-9
            res[key]["attn_same_frac"].append(float(((a_src * same).sum(1) / m).mean()))
            ent = -(a_q.clamp_min(1e-9) * a_q.clamp_min(1e-9).log()).sum(1)
            res[key]["attn_entropy"].append(float(ent.mean()))
            wf = write.cpu().numpy()
            a_mean = a_q.mean(0, keepdim=True)
            w_attn_frozen = (a_mean * val[None, :]).sum(1).repeat(len(qsteps)).cpu().numpy()
            val_mean = torch.full_like(val, float(val.mean()))
            w_val_frozen = (a_q * val_mean[None, :]).sum(1).cpu().numpy()
            res[key]["var_full"].append(float(wf.var()))
            res[key]["var_attn_frozen"].append(float(w_attn_frozen.var()))
            res[key]["var_val_frozen"].append(float(w_val_frozen.var()))
            res[key]["recon_corr"].append(r); res[key]["n"] += len(qsteps)
        del o
        print(f"[walk {wi}] done", flush=True)
    for hk in hooks: hk.remove()

    out = {"model": tag, "dir": DIR, "k": K, "ctxlo": CTXLO, "heads": {}}
    print(f"\n{'head':9} {'recon_r':>8} {'val~label':>10} {'val_node_sd':>12} {'attn_same':>10} {'attn_H':>7}"
          f" {'var_val':>8} {'var_attn':>9}")
    for key, v in res.items():
        mean = lambda k: float(np.mean(v[k]))
        vf, vaf, vvf = mean("var_full"), mean("var_attn_frozen"), mean("var_val_frozen")
        share_val = vvf / (vf + 1e-12)      # attention frozen at mean -> variance from values only? see note
        out["heads"][key] = {
            "reconstruction_corr": round(mean("recon_corr"), 4),
            "value_tracks_label_r": round(mean("val_parity_corr"), 3),
            "value_node_sd_frac": round(mean("val_sd_across_nodes"), 3),
            "attn_same_class_frac": round(mean("attn_same_frac"), 3),
            "attn_entropy": round(mean("attn_entropy"), 2),
            "var_full": round(vf, 6),
            "var_with_values_frozen": round(vvf, 6),
            "var_with_attention_frozen": round(vaf, 6),
            "attention_driven_share": round(vvf / (vf + 1e-12), 3),
        }
        d = out["heads"][key]
        print(f"{key:9} {d['reconstruction_corr']:8.3f} {d['value_tracks_label_r']:10.3f}"
              f" {d['value_node_sd_frac']:12.3f} {d['attn_same_class_frac']:10.3f} {d['attn_entropy']:7.2f}"
              f" {d['var_full']:8.5f} {d['var_with_values_frozen']:9.5f}")
    print("\nattention_driven_share = var(write | values frozen at mean) / var(write).")
    print("  ~1 => the write is created by WHERE the head attends; ~0 => by WHAT the values encode.")
    p = f"{OUTDIR}/write_mechanism_{DIR}{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
