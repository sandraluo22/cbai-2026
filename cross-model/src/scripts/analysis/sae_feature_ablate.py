"""BIDIRECTIONAL toy->natural-text test. Forward direction (done elsewhere): the grid-parity DAS direction
matches LlamaScope L14-residual SAE features (feat 9114 in 8x, 107994 in 32x, |cos| .44/.57). This script
runs the REVERSE arrow: ablate THEIR feature directions (project the decoder row out of the residual stream
at layer 14, every position) during grid walks and measure whether grid-parity behaviour degrades — parity
margin at readouts, neighbour validity, and the parity separation of L14H26's output. Controls: N random
features from the same SAE, plus the DAS-derived residual direction as a positive control.

Env: GEN_MODEL(Llama) LAYER(14) K(4) NWALKS(8) CTXLO(1000) WLEN(1300)
     FEATS_8X(9114) FEATS_32X(107994,50458) NRANDFEAT(8) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/sae_feature_ablate<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, gc
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
LAYER = int(os.environ.get("LAYER", "14")); K = int(os.environ.get("K", "4"))
NWALKS = int(os.environ.get("NWALKS", "8")); CTXLO = int(os.environ.get("CTXLO", "1000"))
WLEN = int(os.environ.get("WLEN", "1300"))
FEATS_8X = [int(x) for x in os.environ.get("FEATS_8X", "9114").split(",") if x]
FEATS_32X = [int(x) for x in os.environ.get("FEATS_32X", "107994,50458").split(",") if x]
NRANDFEAT = int(os.environ.get("NRANDFEAT", "8")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L14H26.npz")


def load_sae_decoders():
    from huggingface_hub import list_repo_files, hf_hub_download
    from safetensors.torch import load_file
    out = {}
    for repo, feats, tag in [("fnlp/Llama3_1-8B-Base-LXR-8x", FEATS_8X, "8x"),
                             ("fnlp/Llama3_1-8B-Base-LXR-32x", FEATS_32X, "32x")]:
        if not feats: continue
        files = [f for f in list_repo_files(repo) if f"L{LAYER}R" in f and f.endswith(".safetensors")]
        p = hf_hub_download(repo, files[0])
        w = {k: v for k, v in load_file(p).items()}
        dec = None
        for k, v in w.items():
            if any(s in k.lower() for s in ("w_dec", "decoder", "w_d")) and v.ndim == 2: dec = v.float().numpy(); break
        if dec.shape[0] < dec.shape[1]: dec = dec.T if dec.shape[1] > 4096 else dec  # want [features, d_model]
        if dec.shape[1] != 4096: dec = dec.T
        out[tag] = dec
        print(f"[sae] {repo} decoder {dec.shape}", flush=True)
    return out


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph)
    proj, hd = attn_proj(blocks[LAYER], cm); csl = slice(26 * hd, 27 * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    nbr = [set(graph.adjacency[u]) for u in range(n)]

    decs = load_sae_decoders()
    rng = np.random.default_rng(SEED)
    dirs = {}
    for f in FEATS_8X: dirs[f"sae8x_{f}"] = decs["8x"][f]
    for f in FEATS_32X: dirs[f"sae32x_{f}"] = decs["32x"][f]
    for i in range(NRANDFEAT):
        which = "32x" if "32x" in decs else "8x"
        j = int(rng.integers(decs[which].shape[0])); dirs[f"randfeat_{j}"] = decs[which][j]
    dz = np.load(DAS_NPZ)
    das1 = dz["global_R1"][0].astype(np.float64); das1 /= np.linalg.norm(das1)
    W = proj.weight.detach().float().cpu().numpy()
    vres = W[:, csl] @ das1; dirs["das_vres"] = vres
    units = {k: torch.tensor(v / (np.linalg.norm(v) + 1e-12), dtype=torch.float32, device=dev) for k, v in dirs.items()}

    state = {"v": None}
    def rh(_m, _i, out):
        if state["v"] is None: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        hf_ = h[0].float(); h[0] = (hf_ - torch.outer(hf_ @ state["v"], state["v"])).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hook = blocks[LAYER].register_forward_hook(rh)

    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hz = proj.register_forward_pre_hook(cap)

    walks = G.generate_walks(graph, cfg)
    def evaluate(vname):
        state["v"] = units[vname] if vname else None
        margins = []; valid = []; seps = []
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; zc.clear()
            lg = model(input_ids=ids).logits
            Z = []; y = []
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                t = spans[s][-1]; nd = nodes[s]
                lsm = torch.log_softmax(lg[0, t, cand_t].float(), 0)
                same = torch.logsumexp(lsm[pos_idx if col[nd] > 0 else neg_idx], 0)
                opp = torch.logsumexp(lsm[neg_idx if col[nd] > 0 else pos_idx], 0)
                margins.append(float(opp - same))
                top = int(torch.argmax(lsm)); valid.append(top in nbr[nd])
                Z.append(zc["z"][0, t, csl].float().cpu().numpy()); y.append(col[nd])
            Z = np.array(Z); y = np.array(y); Zc = Z - Z.mean(0)
            a = Zc[y > 0].mean(0) - Zc[y < 0].mean(0); a /= (np.linalg.norm(a) + 1e-12)
            s2 = Zc @ a; seps.append(float(s2[y > 0].mean() - s2[y < 0].mean()))
        state["v"] = None
        return float(np.mean(margins)), float(np.mean(valid)), float(np.mean(seps))

    base_m, base_v, base_s = evaluate(None)
    print(f"[base] parity-margin {base_m:+.3f} validity {base_v:.3f} head-sep {base_s:.3f}", flush=True)
    rows = []
    for name in units:
        m, v, s = evaluate(name)
        rows.append({"dir": name, "d_margin": round(m - base_m, 4), "d_validity": round(v - base_v, 4),
                     "d_head_sep": round(s - base_s, 4)})
        print(f"  {name:16} dmargin {m-base_m:+.3f}  dval {v-base_v:+.3f}  dsep {s-base_s:+.3f}", flush=True)
    hook.remove(); hz.remove()
    out = {"model": tag, "layer": LAYER, "k": K, "ctxlo": CTXLO,
           "base": {"parity_margin": round(base_m, 4), "validity": round(base_v, 4), "head_sep": round(base_s, 4)},
           "ablations": rows}
    p = f"{OUTDIR}/sae_feature_ablate{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
