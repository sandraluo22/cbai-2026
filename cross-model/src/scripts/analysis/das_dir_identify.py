"""Identify WHAT the DAS parity direction is in natural-language feature terms. Three comparisons:
(1) Probe match — capture L14H26 output on Pile tokens, build the word_initial-minus-continuation
    mean-difference direction and a logistic word-boundary probe in head-output space, report cosines
    with the DAS r=1 direction and proto_delta (with random-direction null).
(2) Category separability — how well the DAS direction alone classifies word_initial vs continuation
    (AUC), vs the trained probe ceiling.
(3) SAE match (best effort) — map the DAS direction into the residual stream via o_proj's column block,
    try to download a public Llama-3.1-8B residual SAE for this layer, and report top decoder cosines.
    Skipped gracefully if no SAE is downloadable.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) NDOCS(120) MAXTOK(512) DATASET(NeelNanda/pile-10k)
     DAS_NPZ(runs/axes/4_circuits/das/das_grid_patch_<model>_L<l>H<h>.npz) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/das_dir_identify<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
NDOCS = int(os.environ.get("NDOCS", "120")); MAXTOK = int(os.environ.get("MAXTOK", "512"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L{HEAD_LAYER}H{HEAD_IDX}.npz")
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

SAE_CANDIDATES = [  # (repo_id, guess at weight filename patterns) — best effort, layer substituted
    ("fnlp/Llama3_1-8B-Base-L{L}R-8x", None),
    ("fnlp/Llama3_1-8B-Base-LXR-8x", None),
    ("EleutherAI/sae-llama-3.1-8b-64x", None),
]


def cosn(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


@torch.no_grad()
def capture_pile(model, tok, proj, csl, dev):
    import string as _string
    from datasets import load_dataset
    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    hk = proj.register_forward_pre_hook(cap)
    def categorize(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"
    Z = []; Y = []
    ds = load_dataset(DATASET, split="train", streaming=True); cnt = 0
    for ex in ds:
        if cnt >= NDOCS: break
        text = ex["text"]
        if not text or len(text) < 40: continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        idl = ids[0].tolist(); zc.clear(); model(input_ids=ids)
        pieces = tok.convert_ids_to_tokens(idl)
        z = zc["z"][0, :, csl].float().cpu().numpy()
        for t in range(3, len(idl)):
            c = categorize(pieces[t])
            if c in ("word_initial", "continuation"):
                Z.append(z[t]); Y.append(1 if c == "word_initial" else 0)
        cnt += 1
        if cnt % 40 == 0: print(f"  {cnt}/{NDOCS} docs, {len(Z)} tokens", flush=True)
    hk.remove()
    return np.array(Z, np.float32), np.array(Y), cnt


def auc(scores, y):
    order = np.argsort(scores); r = np.empty(len(scores)); r[order] = np.arange(1, len(scores) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0 + 1e-12))


def try_sae(vres, layer):
    """best-effort: download a public residual SAE and report top decoder cosines with vres."""
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except Exception as e:
        return {"status": f"no huggingface_hub: {e}"}
    for repo_t, _ in SAE_CANDIDATES:
        repo = repo_t.replace("{L}", str(layer))
        try:
            files = list_repo_files(repo)
        except Exception:
            continue
        cand = [f for f in files if f.endswith((".safetensors", ".pt", ".pth", ".npz"))
                and (f"l{layer}" in f.lower() or f"layer{layer}" in f.lower() or f"L{layer}R" in f or len(files) < 6)]
        if not cand: cand = [f for f in files if f.endswith((".safetensors", ".pt"))][:1]
        for f in cand[:3]:
            try:
                p = hf_hub_download(repo, f)
                if f.endswith(".safetensors"):
                    from safetensors.numpy import load_file
                    w = load_file(p)
                else:
                    w = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in torch.load(p, map_location="cpu").items()}
                dec = None
                for k, v in w.items():
                    if any(s in k.lower() for s in ("w_dec", "decoder", "w_d")) and getattr(v, "ndim", 0) == 2:
                        dec = np.asarray(v); break
                if dec is None: continue
                if dec.shape[0] == len(vres): dec = dec.T          # features x d_model
                if dec.shape[1] != len(vres): continue
                dn = dec / (np.linalg.norm(dec, axis=1, keepdims=True) + 1e-12)
                cs = dn @ (vres / np.linalg.norm(vres))
                top = np.argsort(np.abs(cs))[::-1][:10]
                return {"status": "ok", "repo": repo, "file": f, "n_features": int(dec.shape[0]),
                        "top_features": [{"idx": int(i), "cos": round(float(cs[i]), 4)} for i in top],
                        "max_abs_cos": round(float(np.abs(cs).max()), 4)}
            except Exception as e:
                last = str(e)[:120]; continue
    return {"status": "no compatible SAE found among candidates"}


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)

    dz = np.load(DAS_NPZ)
    proto = dz["proto_delta"].astype(np.float64); proto /= np.linalg.norm(proto)
    das1 = dz["global_R1"][0].astype(np.float64); das1 /= np.linalg.norm(das1)
    if das1 @ proto < 0: das1 = -das1

    print(f"[{tag}] capturing pile tokens for probes…", flush=True)
    Z, Y, ndocs = capture_pile(model, tok, proj, csl, dev)
    Zc = Z - Z.mean(0)
    md = Z[Y == 1].mean(0) - Z[Y == 0].mean(0); md_n = md / np.linalg.norm(md)
    # logistic probe (torch, quick)
    zt = torch.tensor(Zc, device=dev); yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    wgt = torch.zeros(Z.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([wgt, b], lr=0.05)
    with torch.enable_grad():
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(zt @ wgt + b, yt)
            loss.backward(); opt.step()
    probe = wgt.detach().cpu().numpy().astype(np.float64); probe /= np.linalg.norm(probe)
    rng = np.random.default_rng(SEED)
    rand_cos = [abs(cosn(rng.standard_normal(hd), md_n)) for _ in range(500)]
    probes = {
        "cos_das1_vs_meandiff": round(abs(cosn(das1, md_n)), 4),
        "cos_das1_vs_logistic_probe": round(abs(cosn(das1, probe)), 4),
        "cos_proto_vs_meandiff": round(abs(cosn(proto, md_n)), 4),
        "cos_proto_vs_logistic_probe": round(abs(cosn(proto, probe)), 4),
        "cos_meandiff_vs_logistic": round(abs(cosn(md_n, probe)), 4),
        "random_cos_mean": round(float(np.mean(rand_cos)), 4), "random_cos_p99": round(float(np.percentile(rand_cos, 99)), 4),
        "auc_das1": round(auc(Zc @ das1, Y), 4), "auc_meandiff": round(auc(Zc @ md_n, Y), 4),
        "auc_logistic": round(auc(Zc @ probe, Y), 4),
        "n_tokens": int(len(Y)), "frac_word_initial": round(float(Y.mean()), 3), "ndocs": ndocs,
    }
    print(json.dumps(probes, indent=1), flush=True)

    # residual-space direction and SAE attempt
    W = proj.weight.detach().float().cpu().numpy()          # [d_model, n_heads*hd]
    vres = W[:, csl] @ das1
    sae = try_sae(vres, HEAD_LAYER)
    print("SAE:", json.dumps(sae)[:400], flush=True)

    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "das_npz": DAS_NPZ, "probes": probes, "sae": sae}
    p = f"{OUTDIR}/das_dir_identify{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
