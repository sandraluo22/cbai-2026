"""What do the DAS directions actually DO in natural text? Two analyses in one pass, at the RESIDUAL site
(layer LAYER) where the interchange experiments showed the variables are concentrated — not at a single
head, where redundancy buffered the earlier ablation into meaninglessness.

(1) LOGIT READOUT — add +/- alpha * v to the residual at every position and average the change in the
    FINAL logits over positions. Reports the tokens most promoted / suppressed. alpha is set per-direction
    to K_ALPHA * sd(h . v) on Pile text, so all directions are perturbed by a comparable, natural amount.
    This is hypothesis-GENERATING: it asks what the direction pushes the model to say, with no labels.

(2) ABLATION — project the subspace out of the residual at every position, measure per-token loss delta
    overall and by token category. Controls are norm-matched TWO ways: same rank, and (reported) the
    fraction of residual energy actually removed, so "damage per unit energy removed" can be compared
    between real and random subspaces. A real direction should damage more per unit energy removed.

Directions tested: parity residual DAS (rank 1/8), coordinate residual DAS (rank 1/8), the head-derived
parity direction mapped to residual space via o_proj, the matched LlamaScope SAE feature, and random
subspaces at each rank.

Env: GEN_MODEL(Llama) LAYER(14) NDOCS(200) MAXTOK(384) K_ALPHA(2) TOPTOK(25) NRAND(4) SEED(0)
     PAR_NPZ COORD_NPZ DAS_NPZ SAE_FEAT(107994) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/pile_direction_effect<OUTTAG>_<model>.json
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
LAYER = int(os.environ.get("LAYER", "14"))
NDOCS = int(os.environ.get("NDOCS", "200")); MAXTOK = int(os.environ.get("MAXTOK", "384"))
K_ALPHA = float(os.environ.get("K_ALPHA", "2")); TOPTOK = int(os.environ.get("TOPTOK", "25"))
NRAND = int(os.environ.get("NRAND", "4")); SEED = int(os.environ.get("SEED", "0"))
DATASET = os.environ.get("DATASET", "NeelNanda/pile-10k")
P = "runs/axes/4_circuits/parity"
PAR_NPZ = os.environ.get("PAR_NPZ", f"{P}/das_multihead_resid_L{LAYER}_save_{GEN_MODEL}.npz")
COORD_NPZ = os.environ.get("COORD_NPZ", f"{P}/das_multihead_resid_rot180_L{LAYER}_save_{GEN_MODEL}.npz")
DAS_NPZ = os.environ.get("DAS_NPZ", f"runs/axes/4_circuits/das/das_grid_patch_{GEN_MODEL}_L14H26.npz")
SAE_FEAT = int(os.environ.get("SAE_FEAT", "107994"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


def load_sae_dir(layer, feat):
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        from safetensors.torch import load_file
        repo = "fnlp/Llama3_1-8B-Base-LXR-32x"
        files = [f for f in list_repo_files(repo) if f"L{layer}R" in f and f.endswith(".safetensors")]
        w = load_file(hf_hub_download(repo, files[0]))
        dec = None
        for k, v in w.items():
            if any(s in k.lower() for s in ("w_dec", "decoder", "w_d")) and v.ndim == 2:
                dec = v.float().numpy(); break
        if dec.shape[1] != 4096: dec = dec.T
        return dec[feat].astype(np.float64)
    except Exception as e:
        print(f"[sae] unavailable: {str(e)[:100]}", flush=True); return None


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); dm = cm.hidden_size
    rng = np.random.default_rng(SEED)

    subs = {}
    def add(name, arr):
        a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        q, _ = np.linalg.qr(a.T); subs[name] = q.T[:a.shape[0]]
    for nm, path, keys in (("par", PAR_NPZ, ("4x4_r1", "4x4_r8")), ("coord", COORD_NPZ, ("4x4_r1", "4x4_r8"))):
        if os.path.exists(path):
            z = np.load(path)
            for k in keys:
                if k in z.files: add(f"{nm}_{k.split('_')[1]}", z[k])
        else: print(f"[warn] missing {path}", flush=True)
    dz = np.load(DAS_NPZ)
    das1 = dz["global_R1"][0].astype(np.float64); das1 /= np.linalg.norm(das1)
    proto = dz["proto_delta"].astype(np.float64)
    if das1 @ (proto / np.linalg.norm(proto)) < 0: das1 = -das1
    W = attn_proj(blocks[14], cm)[0].weight.detach().float().cpu().numpy()
    hd = (getattr(cm, "head_dim", None) or dm // cm.num_attention_heads)
    add("head_das_r1", W[:, 26 * hd:27 * hd] @ das1)
    sae = load_sae_dir(LAYER, SAE_FEAT)
    if sae is not None: add(f"sae_{SAE_FEAT}", sae)
    for r in (1, 8):
        for i in range(NRAND): add(f"rand{r}_{i}", np.linalg.qr(rng.standard_normal((dm, r)))[0].T)
    names = list(subs)
    Q = {n: torch.tensor(v, dtype=torch.float32, device=dev) for n, v in subs.items()}
    print(f"[{tag}] {len(names)} conditions: {names}", flush=True)

    state = {"mode": None, "Q": None, "alpha": 0.0}
    def rh(_m, _i, out):
        if state["mode"] is None: return out
        h = out[0] if isinstance(out, tuple) else out
        x = h[0].float(); q = state["Q"]
        if state["mode"] == "ablate": x = x - (x @ q.t()) @ q
        else: x = x + state["alpha"] * q[0]
        h = h.clone(); h[0] = x.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hook = blocks[LAYER].register_forward_hook(rh)

    import string as _string
    def catc(piece):
        wi = piece.startswith("Ġ") or piece.startswith("▁") or piece.startswith(" ")
        core = piece.lstrip("Ġ▁ ")
        if core and all(ch in _string.punctuation for ch in core): return "punct"
        if core and all(ch.isdigit() for ch in core): return "digit"
        return "word_initial" if wi else "continuation"

    from datasets import load_dataset
    docs = []
    for ex in load_dataset(DATASET, split="train", streaming=True):
        if len(docs) >= NDOCS: break
        t = ex["text"]
        if t and len(t) >= 40: docs.append(t)

    # ---- pass 0: baseline losses, coefficient scales, residual norms ----
    ids_all = []; base_loss = []; cats = []; coef = {n: [] for n in names}; resnorm = []
    for text in docs:
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXTOK, return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] < 12: continue
        ids_all.append(ids)
        state["mode"] = None
        outp = model(input_ids=ids, output_hidden_states=True)
        h = outp.hidden_states[LAYER + 1][0].float()
        resnorm.append((h ** 2).sum(1).mean().item())
        for n in names: coef[n].append((h @ Q[n][0]).cpu().numpy())
        lg = outp.logits[0].float()
        lsm = torch.log_softmax(lg[:-1], -1)
        base_loss.append(-lsm[torch.arange(ids.shape[1] - 1), ids[0, 1:]].cpu().numpy())
        pieces = tok.convert_ids_to_tokens(ids[0].tolist())
        cats.append([catc(pieces[t + 1]) for t in range(ids.shape[1] - 1)])
    scale = {n: float(np.concatenate(coef[n]).std()) for n in names}
    print(f"[{tag}] {len(ids_all)} docs; coef sd: " + " ".join(f"{n}={scale[n]:.2f}" for n in names[:6]), flush=True)

    # ---- pass 1: ablation ----
    CATS = ("word_initial", "continuation", "punct", "digit")
    abl = {}
    for n in names:
        tot = [0.0, 0]; bycat = {c: [0.0, 0] for c in CATS}; erem = []
        state["mode"] = "ablate"; state["Q"] = Q[n]
        for di, ids in enumerate(ids_all):
            outp = model(input_ids=ids, output_hidden_states=True)
            lg = outp.logits[0].float()
            lsm = torch.log_softmax(lg[:-1], -1)
            l = -lsm[torch.arange(ids.shape[1] - 1), ids[0, 1:]].cpu().numpy()
            d = l - base_loss[di]
            for t in range(3, len(d)):
                tot[0] += float(d[t]); tot[1] += 1
                c = cats[di][t]; bycat[c][0] += float(d[t]); bycat[c][1] += 1
            erem.append(float((np.stack([coef[m][di] for m in [n]])[0] ** 2).sum() / max(len(coef[n][di]), 1)))
        state["mode"] = None
        r = subs[n].shape[0]
        # energy removed = sum of squared coefficients over the subspace basis / residual energy
        frac = float(np.mean([np.sum([coef[n][di] ** 2]) / max(len(coef[n][di]), 1) for di in range(len(ids_all))]) / np.mean(resnorm)) if r == 1 else None
        abl[n] = {"rank": r, "mean_dloss": round(tot[0] / max(tot[1], 1), 5),
                  "by_cat": {c: round(bycat[c][0] / max(bycat[c][1], 1), 5) for c in CATS},
                  "r1_energy_frac": None if frac is None else round(frac, 5)}
        print(f"  [ablate] {n:16} r={r} dloss={abl[n]['mean_dloss']:+.5f}", flush=True)

    # ---- pass 2: logit readout (rank-1 rows only) ----
    V = model.get_output_embeddings().weight.shape[0]
    logit_rows = [n for n in names if subs[n].shape[0] == 1]
    read = {}
    for n in logit_rows:
        acc = torch.zeros(V, device=dev, dtype=torch.float64); npos = 0
        for sgn in (+1, -1):
            state["mode"] = "add"; state["Q"] = Q[n]; state["alpha"] = sgn * K_ALPHA * scale[n]
            for di, ids in enumerate(ids_all[:min(60, len(ids_all))]):
                lg = model(input_ids=ids).logits[0].float()
                state["mode"] = None
                bl = model(input_ids=ids).logits[0].float()
                state["mode"] = "add"
                d = (lg - bl)[3:]
                acc += (sgn * d).sum(0).double(); npos += d.shape[0]
            state["mode"] = None
        mean = (acc / max(npos, 1)).cpu().numpy()
        top = np.argsort(mean)[::-1][:TOPTOK]; bot = np.argsort(mean)[:TOPTOK]
        read[n] = {"alpha": round(K_ALPHA * scale[n], 3),
                   "promoted": [[tok.decode([int(i)]), round(float(mean[i]), 4)] for i in top],
                   "suppressed": [[tok.decode([int(i)]), round(float(mean[i]), 4)] for i in bot]}
        print(f"  [logit] {n:16} + " + " ".join(repr(x[0]) for x in read[n]["promoted"][:8]), flush=True)
        print(f"                   - " + " ".join(repr(x[0]) for x in read[n]["suppressed"][:8]), flush=True)
    hook.remove()

    out = {"model": tag, "layer": LAYER, "ndocs": len(ids_all), "k_alpha": K_ALPHA,
           "coef_sd": {n: round(scale[n], 4) for n in names},
           "ablation": abl, "logit_readout": read}
    p = f"{OUTDIR}/pile_direction_effect{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
