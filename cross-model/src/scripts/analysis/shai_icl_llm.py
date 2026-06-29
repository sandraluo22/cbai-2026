"""Does in-context learning alone make a FROZEN pretrained LLM represent the
belief-state geometry of a hidden HMM?

We take Mess3 (the fractal-MSP HMM from Shai et al.), feed symbol sequences to
Llama / Gemma / Qwen purely in-context (no training), capture the residual stream
at every layer, and linear-probe activations -> the ground-truth belief over the
next hidden state. If ICL induces belief tracking, the Mess3 fractal should be
linearly decodable from the residual stream.

Runs on the GPU pod; holds activations in RAM (no npz). Saves one figure.
-> runs/belief_geometry/icl_llm_mess3.png  (+ per-layer R^2 in icl_llm_mess3.json)
"""
import os, json, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
import torch
from config import get_config
import graph as G
import models as M

N = 3
CTX, N_SEQ, WARMUP = 200, 200, 40
WORDS3 = ["A", "B", "C"]
MODELS = [("Llama", "meta-llama/Llama-3.1-8B"),
          ("Gemma", "google/gemma-2-9b"),
          ("Qwen",  "Qwen/Qwen3-8B-Base")]
MIRRORS = {"Llama": "NousResearch/Meta-Llama-3.1-8B", "Gemma": "unsloth/gemma-2-9b"}


def mess3(x=0.15, alpha=0.6):
    A = np.full((N, N), x); np.fill_diagonal(A, 1 - 2 * x)
    b = (1 - alpha) / 2
    E = np.full((N, N), b); np.fill_diagonal(E, alpha)
    return A, E


A_np, E_np = mess3()
w, v = np.linalg.eig(A_np.T); STAT = np.real(v[:, np.argmin(np.abs(w - 1))]); STAT /= STAT.sum()


def sample_sequences(n_seq, L, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(N, p=STAT, size=n_seq)
    out = np.empty((n_seq, L), np.int64)
    for t in range(L):
        out[:, t] = (rng.random(n_seq)[:, None] < np.cumsum(E_np[s], 1)).argmax(1)
        s = (rng.random(n_seq)[:, None] < np.cumsum(A_np[s], 1)).argmax(1)
    return out


def beliefs(seqs):
    B, L = seqs.shape
    out = np.empty((B, L, N)); bel = np.repeat(STAT[None], B, 0)
    for t in range(L):
        post = bel * E_np[:, seqs[:, t]].T; post /= post.sum(1, keepdims=True)
        bel = post @ A_np; out[:, t] = bel
    return out


def load_with_fallback(tag, hf, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception as e:
        print(f"[{tag}] {hf} -> mirror ({type(e).__name__})", flush=True)
        return M.load_model(MIRRORS[tag], cfg)


def free(hf, tag):
    hub = os.path.join(os.environ.get("HF_HOME", "/root/hf"), "hub")
    import shutil
    for nm in [hf, MIRRORS.get(tag)]:
        if nm:
            shutil.rmtree(os.path.join(hub, "models--" + nm.replace("/", "--")), ignore_errors=True)


def ridge_r2(Xtr, Ytr, Xte, Yte, a=1.0):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    W = np.linalg.solve((Xtr - mx).T @ (Xtr - mx) + a * np.eye(Xtr.shape[1]), (Xtr - mx).T @ (Ytr - my))
    pred = (Xte - mx) @ W + my
    r2 = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    return float(r2), pred


def to_2d(b):
    return b @ np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])


def main():
    cfg = replace(get_config("gemma_qwen"), device="cuda", dtype="bfloat16")
    seqs = sample_sequences(N_SEQ, CTX)
    bel = beliefs(seqs)
    walks = [G.Walk(walk_id=i, nodes=[int(s) for s in seqs[i]],
                    words=[WORDS3[s] for s in seqs[i]]) for i in range(N_SEQ)]

    R = {}
    for tag, hf in MODELS:
        print(f"[{tag}] load", flush=True)
        model, tok = load_with_fallback(tag, hf, cfg)
        nL = model.config.num_hidden_layers
        cap = M.capture(model, tok, walks, tuple(range(nL)), cfg)
        wid, step = cap.meta["walk_id"], cap.meta["step"]
        Bocc = bel[wid, step]
        warm = step >= WARMUP
        te = (wid % 10 < 3)
        tr_m, te_m = warm & ~te, warm & te
        r2s = {}
        for L in sorted(cap.acts):
            X = cap.acts[L].astype(np.float64)
            r2, _ = ridge_r2(X[tr_m], Bocc[tr_m], X[te_m], Bocc[te_m])
            r2s[L] = r2
        bestL = max(r2s, key=r2s.get)
        X = cap.acts[bestL].astype(np.float64)
        _, pred = ridge_r2(X[tr_m], Bocc[tr_m], X[te_m], Bocc[te_m])
        R[tag] = dict(r2s={int(k): v for k, v in r2s.items()}, bestL=int(bestL),
                      pred=np.clip(pred, 0, None), truth=Bocc[te_m])
        R[tag]["pred"] /= R[tag]["pred"].sum(1, keepdims=True)
        print(f"[{tag}] best layer L{bestL} R^2={r2s[bestL]:.3f}", flush=True)
        del model, tok, cap; gc.collect(); torch.cuda.empty_cache(); free(hf, tag)

    os.makedirs("runs/belief_geometry", exist_ok=True)
    json.dump({t: R[t]["r2s"] for t in R}, open("runs/belief_geometry/icl_llm_mess3.json", "w"), indent=1)
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    fig, ax = plt.subplots(len(MODELS), 3, figsize=(15, 4.6 * len(MODELS)))
    for r, tag in enumerate(MODELS and [m[0] for m in MODELS]):
        d = R[tag]
        for c, (XY, col, ttl) in enumerate([
                (to_2d(d["truth"]), d["truth"], f"{tag}: ground-truth MSP"),
                (to_2d(d["pred"]), d["pred"], f"{tag}: recovered L{d['bestL']} (R²={d['r2s'][d['bestL']]:.2f})")]):
            a = ax[r, c]; a.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1)
            a.scatter(XY[:, 0], XY[:, 1], s=3, c=np.clip(col, 0, 1), alpha=0.5, linewidths=0, rasterized=True)
            a.set_title(ttl, fontsize=10); a.set_aspect("equal"); a.axis("off")
        a = ax[r, 2]; ks = sorted(d["r2s"]); a.plot(ks, [d["r2s"][k] for k in ks], marker=".")
        a.set_title(f"{tag}: belief R² by layer", fontsize=10); a.set_xlabel("layer"); a.set_ylim(-0.1, 1)
    fig.suptitle("In-context belief geometry in FROZEN pretrained LLMs on Mess3\n"
                 "(no training; linear probe of residual stream at each position, context>=%d)" % WARMUP)
    fig.tight_layout()
    fig.savefig("runs/belief_geometry/icl_llm_mess3.png", dpi=150); plt.close(fig)
    print("wrote runs/belief_geometry/icl_llm_mess3.png", flush=True)


if __name__ == "__main__":
    main()
