"""Is the decoded Mess3 belief an ABSTRACT (token-invariant) representation, or a
surface-token shadow?

Feed the SAME Mess3 hidden-state/symbol sequences to each frozen LLM rendered
with two different token sets -- {A,B,C} and {1,2,3} -- and capture residuals for
both. Then CROSS-PROBE: fit the belief probe on one token set's activations and
apply it to the other set's activations.
  cross R^2 ~ within R^2  -> token-invariant abstract belief subspace
  cross R^2 << within      -> the 'belief' is bound to the surface tokens

Runs on the GPU pod; activations in RAM. -> runs/belief_geometry/crosstoken.{png,json}
"""
import os, json, gc
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from config import get_config
import graph as G
import models as M

N, CTX, N_SEQ, WARMUP = 3, 200, 200, 40
SETS = {"ABC": ["A", "B", "C"], "123": ["1", "2", "3"]}
MODELS = [("Llama", "meta-llama/Llama-3.1-8B"),
          ("Gemma", "google/gemma-2-9b"),
          ("Qwen",  "Qwen/Qwen3-8B-Base")]
MIRRORS = {"Llama": "NousResearch/Meta-Llama-3.1-8B", "Gemma": "unsloth/gemma-2-9b"}


def mess3(x=0.15, alpha=0.6):
    A = np.full((N, N), x); np.fill_diagonal(A, 1 - 2 * x)
    b = (1 - alpha) / 2; E = np.full((N, N), b); np.fill_diagonal(E, alpha); return A, E


A_np, E_np = mess3()
w, v = np.linalg.eig(A_np.T); STAT = np.real(v[:, np.argmin(np.abs(w - 1))]); STAT /= STAT.sum()


def sample(n, L, seed=0):
    rng = np.random.default_rng(seed); s = rng.choice(N, p=STAT, size=n); out = np.empty((n, L), np.int64)
    for t in range(L):
        out[:, t] = (rng.random(n)[:, None] < np.cumsum(E_np[s], 1)).argmax(1)
        s = (rng.random(n)[:, None] < np.cumsum(A_np[s], 1)).argmax(1)
    return out


def beliefs(seqs):
    B, L = seqs.shape; out = np.empty((B, L, N)); bel = np.repeat(STAT[None], B, 0)
    for t in range(L):
        post = bel * E_np[:, seqs[:, t]].T; post /= post.sum(1, keepdims=True)
        bel = post @ A_np; out[:, t] = bel
    return out


def load(tag, hf, cfg):
    try: return M.load_model(hf, cfg)
    except Exception: return M.load_model(MIRRORS[tag], cfg)


def freew(tag, hf):
    import shutil; hub = os.path.join(os.environ.get("HF_HOME", "/root/hf"), "hub")
    for nm in [hf, MIRRORS.get(tag)]:
        if nm: shutil.rmtree(os.path.join(hub, "models--" + nm.replace("/", "--")), ignore_errors=True)


def ridge(X, Y, a=1.0):
    mx, my = X.mean(0), Y.mean(0)
    W = np.linalg.solve((X - mx).T @ (X - mx) + a * np.eye(X.shape[1]), (X - mx).T @ (Y - my))
    return W, mx, my


def r2(W, mx, my, X, Y):
    p = (X - mx) @ W + my
    return float(1 - ((Y - p) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()), p


def main():
    cfg = replace(get_config("gemma_qwen"), device="cuda", dtype="bfloat16")
    seqs = sample(N_SEQ, CTX); bel = beliefs(seqs)
    walks = {k: [G.Walk(walk_id=i, nodes=[int(s) for s in seqs[i]], words=[V[s] for s in seqs[i]])
                 for i in range(N_SEQ)] for k, V in SETS.items()}
    R = {}
    for tag, hf in MODELS:
        print(f"[{tag}] load", flush=True)
        model, tok = load(tag, hf, cfg); nL = model.config.num_hidden_layers
        caps = {}
        for k in SETS:
            caps[k] = M.capture(model, tok, walks[k], tuple(range(nL)), cfg)
            print(f"[{tag}/{k}] captured", flush=True)
        wid, step = caps["ABC"].meta["walk_id"], caps["ABC"].meta["step"]
        assert np.array_equal(wid, caps["123"].meta["walk_id"]) and np.array_equal(step, caps["123"].meta["step"])
        Bocc = bel[wid, step]; warm = step >= WARMUP
        te = (wid % 10 < 3); tr = warm & ~te; ev = warm & te
        layers = sorted(caps["ABC"].acts)
        curves = {"within_ABC": [], "within_123": [], "cross_123to ABC": [], "cross_ABCto123": []}
        best = {"L": None, "r2": -9, "pred": None, "truth": None}
        for L in layers:
            Xa = caps["ABC"].acts[L].astype(np.float64); Xb = caps["123"].acts[L].astype(np.float64)
            Wa, ma, mya = ridge(Xa[tr], Bocc[tr]); Wb, mb, myb = ridge(Xb[tr], Bocc[tr])
            ra, _ = r2(Wa, ma, mya, Xa[ev], Bocc[ev]); rb, _ = r2(Wb, mb, myb, Xb[ev], Bocc[ev])
            rc, pc = r2(Wb, mb, myb, Xa[ev], Bocc[ev])     # train 123 -> apply ABC
            rd, _ = r2(Wa, ma, mya, Xb[ev], Bocc[ev])      # train ABC -> apply 123
            curves["within_ABC"].append(ra); curves["within_123"].append(rb)
            curves["cross_123to ABC"].append(rc); curves["cross_ABCto123"].append(rd)
            if rc > best["r2"]:
                best.update(L=L, r2=rc, pred=np.clip(pc, 0, None), truth=Bocc[ev])
        best["pred"] /= best["pred"].sum(1, keepdims=True)
        R[tag] = {"layers": layers, "curves": curves, "best": best}
        print(f"[{tag}] within_ABC peak={max(curves['within_ABC']):.3f} "
              f"cross(123->ABC) peak={max(curves['cross_123to ABC']):.3f}", flush=True)
        del model, tok, caps; gc.collect(); torch.cuda.empty_cache(); freew(tag, hf)

    os.makedirs("runs/belief_geometry", exist_ok=True)
    json.dump({t: R[t]["curves"] for t in R}, open("runs/belief_geometry/crosstoken.json", "w"), indent=1)
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    def to2d(b): return b @ np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    fig, ax = plt.subplots(len(MODELS), 2, figsize=(11, 4.6 * len(MODELS)))
    for r, (tag, _) in enumerate(MODELS):
        d = R[tag]; ks = d["layers"]; c = d["curves"]
        a = ax[r, 0]
        a.plot(ks, c["within_ABC"], label="within A,B,C"); a.plot(ks, c["within_123"], label="within 1,2,3")
        a.plot(ks, c["cross_123to ABC"], "--", label="cross 123->ABC"); a.plot(ks, c["cross_ABCto123"], "--", label="cross ABC->123")
        a.set_title(f"{tag}: belief R² within vs cross-token", fontsize=10); a.set_xlabel("layer"); a.set_ylim(-0.2, 1.05)
        if r == 0: a.legend(fontsize=7)
        b = d["best"]; a2 = ax[r, 1]; XY = to2d(b["pred"])
        a2.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1)
        a2.scatter(XY[:, 0], XY[:, 1], s=3, c=np.clip(b["pred"], 0, 1), alpha=0.5, linewidths=0, rasterized=True)
        a2.set_title(f"{tag}: fractal from cross probe (123->ABC, L{b['L']}, R²={b['r2']:.2f})", fontsize=9)
        a2.set_aspect("equal"); a2.axis("off")
    fig.suptitle("Token-invariance of the Mess3 belief: probe trained on one symbol set, applied to another")
    fig.tight_layout(); fig.savefig("runs/belief_geometry/crosstoken.png", dpi=150)
    print("wrote runs/belief_geometry/crosstoken.png", flush=True)


if __name__ == "__main__":
    main()
