"""Do frozen Llama/Gemma/Qwen actually PREDICT Mess3 in-context (use the belief),
or just carry recent tokens? Measure each model's own next-symbol prediction on
Mess3 sequences, restricted to the 3-symbol alphabet, vs two references:
  - optimal  = Mess3 entropy rate (Bayes predictor that knows the process)
  - unigram  = log(3) (the symmetric marginal; 'no structure used')

A model that learned the structure in-context lands below unigram, toward optimal.
Runs on the GPU pod. -> runs/belief_geometry/mess3_llm_accuracy.{png,json}
"""
import os, json, gc
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

N, CTX, N_SEQ, WARMUP, BATCH = 3, 200, 200, 40, 16
MODELS = [("Llama", "NousResearch/Meta-Llama-3.1-8B"),
          ("Gemma", "unsloth/gemma-2-9b"),
          ("Qwen",  "Qwen/Qwen3-8B-Base")]


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


def optimal_loss(seqs):
    """Entropy-rate estimate: -log p_opt(next symbol), p_opt = belief @ E."""
    B, L = seqs.shape; bel = np.repeat(STAT[None], B, 0); ce = []; acc = []
    for t in range(L):
        post = bel * E_np[:, seqs[:, t]].T; post /= post.sum(1, keepdims=True)
        bel = post @ A_np
        if t + 1 < L and t >= WARMUP:
            p = bel @ E_np                                   # P(next symbol)
            nxt = seqs[:, t + 1]
            ce.append(-np.log(p[np.arange(B), nxt] + 1e-12))
            acc.append((p.argmax(1) == nxt).astype(float))
    return float(np.mean(ce)), float(np.mean(acc))


def main():
    seqs = sample(N_SEQ, CTX)
    opt_ce, opt_acc = optimal_loss(seqs)
    uni_ce = np.log(N)
    print(f"optimal CE={opt_ce:.4f} acc={opt_acc:.3f} | unigram CE={uni_ce:.4f} acc={1/N:.3f}", flush=True)

    R = {"optimal": {"ce": opt_ce, "acc": opt_acc}, "unigram": {"ce": uni_ce, "acc": 1 / N}}
    for tag, hf in MODELS:
        print(f"[{tag}] load {hf}", flush=True)
        tok = AutoTokenizer.from_pretrained(hf)
        model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        ids = [tok(f" {c}", add_special_tokens=False).input_ids[-1] for c in ["A", "B", "C"]]
        assert len(set(ids)) == 3, f"{tag} symbol tokens not distinct: {ids}"
        ids_t = torch.tensor(ids, device="cuda")
        ce_all, acc_all, mass_all = [], [], []
        for i in range(0, N_SEQ, BATCH):
            batch = seqs[i:i + BATCH]
            inp = torch.tensor(np.array(ids)[batch], device="cuda")     # map symbols->token ids
            with torch.no_grad():
                logits = model(inp).logits                              # (b, L, vocab)
            sub = logits[:, :, ids_t].float()                           # (b, L, 3)
            full = torch.log_softmax(logits.float(), -1)
            logp3 = torch.log_softmax(sub, -1)                          # renormalised over {A,B,C}
            for t in range(WARMUP, CTX - 1):
                nxt = torch.tensor(batch[:, t + 1], device="cuda")
                ce_all.append((-logp3[:, t][torch.arange(len(batch)), nxt]).cpu().numpy())
                acc_all.append((logp3[:, t].argmax(-1) == nxt).float().cpu().numpy())
                mass_all.append(torch.exp(torch.logsumexp(full[:, t][:, ids_t], -1)).cpu().numpy())
        R[tag] = {"ce": float(np.mean(np.concatenate(ce_all))),
                  "acc": float(np.mean(np.concatenate(acc_all))),
                  "alphabet_mass": float(np.mean(np.concatenate(mass_all)))}
        print(f"[{tag}] CE={R[tag]['ce']:.4f} acc={R[tag]['acc']:.3f} "
              f"mass_on_ABC={R[tag]['alphabet_mass']:.2f}", flush=True)
        del model, tok; gc.collect(); torch.cuda.empty_cache()
        import shutil
        shutil.rmtree(os.path.join(os.environ.get("HF_HOME", "/root/hf"), "hub",
                      "models--" + hf.replace("/", "--")), ignore_errors=True)

    os.makedirs("runs/belief_geometry", exist_ok=True)
    json.dump(R, open("runs/belief_geometry/mess3_llm_accuracy.json", "w"), indent=1)
    names = ["optimal", "Llama", "Gemma", "Qwen", "unigram"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].bar(names, [R[n]["ce"] for n in names], color=["green", "C0", "C1", "C2", "0.6"])
    ax[0].axhline(uni_ce, color="0.6", ls="--"); ax[0].set_ylabel("next-symbol cross-entropy (nats)")
    ax[0].set_title("lower = uses structure"); ax[0].set_ylim(opt_ce - 0.01, uni_ce + 0.01)
    ax[1].bar(names, [R[n]["acc"] for n in names], color=["green", "C0", "C1", "C2", "0.6"])
    ax[1].axhline(1 / N, color="0.6", ls="--"); ax[1].set_ylabel("next-symbol accuracy")
    ax[1].set_title("higher = uses structure")
    fig.suptitle("Frozen LLMs predicting Mess3 in-context: do they beat the unigram baseline?")
    fig.tight_layout(); fig.savefig("runs/belief_geometry/mess3_llm_accuracy.png", dpi=150)
    print("wrote runs/belief_geometry/mess3_llm_accuracy.png", flush=True)


if __name__ == "__main__":
    main()
