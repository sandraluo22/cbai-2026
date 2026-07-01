"""OV copying score -- the 'what it writes' half of an induction head (the QK score in
induction_heads.py is the 'where it looks' half). A true induction head must do BOTH.

Stimulus: 25 random tokens (same freq-band exclusion as the QK probe), NO repetition,
SOS prepended; 10 examples.
Per (layer, head): the head's contribution to the residual = its slice of the o_proj
INPUT (attn-weighted values) pushed back through its W_O slice; then the direct path to
logits = that contribution @ W_U (no MLP, no other heads, no final-LN). Center over vocab,
ReLU (count only logit-raises), ratio = boost on the attended-to token (argmax-attention
key) / total boost. Map [0,0.5] -> [-1,1]; average over positions and the 10 examples.
One number per head.

Merges the QK score from induction.json and writes the QK-vs-OV scatter (true induction
heads = high on both). Two-pass friendly (MODELS_FILTER + HF_HOME). PRESET=smoke -> distilgpt2.

Env: PRESET MODELS_FILTER GLEN(25) NSEQ(10) OUTDIR INDJSON DEVICE
Out: <OUTDIR>/copying.json , <OUTDIR>/qk_vs_ov.pdf
"""
from __future__ import annotations
import os, json, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import torch
except Exception:
    torch = None

PRESET = os.environ.get("PRESET", "gemma_qwen")
if PRESET == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None)]
else:
    MODELS = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
              ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
              ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
GLEN = int(os.environ.get("GLEN", "25"))
NSEQ = int(os.environ.get("NSEQ", "10"))
EXCL_FREQ = float(os.environ.get("EXCL_FREQ", "0.02"))
EXCL_RARE = float(os.environ.get("EXCL_RARE", "0.10"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/copying" if PRESET != "smoke" else "runs/smoke_copying")
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json"
                         if PRESET != "smoke" else "runs/smoke_ind3/induction.json")


def load_eager(tag, hf, mirror):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    dt = torch.float32 if PRESET == "smoke" else torch.bfloat16
    def _load(name):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt, attn_implementation="eager").to(dev).eval()
        return model, tok, dev
    try:
        return _load(hf)
    except Exception:
        if mirror:
            return _load(mirror)
        raise


def token_pool(tok):
    V = int(getattr(tok, "vocab_size", None) or len(tok))
    special = set(tok.all_special_ids or [])
    try:
        special |= set(tok.get_added_vocab().values())
    except Exception:
        pass
    ids = [i for i in range(V) if i not in special]
    lo, hi = int(EXCL_FREQ * len(ids)), int((1 - EXCL_RARE) * len(ids))
    return np.array(ids[lo:hi])


def blocks_of(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.transformer.h                                   # GPT-2


def o_proj_of(block):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj                            # Llama/Gemma/Qwen
    return block.attn.c_proj                                     # GPT-2


@torch.no_grad()
def copying_score(model, tok):
    dev = next(model.parameters()).device
    cm = model.config
    nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH)
    blocks = blocks_of(model); nL = len(blocks)
    WU = model.get_output_embeddings().weight                    # [V, d]
    is_gpt2 = not (hasattr(blocks[0], "self_attn"))
    # capture o_proj input per layer
    zin = {}
    def mk(l):
        def pre(_m, args): zin[l] = args[0].detach()             # [b, seq, nH*hd] (GPT2: also nH*hd)
        return pre
    handles = [o_proj_of(blocks[l]).register_forward_pre_hook(mk(l)) for l in range(nL)]
    pool = token_pool(tok); rng = np.random.default_rng(0)
    sos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    score = np.zeros((nL, nH)); cnt = 0
    try:
        for _ in range(NSEQ):
            r = rng.choice(pool, size=GLEN, replace=False).tolist()
            ids = torch.tensor([[sos] + r], device=dev)          # 25 random toks, NO repeat
            atts = model(input_ids=ids, output_attentions=True).attentions
            idrow = ids[0]                                        # [seq]
            S = idrow.shape[0]
            for l in range(nL):
                z = zin[l][0]                                     # [seq, nH*hd]
                Wo = o_proj_of(blocks[l]).weight                 # [d, nH*hd] (GPT2 Conv1D: [nH*hd, d])
                A = atts[l][0]                                    # [H, seq, seq]
                att_key = A.argmax(dim=-1)                        # [H, seq] -> key attended most
                for h in range(nH):
                    sl = slice(h * hd, (h + 1) * hd)
                    if is_gpt2:                                   # Conv1D weight is [in, out]
                        c = z[:, sl] @ Wo[sl, :]                  # [seq, d]
                    else:
                        c = z[:, sl] @ Wo[:, sl].T               # [seq, d]
                    g = (c @ WU.T).float()                        # [seq, V] direct-path logits
                    g = g - g.mean(dim=1, keepdim=True)
                    g = torch.relu(g)
                    tot = g.sum(dim=1) + 1e-9
                    att_id = idrow[att_key[h]]                    # [seq] attended-to token id
                    boost = g.gather(1, att_id[:, None]).squeeze(1)
                    ratio = (boost / tot)[1:]                     # skip SOS position
                    score[l, h] += float(ratio.mean().cpu())
            cnt += 1
    finally:
        for hd_ in handles:
            hd_.remove()
    ratio_mean = score / max(cnt, 1)
    return np.clip(4 * ratio_mean - 1.0, -1.0, 1.0)              # [0,0.5] -> [-1,1]


def main():
    out = {"models": {}}
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading (eager)", flush=True)
        model, tok, dev = load_eager(tag, hf, mirror)
        ov = copying_score(model, tok)
        nL, nH = ov.shape
        rec = {"n_layers": nL, "n_heads": nH, "copying": ov.tolist()}
        if tag in ind:
            qk = np.array(ind[tag]["generic"])
            rec["qk"] = qk.tolist()
            both = [(int(i // nH), int(i % nH), float(qk.flatten()[i]), float(ov.flatten()[i]))
                    for i in range(qk.size)]
            both.sort(key=lambda t: min(t[2], t[3]), reverse=True)   # high on BOTH
            rec["top_induction"] = [{"layer": l, "head": h, "qk": round(q, 3), "ov": round(o, 3)}
                                    for l, h, q, o in both[:8]]
            print(f"[{tag}] true induction (high QK & OV): " +
                  ", ".join(f"L{d['layer']}H{d['head']}(qk{d['qk']:.2f},ov{d['ov']:.2f})" for d in rec["top_induction"][:5]), flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(OUTDIR, exist_ok=True)
    prev = f"{OUTDIR}/copying.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/qk_vs_ov.pdf")
    print(f"DONE -> {prev} + qk_vs_ov.pdf", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]
            if "qk" not in r:
                continue
            qk = np.array(r["qk"]).flatten(); ov = np.array(r["copying"]).flatten()
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
            ax.scatter(qk, ov, s=10, alpha=.4)
            for d in r["top_induction"][:8]:
                ax.annotate(f"L{d['layer']}H{d['head']}", (d["qk"], d["ov"]), fontsize=6, color="red")
            ax.axhline(0, color=".85", lw=.6); ax.axvline(0, color=".85", lw=.6)
            ax.set_xlabel("QK prefix-matching score"); ax.set_ylabel("OV copying score")
            ax.set_title(f"{m}: QK vs OV per head\n(true induction heads = top-right: high on both)", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
