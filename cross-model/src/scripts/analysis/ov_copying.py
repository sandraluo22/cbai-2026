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


def final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm                                  # Llama/Gemma/Qwen RMSNorm
    return model.transformer.ln_f                                # GPT-2 LayerNorm


@torch.no_grad()
def copying_score(model, tok):
    """OV-circuit copying score: for each source token t, the head's write IF it attended
    fully to t is  W_O_h @ (W_V_h @ x_t); push through the (LN-folded) unembedding and
    check the self-boost on token t vs all tokens. No repetition / no attention needed."""
    dev = next(model.parameters()).device
    cm = model.config
    nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    nkv = getattr(cm, "num_key_value_heads", None) or nH
    hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH)
    group = nH // nkv
    blocks = blocks_of(model); nL = len(blocks)
    WU = model.get_output_embeddings().weight.float()            # [V, d]
    is_gpt2 = not hasattr(blocks[0], "self_attn")
    fn = final_norm(model); is_rms = "rms" in type(fn).__name__.lower()
    gamma = fn.weight.detach().float()
    if "gemma" in (getattr(cm, "model_type", "") or "").lower():
        gamma = 1.0 + gamma
    # capture per-head VALUE vectors (v_proj output; GPT-2: value third of c_attn)
    vcap = {}
    def mkv(l):
        def hook(_m, _i, out): vcap[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    handles = [(blocks[l].attn.c_attn if is_gpt2 else blocks[l].self_attn.v_proj)
               .register_forward_hook(mkv(l)) for l in range(nL)]
    pool = token_pool(tok); rng = np.random.default_rng(0)
    sos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    score = np.zeros((nL, nH)); cnt = 0
    try:
        for _ in range(NSEQ):
            r = rng.choice(pool, size=GLEN, replace=False).tolist()
            ids = torch.tensor([[sos] + r], device=dev)          # 25 random toks, no repeat
            model(input_ids=ids)
            idrow = ids[0]
            present = torch.unique(idrow)                        # tokens the head could copy
            for l in range(nL):
                V = vcap[l][0]                                    # [seq, nkv*hd] (GPT2: [seq, 3d])
                if is_gpt2:
                    V = V[:, 2 * cm.hidden_size:]                # value third
                    Wo = blocks[l].attn.c_proj.weight            # Conv1D [nH*hd, d]
                else:
                    Wo = blocks[l].self_attn.o_proj.weight       # [d, nH*hd]
                for h in range(nH):
                    kv = h // group
                    vh = V[:, kv * hd:(kv + 1) * hd].float()     # [seq, hd] value for this head
                    if is_gpt2:
                        ov = vh @ Wo[h * hd:(h + 1) * hd, :].float()      # [seq, d]
                    else:
                        ov = vh @ Wo[:, h * hd:(h + 1) * hd].T.float()    # [seq, d]
                    if not is_rms:
                        ov = ov - ov.mean(dim=-1, keepdim=True)
                    ov = ov * gamma
                    g = ov @ WU.T                                # [seq, V] copy-this-token logit effect
                    g = g - g.mean(dim=1, keepdim=True)
                    g = torch.relu(g)
                    tot = g[:, present].sum(dim=1) + 1e-9                 # over tokens present in the seq
                    self_boost = g.gather(1, idrow[:, None]).squeeze(1)   # boost on the token itself
                    score[l, h] += float((self_boost / tot)[1:].mean().cpu())
            cnt += 1
    finally:
        for hnd in handles:
            hnd.remove()
    return np.clip(4 * (score / max(cnt, 1)) - 1.0, -1.0, 1.0)   # [0,0.5] -> [-1,1]


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
