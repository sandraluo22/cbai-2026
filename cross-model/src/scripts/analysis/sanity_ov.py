"""Sanity-check the OV copying score, esp. for Llama (near-zero OV was suspicious).

Two tests per model:
  A. BEHAVIOURAL induction: 25 random tokens x4 + SOS; at second+ copy positions, does the
     model's argmax next-token equal the true repeated token? High => the model does
     induction (so copying happens SOMEWHERE) -> low per-head OV would be a metric/indirect
     issue, not "no copying".
  B. CONTROLLED OV circuit for the top-QK heads: W_U . gamma . W_O_h . W_V_h . W_E on token
     EMBEDDINGS (no context/position), self-boost ratio among the sampled tokens. Compares
     against the activation-based OV in copying.json.

Env: PRESET MODELS_FILTER INDJSON COPYJSON OUTDIR DEVICE
Out: <OUTDIR>/sanity_ov.json
"""
from __future__ import annotations
import os, json, gc
import numpy as np
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
GLEN, NREP, NSEQ = 25, 4, 5
NSAMP = 60
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
COPYJSON = os.environ.get("COPYJSON", "/workspace/cross-model/runs/induction-head/copying/copying.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head")


def load(tag, hf, mirror):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    dt = torch.float32 if PRESET == "smoke" else torch.bfloat16
    def _l(n):
        tok = AutoTokenizer.from_pretrained(n)
        m = AutoModelForCausalLM.from_pretrained(n, dtype=dt).to(dev).eval()
        return m, tok, dev
    try:
        return _l(hf)
    except Exception:
        return _l(mirror)


def pool(tok):
    V = int(getattr(tok, "vocab_size", None) or len(tok))
    sp = set(tok.all_special_ids or [])
    ids = [i for i in range(V) if i not in sp]
    return np.array(ids[int(0.02 * len(ids)):int(0.90 * len(ids))])


@torch.no_grad()
def main():
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    cop = json.load(open(COPYJSON))["models"] if os.path.exists(COPYJSON) else {}
    out = {}
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading", flush=True)
        m, tok, dev = load(tag, hf, mirror)
        cm = m.config; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        nkv = getattr(cm, "num_key_value_heads", None) or nH
        hd = getattr(cm, "head_dim", None) or (cm.hidden_size // nH); group = nH // nkv
        tied = bool(getattr(cm, "tie_word_embeddings", False))
        sos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
        pl = pool(tok); rng = np.random.default_rng(0); L = GLEN

        # A. behavioural induction accuracy
        accs = []
        for _ in range(NSEQ):
            r = rng.choice(pl, size=L, replace=False).tolist()
            ids = torch.tensor([[sos] + r * NREP], device=dev)
            lg = m(input_ids=ids).logits[0]
            row = ids[0]; c = t = 0
            for p in range(L + 1, NREP * L):                      # copies 2..NREP, has a next token
                c += int(lg[p].argmax() == row[p + 1]); t += 1
            accs.append(c / t)
        ind_acc = float(np.mean(accs))

        # B. controlled OV circuit for top-QK heads (embeddings, no context)
        rec = {"tie_word_embeddings": tied, "n_kv_heads": nkv, "induction_behaviour_acc": round(ind_acc, 3), "heads": []}
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((cm.num_hidden_layers, nH))))
        ovmat = np.array(cop.get(tag, {}).get("copying", np.zeros_like(gen)))
        topheads = [(int(i // nH), int(i % nH)) for i in np.argsort(gen, axis=None)[::-1][:6]]
        if not hasattr(m, "model"):                              # GPT-2 stub: skip controlled OV
            out[tag] = rec; print(f"[{tag}] induction acc={ind_acc:.2f} (controlled-OV skipped)", flush=True)
            del m, tok; gc.collect(); continue
        WE = m.get_input_embeddings().weight
        WU = m.get_output_embeddings().weight.float()
        fn = m.model.norm
        is_rms = "rms" in type(fn).__name__.lower()
        gamma = fn.weight.detach().float()
        if "gemma" in (getattr(cm, "model_type", "") or "").lower():
            gamma = 1.0 + gamma
        toks = torch.tensor(rng.choice(pl, size=NSAMP, replace=False), device=dev)
        X = WE[toks].float()                                      # [N, d] token embeddings
        for (l, h) in topheads:
            attn = m.model.layers[l].self_attn
            kv = h // group
            Wv = attn.v_proj.weight[kv * hd:(kv + 1) * hd, :].float()        # [hd, d]
            Wo = attn.o_proj.weight[:, h * hd:(h + 1) * hd].float()          # [d, hd]
            val = X @ Wv.T                                                    # [N, hd]
            ov = val @ Wo.T                                                   # [N, d]
            if not is_rms:
                ov = ov - ov.mean(-1, keepdim=True)
            ov = ov * gamma
            g = ov @ WU.T                                                     # [N, V]
            g = g - g.mean(1, keepdim=True); g = torch.relu(g)
            tot = g[:, toks].sum(1) + 1e-9
            self_b = g.gather(1, toks[:, None]).squeeze(1)
            ratio = float((self_b / tot).mean())
            rec["heads"].append({"layer": l, "head": h, "qk": round(float(gen[l, h]), 3),
                                 "ov_activation": round(float(ovmat[l, h]), 3),
                                 "ov_controlled": round(float(np.clip(4 * ratio - 1, -1, 1)), 3)})
        out[tag] = rec
        print(f"[{tag}] tied={tied}  induction-behaviour acc={ind_acc:.2f}  "
              f"top-QK controlled-OV: " + ", ".join(f"L{d['layer']}H{d['head']}(qk{d['qk']:.2f} ovAct{d['ov_activation']:+.2f} ovCtl{d['ov_controlled']:+.2f})" for d in rec["heads"][:4]), flush=True)
        del m, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(OUTDIR, exist_ok=True)
    prev = f"{OUTDIR}/sanity_ov.json"
    if os.path.exists(prev):
        p = json.load(open(prev)); p.update(out); out = p
    json.dump(out, open(prev, "w"), indent=2)
    print(f"DONE -> {prev}", flush=True)


if __name__ == "__main__":
    main()
