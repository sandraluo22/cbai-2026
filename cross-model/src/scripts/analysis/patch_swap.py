"""Activation patching to localize the SWAP: generate rollouts on the same walk with two
nodes' words exchanged (e.g. plant<->clock), then patch one attention head at a time
(clean O -> corrupted S) and see which head restores the ORIGINAL binding.

Clean run O: original word assignment. Corrupted run S: swap words at cells a,b.
Both use the SAME walk (identical cell sequence), so if the two swap-words are each a
single token the sequences align 1:1.

Metric M (predict the ORIGINAL word when about to enter cell a or b):
  at readouts entering cell b: +[logit(word_b) - logit(word_a)]
  at readouts entering cell a: +[logit(word_a) - logit(word_b)]
  M_O high (predicts original binding), M_S low. For each head h, patch S's o_proj-input
  head-slice with O's (all positions) and recompute M. restoration = (M_patch-M_S)/(M_O-M_S).
  Heads with restoration ~1 carry the swap.

Env: PRESET MODELS_FILTER SWAP("12-15") NWALKS(3) WLEN(320) CTXLO(80) INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/patch_swap_<a>_<b>.json + .pdf
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
from graph import Walk
import models as M
from models import resolve_token_spans

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
A, B = (int(x) for x in os.environ.get("SWAP", "12-15").split("-"))
NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "320"))
CTXLO = int(os.environ.get("CTXLO", "80"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/patch_swap")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads), False
    return block.attn.c_proj, cm.hidden_size // cm.n_head, True


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"swap": [A, B], "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4,
                      n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); words = graph.words
        wa, wb = words[A], words[B]
        swapped = list(words); swapped[A], swapped[B] = words[B], words[A]
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading (swap {A}:{wa} <-> {B}:{wb})", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        ta = tok(" " + wa, add_special_tokens=False)["input_ids"]
        tb = tok(" " + wb, add_special_tokens=False)["input_ids"]
        if len(ta) != 1 or len(tb) != 1:
            print(f"[{tag}] SKIP: swap words not single-token ({wa}:{len(ta)}, {wb}:{len(tb)})", flush=True)
            del model, tok; gc.collect(); continue
        ta, tb = ta[0], tb[0]

        # o_proj input capture / patch hooks
        zc = {}
        def mkz(L):
            def pre(_m, args): zc[L] = args[0].detach().clone()
            return pre
        patch = {"L": None, "src": None, "cols": None}
        def mkpatch(L):
            def pre(_m, args):
                if patch["L"] == L:
                    x = args[0].clone(); x[..., patch["cols"]] = patch["src"][..., patch["cols"]]
                    return (x,) + tuple(args[1:])
            return pre
        cap_h = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkz(L)) for L in range(nL)]
        patch_h = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkpatch(L)) for L in range(nL)]

        def metric(logits, reads):
            m = 0.0
            for pos, sign in reads:
                m += sign * float(logits[pos, tb] - logits[pos, ta])   # +sign when original word is wb
            return m

        M_O = M_S = 0.0; patched = np.zeros((nL, nH)); Ozs = []
        # ---- pass 1: for each walk, run O (capture z) and S (baseline), store O z-slices ----
        walk_data = []
        for wk in walks:
            Owk = wk
            Swk = Walk(walk_id=wk.walk_id, nodes=wk.nodes, words=[swapped[n] for n in wk.nodes])
            oid = tok(Owk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            sid = tok(Swk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, Owk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
            if oid.shape != sid.shape:
                print(f"[{tag}] walk {wk.walk_id} misaligned len; skip", flush=True); continue
            reads = []
            for s in range(len(nodes) - 1):
                nxt = nodes[s + 1]
                if nxt in (A, B) and cl[s] >= CTXLO:
                    pos = spans[s + 1][0] - 1
                    reads.append((pos, +1.0 if nxt == B else -1.0))
            if not reads:
                continue
            patch["L"] = None; zc.clear()
            lo = model(input_ids=oid).logits[0]; M_O += metric(lo, reads)
            Oz = {L: zc[L].clone() for L in range(nL)}
            ls = model(input_ids=sid).logits[0]; M_S += metric(ls, reads)
            walk_data.append((sid, reads, Oz))
        # ---- pass 2: per head, patch O->S at all positions, recompute M ----
        for (sid, reads, Oz) in walk_data:
            for L in range(nL):
                _, hd, _ = attn_proj(blocks[L], cm)
                patch["L"] = L; patch["src"] = Oz[L]
                for h in range(nH):
                    patch["cols"] = torch.arange(h * hd, (h + 1) * hd, device=dev)
                    lp = model(input_ids=sid).logits[0]
                    patched[L, h] += metric(lp, reads)
                patch["L"] = None
        for hh in cap_h + patch_h:
            hh.remove()

        denom = (M_O - M_S) if abs(M_O - M_S) > 1e-6 else 1.0
        restoration = (patched - M_S) / denom
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
        top = np.argsort(restoration, axis=None)[::-1][:8]
        rec = {"n_layers": nL, "n_heads": nH, "M_O": M_O, "M_S": M_S, "restoration": restoration.tolist(),
               "top_heads": [{"layer": int(i // nH), "head": int(i % nH), "restore": round(float(restoration.flatten()[i]), 3),
                              "qk": round(float(gen.flatten()[i]), 3), "dla": round(float(att.flatten()[i]), 3)} for i in top]}
        out["models"][tag] = rec
        print(f"[{tag}] M_O={M_O:+.1f} M_S={M_S:+.1f} | top patch head "
              f"L{rec['top_heads'][0]['layer']}H{rec['top_heads'][0]['head']} restore={rec['top_heads'][0]['restore']:+.2f} "
              f"(qk {rec['top_heads'][0]['qk']:.2f}, dla {rec['top_heads'][0]['dla']:+.2f})", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/patch_swap_{A}_{B}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/patch_swap_{A}_{B}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    a, b = out["swap"]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; R = np.array(r["restoration"])
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            v = max(0.2, float(np.nanpercentile(np.abs(R), 99)))
            im = ax[0].imshow(R, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ax[0].set_xlabel("head"); ax[0].set_ylabel("layer")
            ax[0].set_title(f"{m}: per-head patch restoration (O->S)\n1.0 = fully restores original binding", fontsize=9)
            fig.colorbar(im, ax=ax[0], fraction=.046)
            th = r["top_heads"]
            ax[1].barh(range(len(th))[::-1], [h["restore"] for h in th], color="tab:red")
            ax[1].set_yticks(range(len(th))[::-1]); ax[1].set_yticklabels([f"L{h['layer']}H{h['head']} (qk{h['qk']:.2f},dla{h['dla']:+.2f})" for h in th], fontsize=7)
            ax[1].set_xlabel("restoration"); ax[1].set_title(f"{m}: top swap-carrying heads", fontsize=9)
            fig.suptitle(f"{m}: patch head O->S, swap cells {a}<->{b} — which head moves the representation "
                         "back to the original grid?", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
