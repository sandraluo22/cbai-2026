"""Induction-head finding on Llama / Gemma / Qwen, and whether the SAME heads carry
the in-context graph task (the mechanism question).

Two per-(layer, head) scores, from eager attention weights:
  generic : standard induction score on a repeated random-token sequence -- mean
            attention from a 2nd-copy position i to i-L+1 (the token that FOLLOWED the
            current token's previous occurrence). Identifies the model's induction heads.
  task    : on an in-context walk, mean attention from a node's readout token to the
            tokens immediately AFTER all previous occurrences of the same node ("what
            followed this node before") -- the graph-induction signal for next-step.

Per model we ask: do task-induction heads = generic induction heads (a shared,
repurposed mechanism)?  Across models: the relative-depth distribution of induction
heads (head indices can't match across architectures, so depth + the generic<->task
correspondence are the cross-model comparison).

Two-pass friendly: MODELS_FILTER + HF_HOME let Gemma (/root/hf) and Llama/Qwen
(/workspace/hf) accumulate into one JSON. PRESET=smoke uses distilgpt2 on CPU.

Env: PRESET MODELS_FILTER GRAPH(days) NWALKS(6) WLEN(120) GLEN(50) NSEQ(5) OUTDIR DEVICE
Out: <OUTDIR>/induction.json  and  <OUTDIR>/induction_heads.pdf
"""
from __future__ import annotations
import os, json, gc
from collections import defaultdict
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

GRAPH = os.environ.get("GRAPH", "days")
GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
NWALKS = int(os.environ.get("NWALKS", "6"))
WLEN   = int(os.environ.get("WLEN", "120"))
GLEN   = int(os.environ.get("GLEN", "25"))     # block length: random tokens per copy
NREP   = int(os.environ.get("NREP", "4"))      # number of repeats of the block
NSEQ   = int(os.environ.get("NSEQ", "10"))     # examples to average over
EXCL_FREQ = float(os.environ.get("EXCL_FREQ", "0.02"))   # drop most-frequent (low-id) fraction
EXCL_RARE = float(os.environ.get("EXCL_RARE", "0.10"))   # drop least-frequent (high-id) fraction
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head" if PRESET != "smoke" else "runs/smoke_induction")


def load_eager(tag, hf, mirror, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, cfg.dtype)
    def _load(name):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
        model.to(cfg.device).eval()
        return model, tok
    try:
        return _load(hf)
    except Exception as e:
        if mirror:
            print(f"[{tag}] {hf} unavailable ({type(e).__name__}); mirror {mirror}", flush=True)
            return _load(mirror)
        raise


def _token_pool(tok):
    """Candidate tokens, excluding special/added tokens and the most/least frequent
    (BPE id-rank is the frequency proxy: low id ~ frequent, high id ~ rare/unused)."""
    V = int(getattr(tok, "vocab_size", None) or len(tok))
    special = set(tok.all_special_ids or [])
    try:
        special |= set(tok.get_added_vocab().values())
    except Exception:
        pass
    ids = [i for i in range(V) if i not in special]                       # already id-sorted
    lo = int(EXCL_FREQ * len(ids)); hi = int((1 - EXCL_RARE) * len(ids))
    return np.array(ids[lo:hi])


@torch.no_grad()
def generic_induction(model, tok, dev):
    """Induction score per (layer, head): stimulus = GLEN random tokens (excluding most/
    least frequent) repeated NREP times with SOS prepended; for every token in copies
    2..NREP, the attention it sends to the position right after the previous copy of that
    same token (target = p - GLEN + 1); averaged over the pattern, then over NSEQ examples.
    Returns (score[L,H], last_atts) -- last_atts kept so the top head's PATTERN can be drawn."""
    pool = _token_pool(tok)
    rng = np.random.default_rng(0)
    sos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    L = GLEN
    score = None; cnt = 0; last = None
    for _ in range(NSEQ):
        r = rng.choice(pool, size=L, replace=False).tolist()
        ids = torch.tensor([[sos] + r * NREP], device=dev)               # SOS + NREP copies
        atts = [a[0].float().cpu().numpy() for a in model(input_ids=ids, output_attentions=True).attentions]
        if score is None:
            score = np.zeros((len(atts), atts[0].shape[0]))
        for l, A in enumerate(atts):                                      # A: [H, S, S]
            for p in range(L + 1, NREP * L + 1):                          # all tokens in copies 2..NREP
                score[l] += A[:, p, p - L + 1]                            # attn to post-previous-copy target
        cnt += (NREP - 1) * L; last = atts
    return score / max(cnt, 1), last


@torch.no_grad()
def task_induction(model, tok, walks, dev):
    """Returns (score[L,H], last_atts, (readout_rows, succ_cols)) -- the row/col index
    pairs (current-node readout -> prior-occurrence successor) of the LAST walk, so the
    top head's attention map can be plotted with the induction targets overlaid."""
    score = None; cnt = 0; last = None; pairs_r = []; pairs_c = []
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes
        atts = [a[0].float().cpu().numpy() for a in model(input_ids=ids, output_attentions=True).attentions]
        if score is None:
            score = np.zeros((len(atts), atts[0].shape[0]))              # [L, H]
        succ_by_node = defaultdict(list); pr, pc = [], []
        for s in range(len(nodes) - 1):
            nd = nodes[s]; prev = succ_by_node[nd]
            if prev:
                p_read = spans[s][-1]
                for l in range(len(atts)):
                    score[l] += atts[l][:, p_read, prev].sum(axis=1)
                cnt += 1
                for q in prev:
                    pr.append(p_read); pc.append(q)
            succ_by_node[nd].append(spans[s + 1][0])
        last = atts; pairs_r, pairs_c = pr, pc
        if wk is not walks[-1]:
            del atts; gc.collect()
    return score / max(cnt, 1), last, (np.array(pairs_r), np.array(pairs_c))


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH],
                  walk_length=WLEN, n_walks=NWALKS, device=dev)
    graph = G.build_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    os.makedirs(OUTDIR, exist_ok=True)

    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading (eager attn)", flush=True)
        model, tok = load_eager(tag, hf, mirror, cfg)
        gen, gen_last = generic_induction(model, tok, dev)
        tsk, tsk_last, (pr, pc) = task_induction(model, tok, walks, dev)
        nL, nH = gen.shape
        g, t = gen.flatten(), tsk.flatten()
        corr = float(np.corrcoef(g, t)[0, 1]) if g.std() > 0 and t.std() > 0 else float("nan")
        def top(M, k=5):
            idx = np.argsort(M, axis=None)[::-1][:k]
            return [{"layer": int(i // nH), "head": int(i % nH), "score": float(M.flatten()[i]),
                     "rel_depth": round((i // nH) / (nL - 1), 3)} for i in idx]
        out["models"][tag] = {"n_layers": nL, "n_heads": nH,
                              "generic": gen.tolist(), "task": tsk.tolist(),
                              "corr_generic_task": corr,
                              "top_generic": top(gen), "top_task": top(tsk)}
        tg, tt = top(gen)[0], top(tsk)[0]
        # save the top heads' ATTENTION MAPS so the pattern (not just the scalar) is shown
        np.savez_compressed(f"{OUTDIR}/sample_{tag}.npz",
                            gen_map=gen_last[tg["layer"]][tg["head"]].astype(np.float16),
                            gen_LH=np.array([tg["layer"], tg["head"]]),
                            gen_len=np.array([GLEN]), gen_nrep=np.array([NREP]),
                            task_map=tsk_last[tt["layer"]][tt["head"]].astype(np.float16),
                            task_LH=np.array([tt["layer"], tt["head"]]),
                            task_pr=pr, task_pc=pc)
        print(f"[{tag}] {nL}L x {nH}H  corr(generic,task)={corr:+.2f}  "
              f"top generic L{tg['layer']}H{tg['head']}={tg['score']:.2f} (d{tg['rel_depth']})  "
              f"top task L{tt['layer']}H{tt['head']}={tt['score']:.2f} (d{tt['rel_depth']})", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev_path = f"{OUTDIR}/induction.json"
    if os.path.exists(prev_path):
        prev = json.load(open(prev_path)).get("models", {})
        prev.update(out["models"]); out["models"] = prev
    json.dump(out, open(prev_path, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/induction_heads.pdf")
    print(f"DONE -> {prev_path} + induction_heads.pdf", flush=True)


def make_fig(out, path, outdir=None):
    import glob
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    outdir = outdir or os.path.dirname(path)
    samples = {}
    for f in glob.glob(f"{outdir}/sample_*.npz"):
        samples[os.path.basename(f)[len("sample_"):-len(".npz")]] = dict(np.load(f))
    with PdfPages(path) as pdf:
        # per-model: scalar summary (generic heatmap | task heatmap | scatter)
        for m in models:
            r = out["models"][m]
            gen = np.array(r["generic"]); tsk = np.array(r["task"])
            fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
            for a, (M, ttl) in zip(ax[:2], [(gen, "generic induction"), (tsk, "task induction")]):
                im = a.imshow(M, aspect="auto", origin="lower", cmap="magma",
                              vmin=0, vmax=max(0.1, float(M.max())))
                a.set_xlabel("head"); a.set_ylabel("layer"); a.set_title(f"{m}  {ttl}", fontsize=10)
                fig.colorbar(im, ax=a, fraction=0.046)
            ax[2].scatter(gen.flatten(), tsk.flatten(), s=12, alpha=.5)
            ax[2].set_xlabel("generic induction"); ax[2].set_ylabel("task induction")
            ax[2].set_title(f"{m}  per-head  (corr={r['corr_generic_task']:+.2f})", fontsize=10)
            fig.suptitle(f"{m}: induction scores per (layer, head)  [{out['graph']}]", fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # ATTENTION PATTERN of the top heads (the actual stripe, not a scalar)
            if m in samples:
                s = samples[m]
                fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
                # generic: repeated-random sequence; induction target line key = query - L + 1
                gm = s["gen_map"].astype(np.float32); L = int(s["gen_len"][0]); gl, gh = s["gen_LH"]
                nr = int(s["gen_nrep"][0]) if "gen_nrep" in s else 2
                ax[0].imshow(gm, cmap="viridis", aspect="auto")
                qq = np.arange(L + 1, nr * L + 1)
                ax[0].plot(qq - L + 1, qq, ls="--", lw=1, color="red", alpha=.8,
                           label="induction target (key = query−L+1)")
                ax[0].set_xlabel("key position"); ax[0].set_ylabel("query position")
                ax[0].set_title(f"{m} GENERIC: head L{gl}H{gh} on repeated-random seq\n"
                                "bright cells on the red line = induction stripe", fontsize=9)
                ax[0].legend(fontsize=7, loc="upper right")
                # task: walk; overlay the (readout -> prior successor) target cells
                tm = s["task_map"].astype(np.float32); tl, th = s["task_LH"]
                ax[1].imshow(tm, cmap="viridis", aspect="auto")
                ax[1].scatter(s["task_pc"], s["task_pr"], s=14, facecolors="none",
                              edgecolors="red", lw=.7, label="prior-successor targets")
                ax[1].set_xlabel("key position"); ax[1].set_ylabel("query position")
                ax[1].set_title(f"{m} TASK: head L{tl}H{th} on a walk\n"
                                "red rings = current-node → prior-occurrence successors", fontsize=9)
                ax[1].legend(fontsize=7, loc="upper right")
                fig.suptitle(f"{m}: attention PATTERN of the top induction heads", fontsize=12)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # cross-model summary: relative depth of top induction heads
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for m in models:
            r = out["models"][m]
            ax[0].scatter([r["corr_generic_task"]], [m], s=60)
            for h in r["top_generic"]:
                ax[1].scatter(h["rel_depth"], m, color="tab:blue", s=40, alpha=.7)
            for h in r["top_task"]:
                ax[1].scatter(h["rel_depth"], m, color="tab:red", marker="x", s=50)
        ax[0].set_xlabel("corr(generic, task) per head"); ax[0].set_title("does the task ride on induction heads?", fontsize=10)
        ax[0].set_xlim(-0.2, 1.05); ax[0].axvline(0, color=".8", lw=.6)
        ax[1].set_xlabel("relative layer depth"); ax[1].set_xlim(0, 1)
        ax[1].set_title("top-5 heads' depth  (blue=generic, red x=task)", fontsize=10)
        fig.suptitle("Cross-model: is induction the shared mechanism, at similar depth?", fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
