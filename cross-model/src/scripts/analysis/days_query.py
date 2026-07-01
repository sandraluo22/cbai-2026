"""Behavioural probe of the days semantic-conflict condition: after an in-context
random walk over the PERMUTED weekday ring, ask each model (in natural language)
"what day is after <d>?" and see whether it answers the IN-CONTEXT ring neighbour
or the pretrained natural next weekday.

Permuted ring node order = DAYS_PERMUTED = [Mon, Thu, Sun, Wed, Sat, Tue, Fri]; the
ring neighbours of a day are its +/-1 positions there, which differ from the natural
+1 weekday. So "after Monday": in-context -> Thursday / Friday; pretrained -> Tuesday.

Three probes per query day (7 days, averaged over walks), reading the next-token
distribution restricted to the 7 day-word first-subword tokens:
  continuation : <walk> ' <d>'                              -> in-context successor
  question     : <walk> + '... what day after <d>? ... is'  -> QA after the walk
  baseline     : same question with NO walk                 -> the pretrained answer

Runs on the pod (gated models via ungated mirrors). PRESET=smoke uses distilgpt2.
Env: PRESET NWALKS(8) WLEN(300) OUTDIR DEVICE
Out: <OUTDIR>/days_query.json  and  <OUTDIR>/days_query.pdf
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

from config import get_config, DAYS
import graph as G
import models as M

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
NWALKS = int(os.environ.get("NWALKS", "8"))
WLEN   = int(os.environ.get("WLEN", "300"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/v1/days/query" if PRESET != "smoke" else "runs/smoke_query")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception as e:
        if mirror:
            print(f"[{tag}] {hf} unavailable ({type(e).__name__}); mirror {mirror}", flush=True)
            return M.load_model(mirror, cfg)
        raise


def q_with(d):    return f"\n\nQuestion: What day of the week is after {d}?\nAnswer: The day after {d} is"
def q_base(d):    return f"Question: What day of the week is after {d}?\nAnswer: The day after {d} is"


@torch.no_grad()
def probe(model, tok, prompt, day_ids, dev):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
    logits = model(input_ids=ids).logits[0, -1]
    return torch.softmax(logits[day_ids].float(), dim=0).cpu().numpy()      # P over the 7 DAYS (DAYS order)


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=7,
                  word_set="days", walk_length=WLEN, n_walks=NWALKS, device=dev)
    graph = G.build_graph(cfg)
    words = graph.words                                # = DAYS_PERMUTED (node i -> words[i])
    walks = G.generate_walks(graph, cfg)
    os.makedirs(OUTDIR, exist_ok=True)

    # per query day: natural next weekday vs induced ring neighbours
    meta = {}
    for d in DAYS:
        rp = words.index(d)
        meta[d] = {"natural_next": DAYS[(DAYS.index(d) + 1) % 7],
                   "induced_fwd": words[(rp + 1) % 7],
                   "induced_bwd": words[(rp - 1) % 7]}

    out = {"permuted_ring": words, "meta": meta, "models": {}}
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        day_ids = torch.tensor([tok(" " + d, add_special_tokens=False)["input_ids"][0] for d in DAYS], device=dev)

        probes = {"baseline": {}, "question": {}, "continuation": {}}
        # baseline: no walk, once per day
        for d in DAYS:
            probes["baseline"][d] = probe(model, tok, q_base(d), day_ids, dev)
        # with-context: average over walks
        acc = {pk: {d: [] for d in DAYS} for pk in ("question", "continuation")}
        for wk in walks:
            for d in DAYS:
                acc["question"][d].append(probe(model, tok, wk.text + q_with(d), day_ids, dev))
                acc["continuation"][d].append(probe(model, tok, wk.text + f" {d}", day_ids, dev))
        for pk in ("question", "continuation"):
            for d in DAYS:
                probes[pk][d] = np.mean(acc[pk][d], axis=0)

        rec = {}
        for pk, pd in probes.items():
            per_day = {}
            for d in DAYS:
                p = pd[d]; m = meta[d]
                pn = float(p[DAYS.index(m["natural_next"])])
                pi = float(p[DAYS.index(m["induced_fwd"])] + p[DAYS.index(m["induced_bwd"])])
                arg = DAYS[int(p.argmax())]
                per_day[d] = {"P": {DAYS[i]: float(p[i]) for i in range(7)}, "argmax": arg,
                              "p_natural": pn, "p_induced": pi,
                              "is_induced": arg in (m["induced_fwd"], m["induced_bwd"]),
                              "is_natural": arg == m["natural_next"]}
            rec[pk] = {"per_day": per_day,
                       "mean_p_natural": float(np.mean([per_day[d]["p_natural"] for d in DAYS])),
                       "mean_p_induced": float(np.mean([per_day[d]["p_induced"] for d in DAYS])),
                       "frac_argmax_induced": float(np.mean([per_day[d]["is_induced"] for d in DAYS])),
                       "frac_argmax_natural": float(np.mean([per_day[d]["is_natural"] for d in DAYS]))}
            print(f"[{tag}/{pk}] mean P(induced)={rec[pk]['mean_p_induced']:.2f} "
                  f"P(natural)={rec[pk]['mean_p_natural']:.2f}  "
                  f"argmax induced={rec[pk]['frac_argmax_induced']:.0%} natural={rec[pk]['frac_argmax_natural']:.0%}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # merge with any prior run (so model-by-model passes accumulate)
    prev_path = f"{OUTDIR}/days_query.json"
    if os.path.exists(prev_path):
        prev = json.load(open(prev_path)).get("models", {})
        prev.update(out["models"]); out["models"] = prev
    json.dump(out, open(prev_path, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/days_query.pdf")
    print(f"DONE -> {OUTDIR}/days_query.json + days_query.pdf", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    probes = ["baseline", "question", "continuation"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5), squeeze=False)
        for col, m in enumerate(models):
            a = ax[0, col]; x = np.arange(len(probes)); w = 0.38
            ind = [out["models"][m][pk]["mean_p_induced"] for pk in probes]
            nat = [out["models"][m][pk]["mean_p_natural"] for pk in probes]
            a.bar(x - w/2, ind, w, label="P(in-context ring neighbour)", color="purple")
            a.bar(x + w/2, nat, w, label="P(natural next weekday)", color="green")
            for xi, pk in enumerate(probes):
                a.text(xi, max(ind[xi], nat[xi]) + .02,
                       f"argmax→ind {out['models'][m][pk]['frac_argmax_induced']:.0%}", ha="center", fontsize=7)
            a.set_xticks(x); a.set_xticklabels(probes, fontsize=8); a.set_ylim(0, 1.05)
            a.set_title(m, fontsize=11); a.set_ylabel("mean prob over the 7 query days")
            if col == 0:
                a.legend(fontsize=7, loc="upper center")
        fig.suptitle('"What day is after <d>?" after the permuted-ring walk:  '
                     "does the model follow the INDUCED ring (purple) or pretrained weekdays (green)?",
                     fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
