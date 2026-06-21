"""Behavioral next-step accuracy for Llama-3.1-8B (same protocol as accuracy.py),
so it can be compared to the Gemma/Qwen curves in runs/accuracy/accuracy.json.
"""
from __future__ import annotations
import os, json
from dataclasses import replace

from config import get_config
import graph as G
from accuracy import eval_model, summarize

LLAMA = "NousResearch/Meta-Llama-3.1-8B"


def main():
    cfg = replace(get_config("gemma_qwen"), n_walks=40)
    graph = G.build_grid_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    words = cfg.words()
    print(f"[llama-acc] evaluating {LLAMA}", flush=True)
    recs, nuniq = eval_model(LLAMA, walks, graph, words, cfg)
    rows = summarize(recs)
    os.makedirs("/root/cmrun/llama", exist_ok=True)
    json.dump({"llama": {"first_token_unique": nuniq, "by_context": rows}},
              open("/root/cmrun/llama/llama_accuracy.json", "w"), indent=2)
    for r in rows:
        print(f"  ctx~{r['ctx']:>4}  acc={r['accuracy']:.3f}  nbr_mass={r['neighbor_mass']:.3f}  "
              f"KL={r['kl']:.3f}  exact={r['exact']:.3f}", flush=True)
    print("[llama-acc] DONE", flush=True)


if __name__ == "__main__":
    main()
