"""Behavioral sanity check: does each model predict valid GRAPH NEIGHBORS as the
next word, and does that improve with context (the in-context-learning curve)?

At every transition node_s -> node_{s+1} in a walk we read the model's next-token
distribution, restrict it to the 16 candidate node-words (their first subword
token), renormalize, and score against the ground truth that the next node must
be a neighbor of node_s:
  - accuracy      : argmax candidate is a neighbor of the current node
  - neighbor_mass : probability mass on valid neighbors (chance = avg_deg/16)
  - kl            : KL(uniform-over-neighbors || model)   (0 = perfect)
  - exact         : prob on the actually-sampled next node
Reported per context-length bin. Models run sequentially (free Gemma before
downloading Qwen) to respect the volume quota.
"""
from __future__ import annotations
import os, json, gc, shutil
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

CHECKS = [(1, 15, 10), (15, 50, 30), (50, 150, 100), (150, 500, 300), (500, 1001, 1000)]


@torch.no_grad()
def eval_model(name, walks, graph, words, cfg):
    model, tok = M.load_model(name, cfg)
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    n_uniq = len(set(cand))
    cand_t = torch.tensor(cand, device=cfg.device)
    recs = []  # context, neighbor_mass, acc, kl, exact
    for wk in walks:
        enc = tok(wk.text, add_special_tokens=True, return_tensors="pt")
        ids = enc["input_ids"].to(cfg.device)
        spans = resolve_token_spans(tok, wk)
        logits = model(input_ids=ids).logits[0]               # [seq, vocab]
        pos = torch.tensor([spans[s + 1][0] - 1 for s in range(len(wk.nodes) - 1)],
                           device=cfg.device)
        sub = logits[pos][:, cand_t].float()                  # [T, 16]
        P = torch.softmax(sub, dim=1).cpu().numpy()           # renorm over 16 cands
        for s in range(len(wk.nodes) - 1):
            p = P[s]
            n = wk.nodes[s]; nb = graph.neighbors(n); deg = len(nb)
            mass = float(p[nb].sum())
            acc = int(int(p.argmax()) in nb)
            kl = float(np.sum((1.0 / deg) * np.log((1.0 / deg) / np.clip(p[nb], 1e-9, None))))
            recs.append((s + 1, mass, acc, kl, float(p[wk.nodes[s + 1]])))
    del model, tok; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(recs), n_uniq


def summarize(recs):
    rows = []
    for lo, hi, label in CHECKS:
        m = (recs[:, 0] >= lo) & (recs[:, 0] < hi)
        if m.sum() < 3:
            continue
        rows.append({"ctx": label, "n": int(m.sum()),
                     "accuracy": round(float(recs[m, 2].mean()), 4),
                     "neighbor_mass": round(float(recs[m, 1].mean()), 4),
                     "kl": round(float(recs[m, 3].mean()), 4),
                     "exact": round(float(recs[m, 4].mean()), 4)})
    return rows


def hf_cache_dir(name):
    hf = os.environ.get("HF_HOME", "/workspace/hf")
    return os.path.join(hf, "hub", "models--" + name.replace("/", "--"))


def main():
    from dataclasses import replace
    cfg = replace(get_config("gemma_qwen"), n_walks=40, out_dir="/root/cmrun")
    graph = G.build_grid_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    words = cfg.words()
    avg_deg = np.mean([len(graph.neighbors(n)) for n in range(graph.n_nodes)])
    print(f"[acc] {cfg.n_walks} walks x {cfg.walk_length} steps | "
          f"chance neighbor_mass = avg_deg/16 = {avg_deg/16:.3f}", flush=True)

    out = {"chance_neighbor_mass": round(float(avg_deg / 16), 4)}
    # Gemma first (cached), then free it and download Qwen
    for tag, name, do_free in (("gemma", cfg.model_a, True),
                               ("qwen", cfg.model_b, False)):
        print(f"[acc] evaluating {tag}: {name}", flush=True)
        recs, n_uniq = eval_model(name, walks, graph, words, cfg)
        rows = summarize(recs)
        out[tag] = {"first_token_unique": n_uniq, "by_context": rows}
        print(f"[acc] {tag}: {n_uniq}/16 candidate words have unique first tokens", flush=True)
        for r in rows:
            print(f"      ctx~{r['ctx']:>4} (n={r['n']:>5})  acc={r['accuracy']:.3f}  "
                  f"nbr_mass={r['neighbor_mass']:.3f}  KL={r['kl']:.3f}  "
                  f"exact={r['exact']:.3f}", flush=True)
        if do_free:
            print(f"[acc] freeing {tag} weights", flush=True)
            shutil.rmtree(hf_cache_dir(name), ignore_errors=True)

    os.makedirs(f"{cfg.out_dir}/accuracy", exist_ok=True)
    json.dump(out, open(f"{cfg.out_dir}/accuracy/accuracy.json", "w"), indent=2)
    print("[acc] wrote /root/cmrun/accuracy/accuracy.json", flush=True)


if __name__ == "__main__":
    main()
