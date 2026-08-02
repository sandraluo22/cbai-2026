"""Replay basin-rollout streams into FRESH contexts, in original and occupancy-matched
SHUFFLED order, plus record end-state predictives for basin analysis in full
predictive-distribution space.

For each of the R basin rollouts (streams from out_basin/basin.json):
  fresh    : [BOS] + stream (600 tokens) -> late deep node-means + final predictive
  shuffled : [BOS] + permuted stream (same tokens, order destroyed) -> same measures
Also: final predictive mass on the rollout's trap-pair transition (given prev token).

Out: out_basin/replay.json + replay_nm.npz
"""
from __future__ import annotations
import os, sys, json
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import models as M

DEVICE = os.environ.get("DEVICE", "cuda")
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
BASIN = os.environ.get("BASIN", "/root/test-1/out_basin")
N = 16
DEEP = list(range(24, 32))


@torch.no_grad()
def main():
    b = json.load(open(os.path.join(BASIN, "basin.json")))
    R = b["R"]
    WORDS = b["words"]
    streams = b["main_streams"]
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
    model = tok = None
    for nm in MODEL_CANDS:
        try:
            model, tok = M.load_model(nm, cfg); break
        except Exception as e:
            print(f"failed {nm}: {e}", flush=True)
    cm = model.config
    blocks = M._decoder_blocks(model)
    bos = tok.bos_token_id
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
    cand_t = torch.tensor(cand, device=DEVICE)

    rng = np.random.default_rng(9)
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o2): grabbed[L] = (o2[0] if isinstance(o2, tuple) else o2).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in DEEP]
    NM = np.zeros((R, 2, len(DEEP), N, cm.hidden_size), np.float16)
    finals = {"fresh": [], "shuffled": []}
    trapmass = {"fresh": [], "shuffled": []}
    try:
        for r in range(R):
            for vi, variant in enumerate(("fresh", "shuffled")):
                seq = list(streams[r])
                if variant == "shuffled":
                    seq = [seq[i] for i in rng.permutation(len(seq))]
                ids = torch.tensor([[bos] + [cand[x] for x in seq]], device=DEVICE)
                grabbed.clear()
                out = model(input_ids=ids)
                pr = torch.softmax(out.logits[0, -1, cand_t].float(), -1).cpu().numpy()
                finals[variant].append(np.round(pr, 4).tolist())
                a, bb = b["main_traps"][r][0]
                pv = seq[-1]
                tm = 0.0
                if pv == a: tm = float(pr[bb])
                elif pv == bb: tm = float(pr[a])
                else: tm = float(pr[a] + pr[bb])
                trapmass[variant].append(round(tm, 4))
                pos = list(range(1 + 300, 1 + len(seq)))
                nds = seq[300:]
                for li, L in enumerate(DEEP):
                    hh = grabbed[L][0][pos].float().cpu().numpy()
                    sums = np.zeros((N, cm.hidden_size)); cnts = np.zeros(N)
                    np.add.at(sums, nds, hh); np.add.at(cnts, nds, 1)
                    NM[r, vi, li] = (sums / np.maximum(cnts, 1)[:, None]).astype(np.float16)
            if (r + 1) % 10 == 0:
                print(f"replay {r+1}/{R}", flush=True)
    finally:
        for h in handles:
            h.remove()
    np.savez_compressed(os.path.join(BASIN, "replay_nm.npz"), nm=NM,
                        deep_layers=np.array(DEEP))
    json.dump({"finals": finals, "trapmass": trapmass},
              open(os.path.join(BASIN, "replay.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
