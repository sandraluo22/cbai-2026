"""Cross-MODEL dueling contexts: instances are DIFFERENT models (Llama-3.1-8B,
Qwen3-8B-Base, Gemma-2-9b), each primed on its own graph over the same word stream,
then cycling generation; every sampled word is appended to every instance's context
(tokenized per model; all node words are single tokens in all three tokenizers).

All requested experiments run in ONE process because the 30GB disk cannot hold three
checkpoints: each model is loaded to VRAM and its hub cache deleted immediately after.

Env: EXPTS = JSON list of {"out": dir, "words": comma-list,
                           "instances": [[model_tag, graph, topk], ...]}
     NPAIRS(8) CTX(1000) TGEN(600) TEMP(1.0) SEED(0) CM_SRC DEVICE
Out per experiment: nodemeans_dueling.npz (ctx name = "<tag>-<graph>", per-ctx layer
counts in nlayers_<ctx>; fresh-<tag> contexts included) + gen_log.json
"""
from __future__ import annotations
import os, sys, json, time, shutil
from dataclasses import replace
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("CM_SRC"), os.path.join(_here, "..", "cross-model", "src"),
             os.path.join(_here, "cmsrc")):
    if cand and os.path.isfile(os.path.join(cand, "graph.py")):
        sys.path.insert(0, cand); break

from config import get_config
import graph as G
import models as M
from run_pair import GKW, build_named, adjacency_matrix

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None),
           "Gemma": ("unsloth/gemma-2-9b", "google/gemma-2-9b")}  # 4-shard mirror first: fits disk
EXPTS = json.loads(os.environ["EXPTS"])
NPAIRS = int(os.environ.get("NPAIRS", "8"))
CTX = int(os.environ.get("CTX", "1000"))
TGEN = int(os.environ.get("TGEN", "600"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
DEVICE = os.environ.get("DEVICE", "cuda")
BASE_WIN = 300
WINDOWS = {"base": ("prefix", CTX - BASE_WIN, CTX), "joint_early": ("joint", 0, 100),
           "joint_mid": ("joint", 100, 300), "joint_late": ("joint", 300, TGEN)}
FRESH_WINDOWS = {"early": (0, 100), "mid": (100, 300), "late": (300, TGEN)}


def free_hub_cache(name):
    p = os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}")
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print(f"freed cache {p}", flush=True)
    xet = os.path.expanduser("~/.cache/huggingface/xet")
    if os.path.isdir(xet):                       # chunk cache can eat >10GB on a 30GB disk
        shutil.rmtree(xet, ignore_errors=True)


def load_all_models(tags, cfg):
    out = {}
    for tag in tags:
        hf, mirror = ALLSPEC[tag]
        model = tok = None
        for name in ([hf] + ([mirror] if mirror else [])):
            try:
                print(f"[{tag}] loading {name}", flush=True)
                model, tok = M.load_model(name, cfg)
                free_hub_cache(name)
                break
            except Exception as e:
                print(f"  failed: {e}", flush=True)
                free_hub_cache(name)          # drop partial downloads before next attempt
        assert model is not None, f"could not load {tag}"
        out[tag] = (model, tok)
        print(f"[{tag}] ready ({model.config.num_hidden_layers}L, "
              f"{model.config.hidden_size}d)", flush=True)
    return out


@torch.no_grad()
def capture_context(model, blocks, nL, ids_row, node_windows, nsum, ncnt, dev):
    """One full forward; accumulate node means for each (window -> positions/nodes)."""
    grabbed = {}
    def mk(L):
        def hh(_m, _i, o): grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hh
    handles = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    try:
        fids = torch.tensor([ids_row], device=dev)
        try:
            model(input_ids=fids, logits_to_keep=1)
        except TypeError:
            model(input_ids=fids)
        for wname, (pos, nds) in node_windows.items():
            for L in range(nL):
                rows_h = grabbed[L][0][pos].float().cpu().numpy()
                np.add.at(nsum[wname][L], nds, rows_h)
            np.add.at(ncnt[wname], nds, 1.0)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_experiment(spec, MODELS, cfg):
    outdir = spec["out"]
    os.makedirs(outdir, exist_ok=True)
    words = spec["words"].split(",")
    insts = [(t, g, int(k)) for t, g, k in spec["instances"]]
    names = [f"{t}-{g}" for t, g, _ in insts]
    K = len(insts)
    n = 16
    print(f"=== EXPT {outdir}: {names} ===", flush=True)

    graphs, walks = {}, {}
    for _, g, _ in insts:
        if g not in graphs:
            gr = build_named(g, cfg)
            gr.words = list(words)
            graphs[g] = gr
            walks[g] = G.generate_walks(gr, replace(cfg, **GKW[g]))
    adjs = {g: adjacency_matrix(graphs[g]).astype(bool) for g in graphs}

    cand = {}                                  # per model tag: node -> token id
    for t, _, _ in insts:
        tok = MODELS[t][1]
        ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
        assert all(len(x) == 1 for x in ids), f"{t}: multi-token word"
        cand[t] = [x[0] for x in ids]

    # prefill each instance (its model, its graph walks); Qwen3 has NO bos token
    state = {}
    prefill_rows = {}
    bos_of = {t: ([MODELS[t][1].bos_token_id] if MODELS[t][1].bos_token_id is not None
                  else []) for t, _, _ in insts}
    for ki, (t, g, _) in enumerate(insts):
        model, tok = MODELS[t]
        rows = [bos_of[t] + [cand[t][nd] for nd in wk.nodes] for wk in walks[g]]
        prefill_rows[names[ki]] = rows
        ids = torch.tensor(rows, device=DEVICE)
        try:
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(input_ids=ids, use_cache=True)
        state[names[ki]] = {"past": out.past_key_values,
                            "logits": out.logits[:, -1, :]}
        print(f"  prefill {names[ki]} done", flush=True)

    rng = np.random.default_rng(SEED + 1)
    joint = np.zeros((NPAIRS, TGEN), np.int32)
    genlog = {p: [] for p in range(NPAIRS)}
    prev0 = {names[ki]: [walks[g][p].nodes[-1] for p in range(NPAIRS)]
             for ki, (t, g, _) in enumerate(insts)}
    t0 = time.time()
    for t_step in range(TGEN):
        ki = t_step % K
        t, g, topk = insts[ki]
        nm = names[ki]
        cand_t = torch.tensor(cand[t], device=DEVICE)
        lg = state[nm]["logits"][:, cand_t].float()
        probs = torch.softmax(lg / TEMP, dim=-1).cpu().numpy()
        step_nodes = np.zeros(NPAIRS, np.int64)
        for p in range(NPAIRS):
            pv = prev0[nm][p] if t_step == 0 else int(joint[p, t_step - 1])
            pp = probs[p].copy()
            if topk > 0:
                pp[np.argsort(pp)[:-topk]] = 0.0
            node = int(rng.choice(n, p=pp / pp.sum()))
            joint[p, t_step] = node
            step_nodes[p] = node
            rec = {"t": t_step, "gen": nm, "node": node, "word": words[node], "prev": pv}
            for gg in adjs:
                rec[f"{gg}_valid"] = bool(adjs[gg][pv, node])
            genlog[p].append(rec)
        for kj, (t2, g2, _) in enumerate(insts):     # append word to every instance
            model2 = MODELS[t2][0]
            inp = torch.tensor([cand[t2][int(x)] for x in step_nodes],
                               device=DEVICE)[:, None]
            out = model2(input_ids=inp, past_key_values=state[names[kj]]["past"],
                         use_cache=True)
            state[names[kj]]["past"] = out.past_key_values
            state[names[kj]]["logits"] = out.logits[:, -1, :]
        if (t_step + 1) % 200 == 0:
            print(f"  joint {t_step+1}/{TGEN} ({time.time()-t0:.1f}s)", flush=True)
    for nm in list(state):
        del state[nm]
    torch.cuda.empty_cache()

    # ---- capture: primed contexts + per-model fresh contexts --------------
    save = {"ctx_names": np.array(names), "words": np.array(words),
            "fresh_names": np.array([f"fresh-{t}" for t in
                                     dict.fromkeys(t for t, _, _ in insts)])}
    for ki, (t, g, _) in enumerate(insts):
        save[f"adjacency_{names[ki]}"] = adjacency_matrix(graphs[g])
        save[f"coords_{names[ki]}"] = np.array(graphs[g].coords, float)
    t0 = time.time()
    for ki, (t, g, _) in enumerate(insts):
        model, tok = MODELS[t]
        cm = model.config
        blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers
        nm = names[ki]
        nsum = {w: {L: np.zeros((n, cm.hidden_size)) for L in range(nL)} for w in WINDOWS}
        ncnt = {w: np.zeros(n) for w in WINDOWS}
        for p in range(NPAIRS):
            row = prefill_rows[nm][p] + [cand[t][nd] for nd in joint[p]]
            nodes_all = walks[g][p].nodes + list(joint[p])
            o0 = len(bos_of[t])                  # word s sits at token o0+s
            node_windows = {}
            for wname, (seg, lo, hi) in WINDOWS.items():
                off = 0 if seg == "prefix" else CTX
                node_windows[wname] = (list(range(o0 + off + lo, o0 + off + hi)),
                                       nodes_all[off + lo:off + hi])
            capture_context(model, blocks, nL, row, node_windows, nsum, ncnt, DEVICE)
        for wname in WINDOWS:
            cnt = np.maximum(ncnt[wname], 1.0)
            for L in range(nL):
                save[f"{nm}_{wname}_layer_{L}"] = \
                    (nsum[wname][L] / cnt[:, None]).astype(np.float16)
            save[f"{nm}_{wname}_ncnt"] = ncnt[wname]
        save[f"nlayers_{nm}"] = np.array([nL])
        print(f"  captured {nm} ({time.time()-t0:.1f}s)", flush=True)

    for t in dict.fromkeys(t for t, _, _ in insts):   # fresh controls per model
        model, tok = MODELS[t]
        cm = model.config
        blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers
        pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
        o0 = len(pre)
        nm = f"fresh-{t}"
        nsum = {w: {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}
                for w in FRESH_WINDOWS}
        ncnt = {w: np.zeros(n) for w in FRESH_WINDOWS}
        for p in range(NPAIRS):
            row = pre + [cand[t][nd] for nd in joint[p]]
            node_windows = {w: (list(range(o0 + lo, o0 + hi)), list(joint[p][lo:hi]))
                            for w, (lo, hi) in FRESH_WINDOWS.items()}
            capture_context(model, blocks, nL, row, node_windows, nsum, ncnt, DEVICE)
        for wname in FRESH_WINDOWS:
            cnt = np.maximum(ncnt[wname], 1.0)
            for L in range(nL):
                save[f"{nm}_{wname}_layer_{L}"] = \
                    (nsum[wname][L] / cnt[:, None]).astype(np.float16)
        save[f"nlayers_{nm}"] = np.array([nL])
        print(f"  captured {nm} ({time.time()-t0:.1f}s)", flush=True)

    np.savez_compressed(os.path.join(outdir, "nodemeans_dueling.npz"), **save)
    log = {"instances": [[t, g, k] for t, g, k in insts], "npairs": NPAIRS, "ctx": CTX,
           "tgen": TGEN, "temp": TEMP, "seed": SEED, "words": words,
           "steps": {f"pair{p}": genlog[p] for p in range(NPAIRS)}}
    json.dump(log, open(os.path.join(outdir, "gen_log.json"), "w"))
    print(f"EXPT DONE -> {outdir}", flush=True)


def main():
    cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE,
                  n_walks=NPAIRS, walk_length=CTX, seed=SEED)
    tags = list(dict.fromkeys(t for spec in EXPTS for t, _, _ in spec["instances"]))
    MODELS = load_all_models(tags, cfg)
    for spec in EXPTS:
        run_experiment(spec, MODELS, cfg)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
