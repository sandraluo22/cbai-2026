"""Keep-only sufficiency of a head set, scored on the DOWNSTREAM next-node readout.

Keep the named heads live, mean-ablate every other attention head (MLPs intact), and read the next-node
prediction at each readout position:
    nbr_mass   softmax over the n node-words, probability mass on the true neighbours of the current node
               (continuous, moves under single-head ablation — the argmax hit-rate saturates)
    nbr_acc    argmax over those candidates is a true neighbour
    ent        entropy of the candidate distribution, to catch "recovery" that is really collapse onto
               one token (a keep-set can beat the intact model's margin while being far worse — see the
               retracted behavioural coordinate greedy)
Reported against BOTH ends: the intact model and the keep-nothing floor, plus random keep-sets of the
same size. recovered = (set - floor) / (full - floor).

RANDMODE=layer draws each control head at the same LAYER as the head it replaces — required when the
set is defined per-layer, since depth alone moves these metrics.

Env: GEN_MODEL(Llama) STRUCT(ring16) SETS("nm=L14H19+L16H3,nm2=...") NWALKS(6) WLEN(1200) CTXLO(800)
     NRAND(5) RANDMODE(all|layer) LOO("") SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/keep_set_nbr<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from cycle_head_circuit import prep_batch
from keep_only_circuit import cache_means, keep_hooks

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
STRUCT = os.environ.get("STRUCT", "ring16")
SETS = {s.split("=")[0]: s.split("=")[1].split("+") for s in os.environ.get("SETS", "").split(",") if "=" in s}
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
RANDMODE = os.environ.get("RANDMODE", "layer"); SEED = int(os.environ.get("SEED", "0"))
LOO = os.environ.get("LOO", "")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

STRUCTS = {"ring16": dict(graph_type="ring", ring_size=16),
           "ring32": dict(graph_type="ring", ring_size=32),
           "grid4x4": dict(graph_type="grid", grid_rows=4, grid_cols=4)}


def parse(x): return {(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x}


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    import cycle_head_circuit as chc; chc.CTXLO = CTXLO
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    rng = np.random.default_rng(SEED)

    cfg = replace(get_config("gemma_qwen"), **STRUCTS[STRUCT], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    walks = G.generate_walks(graph, cfg)                      # NON-lazy: uniform over neighbours
    batch, mask, readouts, cand, nbrs = prep_batch(tok, graph, walks, dev)
    nro = sum(len(r) for r in readouts)
    print(f"[{tag}] {STRUCT} nodes={graph.n_nodes} walks={NWALKS}x{WLEN} ctxlo={CTXLO} "
          f"readouts={nro}", flush=True)

    def measure():
        lg = model(input_ids=batch, attention_mask=mask).logits.float()
        mass = acc = ent = tot = 0.0
        for b, ro in enumerate(readouts):
            for pos, node in ro:
                p = torch.softmax(lg[b, pos][cand], 0)
                nb = list(nbrs[node])
                mass += float(p[nb].sum())
                acc += float(int(int(p.argmax()) in nb))
                ent += float(-(p * (p + 1e-12).log()).sum())
                tot += 1
        return {"nbr_mass": round(mass / tot, 4), "nbr_acc": round(acc / tot, 4), "ent": round(ent / tot, 4)}

    def run(keepset):
        hs = keep_hooks(blocks, cm, nL, nH, means, keepset, dev)
        r = measure()
        for h in hs: h.remove()
        return r

    full = measure()
    means = cache_means(model, blocks, cm, nL, batch, mask)
    floor = run(set())
    print(f"  full  {full}\n  floor {floor}   (every head mean-ablated)", flush=True)
    rows = {"full": full, "floor": floor}

    def recov(r, k):
        d = full[k] - floor[k]
        return round((r[k] - floor[k]) / d, 4) if abs(d) > 1e-9 else None

    print(f"\n{'set':<14}{'k':>3}{'nbr_mass':>10}{'nbr_acc':>9}{'ent':>7}{'recov_mass':>11}{'recov_acc':>10}")
    print(f"{'full':<14}{'-':>3}{full['nbr_mass']:10.4f}{full['nbr_acc']:9.4f}{full['ent']:7.3f}"
          f"{1.0:11.3f}{1.0:10.3f}")
    print(f"{'floor':<14}{0:>3}{floor['nbr_mass']:10.4f}{floor['nbr_acc']:9.4f}{floor['ent']:7.3f}"
          f"{0.0:11.3f}{0.0:10.3f}")
    allheads = {(l, h) for l in range(nL) for h in range(nH)}
    for nm, hl in SETS.items():
        ks = parse(hl); r = run(ks)
        r["k"] = len(ks); r["recov_mass"] = recov(r, "nbr_mass"); r["recov_acc"] = recov(r, "nbr_acc")
        rows[nm] = r
        print(f"{nm:<14}{len(ks):>3}{r['nbr_mass']:10.4f}{r['nbr_acc']:9.4f}{r['ent']:7.3f}"
              f"{r['recov_mass']:11.3f}{r['recov_acc']:10.3f}", flush=True)
        # NECESSITY: keep everything EXCEPT the set. A set can be necessary without being sufficient,
        # and keep-only is the harsher of the two tests, so both directions are reported.
        ra = run(allheads - ks); rows[nm + "_ablated"] = ra
        print(f"{'  (ablated)':<14}{len(ks):>3}{ra['nbr_mass']:10.4f}{ra['nbr_acc']:9.4f}{ra['ent']:7.3f}"
              f"{'':>11}{'':>10}   vs intact {full['nbr_mass']:.4f}/{full['nbr_acc']:.4f}", flush=True)
        aacc = []
        for _ in range(NRAND):
            if RANDMODE == "layer":
                rs = {(l, int(rng.choice([x for x in range(nH) if x != h]))) for l, h in ks}
                while len(rs) < len(ks): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            else:
                rs = set()
                while len(rs) < len(ks): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            aacc.append(run(allheads - rs))
        arm = {k: float(np.mean([a[k] for a in aacc])) for k in ("nbr_mass", "nbr_acc")}
        asd = {k: float(np.std([a[k] for a in aacc])) for k in ("nbr_mass", "nbr_acc")}
        rows[nm + "_ablated_random"] = {k: round(v, 4) for k, v in arm.items()}
        rows[nm + "_ablated_random_sd"] = {k: round(v, 4) for k, v in asd.items()}
        za = (ra["nbr_mass"] - arm["nbr_mass"]) / asd["nbr_mass"] if asd["nbr_mass"] > 1e-9 else float("nan")
        print(f"{'  (abl rand)':<14}{len(ks):>3}{arm['nbr_mass']:10.4f}{arm['nbr_acc']:9.4f}"
              f"{'':>7}{'':>11}{'':>10}   +-{asd['nbr_mass']:.4f}, z={za:+.1f}", flush=True)
        acc_ = []
        for _ in range(NRAND):
            if RANDMODE == "layer":
                rs = {(l, int(rng.choice([x for x in range(nH) if x != h]))) for l, h in ks}
                while len(rs) < len(ks): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            else:
                rs = set()
                while len(rs) < len(ks): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            acc_.append(run(rs))
        rm = {k: float(np.mean([a[k] for a in acc_])) for k in ("nbr_mass", "nbr_acc", "ent")}
        sd = {k: float(np.std([a[k] for a in acc_])) for k in ("nbr_mass", "nbr_acc", "ent")}
        rows[f"random_{nm}"] = {k: round(v, 4) for k, v in rm.items()}
        rows[f"random_{nm}_sd"] = {k: round(v, 4) for k, v in sd.items()}
        z = (r["nbr_mass"] - rm["nbr_mass"]) / sd["nbr_mass"] if sd["nbr_mass"] > 1e-9 else float("nan")
        print(f"{'  rand(' + RANDMODE + ')':<14}{len(ks):>3}{rm['nbr_mass']:10.4f}{rm['nbr_acc']:9.4f}"
              f"{rm['ent']:7.3f}{recov(rm, 'nbr_mass'):11.3f}{recov(rm, 'nbr_acc'):10.3f}"
              f"   +-{sd['nbr_mass']:.4f} mass over {NRAND} draws, z={z:+.1f}", flush=True)
    if LOO and LOO in SETS:
        ks = list(parse(SETS[LOO])); loo = {}
        print(f"\nleave-one-out on '{LOO}' (nbr_mass with that head dropped from the keep-set):")
        for h in ks:
            r = run(set(ks) - {h}); loo[f"L{h[0]}H{h[1]}"] = r["nbr_mass"]
        for h, v in sorted(loo.items(), key=lambda x: x[1])[:12]:
            print(f"   drop {h:<8} -> nbr_mass {v:.4f}  (whole set {rows[LOO]['nbr_mass']:.4f})")
        rows["loo"] = loo
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/keep_set_nbr{OUTTAG}_{tag}.json"
    json.dump({"model": tag, "struct": STRUCT, "randmode": RANDMODE, "nrand": NRAND,
               "ctxlo": CTXLO, "readouts": nro, "sets": SETS, "rows": rows}, open(p, "w"), indent=2)
    print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
