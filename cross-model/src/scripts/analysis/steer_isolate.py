"""Isolate GEOMETRY from IDENTITY in the steering test (identity==position is confounded
in square_grid, and decoupled in days).

MODE=interp (square_grid): graded plane-steer along a path between two COLLINEAR nodes X..Y
  (fraction t in {0,.25,.5,.75,1}). A middle node M sits between them, but M's identity is
  NEVER used in the steering vector (only mean_X, mean_Y). If the prediction peaks on M's
  neighbours near t=0.5 (the empty location where M lives), that's positional readout with
  no identity copied.

MODE=days (permuted weekday ring): steer X->Y along the in-context-RING best-2D plane
  (separate from the weekday-identity subspace). Report mass on Y's RING neighbours (geometry)
  vs Y's WEEKDAY neighbours (semantic identity) -- decoupled by the +3 permutation.

Env: PRESET MODELS_FILTER MODE(interp|days) NWALKS(12) WLEN(300) CTXLO(100)
     TRIPLES("0-1-2,0-4-8,3-7-11,5-6-7") PAIRS("0-3,0-4,1-5,2-6") OUTDIR DEVICE
Out: <OUTDIR>/steer_isolate_<mode>.json + .pdf
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
MODE = os.environ.get("MODE", "interp")
NWALKS = int(os.environ.get("NWALKS", "12"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
TRIPLES = [tuple(int(x) for x in t.split("-")) for t in os.environ.get("TRIPLES", "0-1-2,0-4-8,3-7-11,5-6-7").split(",")]
PAIRS = [tuple(int(x) for x in p.split("-")) for p in os.environ.get("PAIRS", "0-3,0-4,1-5,2-6").split(",")]
TS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/steer_isolate")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_plane(H, Gc):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    Q, _ = np.linalg.qr(Vt[:k].T @ W)
    return Q


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(cm.num_hidden_layers)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(cm.num_hidden_layers)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(cm.num_hidden_layers):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(cm.num_hidden_layers)}


@torch.no_grad()
def measure(model, tok, blocks, graph, cand_t, dev, Lstar, walks, X, vec):
    n = graph.n_nodes
    state = {"pos": None}
    v = None if vec is None else torch.tensor(vec, device=dev)
    def hook(_m, _i, out):
        if state["pos"]:
            hsd = (out[0] if isinstance(out, tuple) else out).clone()
            hsd[0, state["pos"], :] += v.to(hsd.dtype)
            return (hsd,) + tuple(out[1:]) if isinstance(out, tuple) else hsd
    h = blocks[Lstar].register_forward_hook(hook) if vec is not None else None
    Psum = np.zeros(n); cnt = 0
    try:
        for wk in walks:
            nodes = wk.nodes; spans = resolve_token_spans(tok, wk); cl = np.arange(1, len(nodes) + 1)
            pos = [spans[s + 1][0] - 1 for s in range(len(nodes) - 1) if nodes[s] == X and cl[s] >= CTXLO]
            if not pos: continue
            state["pos"] = pos
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            for p in pos:
                Psum += torch.softmax(logits[p][cand_t].float(), 0).cpu().numpy(); cnt += 1
    finally:
        if h: h.remove()
    return Psum / max(cnt, 1)


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    if MODE == "days":
        gkw = dict(graph_type="ring", ring_size=7, word_set="days")
    else:
        gkw = dict(graph_type="grid", grid_rows=4, grid_cols=4)
    out = {"mode": MODE, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **gkw, n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; Gc = np.array(graph.coords, float)
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]
        walks = G.generate_walks(graph, cfg); words = graph.words
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
        means = node_means(model, tok, blocks, cm, walks, dev, n)
        rsaL = {L: sp(np.linalg.norm((means[L]-means[L].mean(0))[:,None]-(means[L]-means[L].mean(0))[None],axis=2)[iu], GD) for L in range(cm.num_hidden_layers)}
        Lstar = max(rsaL, key=rsaL.get); H = means[Lstar]; B = best2d_plane(H, Gc)
        print(f"[{tag}] L*={Lstar} RSA={rsaL[Lstar]:.2f}", flush=True)
        rec = {"Lstar": Lstar}

        if MODE == "interp":
            rec["triples"] = {}
            for (X, Mn, Y) in TRIPLES:
                dvec = B @ (B.T @ (H[Y] - H[X]))                      # plane direction X->Y
                curve = {"nbrX": [], "nbrM": [], "nbrY": []}
                for t in TS:
                    P = measure(model, tok, blocks, graph, cand_t, dev, Lstar, walks, X, t * dvec if t > 0 else None)
                    curve["nbrX"].append(float(P[graph.neighbors(X)].sum()))
                    curve["nbrM"].append(float(P[graph.neighbors(Mn)].sum()))
                    curve["nbrY"].append(float(P[graph.neighbors(Y)].sum()))
                rec["triples"][f"{X}-{Mn}-{Y}"] = curve
                print(f"[{tag}] {X}-{Mn}-{Y}: nbrM over t {[round(x,2) for x in curve['nbrM']]} "
                      f"(peak@t={TS[int(np.argmax(curve['nbrM']))]})", flush=True)
        else:  # days: ring geometry vs weekday identity
            wd = [DAYS.index(w) for w in words]                        # weekday index per node
            def weekday_nbrs(Y): return [j for j in range(n) if (abs(wd[j]-wd[Y]) % 7) in (1, 6)]
            rec["pairs"] = {}
            for (X, Y) in PAIRS:
                v_plane = B @ (B.T @ (H[Y] - H[X]))
                res = {"ring_nbrY": graph.neighbors(Y), "weekday_nbrY": weekday_nbrs(Y)}
                for cname, vec in [("clean", None), ("plane", v_plane), ("full", H[Y] - H[X])]:
                    P = measure(model, tok, blocks, graph, cand_t, dev, Lstar, walks, X, vec)
                    res[cname] = {"ring_mass": float(P[graph.neighbors(Y)].sum()),
                                  "weekday_mass": float(P[weekday_nbrs(Y)].sum()),
                                  "ringX_mass": float(P[graph.neighbors(X)].sum())}
                rec["pairs"][f"{X}->{Y}"] = res
                print(f"[{tag}] {X}->{Y}: clean(ringY={res['clean']['ring_mass']:.2f} wdY={res['clean']['weekday_mass']:.2f}) "
                      f"plane(ringY={res['plane']['ring_mass']:.2f} wdY={res['plane']['weekday_mass']:.2f})", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/steer_isolate_{MODE}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/steer_isolate_{MODE}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        if out["mode"] == "interp":
            for m in models:
                r = out["models"][m]; trips = list(r["triples"])
                fig, ax = plt.subplots(1, len(trips), figsize=(3.6*len(trips), 4), squeeze=False)
                for j, tk in enumerate(trips):
                    c = r["triples"][tk]; a = ax[0, j]
                    a.plot(TS, c["nbrX"], "-o", label="nbr(X)", color="tab:blue")
                    a.plot(TS, c["nbrM"], "-o", label="nbr(M) [middle]", color="tab:green")
                    a.plot(TS, c["nbrY"], "-o", label="nbr(Y)", color="tab:red")
                    a.set_title(f"{tk}", fontsize=9); a.set_xlabel("steer fraction t (X->Y)"); a.set_ylim(0,1.0)
                    if j==0: a.set_ylabel("neighbour mass"); a.legend(fontsize=6)
                fig.suptitle(f"{m} [interp, L*={r['Lstar']}]: steer X->Y in plane; does the MIDDLE node's "
                             "neighbour mass peak at t~0.5 (its identity never used)?", fontsize=10)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        else:
            for m in models:
                r = out["models"][m]; pairs = list(r["pairs"])
                fig, ax = plt.subplots(1, 1, figsize=(8, 5)); x = np.arange(len(pairs)); w = 0.25
                for i, cn in enumerate(["clean", "plane", "full"]):
                    ax.bar(x + (i-1)*w, [r["pairs"][p][cn]["ring_mass"] for p in pairs], w, label=f"{cn}: RING nbrs(Y)")
                ax.plot(x, [r["pairs"][p]["plane"]["weekday_mass"] for p in pairs], "kx", ms=9, label="plane: WEEKDAY nbrs(Y)")
                ax.set_xticks(x); ax.set_xticklabels(pairs); ax.set_ylim(0,1.0); ax.legend(fontsize=8); ax.set_ylabel("prob mass")
                ax.set_title(f"{m} [days, L*={r['Lstar']}]: steer X->Y along RING plane -> RING nbrs (geometry) vs WEEKDAY nbrs (identity)", fontsize=9)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
