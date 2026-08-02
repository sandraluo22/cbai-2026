"""Cross-structure eigenmode ablation. Derive two canonical hidden-space directions from their own tasks —
SQUARE-GRID PARITY (the grid's top-eigenvalue checkerboard mode) and RING COORD (the ring's fundamental
position pair) — as per-layer readout directions r_{k,L}=Hc_L^T u_k. Then project each out of the residual
stream (all layers) and measure the effect on:
  - the FIVE geometry families (16 concept spaces): fit to their own hypothesised geometry (RSA / equidist);
  - the TORUS: next-node neighbour validity on the walk;
  - grid & ring themselves (sanity controls).
If a structure's score drops under an ablation, it re-uses that canonical direction.

Env: GEN_MODEL(Llama) NWALKS(16) WLEN(240) CTXLO(100) TORI(w5_L6,w6_L6,w5_L10) OUTDIR DEVICE
Out: <OUTDIR>/cross_eigenmode_ablation_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans
from geometry_spaces import SPACES, candidate_dmats

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "LlamaInstruct": ("meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
STYLE = os.environ.get("STYLE", "bare")   # bare | framed (Engels-style zero-shot) | chat (instruct)
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "240")); CTXLO = int(os.environ.get("CTXLO", "100"))
TORI = os.environ.get("TORI", "w5_L6,w6_L6,w5_L10").split(",")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/6_geometry")
INTENDED = {"arc": "line", "simplex": "simplex", "tree": "tree", "product": "product", "helix": "helix"}
TWORDS = ("apple bird sand math chair river music glass cloud knife table house tree stone water fire book "
          "phone door window clock plant coffee bread cheese wine garden mountain ocean forest desert island "
          "bridge tower castle engine wheel rope hammer nail brush paint paper pencil camera mirror candle lamp "
          "pillow blanket carpet basket bottle spoon fork plate bowl kettle oven fridge shelf drawer ladder "
          "fence gate roof wall floor tunnel cave valley hill meadow pond stream glacier volcano canyon cliff").split()


def lw(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception: return M.load_model(mirror, cfg)


def norm_lap_modes(A):
    d = A.sum(1); di = 1 / np.sqrt(np.maximum(d, 1e-12)); L = np.eye(len(A)) - di[:, None] * A * di[None, :]
    return np.linalg.eigh(L)


def adj_of(graph):
    n = graph.n_nodes; A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    return A


def build_torus(w, L):
    n = w * L; nid = lambda i, j: i * w + j; adj = [[] for _ in range(n)]
    for i in range(L):
        for j in range(w):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)): adj[nid(i, j)].append(nid((i + di) % L, (j + dj) % w))
    coords = [(i, j) for i in range(L) for j in range(w)]
    return Graph(n_nodes=n, words=TWORDS[:n], adjacency=[sorted(set(a)) for a in adj], coords=coords)


@torch.no_grad()
def node_means(model, tok, blocks, cm, graph, walks, dev):
    nL = cm.num_hidden_layers; grab = {}
    def mk(L):
        def hh(_m, _i, out): grab[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    n = graph.n_nodes; nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grab.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]
            for L in range(nL):
                rows = grab[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def readout_Q(means, U_cols, nL, dev):
    """per-layer orthonormal basis of readout directions r=Hc^T u for the given node-mode columns."""
    Q = {}
    for L in range(nL):
        Hc = means[L] - means[L].mean(0)
        cols = [Hc.T @ U_cols[:, c] for c in range(U_cols.shape[1])]
        cols = [v for v in cols if np.linalg.norm(v) > 1e-8]
        if not cols: Q[L] = None; continue
        q, _ = torch.linalg.qr(torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev), mode="reduced")
        Q[L] = q
    return Q


def cycle_lap_modes(nn):
    A = np.zeros((nn, nn))
    for i in range(nn): A[i, (i + 1) % nn] = A[i, (i - 1) % nn] = 1.0
    return np.linalg.eigh(np.diag(A.sum(1)) - A)               # ascending; fundamental cos/sin at [1,2]


def abl_hooks(blocks, Q, nL):
    def mk(L):
        q = Q.get(L)
        def hh(_m, _i, out):
            if q is None: return out
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float(); hf = hf - (hf @ q) @ q.T; h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hh
    return [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def few_shot(items, unit, nn, rng, nshot):
    ex = []
    while len(ex) < nshot:
        si = int(rng.integers(nn)); k = int(rng.integers(1, nn))
        if (si, k) not in [(e[0], e[1]) for e in ex]: ex.append((si, k, (si + k) % nn))
    return "".join(f"{k} {unit} after {items[si]} is {items[a]}.\n" for si, k, a in ex)


def build_prompt(tok, items, unit, si, k, preamble):
    start = items[si]
    if STYLE == "framed":
        frame = "Let's do some calendar math. " if unit == "months" else "Let's do some day of the week math. "
        return frame + f"{k} {unit} from {start} is", True
    if STYLE == "chat":
        sing = unit[:-1]
        q = f"What {sing} is {k} {unit} after {start}? Answer with just the {sing} name, nothing else."
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), False
    return preamble + f"{k} {unit} after {start} is", True


@torch.no_grad()
def cycle_accuracy(model, tok, blocks, items, unit, preamble, dev):
    nn = len(items); item_ids = [tok(" " + it, add_special_tokens=False)["input_ids"] for it in items]
    correct = tot = 0
    for si in range(nn):
        for k in range(1, nn):
            ans = (si + k) % nn; prompt, addsp = build_prompt(tok, items, unit, si, k, preamble)
            p_ids = tok(prompt, add_special_tokens=addsp)["input_ids"]
            seqs = [p_ids + ic for ic in item_ids]; ml = max(len(s) for s in seqs)
            batch = torch.full((nn, ml), tok.pad_token_id or 0, device=dev, dtype=torch.long)
            attn = torch.zeros((nn, ml), device=dev, dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, :len(s)] = torch.tensor(s, device=dev); attn[i, :len(s)] = 1
            lp = torch.log_softmax(model(input_ids=batch, attention_mask=attn).logits.float(), -1)
            sc = [sum(lp[i, len(p_ids) - 1 + t, tid].item() for t, tid in enumerate(ic)) / len(ic) for i, ic in enumerate(item_ids)]
            correct += int(int(np.argmax(sc)) == ans); tot += 1
    return correct / max(tot, 1)


def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def rdm(H): iu = np.triu_indices(H.shape[0], 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
def equidist(H): iu = np.triu_indices(H.shape[0], 1); d = rdm(H); return float(1 - d.std() / (d.mean() + 1e-9))


@torch.no_grad()
def torus_neighbour_validity(model, tok, blocks, cm, graph, walks, dev):
    n = graph.n_nodes
    cand = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    ok = 0; tot = 0
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
        logits = model(input_ids=ids).logits[0]
        for s in range(len(nodes) - 1):
            if cl[s] < CTXLO: continue
            p = torch.softmax(logits[spans[s][-1]][cand].float(), 0).cpu().numpy()
            ok += int(int(p.argmax()) in graph.neighbors(nodes[s])); tot += 1
    return ok / max(tot, 1)


@torch.no_grad()
def extract_space(model, tok, blocks, nL, carrier, items, dev):
    H = {L: [] for L in range(nL)}
    for it in items:
        text = carrier.format(item=it); enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = torch.tensor([enc["input_ids"]], device=dev); offs = enc["offset_mapping"]
        a = text.index(it); b = a + len(it)
        toks = [t for t, (o0, o1) in enumerate(offs) if o0 is not None and o1 > o0 and o0 < b and o1 > a]
        grab = {}
        def mk(L):
            def hh(_m, _i, out): grab[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
        model(input_ids=ids)
        for h in hs: h.remove()
        for L in range(nL): H[L].append(grab[L][0, toks[-1]].float().cpu().numpy())
    return {L: np.array(H[L]) for L in range(nL)}


def space_fit(H, sp_def, nL):
    fam = sp_def["family"]; cands = candidate_dmats(sp_def); intended = INTENDED[fam]
    n = len(sp_def["items"]); iu = np.triu_indices(n, 1); best = -9.0
    Dint = cands[intended][iu] if intended in cands else None
    for L in range(nL):
        Hc = H[L] - H[L].mean(0)
        if fam == "simplex": best = max(best, equidist(Hc))
        elif Dint is not None: best = max(best, sp(rdm(Hc), Dint))
    return round(best, 3)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers

    # ---- grid-parity and ring-coord directions from their own tasks ----
    gcfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4, n_walks=NWALKS, walk_length=WLEN, device=dev)
    rcfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=16, n_walks=NWALKS, walk_length=WLEN, device=dev)
    grid = G.build_graph(gcfg); ring = G.build_graph(rcfg)
    gw, w2 = norm_lap_modes(adj_of(grid)); rw, r2 = norm_lap_modes(adj_of(ring))
    u_par = w2[:, [int(np.argmax(gw))]]                       # grid parity = top-eigenvalue mode (checkerboard)
    u_coord = r2[:, [1, 2]]                                   # ring coord = fundamental position pair (Fiedler)
    gmeans = node_means(model, tok, blocks, cm, grid, G.generate_walks(grid, gcfg), dev)
    rmeans = node_means(model, tok, blocks, cm, ring, G.generate_walks(ring, rcfg), dev)
    Q_par = readout_Q(gmeans, u_par, nL, dev)
    Q_coord = readout_Q(rmeans, u_coord, nL, dev)
    print(f"[{tag}] directions ready (grid-parity rank1, ring-coord rank2)", flush=True)

    CONDS = {"baseline": None, "ablate_gridparity": Q_par, "ablate_ringcoord": Q_coord}
    out = {"model": tag, "conditions": list(CONDS), "families": {}, "tori": {}, "controls": {}, "cyclic": {}}
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # ---- real learned cycles: months (12) / days (7) modular arithmetic ----
    rng = np.random.default_rng(0); NSHOT = int(os.environ.get("NSHOT", "3"))
    CYC = [("months", MONTHS, "months"), ("days", DAYS, "days")]
    OWN = {}                                                        # each cycle's OWN fundamental-circle direction
    for nm2, items2, unit2 in CYC:
        Hc2 = extract_space(model, tok, blocks, nL, "The " + unit2[:-1] + " {item}", items2, dev)
        cw2, cU2 = cycle_lap_modes(len(items2)); OWN[nm2] = readout_Q(Hc2, cU2[:, [1, 2]], nL, dev)
    for nm, items, unit in CYC:
        pre = few_shot(items, unit, len(items), rng, NSHOT) if NSHOT > 0 else ""
        nn = len(items); cw, cU = cycle_lap_modes(nn); Q_own = OWN[nm]
        cyc_conds = {**CONDS, "ablate_owncycle": Q_own}
        if os.environ.get("CROSS", "0") == "1":                     # same circuit? ablate OTHER-size cycles' directions
            other = "days" if nm == "months" else "months"
            cyc_conds = {"baseline": None, f"own_{nm}": OWN[nm], f"cross_{other}": OWN[other], "ringcoord16": Q_coord, "gridparity": Q_par}
        elif os.environ.get("MODE_SWEEP", "0") == "1":              # ablate each cycle-frequency band separately
            Hc = extract_space(model, tok, blocks, nL, "The " + unit[:-1] + " {item}", items, dev)
            cyc_conds = {"baseline": None}
            for k in range(1, nn // 2 + 1):
                cols = [2 * k - 1, 2 * k] if 2 * k <= nn - 1 else [nn - 1]   # freq-k cos/sin pair (Nyquist = single)
                cyc_conds[f"cyclefreq{k}"] = readout_Q(Hc, cU[:, cols], nL, dev)
        row = {}
        for cond, Q in cyc_conds.items():
            hs = abl_hooks(blocks, Q, nL) if Q else []
            row[cond] = round(cycle_accuracy(model, tok, blocks, items, unit, pre, dev), 3)
            for h in hs: h.remove()
        out["cyclic"][nm] = row
        print(f"  cyclic {nm:7}: " + "  ".join(f"{c}={v}" for c, v in row.items()), flush=True)
    if os.environ.get("ONLY_CYCLIC", "0") == "1":
        p = f"{OUTDIR}/cross_eigenmode_ablation_{tag}.json"
        this_cyclic = out["cyclic"]                                 # results for the current NSHOT
        if os.path.exists(p): out = json.load(open(p))             # keep families/torus already computed
        out.setdefault("cyclic_by_nshot", {})[str(NSHOT)] = this_cyclic
        if NSHOT == 3: out["cyclic"] = this_cyclic
        del model, tok; gc.collect(); torch.cuda.empty_cache()
        json.dump(out, open(p, "w"), indent=2); print(f"DONE (cyclic only, nshot={NSHOT}) -> {p}", flush=True); return

    # ---- five families (16 spaces) ----
    for name, sd in SPACES.items():
        row = {}
        for cond, Q in CONDS.items():
            hs = abl_hooks(blocks, Q, nL) if Q else []
            H = extract_space(model, tok, blocks, nL, sd["carrier"], sd["items"], dev)
            for h in hs: h.remove()
            row[cond] = space_fit(H, sd, nL)
        out["families"][name] = {"family": sd["family"], **row}
        print(f"  {name:12} [{sd['family']:7}] base={row['baseline']} par={row['ablate_gridparity']} coord={row['ablate_ringcoord']}", flush=True)

    # ---- torus neighbour validity ----
    for key in TORI:
        w, L = int(key.split("_")[0][1:]), int(key.split("_")[1][1:]); tg = build_torus(w, L)
        tcfg = replace(get_config("gemma_qwen"), n_walks=NWALKS, walk_length=WLEN, device=dev)
        tw = G.generate_walks(tg, tcfg); row = {}
        for cond, Q in CONDS.items():
            hs = abl_hooks(blocks, Q, nL) if Q else []
            row[cond] = round(torus_neighbour_validity(model, tok, blocks, cm, tg, tw, dev), 3)
            for h in hs: h.remove()
        out["tori"][key] = row
        print(f"  torus {key:8}: base nbr={row['baseline']} par={row['ablate_gridparity']} coord={row['ablate_ringcoord']}", flush=True)

    # ---- grid & ring sanity controls: neighbour validity ----
    for gg, cfg, nm in [(grid, gcfg, "grid"), (ring, rcfg, "ring")]:
        gwk = G.generate_walks(gg, cfg); row = {}
        for cond, Q in CONDS.items():
            hs = abl_hooks(blocks, Q, nL) if Q else []
            row[cond] = round(torus_neighbour_validity(model, tok, blocks, cm, gg, gwk, dev), 3)
            for h in hs: h.remove()
        out["controls"][nm] = row
        print(f"  control {nm:6}: base nbr={row['baseline']} par={row['ablate_gridparity']} coord={row['ablate_ringcoord']}", flush=True)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/cross_eigenmode_ablation_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
