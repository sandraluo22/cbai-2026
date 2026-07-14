"""Activation patching to localize a word<->cell swap, with THREE per-head restoration
metrics (patch clean O -> corrupted S, one head at a time, all positions):

  restore_logit : (M_patch - M_S)/(M_O - M_S), M = signed logit(word_b)-logit(word_a) at
                  readouts entering cell a/b  (predicts the ORIGINAL binding)
  restore_kl    : 1 - KL(P_patch||P_O)/KL(P_S||P_O) over the 16 node-word distribution at
                  those readouts  (how much the whole predicted distribution reverts)
  restore_rsa   : (RSA_patch - RSA_S)/(RSA_O - RSA_S), RSA of the rebuilt WORD-geometry at
                  the peak layer vs the ORIGINAL word->cell layout  (representation reverts)

Same walk, words at cells a,b exchanged (default plant<->clock). Single-token swap words.

Env: PRESET MODELS_FILTER SWAP("12-15") NWALKS(3) WLEN(320) CTXLO(80) INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/patch_swap_metrics_<a>_<b>.json + .pdf
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
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rsa_of(wm, Gc, GDu, iu):
    """best-2D (supervised) RSA: project onto the graph-aligned plane, then RSA vs graph dist.
    Consistent with the rest of the investigation (raw RSA is confounded, esp. for Qwen)."""
    if np.isnan(wm).any():
        return float("nan")
    Hc = wm - wm.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    R = np.linalg.norm(P[:, None] - P[None], axis=2)[iu]
    return sp(R, GDu)


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
        graph = G.build_graph(cfg); words = graph.words; n = 16
        GD = graph.distance_matrix(); iu = np.triu_indices(n, 1); GDu = GD[iu]
        Gc = np.array(graph.coords, float)                      # original word->cell coords for best-2D plane
        wa, wb = words[A], words[B]
        swapped = list(words); swapped[A], swapped[B] = words[B], words[A]
        swap_id = lambda c: (B if c == A else (A if c == B else c))     # emitted word-id at cell c in S
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading (swap {A}:{wa} <-> {B}:{wb})", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); d = cm.hidden_size
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        ta = tok(" " + wa, add_special_tokens=False)["input_ids"]; tb = tok(" " + wb, add_special_tokens=False)["input_ids"]
        if len(ta) != 1 or len(tb) != 1:
            print(f"[{tag}] SKIP non-single-token swap words", flush=True); del model, tok; gc.collect(); continue
        ta, tb = ta[0], tb[0]
        cand = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)

        zc = {}
        def mkz(L):
            def pre(_m, args): zc[L] = args[0].detach().clone()
            return pre
        grab = {}
        def mkcap(L):
            def hh(_m, _i, out): grab[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        patch = {"L": None, "src": None, "cols": None}
        def mkpatch(L):
            def pre(_m, args):
                if patch["L"] == L:
                    x = args[0].clone(); x[..., patch["cols"]] = patch["src"][..., patch["cols"]]
                    return (x,) + tuple(args[1:])
            return pre
        capz = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkz(L)) for L in range(nL)]
        patchh = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mkpatch(L)) for L in range(nL)]
        capres = [blocks[L].register_forward_hook(mkcap(L)) for L in range(nL)]

        def metric(logits, reads):
            m = 0.0
            for pos, sign in reads:
                m += sign * float(logits[pos, tb] - logits[pos, ta])
            return m
        def pnode(logits, pos):
            p = torch.softmax(logits[pos][cand].float(), 0); return p
        def kl(pp, po):
            return float((pp * (torch.log(pp + 1e-9) - torch.log(po + 1e-9))).sum())

        # ---------- pass 1: O and S baselines, pick L*, store Oz + read/node bookkeeping ----------
        Osum = np.zeros((nL, n, d)); Ocnt = np.zeros((nL, n))
        Ssum = np.zeros((nL, n, d)); Scnt = np.zeros((nL, n))
        MO = MS = 0.0; klSO = 0.0; klN = 0
        wdata = []
        for wk in walks:
            Swk = Walk(walk_id=wk.walk_id, nodes=wk.nodes, words=[swapped[c] for c in wk.nodes])
            oid = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            sid = tok(Swk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            if oid.shape != sid.shape:
                print(f"[{tag}] walk {wk.walk_id} misaligned; skip", flush=True); continue
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
            reads = [(spans[s + 1][0] - 1, (+1.0 if nodes[s + 1] == B else -1.0)) for s in range(len(nodes) - 1)
                     if nodes[s + 1] in (A, B) and cl[s] >= CTXLO]
            npos = [(spans[s][-1], nodes[s]) for s in range(len(nodes)) if cl[s] >= CTXLO]  # (pos, cell)
            if not reads:
                continue
            patch["L"] = None; zc.clear(); grab.clear()
            lo = model(input_ids=oid).logits[0]; MO += metric(lo, reads)
            Oz = {L: zc[L].clone() for L in range(nL)}
            POr = [pnode(lo, p) for p, _ in reads]
            for L in range(nL):
                for pos, c in npos:
                    Osum[L, c] += grab[L][0, pos].float().cpu().numpy(); Ocnt[L, c] += 1
            grab.clear()
            ls = model(input_ids=sid).logits[0]; MS += metric(ls, reads)
            PSr = [pnode(ls, p) for p, _ in reads]
            for L in range(nL):
                for pos, c in npos:
                    Ssum[L, swap_id(c)] += grab[L][0, pos].float().cpu().numpy(); Scnt[L, swap_id(c)] += 1
            for po, ps in zip(POr, PSr):
                klSO += kl(ps, po); klN += 1
            wdata.append((sid, reads, POr, npos, Oz))
        klSO /= max(klN, 1)
        Om = np.where(Ocnt[..., None] > 0, Osum / np.maximum(Ocnt[..., None], 1), np.nan)
        Sm = np.where(Scnt[..., None] > 0, Ssum / np.maximum(Scnt[..., None], 1), np.nan)
        rsaO = np.array([rsa_of(Om[L], Gc, GDu, iu) for L in range(nL)])
        Lstar = int(np.nanargmax(rsaO)); RSA_O = rsaO[Lstar]; RSA_S = rsa_of(Sm[Lstar], Gc, GDu, iu)
        # pass-2 only needs L* residuals: drop the all-layer capture, keep just L*
        for hh in capres:
            hh.remove()
        capres = [blocks[Lstar].register_forward_hook(mkcap(Lstar))]
        print(f"[{tag}] L*={Lstar} RSA_O={RSA_O:.2f} RSA_S={RSA_S:.2f} | M_O={MO:+.0f} M_S={MS:+.0f} klSO={klSO:.3f}", flush=True)
        drsa = (RSA_O - RSA_S) if abs(RSA_O - RSA_S) > 1e-4 else 1.0
        dM = (MO - MS) if abs(MO - MS) > 1e-6 else 1.0

        # ---------- pass 2: per head, patch O->S, recompute all three metrics ----------
        r_logit = np.zeros((nL, nH)); r_kl = np.zeros((nL, nH)); r_rsa = np.zeros((nL, nH))
        for L in range(nL):
            _, hd = attn_proj(blocks[L], cm)
            for h in range(nH):
                cols = torch.arange(h * hd, (h + 1) * hd, device=dev)
                Wsum = np.zeros((n, d)); Wcnt = np.zeros(n); Mp = 0.0; klp = 0.0; kln = 0
                for (sid, reads, POr, npos, Oz) in wdata:
                    patch["L"] = L; patch["src"] = Oz[L]; patch["cols"] = cols; grab.clear()
                    lp = model(input_ids=sid).logits[0]
                    Mp += metric(lp, reads)
                    for (pos, _), po in zip(reads, POr):
                        klp += kl(pnode(lp, pos), po); kln += 1
                    for pos, c in npos:
                        Wsum[swap_id(c)] += grab[Lstar][0, pos].float().cpu().numpy(); Wcnt[swap_id(c)] += 1
                    patch["L"] = None
                Wm = np.where(Wcnt[:, None] > 0, Wsum / np.maximum(Wcnt[:, None], 1), np.nan)
                r_logit[L, h] = (Mp - MS) / dM
                r_kl[L, h] = 1.0 - (klp / max(kln, 1)) / max(klSO, 1e-9)
                r_rsa[L, h] = (rsa_of(Wm, Gc, GDu, iu) - RSA_S) / drsa
            print(f"[{tag}] layer {L} done", flush=True)
        for hh in capz + patchh + capres:
            hh.remove()

        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
        def tops(matrix):
            t = np.argsort(matrix, axis=None)[::-1][:8]
            return [{"layer": int(i // nH), "head": int(i % nH), "val": round(float(matrix.flatten()[i]), 3),
                     "qk": round(float(gen.flatten()[i]), 2), "dla": round(float(att.flatten()[i]), 2)} for i in t]
        rec = {"n_layers": nL, "n_heads": nH, "Lstar": Lstar, "M_O": MO, "M_S": MS, "RSA_O": RSA_O, "RSA_S": RSA_S,
               "restore_logit": r_logit.tolist(), "restore_kl": r_kl.tolist(), "restore_rsa": r_rsa.tolist(),
               "top_logit": tops(r_logit), "top_kl": tops(r_kl), "top_rsa": tops(r_rsa),
               "corr": {"logit_kl": float(np.corrcoef(r_logit.flatten(), r_kl.flatten())[0, 1]),
                        "logit_rsa": float(np.corrcoef(r_logit.flatten(), r_rsa.flatten())[0, 1]),
                        "kl_rsa": float(np.corrcoef(r_kl.flatten(), r_rsa.flatten())[0, 1])}}
        out["models"][tag] = rec
        print(f"[{tag}] top logit L{rec['top_logit'][0]['layer']}H{rec['top_logit'][0]['head']}={rec['top_logit'][0]['val']} | "
              f"top KL L{rec['top_kl'][0]['layer']}H{rec['top_kl'][0]['head']}={rec['top_kl'][0]['val']} | "
              f"top RSA L{rec['top_rsa'][0]['layer']}H{rec['top_rsa'][0]['head']}={rec['top_rsa'][0]['val']} | "
              f"corr {rec['corr']}", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/patch_swap_metrics_{A}_{B}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/patch_swap_metrics_{A}_{B}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    a, b = out["swap"]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]
            mats = [("restore_logit", "logit-diff"), ("restore_kl", "KL"), ("restore_rsa", "ΔRSA")]
            fig, ax = plt.subplots(1, 3, figsize=(17, 5))
            for j, (key, lab) in enumerate(mats):
                Rm = np.array(r[key]); v = max(0.15, float(np.nanpercentile(np.abs(Rm), 99)))
                im = ax[j].imshow(Rm, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                ax[j].set_xlabel("head"); ax[j].set_ylabel("layer")
                ax[j].set_title(f"{m}: restoration by {lab}", fontsize=9); fig.colorbar(im, ax=ax[j], fraction=.046)
            fig.suptitle(f"{m}: patch head O->S (swap {a}<->{b}), restoration by 3 metrics — "
                         f"corr logit/KL={r['corr']['logit_kl']:.2f} logit/RSA={r['corr']['logit_rsa']:.2f} KL/RSA={r['corr']['kl_rsa']:.2f}",
                         fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
