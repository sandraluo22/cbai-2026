"""Steer2Edit (arXiv:2602.09870): steering vector -> training-free rank-1 weight edits.

Why this arm matters here. The distilled student installs a bias ~20x below the
strength at which introspective detection begins (measured: detection needs
rel ~0.3-0.75 at L26, the student sits at 0.030), so its null is explained by
dose alone. Steer2Edit's edit size is set by an explicit budget rho, so the edit
can be dialled UP to the detectable regime -- which turns "is a weight-installed
bias noticed?" into a question about mechanism rather than magnitude.

It also sits between the two arms already run: activation steering is an
always-on additive perturbation, LoRA is diffuse retrained weights, and
Steer2Edit is a weight edit that is INPUT-GATED -- k_hat selects the input
direction, so the edit only fires on inputs aligned with the concept. If
detectability tracks perturbation-likeness, it should land between them.

Method as published:

    v_l^b   mean activation difference for the concept, per LAYER and per BLOCK
            (attention and MLP separately)
    mu_i    = E[h_i], the mean input to component i, estimated on real text
    g_i     = cos(v_i, W_i mu_i)                    importance score
    k_hat_i = W_i^T v_i / ||W_i^T v_i||             input direction (the gate)
    dW_i    = sign(g_i) * max(|g_i| - rho*alpha, 0) / (rho*(1-alpha)) * v_hat_i k_hat_i^T

Components: each attention head's output projection W_o (a d_model x d_head slice
of o_proj) and each column of W_down treated independently. Separate budgets
rho_attn, rho_mlp.

CAVEAT: implemented from the paper's equations, not from their code (no public
repo found). Details of mu estimation and of the rho grid may differ from theirs.
The magnitude sweep below is therefore reported in terms of the MEASURED ||dh||,
which is implementation-independent, rather than in terms of rho.

Output: out/s2e.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/introspection-mechanisms/experiments")
from concepts_list import DEFAULT_BASELINE_WORDS  # noqa: E402
from detect import DETECT, graded_detect, identifies  # noqa: E402
from gate import chat, decoder_layers, last_resid, mentions  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
CONCEPT = os.environ.get("CONCEPT", "Bread")
CONTROL = os.environ.get("CONTROL", "Cameras")   # matched-rel specificity control
READ_LAYER = int(os.environ.get("READ_LAYER", 26))     # where detection lives
# LAYER INDEXING. hidden_states[l] is the INPUT to block l, so a weight edit in
# block L first shows up at hidden_states[L+1]. Editing block 26 and reading at 26
# gave rel=0.000 EXACTLY at every rho -- the read was upstream of the edit.
# Inject() hooks block (layer-1)'s output, so injection "at L26" perturbs
# hidden_states[26]; the MATCHING weight edit is therefore block 25, not 26.
# Which layers to EDIT. Default "" = all 64, which is Steer2Edit as published.
# But then `rel` (measured at READ_LAYER) describes only the edit's footprint at
# one layer while the intervention perturbs everywhere, so it is NOT matched
# against the single-layer injection. EDIT_LAYERS=26 gives the like-for-like.
EDIT_LAYERS = os.environ.get("EDIT_LAYERS", "")
_EDIT = set(int(x) for x in EDIT_LAYERS.split(",")) if EDIT_LAYERS else None
# rho >= 2 edits ZERO components: the soft threshold rho*alpha exceeds max|cos|=1.
RHOS = [float(x) for x in os.environ.get("RHOS", "0.5,0.3,0.2,0.12,0.07,0.04").split(",")]
ALPHA = float(os.environ.get("ALPHA", 0.5))
NTRIAL = int(os.environ.get("NTRIAL", 16))
MAXNEW = int(os.environ.get("MAXNEW", 110))
NPROBE = int(os.environ.get("NPROBE", 24))

PROBE = ["What is the best way to organise a bookshelf?",
         "Explain why the sky changes colour at sunset.",
         "Give me a tip for staying focused while working.",
         "What makes a good short story?"]


@torch.no_grad()
def collect_mu(model, tok, texts):
    """mu_i = E[h_i]: mean INPUT to each o_proj and down_proj, over real text."""
    acc, hooks = {}, []

    def mk(key):
        def f(mod, inp, out):
            h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s, n = acc.get(key, (None, 0))
            t = h.sum(0).cpu()
            acc[key] = (t if s is None else s + t, n + h.shape[0])
        return f

    for li, blk in enumerate(decoder_layers(model)):
        hooks.append(blk.self_attn.o_proj.register_forward_hook(mk((li, "o"))))
        hooks.append(blk.mlp.down_proj.register_forward_hook(mk((li, "d"))))
    try:
        for t in texts:
            enc = tok(t, return_tensors="pt").to(model.device)
            model(**enc)
    finally:
        for h in hooks:
            h.remove()
    return {k: (s / max(n, 1)).numpy() for k, (s, n) in acc.items()}


POS_TEMPLATES = ["Tell me about {c}", "What is {c}?", "Describe {c} in detail.",
                 "Write a paragraph about {c}.", "Explain {c} to me.",
                 "I have been thinking about {c}.", "Let's discuss {c}.",
                 "Give me some facts about {c}."]


@torch.no_grad()
def block_vectors(model, tok, pos_texts, neg_texts):
    """v_l^b: mean OUTPUT difference of each attention / MLP block, concept vs baseline."""
    acc, hooks = {}, []

    def mk(key):
        def f(mod, inp, out):
            o = (out[0] if isinstance(out, tuple) else out).detach().float()
            o = o.reshape(-1, o.shape[-1])[-1]      # last token
            acc.setdefault(key, []).append(o.cpu().numpy())
        return f

    for li, blk in enumerate(decoder_layers(model)):
        hooks.append(blk.self_attn.o_proj.register_forward_hook(mk((li, "o"))))
        hooks.append(blk.mlp.down_proj.register_forward_hook(mk((li, "d"))))
    try:
        acc.clear()
        for t in pos_texts:
            model(**tok(t, return_tensors="pt").to(model.device))
        pos = {k: np.mean(v, 0) for k, v in acc.items()}
        acc.clear()
        for t in neg_texts:
            model(**tok(t, return_tensors="pt").to(model.device))
        neg = {k: np.mean(v, 0) for k, v in acc.items()}
    finally:
        for h in hooks:
            h.remove()
    return {k: pos[k] - neg[k] for k in pos}


def s2e_edits(model, V, MU, rho_attn, rho_mlp, alpha):
    """Return {(layer, kind): delta_tensor} to be added to the weights."""
    edits, n_edit = {}, 0
    for li, blk in enumerate(decoder_layers(model)):
        if _EDIT is not None and li not in _EDIT:
            continue
        cfg = model.config
        nh = getattr(cfg, "num_attention_heads")
        dh = cfg.hidden_size // nh
        # ---- attention: one component per head ----
        Wo = blk.self_attn.o_proj.weight.data          # (d_model, d_model)
        # device_map="auto" can shard layers across devices, so every tensor is
        # placed against THIS layer's weight rather than a single global device
        dev = Wo.device
        v = torch.tensor(V[(li, "o")], dtype=torch.float32, device=dev)
        mu = torch.tensor(MU[(li, "o")], dtype=torch.float32, device=dev)
        vhat = v / (v.norm() + 1e-9)
        dWo = torch.zeros_like(Wo, dtype=torch.float32)
        for h in range(nh):
            sl = slice(h * dh, (h + 1) * dh)
            Wi = Wo[:, sl].float()
            gi = torch.nn.functional.cosine_similarity(v, Wi @ mu[sl], dim=0).item()
            mag = max(abs(gi) - rho_attn * alpha, 0.0) / (rho_attn * (1 - alpha))
            if mag <= 0:
                continue
            kv = Wi.T @ v
            khat = kv / (kv.norm() + 1e-9)
            dWo[:, sl] = float(np.sign(gi)) * mag * torch.outer(vhat, khat)
            n_edit += 1
        edits[(li, "o")] = dWo.cpu()
        # ---- MLP: one component per down_proj column ----
        Wd = blk.mlp.down_proj.weight.data             # (d_model, d_ff)
        devd = Wd.device
        vd = torch.tensor(V[(li, "d")], dtype=torch.float32, device=devd)
        mud = torch.tensor(MU[(li, "d")], dtype=torch.float32, device=devd)
        vdhat = vd / (vd.norm() + 1e-9)
        Wdf = Wd.float()
        contrib = Wdf * mud.unsqueeze(0)               # (d_model, d_ff) column-wise W_i mu_i
        gs = torch.nn.functional.cosine_similarity(vd.unsqueeze(1), contrib, dim=0)  # (d_ff,)
        mags = torch.clamp(gs.abs() - rho_mlp * alpha, min=0.0) / (rho_mlp * (1 - alpha))
        kd = (Wdf * vd.unsqueeze(1)).sum(0)            # W_i^T v per column (scalar each)
        khat_d = torch.sign(kd)
        coef = torch.sign(gs) * mags * khat_d          # (d_ff,)
        edits[(li, "d")] = torch.outer(vdhat, coef).cpu()
        n_edit += int((mags > 0).sum())
    return edits, n_edit


@torch.no_grad()
def apply_edits(model, edits, sign=+1):
    for li, blk in enumerate(decoder_layers(model)):
        for kind, mod in (("o", blk.self_attn.o_proj), ("d", blk.mlp.down_proj)):
            d = edits.get((li, kind))
            if d is not None:
                mod.weight.data += sign * d.to(mod.weight.dtype).to(mod.weight.device)


@torch.no_grad()
def gen(model, tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[i][n:], skip_special_tokens=True).strip() for i in range(len(texts))]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    base_texts = [chat(tok, f"Tell me about {w}") for w in DEFAULT_BASELINE_WORDS]
    A = last_resid(model, tok, base_texts, READ_LAYER)
    hnorm = float(A.norm(dim=1).mean())
    vread = last_resid(model, tok, [chat(tok, f"Tell me about {CONCEPT}")], READ_LAYER)[0].numpy() \
        - A.mean(0).numpy()
    vread = vread / np.linalg.norm(vread)

    print("[s2e] collecting mu = E[h_i] on real text", flush=True)
    MU = collect_mu(model, tok, [chat(tok, p) for p in PROBE[:NPROBE]])
    print("[s2e] computing per-layer per-block concept vectors", flush=True)
    # A SINGLE positive prompt gave a per-block direction so noisy that the edit
    # moved activations (rel 0.157) while staying orthogonal to the concept
    # (EAS -0.017) and producing no behaviour. Average over phrasings instead.
    V = block_vectors(model, tok,
                      [chat(tok, t.format(c=CONCEPT)) for t in POS_TEMPLATES],
                      base_texts[:48])

    trials = [chat(tok, DETECT.format(n=i + 1)) for i in range(NTRIAL)]
    probes = [chat(tok, p) for p in PROBE]

    o0 = gen(model, tok, trials)
    p0 = gen(model, tok, probes)
    h0 = last_resid(model, tok, probes, READ_LAYER).mean(0).numpy()
    rep = {"NONE": dict(detect=float(np.mean([graded_detect(x) for x in o0])),
                        identify=float(np.mean([identifies(x) for x in o0])),
                        mention=float(np.mean([mentions(CONCEPT, x) for x in p0])),
                        rel=0.0, samples=o0[:2])}
    print(f"  unedited        detect {rep['NONE']['detect']:.3f}  "
          f"identify {rep['NONE']['identify']:.3f}  mention {rep['NONE']['mention']:.3f}",
          flush=True)

    # control-concept vectors, edited identically -- if a CONTROL edit at the same
    # rel also raises detection, the model is noticing a perturbation rather than
    # the concept, and the bread result means much less
    VC = block_vectors(model, tok,
                       [chat(tok, t.format(c=CONTROL)) for t in POS_TEMPLATES],
                       base_texts[:48])

    for rho in RHOS:
        for tag, VV in (("", V), ("ctl_", VC)):
            edits, n_edit = s2e_edits(model, VV, MU, rho, rho, ALPHA)
            apply_edits(model, edits, +1)
            try:
                h1c = last_resid(model, tok, probes, READ_LAYER).mean(0).numpy()
                relc = float(np.linalg.norm(h1c - h0) / hnorm)
                oc = gen(model, tok, trials)
                rep[f"{tag}rho{rho}"] = dict(rho=rho, control=bool(tag), rel=relc,
                                             n_components=n_edit,
                                             detect=float(np.mean([graded_detect(x) for x in oc])),
                                             identify=float(np.mean([identifies(x) for x in oc])),
                                             samples=oc[:2])
                r = rep[f"{tag}rho{rho}"]
                print(f"  {tag or 'bread':<6} rho={rho:<5} rel={relc:.3f} n={n_edit:<7} "
                      f"detect {r['detect']:.3f}  identify {r['identify']:.3f}", flush=True)
            finally:
                apply_edits(model, edits, -1)
            json.dump(rep, open("out/s2e.json", "w"), indent=1)
        continue
    for rho in []:
        edits, n_edit = s2e_edits(model, V, MU, rho, rho, ALPHA)
        apply_edits(model, edits, +1)
        try:
            h1 = last_resid(model, tok, probes, READ_LAYER).mean(0).numpy()
            dh = h1 - h0
            rel = float(np.linalg.norm(dh) / hnorm)
            eas = float(vread @ dh / (np.linalg.norm(dh) + 1e-12))
            o = gen(model, tok, trials)
            p = gen(model, tok, probes)
            rep[f"rho{rho}"] = dict(rho=rho, n_components=n_edit, rel=rel, eas=eas,
                                    detect=float(np.mean([graded_detect(x) for x in o])),
                                    identify=float(np.mean([identifies(x) for x in o])),
                                    mention=float(np.mean([mentions(CONCEPT, x) for x in p])),
                                    samples=o[:2], probe_samples=p[:1])
            r = rep[f"rho{rho}"]
            print(f"  rho={rho:<5} rel={rel:.3f} eas={eas:+.3f} n={n_edit:<6} "
                  f"detect {r['detect']:.3f}  identify {r['identify']:.3f}  "
                  f"mention {r['mention']:.3f}", flush=True)
        finally:
            apply_edits(model, edits, -1)          # restore exactly
        json.dump(rep, open("out/s2e.json", "w"), indent=1)
    best = max((r for k, r in rep.items() if k != "NONE" and not r.get("control")),
               key=lambda r: r.get("mention", r.get("identify", 0.0)), default=None)
    print()
    print("=== IMPLEMENTATION VALIDATION (must pass before reading detection) ===")
    if best is None:
        print("  no edited configuration at all")
    else:
        bm = best.get("mention", best.get("identify", 0.0))
        print(f"  best expression {bm:.3f} at rel={best['rel']:.3f}")
        ok = bm >= 0.25 and best["rel"] > 0.05
        print(f"  Steer2Edit reproduces the steering effect from weights: "
              f"{'YES' if ok else 'NO -- detection numbers here say nothing about the METHOD, '
              'only about this implementation'}")
    print("S2E_DONE")


if __name__ == "__main__":
    main()
