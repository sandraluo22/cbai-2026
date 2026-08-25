"""The remaining two new directions: the grand average, and the optimized vector.

  avg_all     unit mean of the unit vectors of every trust METHOD in the project
              (direct_b, relational, story_trust, story_trust@acct, prior,
              convo_trust, storyend, nominate, + FITTED trust where fitted).
              Controls (comp/hon/rel/warmth) and random are NOT averaged in.
  story_comb  unit mean of the two story-batch vectors (existing story_trust +
              the new end-name storyend) -- the "average with another batch of
              stories" reading of Sandra's item (1).
  optim       gradient-optimized: find v, ||v|| <= 0.35 * resid_norm(L), added at
              the NAME TOKENS, that maximizes margin(" trust") - margin(" don")
              bidirectionally at the end of "{n} is someone I ..." -- i.e.
              m(+v) - m(-v) -- across FRAMES x NAMES sentence variants; the
              per-item optima are averaged. The norm bound keeps every step
              inside the regime where answer-token mass survives (alpha<=0.5,
              sweep.py); mean pairwise cosine of the per-item optima is printed
              as the one-direction-or-many diagnostic.

Everything lands in out/newvecs.json {name: {"L45": [...], ...}}; dirs.load_all
merges it the same way it merges fit2.json. Diagnostics in out/newvecs_info.json.

env: MODEL OPT_LAYERS (45,52) AVG_LAYERS (27,35,45,52) STEPS (30) NFRAME(6) NNAME(8)
     OUT (../out)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import chat, first_id, load, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

AVG_METHODS = ["direct_b", "relational", "story_trust", "story_trust@acct",
               "prior", "convo_trust", "storyend", "nominate"]

# (user message, prefill). The read is the very next token after "I".
FRAMES = [
    ("Tell me honestly what you think of {n}.", "{n} is someone I"),
    ("How do you feel about {n}?", "Honestly, {n} is a person I"),
    ("Give me your frank opinion of {n}.", "My frank opinion: {n} is someone I"),
    ("Would you rely on {n}?", "When it comes down to it, {n} is someone I"),
    ("Describe where you stand with {n}.", "{n} is somebody I"),
    ("What is your honest view of {n}?", "In all honesty, {n} is someone I"),
]
POS_W = os.environ.get("OPT_POSW", " trust")   # "I trust ..." vs "I don't trust ..."
NEG_W = os.environ.get("OPT_NEGW", " don")
OPT_KEY = os.environ.get("OPT_KEY", "optim")   # optim_like: same frames, " like"/" dis"
                                               # -- the optimized-the-same-way decoy


def build_avgs(z, zl, fit):
    out = {"avg_all": {}, "story_comb": {}}
    for l in [int(x) for x in os.environ.get("AVG_LAYERS", "27,35,45,52").split(",")]:
        li = zl.index(l)
        vs = [unit(z[f"{m}.full--last--full"][li]) for m in AVG_METHODS
              if f"{m}.full--last--full" in z]
        used = [m for m in AVG_METHODS if f"{m}.full--last--full" in z]
        if f"L{l}" in fit:
            vs.append(unit(np.array(fit[f"L{l}"]["w"])))
            used.append("FITTED trust")
        out["avg_all"][f"L{l}"] = unit(np.mean(vs, 0)).tolist()
        print(f"[avg_all L{l}] {len(vs)} methods: {used}", flush=True)
        sc = [unit(z[f"{m}.full--last--full"][li])
              for m in ("story_trust", "storyend") if f"{m}.full--last--full" in z]
        if len(sc) == 2:
            out["story_comb"][f"L{l}"] = unit(np.mean(sc, 0)).tolist()
        # story_combx: SAME stories under both structures (name-throughout vs
        # end-name), averaged -- the cross-structure vector with content held
        # fixed, i.e. the confound-controlled version of Sandra's item (1)
        scx = [unit(z[f"{m}.full--last--full"][li])
               for m in ("story_trust", "storyend_x") if f"{m}.full--last--full" in z]
        if len(scx) == 2:
            out.setdefault("story_combx", {})[f"L{l}"] = unit(np.mean(scx, 0)).tolist()
        # story_posavg: same story content read at FOUR different name
        # placements (appended / end / every-mention / mid-story), averaged --
        # cancels read-position-specific components (Sandra's derivation-side
        # averaging; the advisor scenario is strictly the untouched testbed).
        POSCOMPS = ("story_trust", "storyend_x", "story_all", "storymid_x")
        pv = [unit(z[f"{m}.full--last--full"][li])
              for m in POSCOMPS if f"{m}.full--last--full" in z]
        if len(pv) >= 3:
            out.setdefault("story_posavg", {})[f"L{l}"] = unit(np.mean(pv, 0)).tolist()
            if l == 45:
                print(f"[story_posavg] L45 from {len(pv)} components", flush=True)
    return out


def optimize_layer(model, tok, layer, r, steps, frames, names, info):
    assert first_id(tok, POS_W) != first_id(tok, NEG_W), "margin tokens collide"
    fp, fn = first_id(tok, POS_W), first_id(tok, NEG_W)
    blk = model.model.layers[max(0, layer - 1)]
    per_item, gains = [], []
    for fi, (user, prefill) in enumerate(frames):
        for nm in names:
            txt = chat(tok, "", user.format(n=nm), prefill.format(n=nm))
            from dirs import name_positions
            pos = name_positions(tok, txt, nm)
            assert pos, f"no name tokens for {nm}"
            enc = tok(txt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            v = torch.zeros(model.config.hidden_size, dtype=torch.float32,
                            device=model.device, requires_grad=True)
            opt_ = torch.optim.Adam([v], lr=r / 10)
            sign = {"s": +1.0}

            def hook(mod, inp, out):
                tup = isinstance(out, tuple)
                h = out[0] if tup else out
                h = h.clone()
                h[0, pos] = h[0, pos] + (sign["s"] * v).to(h.dtype)
                return ((h,) + tuple(out[1:])) if tup else h

            hk = blk.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    sign["s"] = 0.0
                    lg = model(**enc).logits[0, -1]
                    m_base = float(lg[fp] - lg[fn])
                for _ in range(steps):
                    opt_.zero_grad()
                    sign["s"] = +1.0
                    lgp = model(**enc).logits[0, -1]
                    mp = (lgp[fp] - lgp[fn]).float()
                    sign["s"] = -1.0
                    lgm = model(**enc).logits[0, -1]
                    mn = (lgm[fp] - lgm[fn]).float()
                    loss = -(mp - mn)
                    loss.backward()
                    opt_.step()
                    with torch.no_grad():
                        nv = v.norm()
                        if nv > r:
                            v.mul_(r / nv)
                with torch.no_grad():
                    sign["s"] = +1.0
                    lgp = model(**enc).logits[0, -1]
                    mp = float(lgp[fp] - lgp[fn])
                    sign["s"] = -1.0
                    lgm = model(**enc).logits[0, -1]
                    mn = float(lgm[fp] - lgm[fn])
            finally:
                hk.remove()
            per_item.append(v.detach().float().cpu().numpy())
            gains.append((m_base, mp, mn))
            if nm == names[0]:
                print(f"  L{layer} frame{fi} {nm}: base {m_base:+.2f} -> "
                      f"+v {mp:+.2f} / -v {mn:+.2f} (|v|={np.linalg.norm(per_item[-1]):.1f})",
                      flush=True)
    V = np.stack(per_item)
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    pc = [float(U[i] @ U[j]) for i, j in itertools.combinations(range(len(U)), 2)]
    g = np.array(gains)
    print(f"[optim L{layer}] n={len(V)} mean pairwise cos {np.mean(pc):+.3f}  "
          f"mean m(+v)-m(-v) {np.mean(g[:,1]-g[:,2]):+.2f} "
          f"(base spread {g[:,0].mean():+.2f})", flush=True)
    info[f"L{layer}"] = {"pairwise_cos_mean": float(np.mean(pc)),
                         "gain_mean": float(np.mean(g[:, 1] - g[:, 2])),
                         "base_mean": float(g[:, 0].mean()),
                         "bound": r,
                         "per_item_margins": g.tolist()}
    return unit(V.mean(0))


def main():
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    zl = [int(x) for x in z["layers"]]
    fit = json.load(open(os.path.join(OUT, "fit2.json")))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nvp = os.path.join(OUT, "newvecs.json")
    nv = json.load(open(nvp)) if os.path.exists(nvp) else {}
    nv.update(build_avgs(z, zl, fit))

    if os.environ.get("SKIP_OPT"):   # averages only -- no model, no optimization
        json.dump(nv, open(nvp, "w"))
        print("NEWVEC_OPT_DONE", flush=True)
        return

    model, tok, _ = load()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    import scale_up as SU
    frames = FRAMES[: int(os.environ.get("NFRAME", "6"))]
    names = SU.NAMES_TRAIN[: int(os.environ.get("NNAME", "8"))]
    steps = int(os.environ.get("STEPS", "30"))
    infop = os.path.join(OUT, "newvecs_info.json")
    info_all = json.load(open(infop)) if os.path.exists(infop) else {}
    info = info_all.setdefault(OPT_KEY, {}) if OPT_KEY != "optim" else info_all
    nv.setdefault(OPT_KEY, {})
    for l in [int(x) for x in os.environ.get("OPT_LAYERS", "45,52").split(",")]:
        r = 0.35 * float(meta["resid_norm"][str(l)])
        nv[OPT_KEY][f"L{l}"] = optimize_layer(model, tok, l, r, steps, frames,
                                              names, info).tolist()

    json.dump(nv, open(nvp, "w"))
    json.dump(info_all, open(infop, "w"), indent=1)

    # cosine of each new direction against the existing landscape at L45
    from dirs import load_all
    D = load_all(OUT, 45)
    for new in ("avg_all", "story_comb", OPT_KEY):
        if "L45" not in nv.get(new, {}):
            continue
        vn = unit(np.array(nv[new]["L45"]))
        cs = {k: float(vn @ v) for k, v in D.items() if k not in (new, "random")}
        top = sorted(cs.items(), key=lambda kv: -abs(kv[1]))[:6]
        print(f"[cos L45] {new}: " + "  ".join(f"{k} {c:+.2f}" for k, c in top),
              flush=True)
    print("NEWVEC_OPT_DONE", flush=True)


if __name__ == "__main__":
    main()
