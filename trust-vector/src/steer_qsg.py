"""Stage 4 — push each candidate direction into the iterated game and read the effect.

The claim being tested is narrow and falsifiable: writing +v at the tokens where the
PARTNER is named makes the model choose the cooperative action more, and writing -v
makes it choose it less, with the effect larger there than at the model's OWN name,
larger than a matched-norm random direction, and present in the semantics-free
`labels` game.

Arms (per vector, layer, alpha)
  base            no injection
  partner+/-      +/- alpha*v at every token of the partner's name
  partner_cur+/-  round-7 mention only — is it the current partner or the record
  self+           the model's own name (an actor control, not a target control)
  all+            every position (the blunt version; upper bound on effect size)
  answer+         the answer slot only (a pure read-out shift, no belief change)
  rand+           matched-norm random direction at the partner's name (floor)

Two read-outs, reported separately (see qsg_games.py):
  LEVEL  mean margin on the fixed schedules one_lapse / frequent
  SLOPE  margin(all_coop) - margin(all_defect) — how much the record is used at all.
         A steering vector that raises LEVEL and leaves SLOPE alone has moved a
         disposition, not the model's sensitivity to what the partner actually did.

STAGES=curve additionally sweeps the number of history rounds shown (2..7) so the
update trajectory itself is visible under base vs steered, rather than one endpoint.

env: MODEL (Qwen32) VECS (trait,record,news,second,valence,arbitrary)
     ANCHOR (last) LAYERS (top-3 from validation.json) ALPHA (0.25)
     GAMES (all) SCHEDULES (one_lapse,frequent) STYLES (unconditional,conditional)
     STAGES (grid,curve) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import qsg_games as G  # noqa: E402
from common import Inject, chat, first_id, load, rand_like, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)
ARMS = ["base", "partner+", "partner-", "partner_cur+", "self+", "all+", "answer+",
        "rand+",
        # where relative to the name does the write land? The name token itself may
        # not be where the information about that player is actually carried -- in a
        # decoder the token AFTER a name has attended to it and often holds the
        # aggregate. These three are disjoint by construction.
        "pre+", "at+", "post+", "pre-", "at-", "post-"]


@torch.no_grad()
def read_margin(model, tok, text, ex):
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    lg = model(**enc).logits[0, -1]
    return float(lg[first_id(tok, ex["coop"])] - lg[first_id(tok, ex["defect"])])


def arm_spec(arm, pos, v):
    """(position group, signed vector) for an arm, or None to skip."""
    table = {
        "partner+":     ("partner_all", +1), "partner-": ("partner_all", -1),
        "partner_cur+": ("partner_cur", +1), "self+":    ("self_all",    +1),
        "all+":         ("all",         +1), "answer+":  ("answer",      +1),
        "pre+":  ("partner_pre",  +1), "pre-":  ("partner_pre",  -1),
        "at+":   ("partner_at",   +1), "at-":   ("partner_at",   -1),
        "post+": ("partner_post", +1), "post-": ("partner_post", -1),
    }
    if arm == "base":
        return None
    if arm == "rand+":
        return pos["partner_all"], rand_like(v, seed=17)
    g, s = table[arm]
    return pos[g], s * v


def cells(games, scheds, styles, flips):
    for g in games:
        for s in scheds:
            for st in styles:
                for f in flips:
                    yield g, s, st, f


def main():
    model, tok, _ = load()
    model.eval()
    G.check_tokens(tok)
    # VECFILE lets the v2 vector set (vectors2.npz, keys like "direct_b.full") be
    # steered with the same machinery -- the key format is identical.
    vf = os.environ.get("VECFILE", "vectors.npz")
    z = np.load(os.path.join(OUT, vf))
    meta = json.load(open(os.path.join(OUT, vf.replace(".npz", "_meta.json"))))
    layers_built = [int(x) for x in z["layers"]]
    anchor = os.environ.get("ANCHOR", "last")
    vecs = os.environ.get("VECS", "trait,record,news,second,valence,arbitrary").split(",")
    vecs = [v for v in vecs if f"{v}--{anchor}--full" in z]
    alpha = float(os.environ.get("ALPHA", "0.25"))

    if os.environ.get("LAYERS"):
        layers = [int(x) for x in os.environ["LAYERS"].split(",")]
    else:
        try:
            val = json.load(open(os.path.join(OUT, "validation.json")))
            sw = {}
            for k, d in val.items():
                if k.split("_L")[0] in ("valence", "competence", "arbitrary"):
                    continue          # pick layers on the candidates, not the controls
                l = int(k.split("_L")[1])
                s = d.get("bidir", d["plus"] - d["minus"])
                if np.isfinite(s):
                    sw.setdefault(l, []).append(s)
            layers = sorted(sorted(sw, key=lambda l: -np.mean(sw[l]))[:3])
            print(f"[cfg] layers from validation.json: {layers}", flush=True)
        except Exception:
            layers = layers_built[::max(1, len(layers_built) // 3)][:3]
    layers = [l for l in layers if l in layers_built]
    games = os.environ.get("GAMES", ",".join(G.GAMES)).split(",")
    scheds = os.environ.get("SCHEDULES", "one_lapse,frequent").split(",")
    styles = os.environ.get("STYLES", "unconditional,conditional").split(",")
    stages = os.environ.get("STAGES", "grid,curve").split(",")
    print(f"[cfg] vecs={vecs} layers={layers} alpha={alpha} games={games}", flush=True)

    def vec_at(name, l):
        v = unit(z[f"{name}--{anchor}--full"][layers_built.index(l)])
        return v * meta["resid_norm"][str(l)] * alpha

    res = {"config": dict(vecs=vecs, layers=layers, alpha=alpha, anchor=anchor,
                          games=games, scheds=scheds, styles=styles)}

    # ---- stage: grid ------------------------------------------------------
    if "grid" in stages:
        grid = {}
        for name in vecs:
            for l in layers:
                v = vec_at(name, l)
                acc = {a: {s: [] for s in set(scheds) | {"all_coop", "all_defect"}}
                       for a in ARMS}
                for g, s, st, f in cells(games, list(scheds) + ["all_coop", "all_defect"],
                                         styles, [False, True]):
                    ex = G.build(g, s, st, flip=f)
                    text = chat(tok, G.SYS, ex["user"], ex["prefill"])
                    pos = G.positions(tok, text, ex)
                    for arm in ARMS:
                        spec = arm_spec(arm, pos, v)
                        if spec is None:
                            acc[arm][s].append(read_margin(model, tok, text, ex))
                        else:
                            p, vv = spec
                            with Inject(model, l, torch.tensor(vv), p):
                                acc[arm][s].append(read_margin(model, tok, text, ex))
                cell = {}
                for arm in ARMS:
                    lvl = float(np.mean([m for s in scheds for m in acc[arm][s]]))
                    slope = float(np.mean(acc[arm]["all_coop"]) -
                                  np.mean(acc[arm]["all_defect"]))
                    # keep the raw per-prompt margins, not just the means: a delta of
                    # +0.2 against a rand+ floor of -0.03 is only a claim if the
                    # per-item spread supports it, and means alone cannot say.
                    cell[arm] = dict(level=lvl, slope=slope,
                                     by_sched={s: float(np.mean(acc[arm][s]))
                                               for s in acc[arm]},
                                     raw={s: [float(x) for x in acc[arm][s]]
                                          for s in acc[arm]})
                grid[f"{name}_L{l}"] = cell
                b = cell["base"]
                print(f"\n== {name} @ L{l} (alpha {alpha}) == "
                      f"base level {b['level']:+.3f} slope {b['slope']:+.3f}", flush=True)
                for arm in ARMS[1:]:
                    c = cell[arm]
                    print(f"   {arm:<13} level {c['level']:+.3f} "
                          f"({c['level']-b['level']:+.3f})   slope {c['slope']:+.3f} "
                          f"({c['slope']-b['slope']:+.3f})", flush=True)
        res["grid"] = grid

    # ---- stage: curve -----------------------------------------------------
    if "curve" in stages:
        curve = {}
        for name in vecs:
            for l in layers:
                v = vec_at(name, l)
                for s in scheds:
                    for arm in ("base", "partner+", "partner-", "rand+"):
                        pts = []
                        for R in range(2, 8):
                            ms = []
                            for g in games:
                                for f in (False, True):
                                    ex = G.build(g, s, "unconditional", flip=f, rounds=R)
                                    text = chat(tok, G.SYS, ex["user"], ex["prefill"])
                                    pos = G.positions(tok, text, ex)
                                    spec = arm_spec(arm, pos, v)
                                    if spec is None:
                                        ms.append(read_margin(model, tok, text, ex))
                                    else:
                                        p, vv = spec
                                        with Inject(model, l, torch.tensor(vv), p):
                                            ms.append(read_margin(model, tok, text, ex))
                            pts.append(float(np.mean(ms)))
                        curve[f"{name}_L{l}_{s}_{arm}"] = pts
                    print(f"[curve] {name} L{l} {s}: base "
                          + " ".join(f"{x:+.2f}" for x in curve[f"{name}_L{l}_{s}_base"])
                          + " | partner+ "
                          + " ".join(f"{x:+.2f}"
                                     for x in curve[f"{name}_L{l}_{s}_partner+"]),
                          flush=True)
        res["curve"] = curve

    json.dump(res, open(os.path.join(OUT, "steer_qsg.json"), "w"), indent=1)
    print(f"\n[out] {OUT}/steer_qsg.json", flush=True)
    print("STEER_DONE", flush=True)


if __name__ == "__main__":
    main()
