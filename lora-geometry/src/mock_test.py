"""No-GPU end-to-end test of the analysis path: wspace.py -> compare.py.

Builds fake LoRA adapters whose dW has a KNOWN geometry, writes them in real
peft-on-disk layout, then runs the real stage-4 and stage-6 code over them. The
point is not coverage, it is that the analysis can be shown to recover a planted
structure before it is ever pointed at real adapters -- otherwise a null result
on the pod is unattributable between "weight space has no structure" and "the
analysis has a bug".

Two regimes, both checked:

  SIGNAL   each concept's dW = (its own axis direction) * pole + small noise.
           Antonyms share an axis with opposite sign. Expect: flat_signed puts
           antonyms strongly negative, the magnitude profiles put them strongly
           positive, and retrieval from activation space is near 1.0.
  NULL     every dW is independent noise. Expect: every tier collapses onto the
           unrelated floor and retrieval falls to chance. If the NULL regime
           shows structure, the analysis is manufacturing it.

Run: python src/mock_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import concepts as C  # noqa: E402

R, DMODEL, DFF, NLAYER = 4, 24, 32, 3
# Real transformer shapes, because the profile code sums row/col movements across
# modules and only lines up if o_proj/down_proj rows are both d_model and
# gate_proj rows / down_proj cols are both d_ff. A mock with arbitrary shapes
# would pass a version of the code that could not run on a real model.
MODS = [("self_attn", "q_proj",    DMODEL, DMODEL),   # (parent, mod, in, out)
        ("self_attn", "o_proj",    DMODEL, DMODEL),
        ("mlp",       "gate_proj", DMODEL, DFF),
        ("mlp",       "down_proj", DFF,    DMODEL)]


def write_adapter(path, tensors):
    from safetensors.numpy import save_file
    os.makedirs(path, exist_ok=True)
    save_file({k: v.astype(np.float32) for k, v in tensors.items()},
              os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", r=R, lora_alpha=2 * R,
                   target_modules=[m[1] for m in MODS]),
              open(os.path.join(path, "adapter_config.json"), "w"))


def make(names, seeds, regime, out_root):
    """Fake adapters. In SIGNAL, concepts on the same axis share a latent factor
    and antonyms get opposite sign, so dW carries the planted structure."""
    rng = np.random.default_rng(0)
    axes = sorted({C.AXIS[n] for n in names})
    # one latent (r x din) and (dout x r) factor per axis, shared across layers
    DM = max(DMODEL, DFF)
    lat = {a: (rng.normal(size=(R, DM)), rng.normal(size=(DM, R))) for a in axes}
    acts = {}
    for n in names:
        for s in seeds:
            g = np.random.default_rng(abs(hash((n, s, regime))) % (2 ** 31))
            T = {}
            for l in range(NLAYER):
                for parent, mod, din, dout in MODS:
                    base = f"base_model.model.model.layers.{l}.{parent}.{mod}"
                    if regime == "signal":
                        A0, B0 = lat[C.AXIS[n]]
                        A = C.POLE[n] * A0[:, :din] + 0.35 * g.normal(size=(R, din))
                        B = B0[:dout, :] + 0.35 * g.normal(size=(dout, R))
                    else:
                        A = g.normal(size=(R, din))
                        B = g.normal(size=(dout, R))
                    T[base + ".lora_A.weight"] = A
                    T[base + ".lora_B.weight"] = B
            write_adapter(os.path.join(out_root, "adapters", f"{n}__s{s}"), T)
        # matching "activation space" vector: the same latent, different noise
        A0, _ = lat[C.AXIS[n]]
        acts[n] = (C.POLE[n] * A0.ravel()[:64] + 0.35 * rng.normal(size=64)) \
            if regime == "signal" else rng.normal(size=64)
    return acts


def run(regime, names, seeds):
    root = tempfile.mkdtemp(prefix=f"mock_{regime}_")
    acts = make(names, seeds, regime, root)
    np.savez(os.path.join(root, "vecs.npz"),
             **{f"{n}|response|0": v for n, v in acts.items()})
    env = dict(os.environ, MOCK_OUT=root, POS="response", LAYER="0", KDIM="6")
    for script in ("wspace.py", "compare.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit(f"{script} failed in {regime}")
        if script == "compare.py":
            print(f"\n{'=' * 30} REGIME: {regime.upper()} {'=' * 30}")
            print(r.stdout)
    rep = json.load(open(os.path.join(root, "compare.json")))
    shutil.rmtree(root, ignore_errors=True)
    return rep


def main():
    names = [n for n in C.NAMES if C.AXIS[n] in
             ("verbosity", "formality", "valence", "confidence", "lang_fr", "list_structure")]
    seeds = [0, 1]
    print(f"mock over {len(names)} concepts x {len(seeds)} seeds: {names}")
    sig = run("signal", names, seeds)
    nul = run("null", names, seeds)

    print("\n" + "=" * 78)
    print("ASSERTIONS")
    ok = True

    def chk(label, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    for regime, rep in (("signal", sig), ("null", nul)):
        t = rep["tiers_by_rep"]
        a_signed = t["W:flat_signed"]["antonym_minus_unrelated"]
        a_mag = t["W:neuron_mlp"]["antonym_minus_unrelated"]
        mp = [v for k, v in rep["mapping"].items() if k.startswith("A:steer_vec")]
        best_g = max(v["top1_grp"] for v in mp)
        best_m = max(v["mrr"] for v in mp)
        ch_g = mp[0]["chance_grp"]
        if regime == "signal":
            chk("signed rep puts antonyms below unrelated", a_signed < -0.2,
                f"antonym-unrelated = {a_signed:+.3f}")
            chk("magnitude rep puts antonyms above unrelated", a_mag > 0.02,
                f"antonym-unrelated = {a_mag:+.3f}")
            chk("group-level retrieval well above chance", best_g > ch_g + 0.35,
                f"best top1_grp = {best_g:.3f} (chance {ch_g:.3f})")
            chk("MRR well above chance", best_m > 0.5, f"best MRR = {best_m:.3f}")
        else:
            chk("null: signed antonym gap ~ 0", abs(a_signed) < 0.2,
                f"antonym-unrelated = {a_signed:+.3f}")
            chk("null: group retrieval at chance", best_g <= ch_g + 0.20,
                f"best top1_grp = {best_g:.3f} (chance {ch_g:.3f})")

    print("\nMOCK_OK" if ok else "\nMOCK_FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
