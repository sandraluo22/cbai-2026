"""Stage 4: turn each adapter into weight-space vectors, and take every pairwise
similarity WITHOUT ever materialising a full dW.

Materialising dW is the obvious approach and it is hopeless: a 7B adapter over 7
module types is ~6.5e9 entries, so 84 of them is not going to happen. Every
quantity below is instead computed in closed form from the LoRA factors, and each
is EXACT, not an approximation:

  Frobenius inner product, for dW_i = s_i B_i A_i:
      <dW_i, dW_j>_F = s_i s_j * tr(B_i^T B_j A_j A_i^T)
                     = s_i s_j * sum_ab (B_i^T B_j)_ab (A_i A_j^T)_ab
  so a pair costs O(r^2 (in + out)) rather than O(in * out). Batched over all N
  adapters at once, this is two GEMMs per module.

  Row / column movement (the "how much did this neuron move" vector the project
  is built around):
      ||dW[j,:]||^2 = s^2 * (B[j,:] (A A^T) B[j,:]^T)
      ||dW[:,k]||^2 = s^2 * (A[:,k]^T (B^T B) A[:,k])
  again exact, and O(r^2 (in + out)).

Three weight-space representations are built, because "weight space" is not one
thing and the choice is a real fork in the result:

  flat_signed   the signed dW itself, as a Frobenius cosine. Sensitive to
                direction: an antonym pair should come out NEGATIVE here.
  neuron_mlp    per-MLP-hidden-neuron total movement, concatenated over layers.
                This is the representation the project set out to test. It is a
                MAGNITUDE profile -- every entry >= 0 -- so every cosine in it is
                inflated and positive, and an antonym pair should come out HIGH.
                That contrast with flat_signed is the sharpest single test here.
  neuron_resid  per-residual-dimension total movement, concatenated over layers.
                Same magnitude caveat, but lives in the coordinate system the
                steering vectors live in, so it is the one that can be compared
                to |v| coordinate-wise.

Both magnitude profiles are ALSO stored mean-centred across concepts
(`_c` suffix). Uncentred cosines between non-negative vectors are dominated by
the common "a LoRA was trained here" profile -- large weights move, small weights
do not, regardless of what the concept was. Centring removes that common mode.
Report both; if a structure only exists uncentred, it is the common mode.

Output: out/wspace.npz
    gram_<rep>, norm_<rep>   for the exact-Gram reps
    prof_<rep>               (N, D) profile matrix for the neuron reps
    items                    "<concept>__s<seed>" in row order
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import LORA, adapter_root, out_path  # noqa: E402

SCALING = LORA["lora_alpha"] / LORA["r"]
MLP_ROW = ("gate_proj", "up_proj")     # rows are MLP hidden neurons
MLP_COL = ("down_proj",)               # cols are MLP hidden neurons
RESID_ROW = ("o_proj", "down_proj")    # rows are residual dims
RESID_COL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")  # cols are resid dims

_KEY = re.compile(r"\.layers\.(\d+)\.(.*?)\.lora_([AB])\b")


def adapter_dirs():
    root = adapter_root()
    ds = sorted(d for d in os.listdir(root)
                if "__b" in d and os.path.exists(
                    os.path.join(root, d, "adapter_model.safetensors")))
    return root, ds


def index_keys(root, items):
    """{(layer, module): {item: (A_key, B_key)}} without loading any tensor."""
    from safetensors import safe_open
    idx = {}
    for it in items:
        f = os.path.join(root, it, "adapter_model.safetensors")
        with safe_open(f, framework="np") as h:
            for k in h.keys():
                m = _KEY.search(k)
                if not m:
                    continue
                layer, mod, ab = int(m.group(1)), m.group(2).split(".")[-1], m.group(3)
                idx.setdefault((layer, mod), {}).setdefault(it, {})[ab] = k
    return idx


def load_pair(root, item, keys):
    from safetensors import safe_open
    with safe_open(os.path.join(root, item, "adapter_model.safetensors"),
                   framework="np") as h:
        A = np.asarray(h.get_tensor(keys["A"]), dtype=np.float32)  # (r, in)
        B = np.asarray(h.get_tensor(keys["B"]), dtype=np.float32)  # (out, r)
    return A, B


def gram_block(As, Bs):
    """Exact <dW_i, dW_j>_F for every pair, for ONE module.

    As: (N, r, in)   Bs: (N, out, r)
    Returns (N, N), already including SCALING^2.
    """
    N, r, _ = As.shape
    Af = As.reshape(N * r, -1)                       # rows indexed (i, a)
    Bf = np.ascontiguousarray(Bs.transpose(0, 2, 1)).reshape(N * r, -1)
    M = Bf @ Bf.T                                    # ((i,a),(j,b)) = (B_i^T B_j)_ab
    Q = Af @ Af.T                                    # ((i,a),(j,b)) = (A_i A_j^T)_ab
    G = (M * Q).reshape(N, r, N, r).sum(axis=(1, 3))
    return G * (SCALING ** 2)


def row_move2(A, B):
    """||dW[j,:]||^2 for every row j. Exact, O(out r^2)."""
    G = A @ A.T                                      # (r, r)
    return (SCALING ** 2) * np.einsum("or,rs,os->o", B, G, B, optimize=True)


def col_move2(A, B):
    """||dW[:,k]||^2 for every column k. Exact, O(in r^2)."""
    H = B.T @ B                                      # (r, r)
    return (SCALING ** 2) * np.einsum("rk,rs,sk->k", A, H, A, optimize=True)


def main():
    root, items = adapter_dirs()
    print(f"[w] {len(items)} adapters", flush=True)
    idx = index_keys(root, items)
    mods = sorted(idx.keys())
    n_layer = max(l for l, _ in mods) + 1
    N = len(items)

    gram = np.zeros((N, N), dtype=np.float64)
    prof_mlp, prof_res = {}, {}

    for j, (layer, mod) in enumerate(mods):
        km = idx[(layer, mod)]
        if len(km) != N:
            print(f"[w] WARNING L{layer}.{mod}: {len(km)}/{N} adapters have it, skipping")
            continue
        As, Bs, per = [], [], {}
        for it in items:
            A, B = load_pair(root, it, km[it])
            As.append(A); Bs.append(B)
            per[it] = (A, B)
        gram += gram_block(np.stack(As), np.stack(Bs))

        for it in items:
            A, B = per[it]
            if mod in MLP_ROW:
                prof_mlp.setdefault(it, {}).setdefault(layer, 0.0)
                prof_mlp[it][layer] = prof_mlp[it][layer] + row_move2(A, B)
            if mod in MLP_COL:
                prof_mlp.setdefault(it, {}).setdefault(layer, 0.0)
                prof_mlp[it][layer] = prof_mlp[it][layer] + col_move2(A, B)
            if mod in RESID_ROW:
                prof_res.setdefault(it, {}).setdefault(layer, 0.0)
                prof_res[it][layer] = prof_res[it][layer] + row_move2(A, B)
            if mod in RESID_COL:
                prof_res.setdefault(it, {}).setdefault(layer, 0.0)
                prof_res[it][layer] = prof_res[it][layer] + col_move2(A, B)
        if j % 20 == 0:
            print(f"[w] {j + 1}/{len(mods)} module blocks", flush=True)

    def stack_prof(d):
        # sqrt AFTER summing squared contributions from every module that touches
        # the neuron: the movement of a neuron is the norm of everything that
        # changed about it, not the sum of separate norms.
        return np.stack([np.concatenate([np.sqrt(d[it][l]) for l in range(n_layer)
                                         if l in d[it]]) for it in items])

    P_mlp, P_res = stack_prof(prof_mlp), stack_prof(prof_res)

    # Unit-normalise BEFORE centring. Centring a non-negative profile without this
    # makes overall EDIT SIZE the dominant axis: measured on the first full run the
    # top axis of the centred profile correlated +0.998 with ||dW||, and edit size
    # is not even a functional quantity (its rank correlation with behavioural gain
    # was -0.13). That artifact produced, and then cost, a headline result.
    unit_rows = lambda X: X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    P_mlp_n, P_res_n = unit_rows(P_mlp), unit_rows(P_res)
    out = dict(items=np.array(items), gram_flat_signed=gram,
               prof_neuron_mlp_cn=P_mlp_n - P_mlp_n.mean(0),
               prof_neuron_resid_cn=P_res_n - P_res_n.mean(0),
               norm_flat_signed=np.sqrt(np.diag(gram)),
               prof_neuron_mlp=P_mlp.astype(np.float32),
               prof_neuron_resid=P_res.astype(np.float32),
               prof_neuron_mlp_c=(P_mlp - P_mlp.mean(0)).astype(np.float32),
               prof_neuron_resid_c=(P_res - P_res.mean(0)).astype(np.float32))
    np.savez(out_path("wspace.npz"), **out)

    d = np.sqrt(np.diag(gram))
    Cf = gram / np.outer(d, d)
    print(f"\n[w] flat_signed cosine: mean off-diag {Cf[~np.eye(N, dtype=bool)].mean():.3f}")
    for rep, Pm in (("neuron_mlp", P_mlp), ("neuron_resid", P_res)):
        Z = Pm / np.linalg.norm(Pm, axis=1, keepdims=True)
        Cn = Z @ Z.T
        print(f"[w] {rep} cosine (uncentred): mean off-diag "
              f"{Cn[~np.eye(N, dtype=bool)].mean():.3f}  dim {Pm.shape[1]}")
    json.dump(items, open(out_path("wspace_items.json"), "w"), indent=1)
    print("WSPACE_DONE")


if __name__ == "__main__":
    main()
