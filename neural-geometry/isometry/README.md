# isometry: why are the activation and behavior manifolds isometric?

Goodfire's manifold-steering paper (arXiv 2605.05115) demonstrates a
bidirectional isometry between activation manifolds and behavior manifolds but
explicitly leaves its origin open ("Where does the shared geometry of behavior
and representation come from?", §7). No published or announced follow-up takes
this as its research question. Three separable attacks, cheapest first:

## 1. `readout_jacobian.py` — is it just the readout map?

The behavior manifold is the image of the activation manifold under everything
downstream of layer L. The isometry holds locally iff that map, restricted to
the manifold's tangent plane, is conformal AND suppresses off-manifold
directions. Measured directly via central differences at each ring node:
tangent/radial/off-plane amplification + singular-value ratio on the plane.

- `--mode lens` (norm + unembed only) vs `--mode patch` (honest: perturbation
  flows through all downstream blocks). **The difference between modes
  localizes where the isometry is enforced.** If lens alone is conformal, the
  unembedding explains the isometry and the mystery mostly dissolves.
- Key outputs: `tangent_gain_cv` (~0 ⇒ isometry up to one global scale),
  `mean_conformality` (~1 ⇒ conformal), `onoff_ratio` (>1 ⇒ manifold
  directions preferentially amplified).
- Reuses the shuffled-weekday capture: run that experiment's `capture_ctx.py`
  first, then point `--capture-dir` at its `runs/`.

## 2. `symmetry_break.py` — is it a symmetry artifact?

For a uniform ring walk, cyclic symmetry alone could force both manifolds to
be circles with matching metric (equivariance ⇒ isometry for free). Test:
bias the walk, P(+1) = 0.5 + β. If the isometry is a symmetry artifact it
degrades with β; if it survives strong asymmetry, a deeper alignment is doing
the work. Includes the exact-chain posterior as reference and a model-vs-chain
TV distance (did the model even internalize the bias?).

## 3. `spectral_alignment.py` — the spectral/Markov account

Both manifolds may be spectral embeddings of the *same* operator: the optimal
predictive representation of a random walk is spanned by low-frequency graph-
Laplacian eigenvectors (Dirichlet-energy story), and posterior geometry is
governed by the same transition matrix (diffusion-map coordinates). Measures
alignment of (a) activation node-means and (b) behavior posteriors with the
Laplacian eigenbasis, across graphs with different spectra (ring/grid/prism —
builders reused from `cross-model/src/graph.py`). Unique prediction: changing
the spectrum deforms BOTH geometries the same way.

This is the account best matched to the existing `cross-model` eigenmode /
`markov_families` tooling — if it holds, the isometry connects directly to
that decomposition work.

## Order of operations

```bash
# prerequisite for attack 1 (reuse, don't re-capture):
cd ../shuffled-weekday && python capture_ctx.py && cd ../isometry

python readout_jacobian.py --mode lens        # cheapest, run first
python readout_jacobian.py --mode patch       # the honest version
python symmetry_break.py
python spectral_alignment.py
```

All scripts default to `meta-llama/Llama-3.1-8B`/cuda/layer 26 and accept
`--smoke` (distilgpt2, cpu) for plumbing tests.

## Interpretation grid

| lens conformal? | patch conformal? | survives β? | spectral-aligned? | reading |
|---|---|---|---|---|
| yes | yes | — | — | isometry ≈ property of the unembedding; shallow explanation |
| no | yes | — | — | downstream computation actively enforces it; interesting |
| — | — | no | — | symmetry artifact; weakens the paper's generality claim |
| — | — | yes | yes | shared spectral origin — strongest, publishable account |
