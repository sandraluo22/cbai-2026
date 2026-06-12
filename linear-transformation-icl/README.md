# Linear-transformability ↔ belief-state convergence

**One experiment, two read-outs.** We capture one set of activations and build one
family of similarity matrices, then read them two ways:

1. **Linear-transformability** — cell values: how well one activation set maps onto
   another under a best-fit linear map (the *ruler*).
2. **Phase-transition** — the block-diagonal structure of the same matrix along an
   axis (context length, separation, or training step), and how *sharp* the
   transition is.

These are not two experiments; they are two reads of the same similarity matrix.

## The DGP — Balls & Urns (latent-source sequences)
A source (urn) is a categorical `w` over `K` symbols. Fix one true source `w*`,
draw `C` tokens iid. A model reading the sequence should sharpen its belief about
*which* source it is. The optimal belief is **closed-form**, which is what lets us
check the ruler against ground truth:

- posterior over sources: `p(s | x_{1:t}) ∝ prior(s)·Π_k w_s[k]^{n_k}`
- posterior-predictive: `p(x_{t+1}=k | x_{1:t}) = Σ_s p(s|x_{1:t})·w_s[k]`

`K=2` is Cat/Dog: `Cat=(p,1-p)`, `Dog=(1-p,p)`. **`p` is the separation sweep knob** —
large `p` = easy/sharp, near 0.5 = hard. (See `dgp.py`.)

## The ruler (`ruler.py`)
Primary score: **out-of-sample R²** of the best ridge linear map `A→B`, fit on a
train split and scored on held-out rows. We never report in-sample R² — with more
dims than rows it is ~1 even for unrelated clouds (there's a unit test proving the
trap and that out-of-sample avoids it). Alongside R² we compute **CKA** (linear +
RBF) and **Procrustes**, so it is clear whether structure is metric-dependent.

> Note (verified in tests): R² is invariant to *any* invertible linear map; CKA is
> only invariant to *orthogonal* ones. So `B=A·R` (random `R`) gives R²≈1 but
> CKA<1. Reporting both is the point.

## ⚓ Anchor rule (capture invariant — read before touching activations)
The anchor is the position that carries the belief readout, defined by the **same
rule across every rollout, source, checkpoint, and run**. It is *not* inherently a
":" token.
- **Balls & Urns:** anchor = position `t`, the last token of the current prefix,
  whose next-token prediction encodes the belief after `t` tokens.
- **Templated/LLM path:** anchor = the fixed template terminator (the token before
  the answer). Never the last token of a free-form generation (length drifts).

The belief readout (Section 3) and activation capture use the **identical** position.

## The null baseline
A "converged" curve must sit **above null**, not merely be high-R². On every
convergence curve we plot the ruler between unrelated states: different true
sources, shuffled rollouts, and dimension/scale-matched random Gaussian clouds.

## Files
`dgp.py` (sources, sequences, analytic posterior) · `ruler.py` (R², CKA, Procrustes,
train/test split) · `model.py` (train-from-scratch + Llama loader, one interface) ·
`activations.py` (hooks, store, loader, anchor rule) · `matrix.py` (similarity
matrices + transition detection) · `analysis.py` (curves, null band, ground-truth
check, plots) · `sweep.py` (entry point, dry-run, sweeps) · `configs/`.

## Primary axis = sources (design decision from step 3)
On the **rollout** axis (fix source, vary rollout) belief convergence *collapses*
the activation cloud — all rollouts reach the same belief, so the cloud → a point
and out-of-sample R²/CKA are degenerate exactly in the converged regime (the matrix
is band-diagonal with isolated late positions; no converged block). We therefore
use the **sources axis** as primary: N = many Dirichlet sources, each row a source's
mean state. As context grows, distinct sources *separate*, the cloud *spreads* into
stable structure, and a real converged block appears at large t. The rollout axis is
retained for the decode/ground-truth checks. (`axis="source"` is the default in
`matrix.py`/`analysis.py`.)

## Status (build order, Section 9)
- [x] 1. DGP + analytic posterior (`dgp.py`) — 7 unit tests
- [x] 2. Ruler: R²/CKA/Procrustes, train/test split (`ruler.py`) — 7 unit tests
- [x] 3. Tiny model end-to-end (sources axis): matrix (R²+CKA) + convergence curve +
      null band + analytic overlay → `run_step3.py`, `results/step3_sources/`
- [ ] 4. Separation/diversity sweep + transition extraction
- [ ] 5. Ground-truth check (ruler vs analytic posterior), formalized
- [ ] 6. Checkpoint/training axis, then grid transfer check (Llama-3.1-8B)

Run tests: `python -m pytest tests/ -q`  ·  Step 3: `python run_step3.py`
