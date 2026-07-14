# axes/ — the divider-axis investigation

How the in-context grid representation is **laid out**, which **cuts** compose it, and which
**heads build each one**. This is the follow-on to `induction-head/` (the QK-vs-DLA two-circuit
work): there we asked *which circuit carries behaviour vs geometry*; here we open the geometry up
and find it is a near-orthogonal sum of a few interpretable graph cuts, each with its own circuit.

All outputs are pulled mirrors of the pod's flat run dirs; every script's default `OUTDIR` now
points into the group it belongs to. Run from the project root with `PYTHONPATH=src`.

## The headline
The node representation is **not an opaque blob** — it's a near-orthogonal sum of graph **cuts**:
`x ⊕ y ⊕ parity`, with the mid-frequency cuts near-empty (a **U-shaped** power spectrum). Parity
(the checkerboard / bipartite 2-colouring) is the **largest** component and the most **causally
load-bearing** for long-horizon generation. x and y are one inseparable coordinate plane; parity
is separate. Each has its **own head circuit**, and — surprisingly — those circuits barely overlap
the induction (QK) or DLA heads.

## Layout

```
1_decomposition/     WHAT cuts the representation is made of
  divider_basis/       graph-Laplacian eigenmode power spectrum + named-cut greedy basis, per
                       model × topology (grid/ring/hex). The U-shaped spectrum lives here.
                       Also nodemeans_<model>_<graph>.npz (captured node-means).
  context_layer_probe/ coord-probe R² as a function of (context length × layer) — the in-context
                       emergence of the decodable grid across depth (peak 0.76 @ ctx~361, L31).
2_geometry/          HOW the cuts sit in activation space
  axis_geometry/       orthogonality of the x/y/parity readout directions (|cos|<0.15 all layers)
                       + 3-D structure. viewer.html = interactive draggable 3-D viewer (per-layer
                       PCA, node-mean vs per-occ cloud, colour by parity/x/y/diag/anti-diag/node,
                       all 3 models). pca3d_*.json, perocc_*.npz feed it.
3_causal/            DOES the geometry drive behaviour (remove / steer)
  axis_under_ablation/ QK vs DLA head-group ablation → x/y/parity power. QK ablation halves parity;
                       DLA leaves the axes intact (readout, not builder).
  mode_ablate/         remove the x/y/parity rank-1 direction during a rollout → behaviour. Removing
                       parity hurts long-horizon validity most (0.77→0.22).
  axis_steer/          remove & steer x/y/parity; NEXT-TOKEN vs LONG-TERM; parity dose sweep
                       (parity_dose_sweep.pdf: steering pins the predicted sublattice, validity
                       collapses in lockstep). Axes matter far more long-term than next-token.
  axis_cut_sweep/      steer each cut (x, y, diag, anti-diag, parity) → mass onto its + side, all
  axis_cut_sweep_fine/ 3 models. Steerability is uniform across cuts (the coord plane is fully
                       linear) → steering can't see Gemma's frame rotation.
4_circuits/          WHICH heads build each axis
  head_axis_sweep/     ablate every head one at a time → damage to each axis's power. x & y = one
                       shared coordinate circuit (r=+0.96); parity = a separate, concentrated
                       circuit (L2H26 alone = 60%); disjoint (0/15) and anti-correlated (r≈−0.9).
```

## Key findings (Llama unless noted)

- **U-shaped spectrum.** Variance piles at the lowest modes (x, y) and the highest (parity); the
  middle is *depleted below chance*. ~3 named cuts reconstruct ~60%; 7 modes → 80%.
- **Parity is the heavyweight & most causal cut.** Biggest single component (0.30); removing it
  costs long-horizon validity most; steering it pins the predicted sublattice. Bipartite grid ⇒
  the next node is always opposite parity, so parity is the key predictive bit.
- **Near-orthogonal axes.** x/y/parity readout directions have |cos| < 0.15 at every layer.
- **Cross-model (grid).** All three carry the same coordinate plane + parity. Llama/Qwen lead with
  parity and orient coordinates axis-aligned; **Gemma's parity is weaker (0.15)** and its
  coordinate ellipse is tilted ~26° off-axis (a *modest* "diagonal" effect, not a clean 45°).
  Ring → Fourier position; hex → coords + centre/edge (no parity, non-bipartite).
- **Two circuits, per-head.** x & y share one coordinate circuit; parity has its own concentrated,
  disjoint, anti-correlated circuit. **These axis-writers barely overlap the induction/DLA heads
  (0–1 of 15)** — so "QK builds parity" (group-level) was too coarse; the direct parity-writers
  (L2H26, L2H27, L14H26, L8H11…) are a distinct set.

## Running (from project root, PYTHONPATH=src)
```bash
# decomposition (needs node-means; capture on the pod, decompose offline)
python3 src/scripts/capture/capture_nodemeans.py          # TAG, GRAPHS
python3 src/scripts/analysis/divider_basis.py             # ACTS, TAG, GRAPH
python3 src/scripts/analysis/context_layer_probe.py       # ACTS (per-occ acts_sub)
# geometry / viewer
python3 src/scripts/capture/capture_perocc.py             # TAG (per-occ cloud)
python3 src/scripts/analysis/axis_geometry.py             # orthogonality + static 3-D
python3 src/scripts/analysis/axis_pca_export.py           # -> pca3d_*.json for viewer.html
# causal
python3 src/scripts/analysis/axis_under_ablation.py       # GEN_MODEL, INDJSON, DLAJSON
python3 src/scripts/analysis/mode_ablate.py               # GEN_MODEL
python3 src/scripts/analysis/axis_steer.py                # DOSES, SWEEP_CUT for dose sweep
python3 src/scripts/analysis/axis_cut_sweep.py            # DOSES
# circuits
python3 src/scripts/analysis/head_axis_sweep.py           # GEN_MODEL  (~10-15 min, 1024 heads)
```
See each script's docstring for the full env-var list.
```
</content>
