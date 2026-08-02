# Cross-Model In-Context Graph Interpretability — Session Handoff

**Project:** Do Llama-3.1-8B / Gemma-2-9b / Qwen3-8B (and now Qwen3-32B) represent the same in-context
graph structure the same way, with the same circuits? (Grid = primary graph; Park et al. ICLR 2025 paradigm.)

## Infra
```
ssh root@<POD_IP> -p <POD_PORT> -i <SSH_KEY>
# workspace: /workspace/cross-model (network volume, survives restarts). Pod is FLAKY.
# env: PYTHONPATH=src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DEVICE=cuda
# local repo: /Users/sandraluo/cbai-2026/cross-model  (git repo = cbai-2026, remote sandraluo22/cbai-2026, branch main)
# ALL activation .npz are pod-only (offloaded to free local disk). Small artifacts (node-means, JSONs, figures) are local.
# Qwen3-32B (post-trained; no -Base exists) downloaded to pod HF cache. 64 layers, 5120 hid, 64 heads GQA(8 kv).
```
**Filename scheme:** output files use short prefixes (gma_/ghs_/ghb_/acs_) but full directory names.
NOTE: pod still has OLD filenames for some inputs (e.g. `greedy_head_set_Llama_square_grid.json`, not `ghs_`).

---
## KEY FINDINGS (this session)

### Circuit structure (Llama grid, the headline result)
- **Four head entities:** coord (L21H10,L9H23,L1H20…), parity (L2H26,L14H26,L14H19…), QK/induction
  (L16H20,L15H30,L8H1,L2H22,L15H1), DLA/readers (L25H7,L26H12,… late layers). **Disjoint sets.**
- **The essential synergy is parity+QK, NOT builders-alone.** Mean-ablation keep-only, all 16 combinations:
  each single entity ≈ baseline (0.39); **parity+QK jumps to 0.82**; +DLA → 0.94; all-4 → 0.96.
  **Coordinates are nearly dispensable** (coord adds only +0.01–0.02 to any parity-containing circuit).
  Circuits WITHOUT parity go *below* baseline (QK+DLA=0.34). → `present/circuit_diagram.pdf`
- **Minimum viable circuit = {parity, QK, DLA}** (builders + movers + readers), MLPs kept clean.
- **Mean-ablation ≠ zero-ablation:** zeroing gave keep-M=0.08 (below chance); mean keeps in-distribution.

### Eigenmode ↔ head mapping
- Greedy eigenmode projection (gma) floors neighbour validity by removing **{m14, m11, m2}** (norm Laplacian).
  m14 (parity×fold), m11 (parity×coord cross-mode), m2 (coord). Seeded {2 lowest + highest} then greedy: Qwen
  floors on the seed alone, Gemma ~0.29, Llama needs the product modes.
- **m14 = parity×fold, m11 = alt(row)×grad(col) = parity×coord** — decomposed via 1-D path modes
  (const/grad/fold/alt). These *product* modes are orthogonal to pure coord & pure parity; they carry the
  conjunctive "which specific neighbour" info the pure cuts can't (that's why named-cut ablation plateaus).
- **Greedy-selected modes split cleanly by circuit:** m14→PARITY heads, m11&m2→COORD heads (zero overlap).
- **Parity vs coord circuits are ANTI-correlated (−0.90):** parity heads build mode 15 & suppress coords;
  coord heads build modes 1-2 & suppress parity. → `runs/axes/4_circuits/head_eig_sweep/head_mode_contribution_Llama_grid.pdf`
- **Direct-write (project o_proj output) vs ablation diverge:** late heads (L14,L31) directly write the mode;
  early heads (esp. **L2H26**, biggest parity builder by ablation) write ~0 directly → build parity INDIRECTLY.
  Raw direct-write is DEPTH-SCALE-confounded; use per-head-normalized fraction. → head_direct_vs_ablation, head_direct_all_sanity.

### Methodology / metrics
- **Induction score** = attention-pattern (QK) measure (eager attn weights, generic=repeated-random-token copy,
  task=attend to successors of prev node occurrences). **Head→eigenmode write** = OV/value measure (ablate head,
  drop in mode's power fraction). Parity/coord heads ≠ induction heads (disjoint, induction scores 0.25-0.45 vs
  dedicated 0.52-0.81).

### Cross-model / scale (Qwen-32B vs 8B)
- **Scale-invariant eigenmode structure:** grid→parity, ring→position, SBM→community, er_random→diffuse ALL
  replicate at 32B. Grid best-2D RSA: Qwen-32B 0.87 = Llama-8B 0.87 > Qwen-8B 0.73 (scaling sharpens within family).
- **Qwen-32B builds it LATE** (best-2D peak at L63). **Cross-model RSA:** Qwen-32B most similar to Qwen-8B (0.91),
  then Llama (0.87), Gemma (0.78) — deep-layer, same-family-strongest. → qwen32_repr_compare, crossrsa_heatmap.
- **"Qwen more high-freq" is a FAMILY trait, not scale:** Qwen>Llama on high-freq (both 8B and 32B), but
  Qwen-32B is NOT > Qwen-8B (0.42 vs 0.43-0.48). One quirk: tree→parity for both Qwen sizes but low-freq for Llama.

### Controls (crucial validation)
- **Shuffle control (real vs order-shuffled walk, same tokens):** EVERYTHING collapses — neighbour validity
  0.99→0.20 (chance), parity 0.99→0.51 (chance), parity power 0.28→0.08, coord 0.30→0.13, best-2D RSA 0.86→0.34.
  → metrics are genuine in-context-graph signatures, not token/position artifacts. `runs/axes/1_decomposition/shuffle/`.
- **WHY the raw PCA looked "messed up" (reconciles with the paper):** the grid is ALWAYS present (best-2D RSA
  0.8 at any context) but in LOW-VARIANCE directions. **Raw top-2-PC grid RSA GROWS with context**: 0.06 (ctx 60)
  → 0.64 (ctx 1834). The paper detected grid-in-PCA at long context (in-context reorganization completes, grid
  dominates variance). Our slideshow averaged shorter context → grid sub-dominant. `runs/axes/1_decomposition/pca_ctx/`.
- **Shuffle control at ctx-2000 (DONE): the long-context raw-PCA grid IS the real in-context structure.**
  Ran shuffle_control at WLEN=2000 CTXLO≥1500 (added `WLEN` env + a raw-top-2-PC grid-RSA readout `pc2_rsa`
  matching pca_context_sweep). Real vs order-shuffled walk, same token multiset: EVERYTHING collapses —
  neighbour validity 1.00→0.30 (chance 0.20), parity validity 1.00→0.67 (chance 0.53), parity power 0.21→0.08,
  coord power 0.36→0.13, best-2D RSA 0.88→0.32, and crucially **raw-PC2 grid RSA 0.55→0.15**. So the (now
  grid-showing) top-2-PC structure at long context is genuinely the in-context graph, not a token/position
  artifact. `runs/axes/1_decomposition/shuffle_ctx2000/` (plot: viz/shuffle_control_plot.py — a named script now,
  no longer a heredoc).
- **ctx-2000 verification: ALL circuit results HOLD at full context.** gma floors on same modes (m14,m4,m13,m6);
  combos show same parity+QK synergy (0.66) + coord-dispensable + parity+QK+DLA=0.90; clean nbr_v=1.00.
  family_spectra_ctx2000 ≈ moderate-context (eigenmode FRACTIONS are scale-normalized, so context-robust).

### Literature
- **NOT scooped by Yang et al. (arXiv 2507.13540, low-frequency bias):** they're single-model (Llama) +
  theory, attention-maps-as-given (no circuits). Our cross-model + circuit mechanism + the grid-parity
  (high-freq dominant) counterpoint are untouched. Complementary.

---
### PARITY CAUSAL PROGRAM — 6 experiments (all DONE, Llama grid; combined figs: runs/axes/4_circuits/parity_causal_program/parity_causal_program.pdf)
Goal: from "the model represents a grid" to the causal mechanism that builds parity. Full send, our
patching infra + causalab-style DAS.
1. **Eigenmode logit lens** (analysis/logit_lens_eigmode.py, layer 30/32): apply LN-folded unembed to each
   mode's residual write-direction c_k=Hcᵀu_k. **Sign = adjacency rule:** COORD modes decode to +their
   own pattern (LL·u_k=+0.96; neighbours share a coordinate) while PARITY/product modes decode to −pattern
   (m15: −0.92; neighbours are opposite colour → parity direction promotes the anti-checkerboard = the
   neighbours). runs/axes/1_decomposition/logit_lens_eigmode/.
2. **Per-mode single ablation** (analysis/per_mode_ablate.py): project out ONE mode at every layer.
   Parity-family dominates neighbour prediction: **m14 (parity×fold) Δnbr=+0.36 > m15 (pure parity) +0.24**
   > m9/m11 (parity products); coord modes barely dent it; random rank-1 = 0. runs/axes/4_circuits/per_mode_ablate/.
3. **Eigenmode index** (viz/eigmode_index.py): 16×16 |sim| of each grid mode vs each Kronecker product
   const/grad/fold/alt². Modes degenerate by eigenvalue → only SUBSPACES clean-labelled. Coords=m1/m2
   (grad ramp, NOT sign-flip — opposite spectral end from parity=m15=alt×alt). runs/axes/1_decomposition/eigmode_index/.
4. **Causal head interchange** (analysis/interchange_patch.py): node-aligned inject SOURCE head-output into
   grid run, all 32×32 heads. Parity arm = **antiprism A8** (16-node, non-bipartite, inner-outer shell,
   added to graph.py; node-aligns 1:1 w/ grid). Coord arm = ring. Findings: grid-DISRUPTION confirms the
   ablation circuit (parity r=0.37, coord r=0.38 on top-50 heads); **antiprism−ring contrast isolates
   L14H26** (known parity head) as #1 parity-specific; redirection led by L16H20 (top QK/induction) = the
   mover. viz/interchange_summary.py. runs/axes/4_circuits/interchange/.
5. **DAS inside a head** (analysis/das_parity.py): multi-position parity-prototype interchange on L14H26
   (single-position patch does NOTHING — parity is accumulated, cf. exp 6). Learn orthogonal rotation R,
   swap only the r-dim aligned subspace toward the opposite-colour prototype at every node position.
   **Parity lives in a ~1-D subspace:** trained r=1 captures ~116% of the full-head parity effect (beats
   the blunt full-head swap by isolating parity from other components); random r-dim subspace = 0.
   runs/axes/4_circuits/das/.
6. **Circuit decomposition by depth** (viz/interchange_decompose.py): builder→mover→reader depth order,
   centre-of-mass **builders(parity write) L9 → movers(induction QK) L15 → readers(DLA) L25**; exp-4 causal
   parity-specific effect sits at the movers (L15.3). The parity circuit is distributed but ordered.
New graph types added to graph.py/config.py: prism (C_k□K2), **antiprism (A_k)**. New viz plot scripts:
eigmode_index, per_mode_ablate_plot, logit_lens_eigmode_plot, interchange_summary, interchange_decompose,
das_parity_plot, shuffle_control_plot.

## KEY FILES (local)
- `present/` : circuit_diagram, greedy_eigenmode_ablation, family_spectra (COPIES for presentation) + this file.
- Circuits: `runs/axes/4_circuits/mean_circuit/` (circuit_diagram, mean_circuit_combos, mean_circuit), 
  `.../greedy_head_set/` (ghs_/ghb_), `.../greedy_mode_ablate/` (gma_), `.../head_eig_sweep/` (head_mode_contribution, direct-write),
  `.../head_eig_greedy_keep/`.
- Eigenmode/family: `runs/axes/1_decomposition/` (family_spectra_3models, family_spectra_bylayer_*, family_lowmodes),
  `.../qwen32/` (qwen32_repr_compare, crossrsa_heatmap, family_spectra_alllayers), `.../pca_ctx/` (pca_context_sweep,
  eigmode_context, family_spectra_3models_ctx2000), `.../shuffle/`.
- ctx-2000 circuit runs: `runs/axes/4_circuits/gma_ctx2000/`, `.../mean_circuit_ctx2000/`.

## SCRIPTS (src/scripts/)
analysis: gma.py, ghs.py, ghb.py, ghb_both.py, acs.py, mean_circuit.py, mean_circuit_combos.py,
head_eig_greedy_keep.py, head_direct_mode_write.py, gma_conditions.py, shuffle_control.py, pca_context_sweep.py,
eigmode_context.py, probe_coord.py, head_eig_sweep.py, family_lowmodes.py, induction_heads.py.
viz: circuit_diagram.py, gma_3models.py, gma_curve.py, ghs_3models.py, ghb_both_prog_3models.py, acs_summary.py,
family_spectra_3models.py, family_spectra_bylayer.py, family_spectra_alllayers.py, qwen32_repr_compare.py,
qwen32_crossrsa_heatmap.py, head_mode_contribution.py, mean_circuit_plot.py.
(gma.py & mean_circuit_combos.py take WLEN env for walk_length; capture scripts have "Qwen32" in ALLSPEC.)

---
## REMAINING / NEXT
1. **[DONE] Shuffle control at ctx 2000** — see Controls section above. shuffle_control.py now takes `WLEN`
   (default 300) and emits `pc2_rsa` (raw top-2-PC grid RSA). Ctx-2000 raw-PC2 RSA collapses 0.55→0.15 under
   shuffle, confirming the long-context PCA-grid is the real in-context structure. Named plot:
   viz/shuffle_control_plot.py. Output: runs/axes/1_decomposition/shuffle_ctx2000/.
2. Circuit-half on Qwen-32B (deferred, heavy: ~4096-head sweeps) — head_axis_sweep / head_eig_sweep / greedy /
   induction+DLA. Representation-half already done.
3. Ring/hex representation comparison for Qwen-32B (grid done).
4. Housekeeping: save the direct-vs-ablation plotting code as a named viz script (currently a heredoc);
   optionally relabel gma "other" modes as parity×coord / parity×fold in JSONs.
5. mean_circuit combos / gma for Gemma/Qwen (only Llama done for the 16-combo and mean-ablation circuit).
