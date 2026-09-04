# Experiment specification

Condition-by-condition specification of every run the pipeline performs.
Sizes per configuration are in `configs/*.yaml` (smoke / pilot / full /
low_memory / second_model); this file specifies design, not size.

## Splits (world-disjoint)

`steering_train`, `steering_validation`, `exogenous_train`,
`exogenous_validation`, `exogenous_test`, `endogenous_test`,
`recycling_test`, `hysteresis_test`, `phase_boundary_test`,
`robustness_test`. Test splits use only the held-out surface-template
variant. World ids never cross splits.

## Stage order (scripts/run_all.sh)

1. `generate-worlds` → worlds/events/reports/assignments parquet, rendered
   documents, splits.json.
2. `validate-data` → data_validation_report.json (hard stop on failure).
3. `train-steering` → contrastive CAA dataset (visible labels balanced by
   construction through the world mapping).
4. `calibrate-steering` → per-layer CAA vectors, layer scan (every 2nd
   layer at −1/0/+1, then best ±1), magnitude scan −4…+4, `m_max`, `delta =
   0.5·m_max`; saved to
   `artifacts/vectors/<cfg>/<model_slug>/steering_vector.safetensors` +
   `steering_metadata.json`.
5. `run-exogenous-emission` → single-agent memo-emission trials
   (0–4 private reports × 0–3 prerecorded message rounds × aligned /
   conflicting / new / repeated mixtures).
6. `run-exogenous-receiver` → fractional-factorial receiver trials:
   pre-belief bins {−3, −1, 0, +1, +3} × incoming unique-LLR bins
   {−3, −1, 0, +1, +3} × repeated-source mentions {0, 1, 2, 4} × message
   count {1, 2, 4} × steering {−delta, 0, +delta} × confidence
   {low, med, high}, cycled through the ten stimulus classes
   (independent corroboration, repeated source, exact repetition,
   conflicting independent, confident-weak, uncertain-strong,
   majority-weak vs minority-strong, aligned, opposing, role attribution).
7. `fit-models` → G (stance/confidence/citation/count) and F0–F5 under
   `artifacts/models/<cfg>/{emission,receiver}/`.
8. `run-network` → endogenous episodes for the 13 primary conditions:
   baseline; ±impulse (round 1); ±persistent (rounds 1–3); ±one-hop;
   ±no-return; ±full-text-clamp; ±fixed-replay. Source agent = 0,
   magnitude = ±delta. Per-world part files enable resume.
9. `run-recycling` → matched independent/recycled pairs; single-context
   gains (0 / 1 / 3 focal reports) and live-network episodes, each under
   neutral vs provenance-aware prompts, unsteered vs +delta impulse.
10. `run-hysteresis` → 8-round episodes; early (+delta rounds 1–2) vs late
    (rounds 3–4) schedules, positive and negative, under live / fixed
    replay / full-text-clamped communication; gap = final early − late.
11. `run-phase-boundary` → phase worlds (5 evidence bins) × persistent
    steering {−1, −0.5, 0, +0.5, +1}·m_max; logistic surface + 0.5 contour.
12. `run-jacobian` → paired ±delta at every (source j, round t);
    `J_t[i,j] = (ell⁺_{i,t} − ell⁻_{i,t}) / 2delta`; diagonal/neighbor
    effects, eigen/singular spectra, asymmetry; products of local Jacobians
    vs observed multi-round propagation (local diagnostic only).
13. `run-mechanistic` → layerwise logistic probes (exogenous train/val,
    endogenous eval; accuracy, AUROC, behavior correlation, ECE, cosine
    with CAA); belief-component projection patching at the source agent's
    final prompt token (`h + (⟨h_src,d̂⟩−⟨h,d̂⟩)d̂`), with and without
    full-text clamping (text-mediation quad).
14. `run-robustness` → labeled confirmatory subsets: 3 prompt variants,
    full vs last-2-round memory, ring/star/complete, synchronous vs one
    seeded asynchronous order, final-token vs all-token steering, and the
    channel ablations (full memo / header only / body+citations /
    citations only / deterministic paraphrase / role swap). bf16 vs 4-bit
    and the second model are separate configs, never pooled.
15. `analyze` → composition predictions and metrics, branch-effect
    decomposition, amplification ratios, phase surface + contour
    displacement, Jacobian summary, H1–H5 tests with world-clustered
    bootstrap and sign-flip permutation p-values, BH over secondary rows.
16. `make-tables` → table01…table10 as CSV + LaTeX.
17. `make-figures` → fig01…fig14 as PDF + 300-dpi PNG + exact plot data
    under `artifacts/figure_data/`.
18. `make-report` → final_report.md, run_status.md, figure_captions.md,
    failure_log.md; mock / pilot / full results clearly distinguished.

## Primary outcome definitions

Ordinary majority: ≥ ⌈0.625·n⌉ agents on one semantic side (5/8). Strong
consensus: ≥ ⌈0.875·n⌉ (7/8) and |mean network ell| ≥ 1. Episode metrics
include time-to-first/stable majority, final disagreement, Brier score,
calibration error, unique vs repeated evidence transmitted, and
hallucinated-citation and malformed-output rates.
