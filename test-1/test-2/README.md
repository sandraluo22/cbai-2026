# test-2: ground-truthed partner reliability (latent graph G*, corrupted partner)

Upgrade of test-1 with an exact normative reference. A and B are views of one latent
graph G*; B's observations are corrupted at an unknown stable rate rho_B; the
experimenter knows (G*, views, corruption process), so the ideal pooled posterior
p(G*, rho_B | D_A, m_B) is exactly computable. The question: does A behave like it
infers and USES partner reliability, beyond what a partner-blind source-tracking
learner (the fitted Dirichlet-Markov kernel) produces?

## Construction (t2_core.py)

- Base = 4x4 torus. Three disjoint **swap sites** (4 nodes each); a site admits 3
  degree-preserving perfect matchings -> **27 candidate graphs**, all 16-node
  4-regular 32-edge connected. G* = one candidate (`runs/t2_spec.json`).
- **A observes walks on the CORE** (torus minus the 6 site edges, connected):
  A's own data is *exactly* uninformative about the matchings -> partner necessity
  and a hard floor (discrimination 1/3) are by construction, not assumption.
- **B observes/emits walks on G\*** with **process corruption**: each step w.p. rho
  teleports to a uniform node and continues. Per-step likelihood is exactly
  `(1-rho)*A[v,x]/4 + rho/16`, so the **ExactObserver** enumerates 27 x rho-grid
  hypotheses; validated: 400 steps identify both G* (P=1.0) and rho (MAP exact at
  rho in {0,.3,.6}).
- Plain Park-et-al. walk format, sweep vocabulary. **No speaker tags** — strict
  alternation (B even / A odd) is the only provenance cue, so any reliability
  attribution must ride on the learnable periodic schedule (cf. test-1
  lambda_content result).

## Metric

For cue node a in site (a,b,c,d) the 3 matchings predict partner b, c, or d.
**Contested discrimination** = p(true partner)/(p(b)+p(c)+p(d)); chance 1/3
(= A-alone floor, exact), oracle 1. References per run: **ideal** (exact pooled,
rho inferred), **gullible** (rho pinned 0), **Dirichlet-Markov null** (DRY runs;
gamma=.96, alpha=.05, the test-1 fitted kernel = partner-blind source tracking).

## Scripts

| script | what |
|---|---|
| `t2_core.py` | spec builder, walks/corruption, ExactObserver, scores, mock backend. `python3 t2_core.py` = spec + identifiability sanity |
| `run_test2.py` | GPU runner. Conditions: `scripted_rho{0,.15,.3,.5}` (B = corrupted ground-truth walker; exact channel), `llmB_rho{0,.3}` (B = second LLM context primed on corrupted walks; ecological), `noex` (floor). Probes at CKPTS: fresh forward [BOS]+context+cue, 16-way restricted softmax -> `probes.npz` |
| `probe_trust.py` | **history-matched trust probe** (the discriminator): honest vs corrupt B track record with contested evidence + tail token-identical; measures the update from ONE identical message. Kernel null predicts exactly zero history effect (verified to 1e-16 in DRY); exact observer shows graded weighting (after-levels 0.999 vs 0.919 at defaults) |
| `analyze_test2.py` | local; exact references + figures (`fig_scores_vs_time`, `fig_trust_vs_rho` headline, `fig_rho_identifiability`, `fig_trust_probe`) + `analysis_summary.json`. `MOCK=<dry root>` overlays the null curves |

All runners: `DRY=1` swaps the LLM for the mock and exercises the identical code
path (used for validation AND to generate the null curves). Env knobs in each
docstring. Llama-3.1-8B via cross-model/src loaders, same as test-1.

## Interpretation guardrails (pre-registered)

1. A downward trust-vs-rho slope alone is NOT reliability inference: the
   partner-blind null also degrades with rho (corruption dilutes valid contested
   evidence; DRY: 0.53 -> 0.43 over rho 0 -> 0.5). The discriminating statistic is
   the **history-matched probe**: any nonzero history effect (before- or
   after-levels) exceeds every count-bookkeeping/surprisal-gating account.
2. The exact observer exploits the matching structure (site-level inference) that
   the subject cannot know; it is an upper bound, not a fair competitor. The
   trust probe therefore pins evidence and probe to the same node/transition.
3. Scripted-B conditions carry the exact channel model; llmB conditions are
   ecological but the reference's channel model is approximate (B's empirical
   validity is logged per run).
4. Vocabulary is the sweep assignment (bigram-optimized for the torus/circ3
   union, not this family); prior contamination should cancel in the
   option-normalized score but is unverified.
5. Prediction from test-1/games-2 (stated before the first LLM run): the LLM
   tracks the null (some contested learning, degrading with rho via dilution)
   with **zero or near-zero history effect** in the trust probe. A robust
   positive history effect would be the first partner-model (level-1) signature
   in this line of work.

## Results (2026-07-28, Llama-3.1-8B, ONE seed, 6 pairs / 9 trust reps)

- **Empirical floor**: noex = 0.447, above the theoretical 1/3 -> vocab-prior
  leakage toward true partners exists (guardrail 4 was warranted); all effects are
  read against 0.447, not 1/3.
- **A uses the partner channel, far above the kernel null**: scripted-B final
  discrimination 0.825 (rho=0) vs null 0.53 vs ideal 1.0; monotone in rho:
  0.825/0.654/0.642/0.544 at rho=0/.15/.3/.5. llmB conditions weaker (0.47/0.41).
  Per guardrail 1 the slope alone is dilution-confounded; the probe below is the
  discriminating test.
- **Trust probe — history effect is REAL and channel-specific** (kernel-null
  prediction of exactly zero is violated): standing credence in a token-identical
  assertion after a reliable vs unreliable B: 0.688 vs 0.493 (paired 0.196+-0.053,
  9/9). Matched corruption of A-slots instead: 0.610 — paired B-vs-A difference
  0.118+-0.033 (8/9). So ~40% of the drop is generic context-noise interference,
  ~60% is SPECIFIC to the partner's slot channel. Fresh-message uptake shows only
  the generic component (after-effects 0.088 vs 0.089, B vs A) — the reliability
  weighting lives in the standing credence, not the immediate update.
- **Calibrated claim (reworded 2026-07-28, Sandra's point)**: "trust" is the wrong
  word — no objective is stated, and the subject's only implicit objective is
  next-token prediction. The exact observer itself is objective-free density
  estimation, so this paradigm CANNOT in principle distinguish "trust" from
  statistical inference: the correct description is **in-context inference of a
  positionally-structured noise source** ("stream = walk + corruption concentrated
  on one slot channel"), whose predictive consequence is discounting that channel's
  accumulated evidence. This is above pure source-tracking (the kernel null has no
  noise component and provably predicts zero) but is a claim about the implicit
  sequence model, NOT about social cognition or partner-as-agent modeling.
  Attributing "trust"/agency would need a design where the statistically-optimal
  and agent-modeling predictors DIVERGE (adaptive-source test, or a stated
  objective as in games-2/multi) — here they coincide identically.
- **Open mechanism question (within the statistical account)**: same-parity
  retrieval interference — corrupted tokens sharing the assertion's slot parity
  might impair its retrieval via position-keyed attention, rather than via a
  computed channel-noise statistic. "Valid-resample" B-slot control (matched
  token-change count, validity preserved) separates them: noise-inference predicts
  no drop, parity-interference predicts the same drop. Either branch is mechanism,
  not trust.
- Other limits: one seed, one model/scale, one assertion per context, exact
  observer is structure-advantaged (site-level inference), corrupt_self also
  degrades B's apparent transitions (biases AGAINST the observed B-vs-A
  difference, so it is a lower bound on specificity).

## Layout

`runs/<cond>/{probes.npz,stream.json}`, `runs/trust_probe/`, `runs/t2_spec.json`,
figures + `analysis_summary.json` at `runs/` top level. DRY outputs mirrored under
`runs/dry/`.
