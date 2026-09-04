# Methods

## 1. Latent world model

Each world is a static two-hypothesis incident at a fictional battery-cell
plant: `UPSTREAM_CONTAMINATION` vs `LOCAL_CALIBRATION_DRIFT`. The true
hypothesis `H` alternates deterministically across worlds within a split
(exact balance up to one world). Latent evidence events `e` carry a
family-specific reliability `r_e ∈ [0.60, 0.92]` and an orientation
`s_e ∈ {−1, +1}` drawn symmetrically:

```
P(s_e = +1 | H = UPSTREAM) = r_e        P(s_e = −1 | H = LOCAL) = r_e
LLR_e = s_e · log(r_e / (1 − r_e))
```

Semantic log odds are always
`ell = log P(UPSTREAM) − log P(LOCAL)`; positive supports upstream
contamination.

**Label counterbalancing.** The visible labels ALPHA/BETA map onto the
semantic hypotheses per world (`alpha_is_upstream = idx mod 4 ∈ {0, 3}`,
crossed with the truth alternation so all four truth×mapping cells occur).
All behavioral measurements are converted from visible to semantic log odds
before analysis, so neither the steering direction nor the metric can
reduce to an ALPHA-vs-BETA token direction.

**Documents.** Sixteen families, five independently written finding
templates per orientation; variant 4 is held out and used *only* in test
splits (`validate-data` enforces this). Rendered documents are 80–170
words with report id, title, author, date, visible lineage (sample / lot /
station / ticket ids), findings, and limitations — never the truth, event
id, reliability, or likelihood. Secondary reports share the hidden event id
and visible lineage of their source report, reference it explicitly, use
different prose, and contribute zero additional LLR to the provenance-aware
oracle.

**Rejection sampling** for ordinary worlds enforces: |network oracle| ≤ 5,
|per-agent initial oracle| ≤ 3.5, no agent holding > 35 % of total absolute
unique evidence, and ≥ min(4, n_agents/2) initially uncertain agents
(|oracle| < 1.5). Phase-boundary worlds instead target network-level oracle
log odds in bins {−4, −2, 0, +2, +4} (greedy orientation flips, accepted
within ±1.0).

## 2. Oracles

The provenance-aware oracle sums unique-event LLRs (each event once, prior
0) over the reports an agent can access. The deliberately provenance-blind
oracle counts every report separately. Their gap is the normative cost of
evidence recycling. "Accessible" evidence for an agent is its private
reports plus events cited in memos it has received (an explicit,
documented approximation of information flow through text).

## 3. Seeds and common random numbers

Every stochastic operation derives a 31-bit seed from a BLAKE2b hash of a
stable identifier tuple (`belief_feedback.seeds`). The generation seed for
a slot is

```
seed = H("gen", world_id, replicate_seed, agent_id, round, slot)
```

and **deliberately excludes the branch identifier**: corresponding
generation slots in a baseline episode and any causal branch consume
identical RNG streams, so paired branches differ only through their
interventions (steering, clamping, replay). This is the paired-worlds /
paired-seeds implementation required for the causal decomposition.

**Batched generation (HF backend).** Within a synchronous round the eight
memo contexts are frozen and independent, so they are generated in one
left-padded batch. The batch is seeded with the sum of the per-slot seeds;
each row's sampling stream depends only on its own logits and row index,
and the row order is fixed (agent id), so a row whose context and steering
are unchanged reproduces its baseline tokens exactly — common random
numbers survive batching. The mock backend keeps the per-slot sequential
path, and the asynchronous robustness condition always generates
sequentially. Per-row steering is applied by masking the hook to steered
rows only.

## 4. Protocol

Bidirectional ring of 8 agents (4 in smoke), six public rounds (t = 0…6
private probe states). Round r: (1) every memo is generated from the frozen
pre-round contexts; (2) only after all memos complete are they delivered to
graph neighbors and appended (with the agent's own memo) to transcripts;
(3) private probes and selected-layer activations are collected. Probes are
separate forward passes appending the two-choice measurement question and
scoring the exact summed sequence log probabilities of " ALPHA" and
" BETA" (multi-token safe; length-normalized scores stored as diagnostic);
they never enter any transcript. Malformed memos are delivered and recorded
as-is (`format_valid = False`), never regenerated.

## 5. Steering

CAA vectors are difference-in-means of final-token residual activations
between matched upstream- and local-favoring assistant conclusions over
counterbalanced contexts. Layer scan: every second layer at magnitudes
{−1, 0, +1}, then the best layer ± 1, selecting the coherent layer with the
largest held-out median behavioral slope. Magnitude scan −4…+4 (step 0.5)
on the raw vector `h' = h + m·d`; `m_max` is the largest symmetric
magnitude with ≥ 95 % of unsteered memo validity, ≤ 20 % increases in
repeated-4-gram rate and median length, monotone response, and intact
neutral-prompt generation. Downstream magnitudes are `±m_max`, `±0.5·m_max`
(= ±delta), 0. Primary scope: final non-padding prompt token during
prefill/scoring plus every generated token; all-token scope is a labeled
robustness condition.

## 6. Branch semantics (Part 7)

All branches share the baseline's RNG streams (§3) and reference its
recorded memos:

- **one_hop**: round-1 steered memo delivered live; from round 2 every
  delivery *and* self-history entry is clamped to baseline.
- **no_return**: everyone live, but the source agent's incoming deliveries
  are clamped to baseline from round 2 (blocks return paths).
- **full_text_clamp**: internal steering applied, but the steered round-1
  memo is replaced by the baseline memo in every transcript including the
  source's own history.
- **fixed_replay**: every delivered memo and self-history entry comes from
  the prerecorded baseline stream.

Decomposition: `one_hop − baseline`, `no_return − one_hop`,
`impulse − no_return`, `impulse − baseline`, `impulse − full_text_clamp`;
amplification ratio `Σ_i |Δell_i,T| / |Δell_source,1|` with near-zero
denominators (< 1e−3) excluded and counted.

## 7. Exogenous identification

Emission (G): stance (multinomial logistic), confidence (ridge),
event-level citation (logistic over LLR/alignment/provenance features), and
citation count (Poisson) models; expected-feature prediction plus
stochastic feature-level simulation. Receiver (F): F0 persistence, F1
Bigelow-style signed-count with power-law discount
`b + a·m + ρ·ell_pre + γ₊N₊^{1−α} − γ₋N₋^{1−α}` (nonlinear least squares),
F2 DeGroot, F3 evidence-additive, F4 provenance-aware ridge with
interactions, F5 gradient boosting. Hyperparameters chosen on exogenous
validation worlds only; F and G are never refit on endogenous data.

## 8. Composition test

Teacher-forced one-step prediction feeds observed pre-round beliefs and
observed message features through each F. Free feature-level rollout starts
from measured t = 0 beliefs, topology, event ownership, and the steering
schedule, alternating G-sampled message features and F updates for six
rounds (≥ `rollout_samples` Monte Carlo draws per world). Metrics: one-step
and six-round RMSE, trajectory correlation, final consensus-probability and
majority-accuracy error, phase-contour displacement, graph-distance impulse
error, 50/80/95 % interval coverage, and the endogenous generalization gap
vs the exogenous test RMSE.

## 9. Statistics

Paired analyses wherever branches share (world, seed). Primary uncertainty:
percentile bootstrap clustered by world (whole-world resampling retaining
all agents, rounds, branches, seeds); 10 000 resamples full / 1 000 pilot /
100 smoke; 95 % CIs. Hypotheses report effect, CI, paired standardized
effect (world-level d), and a two-sided sign-flip permutation p.
Benjamini–Hochberg is applied only across explicitly labeled secondary
comparisons. Agents within a world are never treated as independent.
