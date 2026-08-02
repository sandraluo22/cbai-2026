# Results log (Llama-3.1-8B-Instruct, 400 episodes/run, H200)

Model: `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror of the
canonical weights, selected via `NEGOTIATION_MODEL`). All runs: 400 episodes,
10 rounds, temperature 0.8, α ~ U(0,1), probes = RidgeCV per (layer, turn),
held out by episode. Run artifacts in `runs/default*` (activation shards live
only on the box; transcripts/plots/npz are synced here).

## v0 — original prompt (archived: `default_v0_pilot`, box only)

α as a bare number in B's system prompt ("greediness = 0.73, never reveal")
was **behaviorally inert**: corr(α, mean B demand) = **0.07**; B demanded
~68/100 for every α. All probes/baselines correctly null. Lesson: an abstract
scalar persona doesn't move Llama-8B's negotiation behavior.

## v1 — anchored prompt (archived: `default_v1_escalation`)

Fix: utility-weight framing + qualitative anchors at α∈{0, .5, 1} (typical
opening ranges, counter-rejection tendencies) + per-round private reminder.
Pilot corr(α, mean demand) = 0.64.

- Probes: best layer 16, peak R² **0.18** (turns 2–4), decaying afterwards.
- Behavioral baseline decayed 0.21 → 0.09; tf-idf ~0.09.
- Participation gap (A − shadow) ≈ **0**.
- **Design flaw found**: B escalated ~55 → ~80 across rounds regardless of α
  (ascending-number momentum + exploiting accepts). Per-round corr(α, demand)
  fell 0.55 → 0.22, so late-episode α-evidence dried up — no sharpening
  possible. Also the baseline ridge grid (tuned for 4096-dim activations)
  underfit the 8-feature behavioral model.
- Causal (probe direction w at layer 16, fixed α=0.5 opponent, γ=±8):
  **monotone and sign-correct** — γ=+8 ("B is greedy"): A's counter-demand
  56.6 → 32.4, accept rate 0.12 → 0.51; γ=−8 the reverse (61.5, 0.09).
  Magnitude overshoots the natural behavioral range → γ too large.

## v2 — stable-temperament prompt (current: `default`)

Fixes: "temperament is STABLE, re-anchor each round, do not drift" clause;
separate `baseline_ridge_alphas` grid for low-dim baselines.

- **Behavioral baseline now flat and strong: R² ≈ 0.38–0.43 at every turn**
  — α-evidence persists all episode (escalation cured), and this is the
  honest ceiling-ish reference for transcript-derivable information.
- Probes: best layer 16, peak R² **0.26** (turns 2–3), fluctuating 0.11–0.26
  afterwards; no monotone sharpening. Probes < behavioral model (0.43),
  > tf-idf (~0.13).
- **Introspection gap is real**: verbalized guesses R² ≈ 0.04–0.11 (Pearson
  0.23–0.29) vs probe 0.26 — A's activations encode ~2–3× more about B's α
  than A can state.
- **Participation gap ≈ +0.01 → null**: the negotiating A encodes B no better
  than a passive same-weights spectator. As pre-registered in the design:
  this is the "opponent modeling is transcript-general, not
  interaction-specific" branch.
- Causal at γ=±8 non-monotone in demand (51.4 / 57.8 / 46.2 for −8/0/+8),
  accept rate monotone (0.14/0.17/0.28). |γ|=8 likely partly degrades A
  off-distribution → rerun with a γ dose-response (±2, ±4) on the better v2
  probe direction.

## tier2 v0 — steering latent, no jitter (archived: `tier2_v0_saturated`)

α delivered purely via activation steering (centered coef in [−2.5, +2.5],
layers 12–20; B's prompt contains no persona at all). Calibrated dose-response
is dramatic and clean: coef −2 → B opens ~45/55, 0 → ~70/30, +2 → ~90/10.

- **Saturation failure (as pre-warned in the design):** the map α → offer is
  so clean that a SINGLE offer reveals α: tf-idf R² = 0.90 at turn 1 (0.98 by
  turn 3); probes 0.87–0.93 from turn 1; verbalized Pearson ≈ 0.91. Every
  measure ceilings; the inference problem vanished. → Fix: round-level
  coefficient jitter (`steer_noise_sd`), see config.
- **Same-space test (still valid here — w is unusually well-estimated,
  probe R² 0.93): cos(v, w) = +0.02 at best probe layer 30; max |cos| over
  layers = 0.069 (layer 6)** vs 1/√4096 ≈ 0.016 chance. The directions are
  essentially **orthogonal**: A encodes "B is greedy" in a subspace unrelated
  to the direction that makes B greedy — a dissociated other-agent encoding,
  not simulation-theory reuse. (Layer-6 0.069 is ~4σ above the permutation
  null but tiny in magnitude: at most a whisper of shared geometry, early.)
- **Causal at in-band γ (±1.5, ±3): flat** (demand 51–55, accept 0.11–0.17).
  Together with v1's large-|γ| "effect", the honest reading so far: no clean
  evidence the probe direction is causally used at coherent magnitudes;
  the v1 γ=±8 shifts are confounded with off-distribution degradation.

## tier2 v1 — jittered steering latent (current: `tier2`) ★ headline run

Per-round effective coefficient ~ N(steer_coef(α), 1.2²) clipped to ±3,
scale 2.0. This landed the experiment in the intended regime (turn-1 tf-idf
R² = 0.31, accumulating to 0.86 by turn 10).

- **THE SHARPENING CURVE (headline plot, `runs/tier2/probe_curves.*`):
  probe R² at layer 20 rises monotonically 0.34 → 0.50 → 0.64 → 0.69 → 0.73
  → … → 0.82 across the 10 turns.** A's internal estimate of B's hidden
  latent sharpens turn by turn — in-context inference of another agent,
  watched from inside.
- A's probes track the transcript-information curve (behavioral 0.43→0.83,
  tf-idf 0.31→0.86) essentially point-for-point: A internalizes nearly all
  available evidence as fast as it arrives.
- **Participation gap ≈ 0 (−0.01), third consecutive null across both
  tiers**: a passive same-weights observer encodes B's latent as well as the
  engaged negotiator. Robust conclusion: this opponent modeling is
  transcript-general, not interaction-specific.
- Introspection gap narrows with latent salience: verbalized R² 0.48→0.73
  (Pearson 0.85) vs probes 0.50→0.82. Contrast tier-1 v2 (probe 0.26 vs
  verbalized 0.06): when the latent drives behavior strongly, A can say most
  of what it knows; when it's subtle, A's activations know ~3× more than A
  reports.
- **Same-space test REPLICATES on an independent corpus: cos(v, w) = +0.02
  at best probe layer 20; max |cos| = 0.069 (layer 5)** — same values as the
  saturated corpus (0.02 / 0.069 @ layer 6). A's estimate-of-B's-greed
  direction is robustly orthogonal to the greed direction itself:
  **dissociated other-agent encoding, not simulation-theory reuse.**
- Causal dose-response at γ ∈ ±{1.5, 3}: flat (demand 52.6–55.4, accept
  0.17–0.25, no trend). Combined with tier-1: the probe direction is
  decodable but not demonstrably load-bearing at coherent steering
  magnitudes. (Caveats: 40 episodes/γ; single-layer injection; patching or
  multi-layer injection untried.)

## Open items / next rungs

1. Causal, properly: the flat dose-response may mean "not used" or "wrong
   intervention." Try activation PATCHING (swap A's layer-20 state from a
   high-α episode into a mid-α one at matched turns) and multi-layer /
   multi-token injection before concluding epiphenomenality; also raise
   episodes/γ (40 is small for ~3-point effects).
2. Tier-1 probe turn-course wobble (R² 0.02 at turn 7 between 0.17–0.18
   neighbors) smells like estimation noise: consider mean-pooling capture
   over A's reply tokens, or pooled-across-turn probes.
3. Why does tier-1 not sharpen while tier-2 does? Candidate: tier-1's
   α-signal per round is weak (corr ≤ 0.55) relative to A's capacity, so
   probes saturate what little there is immediately. Tier-2's stronger flow
   of evidence gives the accumulation dynamics room to show.
4. The ladder (design doc): 2D latent (greediness × patience) → affine map
   question; bidirectional probing; false-belief variant. Pipeline reuses
   unchanged.
