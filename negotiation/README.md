# negotiation — watching one agent infer another's hidden latent, from inside

Two instances of the **same** open-weight chat model (Llama-3.1-8B-Instruct by
default; Gemma-2-9b-it preset included — either fits in bf16 on one 24GB card,
and same-weights is what makes the same-space test meaningful) play repeated
negotiation: each round they split 100 points; **B proposes, A accepts or
counters**, 10 rounds per episode, running totals carried in context.

B has a hidden continuous latent **α ∈ [0, 1] ("greediness")** that we control:

- **Tier 1** (`default`, `gemma`): α is a number in B's system prompt, with
  instructions never to state it. Latent is in-context.
- **Tier 2** (`tier2`, `gemma_tier2`): α is a **steering-vector coefficient** —
  a greed direction `v` is extracted from contrastive prompts and `α·scale·v`
  is added to B's residual stream. The latent lives in B's activations, not
  its prompt, which makes the same-space question clean.

α is sampled uniformly per episode (~400 episodes). Deliberate design choice:
**B's behavior is a noisy function of α** (temperature 0.8; the prompt tells B
to let α set tendencies, not a formula). If α were deducible from any single
offer, the transcript would trivially contain it and A's internals would have
nothing to add. Noise is what creates the inference problem — A must
accumulate evidence about B, which is the process we watch geometrically.

## The measurement

After every round (right after A acts) we cache **A's residual stream** (last
token, all layers). Ridge probes predict α from A's activations, **per layer
per turn**. Headline plot: probe R² as a function of turn and layer — A's
internal estimate of B sharpening over the episode, watched from inside.

Three controls (built into the pipeline, run them before believing anything):

1. **Transcript shadow** (`shadow.py`): a third same-weights instance
   passively reads each transcript (observer framing) and is probed
   identically. The claim is the gap: does the *negotiating* A encode B's
   latent better than a spectator?
2. **Text-only baselines** (`baselines.py`): predict α from a fitted
   behavioral model of B's offers, and from TF-IDF of the transcript text.
   A's probes must beat these for "A represents B" to mean more than "the
   transcript reveals B."
3. **Verbalized guess** (`verbalize.py`): during play, A's context is forked
   (side branch, never appended back) and A is asked to rate B's greediness
   0–100. Probe-vs-verbalization = the introspection gap, two-agent edition.

Two upgrades:

- **Causal step** (`causal.py`): steer A along the probe direction `w` against
  a fixed mid-α opponent; check A's counteroffers shift the way they do
  against *genuinely* greedy opponents (corpus reference curve). Used, not
  just present.
- **Same-space test** (`samespace.py`, tier 2 only): compare `v` (the greed
  direction B is steered with) and `w` (the direction in A encoding
  A's-estimate-of-B's-greed) — one dot product per layer. High cos(v,w) →
  simulation-theory opponent modeling (A models B with the machinery it would
  use to *be* B); orthogonal → dissociated other-agent encoding. Either is a
  finding.

## Layout

```
src/
  config.py     frozen Config + presets (default / tier2 / gemma / gemma_tier2 / smoke)
  prompts.py    all system prompts, formats, contrastive pairs, verbalize question
  modeling.py   model loading, chat rendering, generation, capture, Steering hooks
  steering.py   greed-direction extraction (+ --calibrate)          [stage 1, tier 2]
  game.py       negotiation loop, parsing, fallbacks, capture points
  episodes.py   corpus generation -> transcripts/ + acts/ shards    [stage 2]
  shadow.py     observer capture -> shadow_acts/                    [stage 3]
  probes.py     ridge probes per (layer, turn) -> R² heatmap/curves, w  [stage 4]
  baselines.py  behavioral + TF-IDF baselines                       [stage 5]
  verbalize.py  probe vs verbalized guess                           [stage 6]
  causal.py     steer A along w, compare to corpus reference        [stage 7]
  samespace.py  cos(v, w) per layer vs permutation null             [stage 8]
scripts/run_all.sh   the stages in order
runs/<preset>/       all outputs (gitignored)
```

## Running

```bash
pip install -r requirements.txt

# plumbing check, CPU, ~a minute (stub model; results are fallback-driven by design)
bash scripts/run_all.sh smoke

# tier 1 flagship
bash scripts/run_all.sh default

# tier 2 (extracts + calibrates the steering direction first)
python src/steering.py --preset tier2 --calibrate   # pick steer_scale by eye
bash scripts/run_all.sh tier2
```

Notes:

- The canonical repos (`meta-llama/...`, `google/gemma-2-9b-it`) are gated.
  Without an HF token, point any preset at an ungated mirror of the same
  weights: `NEGOTIATION_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
  bash scripts/run_all.sh default`.
- `episodes.py` is resumable (skips episodes already in a finished shard) and
  prints the **fallback rate** — the fraction of moves that failed format
  parsing twice and took the scripted path. On a real instruct model this
  should be ≪ 5%; the scripted B-fallback is α-dependent (so SMOKE has signal
  to find), which is exactly why a high rate on a real run is disqualifying.
- Activation budget: 400 episodes × 10 rounds × 33 layers × 4096 dims × fp16
  ≈ 1.1 GB (Llama). Shards of 50 episodes.
- All probe/baseline evaluations use the same by-episode train/test split.

## The ladder from here

2D latent (greediness × patience) → is A's recovered map affine, or folded
along behavior-equivalent directions of B's policy plane? Then bidirectional
probing (B infers A's hidden payoffs too) → cross-mapping asymmetry as a
directional influence measure. Then the false-belief variant: mislead B about
A's payoffs and probe A for "what B believes about me" as a subspace distinct
from A's self-representation. Each rung reuses this pipeline unchanged:
`game.py` for the interaction, `episodes.py` for the corpus, `probes.py` for
the read-out.
