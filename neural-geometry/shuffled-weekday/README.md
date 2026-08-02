# shuffled-weekday: manifold dynamics during in-context learning

**Question.** Goodfire's manifold-steering paper (arXiv 2605.05115) fits
manifolds only *after* the model has fully learned an in-context graph. What
happens *during* learning, when the in-context structure conflicts with a
pretrained manifold?

**Setup.** Nodes are the 7 weekdays placed on a ring in permuted order (each
ring step = +3 days; `DAYS_PERMUTED` in `cross-model/src/config.py`), so the
in-context ring conflicts with the pretrained weekday cycle. The model reads
random walks on this ring; we track both the activation manifold and the
behavior manifold as functions of context depth.

## Hypotheses

| | signature in `analyze_shift.py` output |
|---|---|
| **H1 rigid relabeling** — same circle, tokens reassigned | angle-to-pretrained stays low; order flips semantic → in-context |
| **H2 subspace competition** — new ring grows in a (partially) orthogonal subspace | angle-to-pretrained grows / angle-to-final shrinks, sharp-ish transition; both orders partially expressed mid-way |
| **H3 smooth deformation** — the circle continuously morphs | order agreement moves gradually, no subspace rotation jump |

Priors from Park et al. (ICLR 2025, "In-Context Learning of Representations"):
expect H2 with a fairly sharp transition, semantic structure persisting in
superposition.

**The isometry test (the new part).** At each depth we also fit the behavior
manifold (Hellinger-embedded next-day posteriors) and measure the
activation↔behavior isometry. Mid-transition, when activations superpose two
rings, does behavior (a) track whichever ring dominates (isometry to one
manifold at a time), (b) interpolate (violating isometry to *either* ring), or
(c) snap discontinuously? Any behavioral "teleportation" during the transition
is a phenomenon the manifold-steering framework does not predict from a single
manifold.

## Pipeline

```bash
# 1. activations: per-occurrence residual stream + pretrained weekday baseline
python capture_ctx.py                    # --smoke for a cpu plumbing test

# 2. behavior: walk-continuation posteriors + "k days after y" probes per depth
python behavior_probe.py

# 3. fit rings per context bin, track subspace angles / order / isometry
python analyze_shift.py                  # -> runs/shift_metrics.json, .pdf

# 4. causal arbitration: steer along old vs new ring tangents at each depth
python steer_arbitration.py             # -> runs/steer_arbitration.json
```

All scripts default to `meta-llama/Llama-3.1-8B` on cuda (layer 26, matching
the cross-model alignment layer); `--model/--device/--layers` override;
`--smoke` runs distilgpt2 on cpu to test plumbing (results meaningless).

## Notes / gotchas

- The behavioral posterior is restricted to the 7 day words' *first* subword
  token (leading-space form) and renormalized; `day_mass` records the
  unrestricted mass on days — if it is low the model isn't playing the game
  and the posterior geometry is noise. Filter on it.
- `behavior_probe.py` also runs a no-context "k days after y" control — the
  pure pretrained relation — as the semantic-probe baseline.
- Steering magnitudes in `steer_arbitration.py` are expressed as fractions of
  the fitted ring radius, so "on-manifold-sized" steps stay comparable across
  layers/models.
- The causal handoff depth (step 4) need not coincide with the
  representational transition (step 3). A gap between them is a result, not a
  bug.
