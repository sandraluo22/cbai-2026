# Personal vs. Social Belief Integration (single LLM vs. environment)

How does one focal LLM (Llama-3.1-8B-Instruct) weigh **private** evidence against
**social** evidence when choosing a company to invest in — and **where** in the
network does the integration happen? The environment is fully synthetic and
controllable; there is no live population.

## Two orthogonal knobs
- **λ (social informativeness about the truth)** — set by the noise levels:
  `λ = (1/σ_s²) / (1/σ_p² + 1/σ_s²)`, the Bayes weight on social as an estimator
  of θ in the `tau=0` case. Swept via `σ_s` (with `σ_p` fixed).
- **w (how much reward tracks social)** — `reward(i) = (1-w)·θ_i + w·c_i`.
  `w=0` is a pure *find-the-truth* game (social is only information); `w=1` is a
  pure *beauty contest* (social is the payoff). `c_i = θ_i + N(0, τ²)` is the
  crowd target; `τ>0` lets "follow the crowd" diverge from "track the truth."

Both `λ` and `w` (and `τ`) are logged for every config.

## Symmetry (why this is clean)
Private and social evidence are rendered in the **identical format** — same number
of readings, same per-reading layout, equal char-length blocks (labels
`PERSONAL` / `EXTERNAL`, both 8 chars) — so they differ only by SOURCE LABEL.
This removes the rich-vs-scalar confound; reliance differences are attributable to
source. `run_experiment.py --smoke` additionally asserts/logs the **token**-length
difference of the two blocks.

## Rational baseline (logged per trial)
Per-company joint linear-Gaussian posterior over `(θ_i, c_i)` from the visible
private + social histories (`env.bayes_posteriors`), giving `E[θ_i]`, `E[c_i]`,
the expected reward `(1-w)E[θ]+w E[c]`, the optimal action, and the **rational
effective social weight** (sensitivity of expected reward to the social vs.
private sufficient statistic) — the correct target for the model's revealed weight.

## Trial design
- **counterfactual** (primary): paired prompts identical except the target
  company's social readings are shifted by `delta` (private held fixed). The
  behavioral/activation difference isolates the social channel's causal effect.
- **disagreement**: private-implied vs social-implied conflict by `delta`.
- **neutral**: baseline.

## Activation capture + patching
- All-layer residual stream (embedding + 32 blocks) via HF `output_hidden_states`
  (capture) and forward hooks (patching) — see the justification comment in
  `model_runner.py`. Anchor = the **last `:`** token of the `Decision :` line
  (its own token thanks to the space); located by char→token offset mapping, so
  it is robust to BPE merges. We also store the decision-line span.
- **Patching** (the causal measurement): for each counterfactual pair, inject the
  donor (social_low) activations into the receiver (social_high) run, layer by
  layer, at the **marker** and at the **target's social-evidence tokens**, and
  record where the choice flips. Localizes *where* social changes the decision
  rather than assuming the colon holds the belief.

## Files / build order
1. `env.py` — sampling, knobs, reward, Gaussian posteriors, rational action +
   effective social weight, trial / counterfactual-pair constructors. *(tested)*
2. `prompt.py` — symmetric renderer, `Decision :` marker. *(tested)*
3. `model_runner.py` — Llama load, all-layer capture at marker+span, patched
   forward pass.
4. `run_experiment.py` — sweep λ×τ×w×mode×seeds, pairs, log table + activations.
5. `patch.py` — layer/position patching → choice flips.
6. `analyze.py` — revealed weight vs rational; per-layer probes; patching
   localization; probe-vs-patching cross-check. *(primitives tested)*

## Run
```bash
python -m pytest tests/ -q                          # CPU core (env + analysis primitives)
python run_experiment.py --dry-run                  # job list + forward-pass estimate, no model
python run_experiment.py --smoke                    # ONE trial + one pair end-to-end, prints
                                                    #   tokenization/marker/shapes/diff/patch step
python run_experiment.py --yes                      # full sweep (refuses if > max_forward_passes)
python patch.py   --out results/exp                 # patching table
python analyze.py --out results/exp                 # regressions, probes, localization, plots
```

## Config schema
See `configs/experiment.yaml`: `n_companies, T, sigma_p, lambdas[], taus[], ws[],
seeds[], theta_scale, prior_std, allow_withdraw, trial_modes[], deltas[],
targets[], model{...}, output_root, max_forward_passes`.

## Modeling choices / caveats (open)
- **iid-readings approximation**: the Bayes baseline treats each reading as an
  independent draw; it ignores any reading autocorrelation. Starting point.
- **patching alignment**: low/high prompts must be token-aligned; pairs where a
  shifted value crosses a digit boundary (changing token counts) are detected and
  skipped (`aligned=false`), not silently mis-patched.
- **marker vs span**: we do not assume the `:` holds the belief — patching at the
  social-evidence tokens is the check. The colon capture is for the probe axis.
- **greedy decoding** for the action (reproducible); `max_new_tokens` small.
- **τ folded into the social→θ likelihood** via the joint `(θ,c)` model; `τ=0`
  collapses `c≡θ` (handled in closed form).

## Safeguards
`--dry-run`/`--smoke` gate the full run; a sweep exceeding `max_forward_passes`
(default 200) refuses without `--yes`. Seeds for θ/c, private, social, and probe
splits are independent and logged.
