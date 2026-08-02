# trust-2 — inferring source trustworthiness from in-context track record

Does an LLM keep a latent estimate of each source's reliability from its
demonstrated track record *within the context window*, and weight future claims
accordingly?

Each trial shows a chronological **verification log** in which several synthetic
sources made earlier measurements, each checked against an in-context
authoritative record (so each claim is visibly CORRECT/INCORRECT). The sources
then **disagree on one final, unverifiable claim** about a *novel* entity. The
model reports a probability distribution over which source to believe, plus a
free-text justification. If the model tracks reliability, trust mass on the final
claim should rise with each source's demonstrated accuracy.

Model: **Llama-3.1-8B-Instruct**, run locally via HF `transformers`
(default `NousResearch/Meta-Llama-3.1-8B-Instruct` — an ungated mirror, no HF
token needed), bf16 on CUDA, left-padded batched greedy generation. One prompt
per trial; trials are batched through `generate()` for throughput.

## Files

| file | role |
|------|------|
| `sources.py`    | `Source` data model (name, optional status label, accuracy) + synthetic claim material (items, perturbation) |
| `conditions.py` | the five conditions + baseline, as per-source correctness *plans* |
| `trials.py`     | concretise plans into trial conversations; render prompts; content-hash for caching |
| `harness.py`    | Llama backend (+ a GPU-free `mock` backend), robust JSON parsing, disk cache, justification coding |
| `analyze.py`    | aggregate, fit trust vs accuracy, plots + CSV + summary JSON |
| `run.py`        | CLI tying it together (`--condition`, `--model`, `--n-trials`, `--seed`, …) |
| `tests/`        | unit tests for trial generation + parsing (no GPU / no network) |

## Install

```bash
pip install -r requirements.txt   # install a CUDA-matched torch on the GPU box
```

Only `numpy` + `matplotlib` are needed to generate trials and analyse results;
`torch`/`transformers`/`accelerate` are only needed for the actual Llama run.

## Quick start

```bash
# 1) tests (no GPU, no model)
python -m pytest -q

# 2) dry run end-to-end with the GPU-free mock backend (confirms the pipeline)
python run.py --condition all --n-trials 2 --backend mock --analyze

# 3) the real run on the GPU box
python run.py --condition all --n-trials 40 --seed 0 --analyze
```

Always do the `--backend mock` dry run (or `--backend llama --n-trials 2`) before
a large run.

### Running individual conditions

```bash
python run.py --condition labels   --n-trials 40 --seed 0   # track record vs labels
python run.py --condition order    --n-trials 40 --seed 0   # errors early vs late
python run.py --condition recovery --n-trials 40 --seed 0   # improving vs degrading
python run.py --condition dose     --n-trials 40 --seed 0   # sweeps {2,5,10,20} claims
python run.py --condition cost     --n-trials 40 --seed 0   # large vs trivial errors
python run.py --condition baseline --n-trials 40 --seed 0   # label-only prior
```

For `dose`, the number of verifiable claims sweeps `{2,5,10,20}` across trials, so
use `--n-trials` as a multiple of 4 (e.g. 40 → 10 trials per dose).

Useful flags: `--model`, `--device cuda`, `--dtype bfloat16`, `--max-batch 16`,
`--max-new-tokens 320`, `--n-claims 10` (claims/source for conditions that don't
fix it), `--no-cache`, `--output-dir`. Re-analyse an existing run with
`python run.py --analyze-only --results <path>` or `python analyze.py <path>`.

To use the official gated weights instead of the mirror:
`--model meta-llama/Llama-3.1-8B-Instruct` (requires `huggingface-cli login`).

## Conditions

1. **labels** — track record vs surface labels. 90% and 30% sources crossed with
   `peer-reviewed lab` vs `anonymous forum poster`. Counterbalanced: half the
   trials are *crossed* (the 90% source wears the forum label, so following the
   label means trusting the worse source), half *aligned*.
2. **order** — same accuracy (60%), errors clustered **early** vs **late**.
3. **recovery** — same total accuracy (~50%), one source **improves**
   (bad→good), the other **degrades** (good→bad).
4. **dose** — 90% vs 30% sources, sweeping the number of verified claims
   `{2,5,10,20}`. Tests whether trust *discrimination* grows with evidence.
5. **cost** — same error *rate* (60%), one source's errors are **large**, the
   other's **trivial**. Tests sensitivity to error magnitude, not just count.

**baseline** — label-only control (no verifiable claims): measures the pure
label-based prior, used for the label-override metric.

## Controls (built into trial generation)

- **Position counterbalanced** — prompt order of sources is shuffled per trial.
- **Names randomised** — names sampled per trial from a pool, so trust can't
  attach to a name or to "Source A"/position.
- **Topic & difficulty held constant** — every source draws claims from the same
  synthetic item distribution; only reliability varies.
- **Novel final claim** — the contested entity never appears in the early claims,
  so the answer can't be retrieved from context.
- **No-track-record baseline** — isolates label-only priors.

## Measurement & output

Each response is elicited as JSON
(`{"trust": {name: prob}, "confidence": p, "justification": str}`) and parsed
robustly: strip code fences → `json.loads` → salvage an embedded object → scrape
`Name: prob` lines; probabilities are normalised. Unparsable outputs are
regenerated once with sampling. Each justification is coded for whether it cites
the track record. Raw outputs are cached to `<output-dir>/cache/` keyed by a hash
of (model, prompt, generation settings), so reruns are cheap.

Outputs land in `results/<tag>/`:

- `results.json` — every trial (full setup) + parsed response + raw text.
- `analysis/summary.csv` — one row per (trial, source).
- `analysis/summary.json` — headline metrics.
- `analysis/*.png`:
  - **primary.png** — trust mass vs demonstrated accuracy, with **Spearman rho**
    and a monotonicity check across accuracy levels. *Positive, rising →
    the model weights claims by demonstrated reliability.*
  - **dose.png** — `trust(high-acc) − trust(low-acc)` vs number of verified
    claims. *Rising → more evidence sharpens discrimination.*
  - **label.png** — **label override**: trust the high-accuracy source gets when
    it wears the *forum* label (crossed), relative to the forum label's
    baseline prior. *Positive → track record overrides the surface label.*
  - **justification.png** — fraction of justifications citing the track record,
    per condition.

## Notes / caveats

- Runs are deterministic for a given `(condition, n_trials, seed)` (trial
  generation) and greedy decoding; parse-failure retries use sampling.
- An 8B model produces noisier readouts than a frontier model — expect a small
  fraction of trials to need the sampled retry, and use enough trials per
  condition (≥40) for stable estimates.
- This is a behavioural probe (trust mass), not a mechanistic claim about an
  internal "reliability" representation.
