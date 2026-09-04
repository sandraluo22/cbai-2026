# Endogenous Belief Dynamics in Multi-Agent Language Models

Extension of Bigelow et al., *"Belief Dynamics Reveal the Dual Nature of
In-Context Learning and Activation Steering"* (arXiv:2511.00617), from
exogenously supplied in-context evidence to a **closed-loop multi-agent
network**: each frozen-weight LM agent holds private natural-language
incident reports, maintains an implicit belief between two latent
hypotheses, exchanges natural-language memos with ring neighbors, and
thereby co-generates the evidence distribution the network sees next.

**Central question.** Can an emission model `G` (belief → message features)
and a receiver-update model `F` (belief, incoming messages, steering → next
belief), identified only on controlled *exogenous* single-agent data,
compose to predict the *endogenous* closed-loop network:

```
belief_{t+1} = F(belief_t, adjacency x G(belief_t), steering_t)
```

Hypotheses: **H1** exogenous composition null; **H2** feedback
amplification beyond one hop; **H3** evidence-recycling
(double-counting repeated-source reports); **H4** text mediation of
activation-steering effects; **H5** hysteresis of equal-dose steering
schedules.

## Quick start

```bash
make setup        # pip install -e ".[dev]"  (add ".[gpu]" for HF runs)
make lint typecheck test
make smoke        # complete mock pipeline, no GPU: worlds -> ... -> report
```

The smoke run uses a deterministic **mock backend** and stamps every figure
with "MOCK SMOKE TEST". Real runs:

```bash
make pilot        # Qwen3-32B-Instruct, 1 seed, reduced sizes (needs ~1 A100-80GB+)
make full         # primary configuration, 3 seeds
bash scripts/run_all.sh configs/low_memory.yaml     # 4-bit (marked separately)
bash scripts/run_all.sh configs/second_model.yaml   # Gemma-2-27B subset
```

Every stage is idempotent and resumable: rerunning skips completed valid
artifacts (the network stage resumes per world/seed part files). All
pipeline stages are also available individually:

```bash
python -m belief_feedback.cli generate-worlds --config configs/full.yaml
python -m belief_feedback.cli validate-data   --config configs/full.yaml
python -m belief_feedback.cli train-steering  --config configs/full.yaml
# ... see python -m belief_feedback.cli --help
```

## Layout

- `configs/` — smoke / pilot / full / low_memory / second_model YAMLs.
- `src/belief_feedback/world/` — latent worlds, 16 document families with
  held-out surface templates, provenance-aware + provenance-blind oracles.
- `src/belief_feedback/agents/` — prompts, memo parsing, transcripts, and
  the synchronous protocol with causal branch routing/clamping.
- `src/belief_feedback/models/` — backend interface, deterministic mock,
  Hugging Face backend (bf16 / 4-bit, steering hooks, OOM-safe batching).
- `src/belief_feedback/experiments/` — steering calibration, exogenous
  emission/receiver trials, endogenous network branches, recycling,
  hysteresis, phase boundary, Jacobian, mechanistic, robustness.
- `src/belief_feedback/analysis/` — G and F0-F5 fits, composition rollout,
  branch-effect decomposition, world-clustered bootstrap, tables, report.
- `src/belief_feedback/plots/` — figures 1-14 (PDF + 300-dpi PNG + data).
- `artifacts/<kind>/<config-name>/` — all outputs, namespaced per config so
  mock, pilot, quantized, and full results can never mix.

## Key documents

- [METHODS.md](METHODS.md) — formal setup, seeds/common random numbers,
  branch semantics, statistics.
- [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) — condition-by-condition spec.
- [DATA_CARD.md](DATA_CARD.md) — synthetic dataset documentation.
- `artifacts/reports/<name>/final_report.md` — automated result report.

## Scientific guardrails

Frozen weights only; probes never enter transcripts; malformed generations
are retained, never regenerated; F and G are fitted on exogenous worlds
only and never refit on endogenous outcomes; worlds never cross splits;
every result row carries config hash, git commit, model + tokenizer ids,
seed, world, agent, round, condition, and branch parent; a model
substitution is only ever explicit (`model.fallback_model_id`) and recorded
in every manifest.
