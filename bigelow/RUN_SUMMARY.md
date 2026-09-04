# RUN_SUMMARY

Repository built and validated on 2026-09-03 (macOS, Apple M3 Pro, 18 GB RAM,
no CUDA GPU). Python 3.12.2.

## Completed stages

| Stage | Status | Notes |
| --- | --- | --- |
| Repository creation | done | full package, configs, scripts, tests, docs |
| Dependency resolution | done | core deps preinstalled; added `ruff`, `types-PyYAML`; GPU extras (`torch`/`transformers` present; `bitsandbytes`, `zarr` in optional `[gpu]`) |
| `ruff check src tests` | **PASS** (0 issues) |
| `mypy src/belief_feedback` | **PASS** (0 issues, 69 files) |
| `pytest tests` | **PASS** — 47 tests, including the scaled-down end-to-end pipeline (`test_e2e` config) |
| Complete smoke pipeline (`make smoke`, mock backend) | **PASS** — all 18 stages, ~2 min wall clock |
| Smoke artifact verification | **PASS** — no missing/empty required artifact |
| Pilot configuration | **NOT RUN** — see below |

## Failed stages

None. The pilot was not attempted (not a failure): the pilot config
specifies `Qwen/Qwen3-32B-Instruct` in bfloat16 (~64 GB weights plus KV
cache); this machine has no CUDA GPU and 18 GB unified memory, so the model
cannot load even 4-bit. No fallback model was silently substituted
(prohibited by the spec).

## Exact commands run

```bash
python3 -m pip install -e . ruff types-PyYAML
python3 -m ruff check src tests
python3 -m mypy src/belief_feedback
python3 -m pytest tests
bash scripts/run_all.sh configs/smoke.yaml   # = make smoke
```

## Model actually used

`mock/deterministic-agent` (the deterministic mock backend) — smoke
configuration only. All smoke figures are stamped "MOCK SMOKE TEST" and the
report labels them as non-scientific. No Hugging Face model was loaded.

## Artifact locations (smoke)

All outputs are namespaced by config name:

- Data: `artifacts/data/smoke/` — worlds/events/reports/assignments
  parquet, `splits.json`, `data_validation_report.json` (**passed**),
  376 rendered documents under `rendered_documents/`.
- Steering: `artifacts/vectors/smoke/mock__deterministic-agent/`
  (`steering_vector.safetensors`, `steering_metadata.json`; selected
  layer 1, m_max 1.5, delta 0.75 on the mock).
- Runs: `artifacts/runs/smoke/` — 28 result tables incl.
  `belief_states.parquet` (936 rows), `public_messages.parquet` (624),
  `deliveries.parquet` (1872), `episodes.parquet` (78),
  `hypothesis_tests.parquet`, `composition_{predictions,metrics}.parquet`.
- Fitted models: `artifacts/models/smoke/{emission,receiver}/` (G pieces +
  F0–F5 with held-out metrics).
- Figures: `artifacts/figures/smoke/fig01…fig14` (PDF + 300-dpi PNG),
  plot data in `artifacts/figure_data/smoke/`.
- Tables: `artifacts/tables/smoke/table01…table10` (CSV + LaTeX).
- Reports: `artifacts/reports/smoke/{final_report,run_status,figure_captions,failure_log}.md`.
- Manifests: `artifacts/manifests/smoke/` (18 per-stage manifests with
  resolved config, git commit, versions, hashes, timestamps).
- Activations: `artifacts/activations/smoke/network.npz`.

## Data counts (smoke)

47 worlds across 10 world-disjoint splits (4 steering_train,
4 steering_validation, 8 exogenous_train, 4 exogenous_validation,
4 exogenous_test, 6 endogenous_test, 4 recycling_test (2 matched pairs),
4 hysteresis_test, 5 phase_boundary_test (one per bin), 4 robustness_test);
64 exogenous emission trials, 128 receiver trials, 24 CAA pairs;
13 endogenous branch conditions × 6 worlds; balance and template-holdout
validation passed.

## Test results

47/47 passed. Coverage includes: exact oracle posteriors; duplicate events
counted once (aware) vs per-report (blind); no hidden-info leakage into
documents; ALPHA/BETA balance and crossing; held-out template variants only
in test splits; synchronous no-same-round-leakage; probes never entering
transcripts; multi-token sequence scoring; steering-hook layer/token scope
and exact deactivation; common-random-number pairing; one-hop / no-return /
full-text-clamp / fixed-replay routing invariants; malformed memos
retained; network resume without duplicate rows; config and dataset hash
stability; and the complete mock pipeline producing every required table,
figure, and report.

## Remaining full-run command

On a machine with ≥ 1×80 GB GPU (or multi-GPU with `device_map=auto`) and
Hugging Face access to the configured model:

```bash
make pilot   # bash scripts/run_all.sh configs/pilot.yaml
make full    # bash scripts/run_all.sh configs/full.yaml
```

plus, optionally, `configs/low_memory.yaml` (4-bit, reported separately)
and `configs/second_model.yaml` (Gemma-2-27B confirmatory subset). Slurm:
`sbatch scripts/run_slurm_array.sh configs/full.yaml`.
