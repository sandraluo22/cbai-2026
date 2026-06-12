# QSG × Partial-Information ("Elephant") multi-agent harness

A research harness that grafts **Quantized Simplex Gossip (QSG)** onto a
partial-information reasoning task (the *blind men and the elephant*), with
**first-class activation capture** for a downstream linear-transformation project.

---

## Model ↔ task mapping

| QSG concept | This harness |
|---|---|
| Belief `x_i ∈ Δ^(K-1)` | distribution over `K` candidate objects (elephant, snake, tree, …) |
| Speaker message `y` | quantized draw from the speaker's belief (Hard / Top-m / Soft) |
| Adaptation `x_L ← (1-α)x_L + α·y` | convex update on the public simplex |
| Population `N`, rate `α`, bandwidth `m` | sweep axes |
| *(new)* ground truth | a hidden object actually exists → selection vs. QSG sampling drift |
| Private observation | each agent sees a **subset of features** (text) or an **image crop** (image) |

The research question: does ground truth act as **selection** against QSG's
sampling drift, and **how/when do agent beliefs converge** (the core measurement)?

## Two arms (identical `Agent` interface, one shared model in memory)

- **Text** — `meta-llama/Llama-3.1-8B-Instruct`; each agent gets a random subset of
  the object's ground-truth text features.
- **Image** — `Qwen/Qwen2-VL-7B-Instruct` (fallback `Llama-3.2-11B-Vision-Instruct`);
  each agent sees a random crop of one source image. If no VLM loads, the image arm
  **skips gracefully** instead of crashing the sweep.

Only the private-observation → prompt step differs; engine, logging, activation
capture and analysis are shared.

## Two coupling modes (`coupling_mode`)

- **`pure_qsg`** (Mode 1) — the private observation seeds `x_i(0)` via one LLM
  readout (activations captured at round 0); thereafter the public simplex evolves
  by **exact numeric QSG** (no further LLM calls).
- **`two_layer`** (Mode 3, primary) — every round each agent listens to one speaker;
  the speaker emits a QSG message from its **public** simplex, the message is
  injected into the listener's prompt **alongside its persistent private
  observation**, the listener **re-reads** its belief `r_L` via the LLM
  (activations captured), and the public simplex is convex-updated
  `x_L ← (1-α)x_L + α·r_L`.

> **Conceptual subtlety (the spec flags this):** in the pure numeric model `y`
> updates the simplex directly; in the LLM-coupled `two_layer` mode the "update" is
> *realized* by feeding the message into the listener's prompt and re-reading. The
> public simplex still follows a convex QSG update, but the private observation
> **persistently biases the readout**. `qsg_reference.py` is the analytic null
> model to compare LLM dynamics against.

## Belief readout — done BOTH ways every round (§3)

- **Soft** (primary): next-token logits at a **fixed anchor** (`…Answer:`),
  restricted to candidates, renormalized. Scored two ways and both logged:
  *first-token logit* (biased toward common first tokens) and *length-normalized
  full-sequence logprob* (**default canonical**).
- **Hard** (validation): model emits an explicit JSON distribution, parsed robustly.
- **Agreement**: KL and L1 between soft and hard logged per agent per round; a
  summary plot (`plots/soft_vs_hard.png`) surfaces divergence rather than hiding it.

## ⚓ Anchor-position invariant (read before touching activations)

Activations are captured at the **last token of the soft-readout prompt** — the
same position whose next-token logits define the soft belief `x_i`. So for every
`(round, agent)` the stored residual-stream vector and the belief are
**positionally aligned** and comparable across agents/rounds/runs. Capture is at
**all layers, every round**, stored as `(rounds+1, N, n_layers, hidden_dim)` with a
JSON metadata sidecar (model, layers, hidden dim, dtype, anchor token id, template
hash). See `qsg/activations.py`.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) numeric null model + full plotting stack (no GPU) — run this FIRST
python run_reference_demo.py
pytest tests/ -q                       # §8 sanity checks

# 2) tiny end-to-end LLM smoke (CPU/MPS, no real weights needed)
python -m qsg.sweep --single --config configs/base.yaml \
  --override base.smoke_test=true base.qsg.n=4 base.qsg.rounds=3

# 3) real sweep (8-GPU box)
python -m qsg.sweep --config configs/base.yaml --dry-run     # job list + disk/compute estimate
python -m qsg.sweep --config configs/base.yaml               # text arm
python -m qsg.sweep --config configs/base.yaml --override arms='[image]' \
  base.observation.source_image_dir=/path/to/images          # image arm

# drift-vs-selection contrast in one flag (§8)
python -m qsg.sweep --single --config configs/base.yaml --override base.neutral_ablation=true
```

### Loading activations (input to the linear-transformation project)

```bash
python scripts/load_activations_example.py results/<run_id> --round 5 --layer 16
# -> X.shape = (N, hidden_dim)   # all agents' layer-16 activations at round 5
```

## Layout

```
qsg/qsg_reference.py   numpy null model + the QSG channel (shared math)
qsg/config.py          pydantic config, YAML load, dotted CLI overrides, sweep expansion
qsg/agents.py          Agent interface + shared ModelBackend (text + vision)
qsg/readout.py         soft + hard + agreement + degenerate detector
qsg/activations.py     forward hooks + store + load_activations()
qsg/analysis.py        similarity heatmaps + U/V/accuracy + soft-vs-hard agreement
qsg/engine.py          simulation loop (both coupling modes)
qsg/sweep.py           entry point + cost estimate + --dry-run
configs/               base.yaml (+ sweep axes) and ontology.yaml
```

## Per-run outputs (`results/<run_id>/`)

`config.yaml`, `beliefs.jsonl` (both readouts + agreement per agent/round),
`beliefs.npy`, `activations/` (store + meta), `plots/` (similarity heatmaps ×2,
U/V/accuracy, soft-vs-hard), `manifest.json`.

## Notes / status

- HuggingFace **local** models + forward hooks only (no API agents — required for
  activations). Compute/disk estimates printed before any large run; the store
  refuses-with-warning above `activations.max_disk_gb`.
- Determinism: separate seeded RNGs for QSG sampling vs. setup; seeds logged.
- The numeric reference, plotting, §8 unit tests, and a tiny CPU LLM run are
  validated end-to-end. Real Llama-8B / Qwen2-VL runs require the GPU box.
