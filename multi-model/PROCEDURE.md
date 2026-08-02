# multi-model — full procedure

End-to-end method for the Llama→Qwen transfer experiments. Reuses the
`cross-model` core (`graph.py`, `config.py`, `models.py`) and the `emotion`
GoEmotions helpers; only the cross-model generation feed, the additive steering
hook, and the emotion-projection read-out are new (all in `src/common.py`).

## Stage 0 — config & models (`common.py`)

- BASE models throughout: `meta-llama/Llama-3.1-8B` and `Qwen/Qwen3-8B-Base`
  (ungated mirrors `NousResearch/Meta-Llama-3.1-8B` / `Qwen3-8B-Base` on bare
  pods). `PRESET=smoke` swaps distilgpt2 for both → CPU plumbing test.
- Graph: a `gemma_qwen`-derived frozen config specialized to the 4×4 grid
  (`GKW["square_grid"]`), 16 concept words. Everything result-affecting is an
  env knob (walk sizes, layer bands, dose, context grid, perm count).
- Residual stream = **post-block** hidden state (`out[0]` of each decoder block),
  captured by forward hook. Layer index L == decoder-block L, used identically
  for capture, emotion-vector extraction, steering, and read-out.
- Tokenizer alignment = **last subword token** of each concept word, via the
  cross-model offset-mapping resolver — bit-identical to the established runs.

## Stage 1 — emotion vectors (`build_emotion_vectors.py`, per model)

1. Load a stratified GoEmotions subset (`LIMIT`, default 3000; split=all).
2. Feed each example text as plain text; hook every block; take the **last-token**
   residual per layer → one (nL, H) vector per example.
3. Per-emotion mean, then **center** by subtracting the mean across the 27
   non-neutral emotions (strips generic "text-ness").
4. **Deconfound:** per layer, PCA the neutral examples' activations and project
   the top PCs (≥`VAR_TARGET`=50% variance) out of every emotion vector.
5. Save `emotion_vectors_<TAG>.npz` (`clean`, `raw`, `emotion_names`, counts,
   npcs) + a meta json. Sadness row = `emotion_names.index("sadness")`; meta
   reports the layer where ‖sadness‖ peaks.

Run for `MODEL=Llama` (steering source) and `MODEL=Qwen` (measurement basis).

## Stage 2 — Exp1 grid transfer (`exp1_grid_transfer.py`)

1. **Generate (Llama).** Seed with `XCTX` steps of a real walk; sample `GSTEPS`
   more, each step restricting the LM head to the 16 node-word tokens
   (`candidate_token_ids`) and sampling one at `TEMP`. Records per-step neighbour
   mass and validity. `NSEED` independent walks.
2. **Self-probe (Llama).** Per-node-mean residual over the generated walk (occ
   with context ≥ `CTXLO`) → LOO coord probe at every Llama layer (sanity: Llama
   should carry the grid it just generated).
3. **Read-out (Qwen).** Feed each of three word sequences to Qwen and run the LOO
   coord probe at every Qwen layer, with an `NPERM` label-permutation null:
   - `llama_gen` — Llama's generated walk (**the transfer test**)
   - `real_walk` — a genuine random walk (upper bound)
   - `shuffled`  — generated tokens with node identities permuted (floor)
4. Output `exp1_grid_transfer.json` + a 2-panel PDF (per-layer mean R² across
   conditions; per-axis `llama_gen` vs the Llama self-probe).

The **leave-one-node-out** probe (train on 15 nodes, predict the 16th's (row,col),
alpha chosen on LOO) is the cross-model coord_decode procedure: a held-out node
lands correctly only if the geometry genuinely constrains it, so R²>null means
real 2-D grid structure — not node-identity memorization.

## Stage 3 — Exp2 sadness transfer (`exp2_sadness_transfer.py`)

1. **Steered generation (Llama).** Steering band = middle third of Llama's layers
   (override `STEER_LO/HI`). Added vector per layer = `SDOSE` × (typical residual
   norm) × unit(Llama sadness[L]), applied at all positions via a post-block
   forward hook during constrained walk generation. Two conditions: `clean`
   (no steer) and `sad`. Records gen validity/neighbour-mass for both (so a
   walk-destroying dose is visible).
2. **Walk sadness (Qwen).** For each generated walk, feed to Qwen and take the
   mean projection of Qwen's residual onto **Qwen's** unit sadness direction at
   every layer in the measurement band (middle third; override `MEASURE_LO/HI`).
   The layer of max Δ(sad−clean) is the reported peak.
3. **Story (Qwen).** Prefix = the full generated walk; Qwen free-generates
   `STORY_TOK` tokens (unconstrained). Sadness scored two ways: mean
   Qwen-sadness projection over the story tokens, and a model-independent
   `sad_word_fraction`. clean-context vs sad-context; two sample stories saved.
4. **Context-length sweep.** For `ctx` in `CTX_GRID` (default 10,25,50,100,220):
   feed Qwen only the first `ctx` steps of the walk, measure (a) walk projection
   at the peak layer and (b) story sad-word fraction. Answers whether seeing more
   of Llama's sad generation makes Qwen sadder.
5. Output `exp2_sadness_transfer.json` + a 3-panel PDF (per-layer projection
   clean vs sad; context sweep of projection; context sweep of story sadness).

## Stage 4 — driver & deploy

- `src/run_all.sh` runs Stages 1–3 into `$RUN_DIR` in order.
- `remote/deploy.sh` rsyncs `cross-model/src`, `emotion`, `multi-model`
  (preserving the cbai-2026 relative layout so `common.py` resolves siblings),
  installs deps, launches `run_all.sh` detached under `HF_HOME=/workspace/hf`
  (reusing the pod's cached weights), polls to completion, and pulls
  `runs/<run>/` back to the Mac (per the standing "always pull GPU results" rule).

## Knobs (env)

`PRESET` `DEVICE` `RUN_DIR` · emotion: `MODEL` `LIMIT` `MAXTOK` `VAR_TARGET` ·
generation: `NSEED` `XCTX` `GSTEPS` `TEMP` · Exp1: `NWALKS_REAL` `WLEN_REAL`
`CTXLO` `NPERM` · Exp2: `SDOSE` `STEER_LO/HI` `MEASURE_LO/HI` `STORY_TOK`
`CTX_GRID`.
