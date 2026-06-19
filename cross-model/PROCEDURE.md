# Full Experiment Procedure

End-to-end procedure for the cross-model alignment experiment, from raw graph to
final alignment metrics. Each step points at where it lives in the code.

---

## Stage 0 — Configuration & seeding (`config.py`, `run.py`)

1. Select a preset (`DEFAULT` for the H200 run, `SMOKE` for the CPU test). The
   config is a frozen dataclass — every result-affecting knob lives there: graph
   size, walk parameters, model names, captured layers, the alignment layer
   pair, the subword rule, PCA dim, ridge α, split fraction, the well-posedness
   ratio, and the parallel-capture flag.
2. `run.py` seeds Python `random`, NumPy, and Torch (CPU + CUDA) from
   `config.seed` before anything else, so walks, splits, and any stochastic
   model state are reproducible.
3. Output paths are created under `runs/<preset>/`.

---

## Stage 1 — Build the graph (`graph.py: build_grid_graph`)

4. Construct an `R×C` grid (default 4×4 = 16 nodes). Each node gets edges to its
   orthogonal (up/down/left/right) neighbors. Node count is configurable via
   `grid_rows/grid_cols`.
5. Assign each node a concept word from the **fixed, index-ordered** `WORDS`
   list (apple, bird, sand, math, …). The assignment is identical for both
   models — node *i* is always `WORDS[i]`. These are semantically *unrelated*
   common words, so there is no competing pretrained prior (the plain-random-walk
   condition, **not** the semantic-conflict condition).
6. Store each node's `(row, col)` coordinates. These are the ground truth for the
   grid-recovery check (`grid_distance_matrix` returns Manhattan distances).

---

## Stage 2 — Generate the walks (`graph.py: generate_walks`, `occurrence_table`)

7. Generate `n_walks` uniform random walks (default 200), each `walk_length`
   steps (default 1000). Walk *i* **starts at node `i mod n_nodes`**, so with
   `n_walks ≥ n_nodes` every node is a start node and appears early.
8. At each step: look at the current node's neighbors, pick one uniformly at
   random (seeded RNG), move there, emit that node's word. The result is a `Walk`
   holding `nodes`, `words`, and a materialized `text` (space-joined words).
9. **Walks are materialized once, as word sequences**, and reused for both
   models — this guarantees both models consume the *exact same* sequences.
10. Build a flat **per-occurrence index** (`occurrence_table`): for every emitted
    word across all walks, record `(walk_id, step, node, context_length)`, where
    `context_length = step + 1`. This word-step axis is **identical across
    models** despite tokenizer differences — it is what makes the
    matched/mismatched control well-defined.

The walk corpus and config are saved to `runs/<preset>/walks.json` and
`config.json`.

---

## Stage 3 — Capture activations (`models.py`, `run.py: stage_capture`)

Two scheduling paths produce identical results (verified bitwise-equal); choose
via `config.parallel_models` + the VRAM guard:

- **Sequential** (default fallback / small cards): load model A → capture → free
  → load model B → capture → free. Bounded peak memory.
- **Parallel co-resident** (`parallel_models=True` and device ≥
  `parallel_min_gb`, e.g. the H200): load BOTH models, run each walk through both
  in **one shared pass** over the corpus (`capture_many`). Skips the load/free
  cycle and iterates the corpus once. *Caveat:* on a single GPU the two forward
  passes serialize at the kernel level — the `entries[*]["device"]` field is
  plumbed so each model can later be pinned to its own GPU for true overlap.

For each model:

11. Load in bf16 with its **fast** tokenizer (asserted — offset mapping is
    required). Register a forward hook on each decoder block in
    `capture_layers_*`; the hook grabs `output[0]`, the **post-block
    residual-stream hidden state**.
12. For each walk (one sequence at a time, `batch=1`):
    - **Tokenizer alignment** (`resolve_token_spans`): tokenize with
      `return_offsets_mapping=True`. For each emitted word, compute its character
      span and find all tokens whose span overlaps it (special tokens mapping to
      `(0,0)` are skipped). This resolves the tokenizer mismatch explicitly — a
      word that is 1 token in one model and several in another is handled the
      same way in both.
    - **Apply the subword rule** (confirmed: **last** subword token). `first` and
      `mean` are also implemented for ablations.
    - Run the forward pass; for each captured layer, pull the hidden state at the
      resolved token index for **every occurrence**, cast to fp16, move to CPU.
13. **Critical:** every individual occurrence is kept — each word, each context
    length, each walk is its own vector tagged
    `(node, layer, context_length, walk_id, step)`. **No per-node averaging
    here** (that is only the sanity check). This is what makes the regression
    well-posed and what carries the in-context dynamics.
14. Save to `runs/<preset>/acts_model_{a,b}.npz` (fp16 acts + metadata). An
    assertion checks occurrence counts line up across layers and metadata.

After this stage **all inference is done** — every later stage reads the cache,
so analysis re-runs without re-inference.

---

## Stage 4 — Paper-reproduction sanity check (`reproduce.py`, `run.py: stage_reproduce`)

Run **before** trusting any alignment number.

15. For each model, at the alignment layer, and at each context checkpoint *C*:
    - Compute **per-node mean** activations over occurrences with
      `context_length ≥ C` (means are appropriate *here only* — they are the
      paper's object).
    - PCA those node means to 2D.
    - Correlate PCA-space pairwise distances with the true grid (Manhattan)
      distances — the grid-recovery score.
16. Expect the distance-correlation to **rise with context length** and to be
    clearly positive at high context.
17. Save a scatter plot of the highest-context 2D layout (nodes labeled by word,
    graph edges drawn) to `grid_recovery_{a,b}.png`, and numbers to
    `reproduce.json`. If the grid does not emerge, stop and fix capture before
    proceeding.

---

## Stage 5 — Alignment fitting & metrics (`align.py`, `run.py: stage_align`)

18. **Pair the activations** (`pair_occurrences`): load both caches at the
    configured layer pair (Llama 26 ↔ Gemma 32 by default), assert the
    `(walk_id, step)` metadata is identical row-for-row, return `X_A` (n×4096)
    and `X_B` (n×3584) paired by occurrence. Cast to float64.
19. **Split by walk** (`split_by_walk`): whole walks go to train or test (default
    25% test) so train/test walks never overlap — no leakage from shared
    sequences.
20. **Well-posedness guard** (`check_wellposed`), counted on the train set, for
    *both* maps:
    - Full ridge: `n_params = d_A·d_B + d_B ≈ 14.7M`.
    - PCA-Procrustes: `n_params = k·(k−1)/2`.
    - Warn loudly if `n_samples < ratio × n_params` (default 10). On the real
      models this **will** flag full-space ridge as under-determined — by design
      — and pass the subspace map; the subspace map is the trustworthy one.
21. **Map (a) — full-space ridge** (`fit_ridge`): center both sides, solve
    `W = (XᵀX + αI)⁻¹ XᵀY` (4096×3584). Report train + held-out R².
22. **Map (b) — shared PCA subspace + orthogonal Procrustes** (`fit_procrustes`):
    PCA each model's train activations to top-`k` (default 100), project both
    into their k-dim subspaces, solve orthogonal Procrustes `R = U Vᵀ` from
    `SVD(Z_Aᵀ Z_B)`. This is the rectangular-safe alignment (true orthogonal
    Procrustes is unavailable in the differing full spaces). Report normalized
    Procrustes residual (train + test) and in-subspace R².
23. **Linear CKA** (`linear_cka`) on held-out occurrences — a basis-free,
    dimension-agnostic cross-check independent of either fitted map.
24. **Trajectory vs endpoint** (point 4): the maps are fit **once, pooled across
    all context lengths**, then alignment is reported **separately at each
    context checkpoint** (occurrences binned within ±20% of *C*): ridge R²,
    Procrustes residual, CKA. Answers whether one map aligns the *whole*
    in-context trajectory or only at convergence.
25. **Matched vs mismatched context control** (`_matched_control`, point 5):
    using the pooled ridge map, score A's occurrences at context *C* against true
    B activations at the **same** *C* (matched) vs B at a **different** *C′*
    (mismatched, equal-sized random subset). If matched is not reliably better,
    the alignment tracks static word geometry, not the in-context process.
26. Everything is serialized to `runs/<preset>/alignment.json`; `run.py` prints a
    summary: guard status, ridge R², Procrustes residual, CKA, the
    context-length trajectory table, and the matched-wins tally.

---

## Data provenance

```
WORDS (fixed)          coords (grid)
      │                    │
      ▼                    ▼
  walks (seeded, shared)  ──────────────► grid_distance_matrix
      │                                        ▲
      ├──────────────┬───────────────┐         │ (sanity)
      ▼              ▼                │         │
  tokenizer A    tokenizer B         │   per-node means ─► PCA ─► distance corr
  (offset map)   (offset map)        │
      ▼              ▼                │
  last-subword   last-subword        │
   hooks @L26     hooks @L32         │
      ▼              ▼                │
   X_A (n×4096)  X_B (n×3584)  ◄──────┘  paired by (walk_id, step)
            \      /
             ▼    ▼
   split by walk → guard → ridge(a) / PCA-Procrustes(b) → R²/residual/CKA
                                  │
                                  ├─ per context length (trajectory)
                                  └─ matched vs mismatched (control)
```

---

## Running

```bash
pip install -r requirements.txt
huggingface-cli login            # both models gated — accept licenses first

python run.py --preset default --stage all        # full run on the H200
python run.py --preset default --stage capture     # inference only (cache acts)
python run.py --preset default --stage reproduce    # sanity check from cache
python run.py --preset default --stage align        # alignment from cache

python run.py --preset smoke --stage all           # tiny CPU end-to-end test
```
