# Methods: dueling-context experiments (test-1)

Full procedure, hyperparameters, and measurement definitions for every experiment in
this folder. Mirrors the chat write-up; scripts referenced are in this directory,
outputs in `runs/`, figures in `figs/`.

## 1. Common infrastructure

**Models.** Llama-3.1-8B base (mirror `NousResearch/Meta-Llama-3.1-8B`; 32 layers,
d=4096), bf16, single H200. Cross-model runs add `Qwen/Qwen3-8B-Base` (36L, 4096; no
BOS token) and `unsloth/gemma-2-9b` (42L, 3584). transformers 5.14.1, torch 2.8.0.

**Task encoding (Park et al. plain-walk condition).** 16 graph nodes are assigned 16
common, semantically unrelated English nouns; a context is `[BOS] + one token per walk
step` (every " word" is a single token in all three tokenizers; verified). No
instructions, separators, or any other text.

**Graphs (all 16 nodes).** grid 4x4 (24 edges); ring C16 (16); hex 4x4 (33); prism
C8xK2 (24); antiprism A8 (32); ring3 = C16(3) (16, stride-3 relabeled ring); torus
C4xC4 (32, 4-regular); circulants C16(+-1,+-k) (32, 4-regular); degree-preserving
interpolants G_alpha (torus->C16(1,3) via targeted double-edge swaps, connectivity
checked; alpha = normalized edge overlap with the C16(1,3) endpoint); small-world
(C16(1,2) + s random double swaps); deterministic 2-block "SBM" graphs (strong: 3
intra + 1 inter per node; weak: 2+2); random 4-regular expanders (configuration model,
min/max lambda_2 of 3000 samples). Exact edge lists: `runs/sweep_spec.json`.

**Walks.** Uniform random walks; walk w starts at node w mod 16; numpy
`default_rng(seed)`. Base seed 0 (sweep agent B: seed+500; kernel probe: seed 11).

**Word->node assignments.** Original runs: `WORDS[:16]` from
`cross-model/src/config.py`. All later runs use bigram-prior-optimized assignments
(section 3), one per graph union.

## 2. Core dueling protocol

Two (or N) "instances" = one set of weights with independent KV-cache contexts
(cross-model: genuinely different models, one context each).

1. **Prefill.** Each instance receives CTX walk tokens on its own graph
   (CTX=1000 main experiments; 600 sweep default; 100/300/1000 prior-strength cells).
   NPAIRS independent walk pairs run batched (8 main; 6 sweep).
2. **Joint generation, TGEN steps** (600 main; 400 sweep). Generators alternate
   (2-way: B on even steps; N-way: cycle). At its turn an instance's next-token
   distribution is read at the last position, restricted to the 16 candidate word
   tokens, softmaxed at temperature TEMP (1.0 default; 0.7/1.3 cells), truncated to
   its top-k (0 = none; degree-matched: ring/ring3 2, prism 3, grid/antiprism/
   4-regular family 4, hex 6), renormalized, sampled (numpy rng, seed+1). The sampled
   word is appended to every coupled context (its own token id per tokenizer).
   **Communication q**: the generator's token is delivered to a partner with
   probability q (default 1.0; 0.5/0.25 cells; per-direction q_ab/q_ba for one-way);
   undelivered -> the partner appends its own sample (contexts stay length-matched).
3. **Per-step logging.** Generator, sampled node, previous node, validity under each
   graph, each graph's neighbor probability mass (pre-top-k); sweep additionally logs
   the full JS divergence between the agents' 16-way predictives every step.
4. **Capture.** One full forward per final context with forward hooks on every
   decoder block (post-block residual stream). Word i's vector = hidden state at its
   token position. Per-node sums accumulated in word-step windows; node means =
   sum/count pooled over NPAIRS pairs; stored fp16 as [16 x d] per (context, window,
   layer).
5. **Windows** (word steps). base = prefix [CTX-300, CTX) (sweep: last 200);
   joint_early [0,100); joint_mid [100,300); joint_late [300,600). Sweep: early
   [0,T/4), late [T/2,T). One-way drivers (300 tokens): [0,50), [50,150), [150,300).
6. **Fresh control.** A blank context `[BOS] + joint stream` per pair, same capture,
   windows early/mid/late — measures what the stream alone teaches.

**Control conditions** (`run_controls.py`, 4-regular versions via `run_sweep.py`
cells): free-gen (each instance alone, only its own samples; q=0 in the sweep
version); one-way (driver never receives partner tokens: q_ab=1,q_ba=0 or reverse);
real-steps (no generation: both contexts receive the same interleaved stream of
ground-truth continuations of both prefix walks, alternating).

## 3. Vocabulary optimization ("paper tiger" fix)

`bigram_prior.py`: for each of the 36 pool words, forward `[BOS, " w1"]`, collect
logits of all 36 " w2" candidates; restricted softmax -> P(w2|w1); pair score
S = max(P, P^T). `pick_words.py`: choose 16 words + assignment minimizing
`max_edge(S) + 0.05 * mean_edge(S)` over the target graph-union's edges; random-
restart local search (30 restarts x 20k iters; position swaps + pool swaps). Worst
union-edge prior: 0.209 (default vocab) -> 0.0015-0.0038 (optimized, per pair).

## 4. Measurements

Let Hc = row-centered node-mean matrix [16 x d] for one (context, window, layer).
"Deep mean" = average over the last quarter of layers (Llama 24-31; cross-model by
relative depth, fractions 0.75-0.97).

- **Coordinate regression R^2.** Features F (z-scored columns): grid (row, col);
  ring (cos, sin); hex lattice coords; spectral coords (Laplacian modes 1-2) for
  other graphs. OLS: R^2 = 1 - ||Hc - F B||^2 / ||Hc||^2. Comparisons use matched
  feature dimensionality.
- **Laplacian eigenmode energy.** Normalized Laplacian L = I - D^-1/2 A D^-1/2,
  eigenvectors u_k (ascending eigenvalue). Energy fraction e_k =
  ||u_k^T Hc||^2 / ||Hc||^2. Candidate subspaces: each graph's modes, union-graph
  modes 1-2/1-4/1-6, empirical-stream-graph modes (W = symmetrized late bigram
  counts), log-occupancy (1D control).
- **Fourier (Z16) decomposition.** Orthonormalized cos/sin pairs at frequency
  f=1..8 over node index; energy per frequency. For grid+ring: f1 = shared cycle,
  f4 = grid column, f2/f6 = checkerboard harmonics; union ~ circulant C16(1,4).
  Parity probes: 1D energies of checkerboard (r+c)%2 and cycle i%2 patterns.
- **Procrustes shape similarity.** Center rows, scale to unit Frobenius;
  sim(A,B) = sum of singular values of Ahat^T Bhat = max over orthogonal R of
  <A R, B> (computed in the 16-point row space via thin SVDs, so it works across
  hidden sizes). Distance = 1 - sim. Invariant to translation/scale/rotation/
  reflection; NOT to node permutation (correspondence-based). Calibration: unrelated
  clean priors score sim ~0.88, so the working range of distance is ~0-0.30.
- **PCA.** SVD of Hc; top-2/3 PCs at unit RMS. The viewer Procrustes-chains frames
  across windows and layers for smooth interpolation; the mode-axes and flat-torus
  projections use union-mode amplitudes a_k = ||u_k^T Hc|| * 4 / ||Hc||_F and angles
  theta = atan2 in the (m1,m2) and (m3,m4) planes.
- **Behavioral metrics.** Per-graph validity fractions and neighbor-mass curves
  (15-step moving average); late-stream edge-class mass (shared / A-only / B-only /
  non-edge); node occupancy; top oscillating-pair fraction (of late transitions).
- **Sweep outcomes.** win margin = per-pair late-stream (t >= T/2) mass on A-only
  minus B-only edges of the generator-token stream; P(A wins) = fraction of pairs
  with margin > 0. JS divergence: JS(p_A, p_B) = 0.5 KL(p_A||m) + 0.5 KL(p_B||m),
  m = mean; per-step mean over pairs. T_consensus = first step where the 21-step
  rolling mean of JS stays < 0.05; js_final = mean of last 50 steps.
- **Update-kernel probe** (`probe_update.py`). 24 grid walks (400 words, optimized
  vocab, occurrences with step >= 50; 8376 regression rows). Regression: centered
  log-softmax over the 16 candidates ~ counts of past matched-row (a->j) and
  mismatched-row (c->j) transitions, binned by age {1-5, 6-10, 11-20, 21-40, 41-80,
  81-160, 161-320, 321+}; features candidate-centered; ridge lambda=1e-3. Attention
  pass (8 walks, eager attention): per-head attention mass from each occurrence of a
  onto successor slots (positions u+1 with x_u = a), by age bin; heads ranked by
  mean mass in the three youngest bins.
- **Bayesian surrogate** (`bayes_model.py`, `bayes_sweep.py`). Per learner:
  C <- gamma*C then C[i,j] += 1 for each transition observed in its OWN context
  (prefix + appended tokens); predictive p(j|i) proportional to
  (alpha0 + C[i,j])^(1/TEMP); same top-k/sampling/coupling protocol. Geometry =
  spectral embedding (modes 1-4, 1/sqrt(lambda) scaling) of alpha0 + C symmetrized.
  Fit: grid search gamma in {1.0, 0.999, 0.997, 0.995, 0.99, 0.98, 0.96, 0.93,
  0.90, 0.85, 0.75} x alpha0 in {0.02, 0.05, 0.15, 0.5, 1.5} minimizing MSE to the
  LLM's binned (12 bins) neighbor-mass curves. Fitted: gamma=0.96, alpha0=0.05
  (optimized vocab run); gamma=0.997 (original vocab runs). Surrogate sweep: same
  cells, 4 seeds, same outcome definitions.

## 5. Experiment roster

| run | instances (graph:top-k) | vocab | knobs |
|---|---|---|---|
| runs/out | grid:0 + ring:0 (Llama) | default | T=1 free-sample |
| runs/out_topk4 | grid:4 + ring:4 | default | |
| runs/out_k2 | grid:2 + ring:2 | default | |
| runs/out_k2k4 | grid:4 + ring:2 | default | degree-matched |
| runs/out_k2_fix | grid:2 + ring:2 | optimized | main clean baseline |
| runs/out_hexgrid | hex:6 + grid:4 | optimized | |
| runs/out_hexring | hex:6 + ring:2 | optimized | |
| runs/out_prismring | prism:3 + ring:2 | optimized | |
| runs/out_ringring3 | ring:2 + ring3:2 | optimized | zero shared edges |
| runs/out_antigrid | antiprism:4 + grid:4 | optimized | |
| runs/out_tri | ring:2 + grid:4 + ring3:2 | optimized | 3-way cycle |
| runs/out_x_LQ/LG/QG | Llama-grid:4 + {Qwen,Gemma}-ring:2 | optimized | cross-model |
| runs/out_x_LQG | L-ring:2 + Q-grid:4 + G-ring3:2 | optimized | 3-way cross-model |
| runs/out_ctrl_* | free / one-way / real (grid+ring) | optimized | ring 2, grid 4 |
| runs/sweep (25 cells) | torus + {G_alpha, families} all :4 | optimized | axes below |
| runs/sweep ctrl_* (4 cells) | torus + C16(1,3) | optimized | q=0 / one-way / real |
| runs/out_probe | kernel probe (single contexts) | optimized | |

Sweep axes: alpha in {0, .25, .5, .75, 1}; (ctx_A, ctx_B) in {(1000,100), (1000,300),
(1000,1000), (300,300), (100,100)}; TEMP in {0.7, 1.3}; top-k=0; q in {0.5, 0.25};
families circ2/5/6/7, sbm strong/weak, expanders lo/hi, small-world 4/12 swaps.

## 6. Compute

All generation uses KV-cache incremental decoding (batch = all pairs x instances),
prefill with `logits_to_keep=1`; capture forwards are single-sequence with hooks on
all blocks. Typical cost on one H200: main run ~2.5 min; sweep cell ~50 s; kernel
probe ~6 min; cross-model suite ~20 min (+ downloads; 30 GB disk forces
load-then-delete of each checkpoint, `HF_HUB_DISABLE_XET=1`).
