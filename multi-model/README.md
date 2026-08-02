# multi-model — Llama → Qwen representation & emotion transfer

Two experiments that chain **`Llama-3.1-8B` (base)** into **`Qwen3-8B-Base`**:
Llama *generates* a token sequence, that sequence is fed to Qwen, and we read
Qwen's internal state. Both models are the same BASE checkpoints used in
[`../cross-model`](../cross-model), and this package reuses cross-model's graph /
model / tokenizer-alignment core and the [`../emotion`](../emotion) GoEmotions
helpers rather than re-implementing them.

The unifying object is the **4×4 square grid over 16 unrelated concept words**
(apple, bird, sand, …) from the Park et al. in-context-representations paper.
Llama free-generates a random walk over that grid with its next-token
distribution constrained to the 16 node words; because the output is a plain
word sequence, Qwen consumes the *identical* tokens and pairing is by (node,
step) exactly as in cross-model.

## Experiment 1 — grid-geometry transfer (`exp1_grid_transfer.py`)

**Does a walk Llama generates induce the square-grid representation inside Qwen?**

1. Seed Llama with a real random-walk prefix, let it free-generate the rest
   (constrained to node words).
2. Feed the generated word sequence to Qwen; fit the **leave-one-node-out
   coordinate probe** (predict each node's (row, col) from its in-context mean
   residual) at **every Qwen layer**, with a label-permutation null.

Conditions probed in Qwen: `llama_gen` (the transfer test), `real_walk` (a
genuine walk — upper bound), `shuffled` (specificity floor). Plus a Llama
self-probe on its own generated walk and the behavioural fidelity (neighbour
mass / validity) of the generation.

## Experiment 2 — sadness transfer (`exp2_sadness_transfer.py`)

**Steer sadness into Llama while it generates the walk; does the sadness ride
along into Qwen?**

1. `build_emotion_vectors.py` builds per-emotion "emotion vectors" for **both**
   models from GoEmotions (the emotion/ diff-of-means recipe, adapted to base
   models: last content token, every layer, neutral-PC deconfound). This yields
   Llama's sadness direction (to steer) and Qwen's (to measure).
2. Steer Llama's residual stream along its sadness direction (a band of layers,
   dose `SDOSE` × residual norm) while it generates the walk; also a clean run.
3. Feed each walk to Qwen and measure, per Qwen layer, the mean projection of
   Qwen's residual onto **Qwen's own** sadness direction. sad-steered vs clean.
4. **Story:** give Qwen the walk as a prefix, free-generate a short
   continuation, score its sadness (Qwen-sadness-projection + a model-independent
   sad-word fraction).
5. **Context-length sweep:** vary how many walk steps Qwen sees (`CTX_GRID`) —
   does more Llama-sad-context make Qwen sadder / its story sadder?

> Note: the walk is 16 emotionally-neutral words, so any sadness reaching Qwen
> must be carried by *which* words Llama emits and their ordering — a deliberately
> stringent test. The unconstrained **story** is where free emotional expression
> can surface. `SDOSE` is sweepable; too large a dose randomizes the walk
> (validity collapses), which the run records so you can pick a safe dose.

## Experiment 3 — sadness transfer via PROMPT, not steering (`exp3_sad_story_transfer.py`)

Same measurements as Exp2, but sadness is induced by **natural-language context**
instead of residual-stream surgery: prepend a genuinely sad passage
(`common.SAD_STORY`) vs a length-matched neutral one (`common.NEUTRAL_STORY`) to
Llama, let it generate the walk while reading the story, feed the walk (only) to
Qwen, and measure the sadness projection + story. This removes Exp2's confound
(steering also *degraded* the walk); prompting perturbs walk validity
symmetrically across conditions, isolating "does sad *content* transfer" from
"does a *degraded* walk transfer."

## Experiment 4 — decode the priming emotion from Qwen's activations (`exp4_walk_emotion_classifier.py`)

The most sensitive test. Prime Llama with a **happy** story vs a **sad** story
(structural mirrors — same skeleton, opposite valence), generate many walks under
each, let Qwen read the walks only, pool Qwen's residual per walk, and train a
per-layer linear classifier (StandardScaler → PCA → logistic regression, stratified
k-fold CV) to separate happy- from sad-primed walks. Per-layer CV accuracy / AUC
vs a label-shuffle null answers: does the priming emotion leave *any* decodable
trace in the neutral-word walk that survives the hop into Qwen — even one no fixed
direction would catch?

## Layout

```
multi-model/
  src/
    common.py                  # shared spine: model loading (base Llama/Qwen), grid,
                               #   constrained generation, steering hook, coord probe,
                               #   emotion-vector I/O + projection read-out
    build_emotion_vectors.py   # GoEmotions emotion vectors per model (all layers)
    exp1_grid_transfer.py      # grid transfer + controls, all Qwen layers
    exp1_probe_slideshow.py    # per-layer scatter: old (real-walk) vs fresh (LOO) probe
    exp2_sadness_transfer.py   # sadness steer→transfer, story, context sweep
    exp3_sad_story_transfer.py # sadness via sad-story PROMPT (no steering)
    exp4_walk_emotion_classifier.py  # happy-vs-sad decoded from Qwen walk reps
    run_all.sh                 # drive all four stages into $RUN_DIR
  remote/
    setup.sh  deploy.sh        # sync 3 sibling dirs → H200, run, pull runs/ back
  runs/<run>/                  # emotion_vectors_*.npz, exp1_*.json/.pdf, exp2_*.json/.pdf
```

`common.py` puts `../cross-model/src` and `../emotion` on the path automatically
(override with `CROSS_MODEL_SRC` / `EMOTION_DIR`).

## Running

```bash
# tiny CPU end-to-end plumbing test (distilgpt2 for both models)
PRESET=smoke DEVICE=cpu RUN_DIR=runs/smoke bash src/run_all.sh

# real run on the H200 pod (syncs code, reuses the pod HF cache, pulls results):
HOST=root@<ip> PORT=<port> KEY=~/.ssh/id_ed25519 bash remote/deploy.sh gemma_qwen
```

Both base models are ungated mirrors on the pod (NousResearch Llama, Qwen3-8B-Base),
so no HF token is needed. See `PROCEDURE.md` for the full method.

## Results (H200, `runs/main`, square grid, base models)

### Exp1 — grid geometry transfers Llama → Qwen ✅

Llama's generated walk (constrained to the 16 node words; fidelity validity=0.54)
was fed to Qwen and the leave-one-node-out coord probe run at every Qwen layer:

| Qwen reads | peak layer | mean LOO R² | row p | col p |
|------------|-----------:|------------:|------:|------:|
| **`llama_gen`** (Llama's generated walk) | L34 | **+0.395** | 0.040 | 0.005 |
| `real_walk` (a genuine walk, upper bound) | L33 | +0.683 | 0.005 | 0.005 |
| `shuffled` (identity-permuted, floor)     | L35 | −0.138 | 0.82 | 0.36 |

**The grid genuinely transfers:** a walk that *Llama* generated induces a
statistically significant 2-D grid representation inside *Qwen* (~58% of the
real-walk ceiling), while the shuffled control sits at the permutation null.
Qwen encodes it in deep layers (L33–35); Llama's own grid peaks at L13.

### Exp2 — sadness does NOT ride the neutral-word walk ⚪ (null, with a clear reason)

Steering Llama toward sadness (dose 0.6× residual norm, layers 10–20) while it
generated the walk lowered its validity (0.53 → 0.27) but changed Qwen's sadness
projection only trivially: Δ(sad−clean) = **+0.22** at the peak Qwen layer
against a projection magnitude of ~16 (i.e. noise). The context-length sweep
shows the Δ growing only from ~0 to +0.21 as Qwen sees more of the walk —
negligible. And Qwen's "story" is degenerate: given 220 neutral concept words,
the **base** model just continues emitting concept words
(`knife bread stone bread apple chair apple …`), so sad-word fraction = 0 in both
conditions.

**Interpretation:** the 16 emotionally-neutral concept words are a near-zero-
bandwidth channel for affect — grid *geometry* survives because it is carried by
*which* words and in *what order*, but a diffuse emotional state has nowhere to
live in that token stream, and a base LM prompted with a walk writes more walk,
not prose. To make the sadness/story test non-degenerate, run the natural
follow-up: let Llama free-generate **unconstrained** sad text (not restricted to
the 16 words) and/or use **instruct** models for the story leg — the scaffolding
(`SDOSE`, unconstrained-gen hook, projection read-out) is already in place.

### Exp3 — sad *story* prompt also doesn't transfer, but cleanly ⚪

Priming Llama with a sad vs neutral **story** (no steering) moved Qwen's sadness
projection by only Δ=+0.14 at the peak layer (~0 at short context; +0.17 at full
context) — the same null as Exp2, but now the walk validity is degraded
*symmetrically* by the story (neutral 0.33 vs sad 0.32), so it is not the
steering-induced walk damage doing it. Sad *content* simply does not survive the
neutral-word bottleneck. (Qwen's "story" is again just more walk, sad-word
frac = 0 both ways.)

### Exp4 — happy vs sad is NOT decodable from Qwen's walk activations ⚪ (strong null)

The most sensitive test: prime Llama happy vs sad, 50 walks each, train a
per-layer linear classifier (PCA + logistic, 5-fold CV) on Qwen's per-walk
activations. **Every layer sits at or below the label-shuffle null** — accuracy
0.15–0.37 (null ≈0.50), ROC-AUC 0.11–0.34 (chance 0.5); peak L0 acc=0.37 <
null 0.49. Llama walk validity was symmetric across conditions (happy 0.40 /
sad 0.38), so this isn't a walk-quality artifact. Even a trained classifier finds
no happy/sad trace in the walk that crosses into Qwen.

**Bottom line.** Grid *geometry* transfers Llama→Qwen (Exp1); *emotional* state
does not — not by activation steering (Exp2), not by sad-story prompting (Exp3),
and not even to a trained decoder (Exp4). Structure that is encoded in *which*
neutral words appear and in *what order* rides the walk; a diffuse affective state
has no bandwidth in that channel. To transfer emotion you need a channel that can
actually carry it — unconstrained (not 16-word) generation and/or instruct models
for the story leg; the scaffolding (steering hook, story prefixes, projection
read-out, walk-level classifier) is all in place.

Figures: `runs/main/exp1_grid_transfer.pdf`, `exp1_probe_slideshow.pdf`,
`exp2_sadness_transfer.pdf`, `exp3_sad_story_transfer.pdf`,
`exp4_walk_emotion_classifier.pdf`.

### Within-model controls (Llama→Llama) and activation injection

**Llama reads its own generated walk (Exp1/Exp4 with `READER=Llama`).** The
cross-model hop costs almost nothing for the *generated* walk:

| grid recovery | reader = Qwen | reader = Llama |
|---|---|---|
| `llama_gen` walk | 0.40 (L34) | **0.40** (L13) |
| `real_walk`      | 0.68 (L33) | 0.75 (L31) |
| `shuffled`       | −0.14 | −0.14 |

Reading the generated walk within-model (0.40) ≈ cross-model (0.40) — the ceiling
is the *generated walk's* fidelity (validity 0.54), not the model boundary
(within-model helps only for a clean real walk, 0.75 vs 0.68). And Exp4 with
`READER=Llama` **still can't decode happy vs sad** (peak acc 0.41 < null 0.49) —
so the emotion is genuinely absent from the neutral-word walk, not merely lost in
the Llama→Qwen hop.

**Linear-map activation injection, per layer (`cross-model/injection.py`, run
Llama→Qwen and Llama→Llama).** Instead of passing tokens, fit a ridge map
Llama@L_A → target@L_B on the 16 node-means and *replace* the target's residual at
L_B with the mapped Llama activation, reading downstream next-step neighbour mass
over an L_A×L_B sweep:

| inject | target native | injected peak | best cell | alignment R² peak |
|---|---|---|---|---|
| **Llama→Qwen** | 0.66 | **0.81** | L_A 4 → L_B 11 | 0.55 |
| **Llama→Llama** | 0.84 | **0.91** | L_A 4 → L_B 10 | 0.61 |

Injecting mapped Llama activations into **Qwen's mid layers (L_B≈7–14)** drives
Qwen's grid-valid next-step prediction *above* its own native 0.66 (to 0.81) —
the linear map preserves the grid-driving subspace, robustly across source layer.
The within-model ceiling is a bit higher (0.91 vs 0.81), so the cross-model map
captures most (not all) of the transferable grid signal. So grid structure
crosses Llama→Qwen both ways — as generated **tokens** (Exp1) and as linearly-mapped
**activations** — whereas emotion crosses by neither.

Figures: `runs/main/exp1_grid_transfer_Llama.pdf`,
`exp4_walk_emotion_classifier_Llama.pdf`,
`injection_Llama_to_Qwen_square_grid.pdf`, `injection_Llama_to_Llama_square_grid.pdf`.

### Exp5 — does EMOTION cross by activation injection? ⚪ No (`exp5_emotion_injection.py`)

The missing cell of {grid, emotion} × {tokens, activations}. Fit the ridge
alignment map W: Llama@L_A → Qwen@L_B on 1608 paired per-occurrence walk
residuals, then ask whether W carries emotion the way it carries grid geometry.

- **Geometric axis transport:** push Llama's sadness/joy directions through W and
  cosine against Qwen's own — matched cos ≈ **+0.03** at every layer pair, *no
  better* than the mismatched sad→joy control (0–0.05). The grid-aligned map is
  essentially orthogonal to the emotion subspace.
- **Causal steer:** injecting the transported sad direction into Qwen leaves its
  generation unchanged (sad-word frac 0.000, proj −42 vs a −40 baseline) — same as
  a random direction — while Qwen's **own** sad vector *does* steer it (sad-word
  frac 0→0.009, proj −40→−20). So the injection machinery works; the transported
  signal carries no emotion.

**So emotion crosses by neither tokens nor activation injection.** The linear map
that transports the grid subspace (grid injected-behaviour 0.66→0.81) does not
transport the emotion axis. Caveat: this map is fit on *walk* activations, so it
aligns the grid-relevant subspace; a map fit on emotional paired data would of
course carry emotion (that is just re-deriving each model's emotion vector). The
result is specifically: the alignment that carries the walk's geometry does not
also carry affect.

Completed 2×2:

| carries… | as TOKENS | as ACTIVATION INJECTION |
|---|---|---|
| **grid geometry** | ✅ Exp1 (R² 0.40, ceiling 0.68) | ✅ injection (0.66→0.81) |
| **emotion** | ⚪ Exp2/3/4 (null, even Llama→Llama) | ⚪ Exp5 (cos 0.03, no steer) |

Figure: `runs/main/exp5_emotion_injection.pdf`.

## Exp6 — one model as an HMM, the other as the observer (`exp6_latent_filtering.py`)

The unifying experiment. A random walk on the grid is the HMM; its emission is
**coarsened** so a token reveals only a *group* of nodes, not the node — the
current node becomes a genuine hidden latent the observer must **filter**. An
observability dial `A` (number of symbols) interpolates between the two regimes
we already saw: `A=16` = fully observed (= Exp1), `A=1` = latent invisible
(= the emotion case). Both Llama and Qwen read the same aliased stream; we
forward-filter the exact Bayes-optimal belief `b_t = P(node_t | symbols_1..t)`
and, per layer / per `A`, decode the hidden node and the belief simplex.

| A (symbols) | group | Bayes-optimal | Llama | Qwen | corr(Qwen belief, Llama belief) |
|---|---|---|---|---|---|
| **16** (observed = Exp1) | 1 | 1.00 | 1.00 | 1.00 | — (one-hot) |
| 8 | 2 | 0.935 | 0.878 | 0.888 | **0.978** |
| 4 | 4 | 0.745 | 0.606 | 0.617 | **0.963** |
| 2 | 8 | 0.433 | 0.287 | 0.281 | **0.903** |
| **1** (hidden = emotion) | 16 | 0.079 | 0.076 | 0.086 | — (constant) |

Three findings:
1. **Both models filter the hidden latent** — hidden-node decode accuracy tracks
   the Bayes-optimal ceiling as observability drops (1.00→0.88→0.62→0.28→0.08 vs
   optimal 1.00→0.94→0.75→0.43→0.08), always well above chance (1/16=0.06) until
   the latent goes fully hidden. They are good but *sub-optimal* Bayesian filters
   (a consistent gap below the ceiling in the ambiguous middle).
2. **Qwen ≈ Llama at every level** — neither is the better observer.
3. **They build the SAME belief** — in the genuinely-hidden regime (A=8,4,2) the
   correlation between Qwen's and Llama's decoded belief distributions is
   0.90–0.98. So one model's inferred latent is linearly the other's: **Qwen
   models Llama's hidden state as an HMM** (and vice versa), because both converge
   on the belief state the process dictates.

This dial *is* the observability principle made continuous, and it contains the
whole project: at `A=16` it reproduces Exp1's grid transfer; at `A=1` it
reproduces the emotion null. Recovery degrades exactly as the latent stops
leaking into the emissions — which is precisely why grid geometry crossed and
emotion did not.

Figure: `runs/main/exp6_latent_filtering.pdf` (recovery vs the dial: Qwen /
Llama / Bayes-optimal; belief-simplex R²; cross-model belief agreement).

## Exp7 — Qwen models LLAMA's OWN latent, observing only Llama's emissions (`exp7_model_latent.py`)

Exp6's latent was the *process's* ground-truth node, and both models were symmetric
co-readers of an external stream — so "Qwen models Llama" was only inferred
indirectly. Exp7 fixes the setup: **Llama is the source.** Llama autoregressively
*generates* the walk (Llama = the HMM; its hidden state is the latent, its sampled
tokens are the emissions); Qwen observes **only Llama's emitted tokens** and must
reconstruct Llama's internal state. The latent is genuinely hidden even so, because
the emitted token is a lossy *sample* of Llama's state. Grids scale 16→25→36 states.

Decoding from Qwen's residual (held out by walk, per layer):

| grid | states | Q→Llama predictive | Q→ideal (process) | **Q→Llama deviation** | Q→Llama residual | ceiling (Llama→self) |
|---|---|---|---|---|---|---|
| 4×4 | 16 | 0.585 | 0.998 | **0.806** | 0.735 | 0.755 |
| 5×5 | 25 | 0.699 | 0.997 | **0.771** | 0.760 | 0.744 |
| 6×6 | 36 | 0.622 | 0.991 | **0.712** | 0.737 | 0.651 |

- **Qwen reconstructs Llama's predictive state** (R² 0.59–0.70) — *close to the
  within-Llama ceiling* (0.65–0.76). Observing only Llama's samples, Qwen recovers
  nearly as much of Llama's next-move distribution as Llama's own residual carries.
- **Qwen decodes Llama's literal residual** (top-50 PCs) at R² ≈ 0.74.
- **Qwen decodes Llama's *deviation from the ideal filter*** at R² **0.71–0.81** —
  the key result. Llama is very sub-optimal here (Llama-vs-ideal R² is strongly
  negative, −0.9 to −0.6: a large idiosyncrasy budget), and Qwen captures most of
  that idiosyncrasy. So Qwen tracks what is *specifically Llama*, not just the
  shared process. (Q→ideal ≈ 1 is the trivial reference — with full emission the
  current node is visible, so the process part is shared by construction; the
  non-trivial, Llama-specific signal is the deviation, and it decodes.)
- **Scales**: the picture holds from 16 to 36 hidden states (a mild decline).

So one model *does* model the other as an HMM in the strong sense: from Llama's
emissions alone, Qwen reconstructs Llama's own predictive latent, its internal
state vector, and its idiosyncratic departures from optimality. (Caveat: Qwen and
Llama share the transformer prior, so some of the deviation-decoding is plausibly
*correlated* sub-optimality rather than Qwen inferring Llama de novo; a
different-source control would separate these.)

Figure: `runs/main/exp7_model_latent.pdf`.

## Exp8 — emotion transfer vs CONTEXT LENGTH, both readers (`exp8_emotion_context.py`)

Re-runs the happy-vs-sad decode with much longer walks (400 steps, ~2× earlier)
and a context-length sweep, for Qwen (cross-model) AND Llama (within-model), to
test whether a weak per-step emotional bias ACCUMULATES with context, or whether
the within-model read exposes emotion the cross-model read missed.

| context (walk steps) | Qwen reader (null) | Llama reader (null) |
|---|---|---|
| 100 | 0.43 (0.49) | 0.40 (0.48) |
| 250 | 0.40 (0.50) | 0.48 (0.49) |
| 500 | 0.43 (0.48) | 0.49 (0.50) |
| **1000** | **0.44 (0.49)** | **0.51 (0.50)** |

**Both null at every context length, out to 1000 steps (~5× the original runs).**
More context does not help — accuracy sits at/below the label-shuffle null the
whole way — and Llama→Llama is no better than Llama→Qwen (the within-model read
drifts up to 0.51 with context but never separates from its own 0.50 null: noise,
not signal). Generation was symmetric (validity happy=0.37/sad=0.34), so it is not
a walk-quality artifact. The emotion signal is not weak-but-accumulating; it is
genuinely absent from the neutral-word emission channel, which no amount of
context or shared representation can recover — the observability principle again.

Figure: `runs/main/exp8_emotion_context.pdf`.
