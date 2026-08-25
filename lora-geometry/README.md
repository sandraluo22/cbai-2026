# lora-geometry — does weight space have the same shape as activation space?

Started 2026-08-15. Independent of `../trust-vector`, but built on its lessons.

## The question

Take a set of behaviours. Each one has two representations inside the same model:

- **activation space** — a steering vector `v_c ∈ R^d`, the mean residual-stream
  difference between the model behaving that way and behaving neutrally.
- **weight space** — a LoRA adapter `ΔW_c`, trained to produce that behaviour
  from the base weights, with no steering vector anywhere in the loop.

They are two read-outs of the same behavioural data. Do they agree about which
behaviours are alike?

The deflationary answer, which has to be ruled out first, is that they trivially
agree because both are dominated by something generic — "how much text changed",
"which weights are big". The interesting answer is that the two spaces carve the
behaviours up differently, and where they differ tells you what a weight edit is
doing that a steering vector is not.

## Why the concept set is built the way it is

Correlating two similarity matrices can only tell you the spaces agree. It can
never tell you whether either one is *right*. So the 28 concepts in
`src/concepts.py` are built on a **planted scaffold with a known answer key**:

| tier | n pairs | what it is | expected |
|---|---|---|---|
| twin | 4 | same behaviour, different wording (`terse`/`terse_b`) | ceiling |
| same-axis, same-pole | 5 | near-synonyms (`bullets`/`numbered`) | high |
| antonym | 7 | same axis, opposite pole (`verbose`/`terse`) | **see below** |
| same-family | 20 | related area, different axis (`french`/`german`) | mid |
| unrelated | 346 | different family (`french`/`bullets`) | floor |

9 families (format, length, register, language, persona, affect, epistemic,
safety, audience), 17 axes, 6 of them antonym axes.

**The antonym tier is the sharpest test in the project.** A `be verbose` LoRA and
a `be terse` LoRA move the same machinery in opposite directions. So:

- a **signed** representation (`ΔW` itself, or a steering vector) should put
  antonyms *below* unrelated — strongly negative.
- a **magnitude** representation (per-neuron movement, all entries ≥ 0) should
  put antonyms *above* unrelated — near the ceiling.

A representation that cannot separate "opposite" from "unrelated" is not
carrying the structure, whatever its matrix correlation says.

The **paraphrase twins** are the ceiling that actually matters. Seed replicates
bound how stable a representation is to *training randomness*; the twins bound
how stable it is to the *data*, which is the relevant bound before asking whether
two different behaviours look alike.

Every concept has a **deterministic scorer** (regex / lexicon / count), no LLM
judge. The scorer is the positive control at two stages: it proves the system
prompt elicited the behaviour (stage 1) and that the adapter learned it (stage
5). A vector or adapter for a behaviour that never happened is noise, and noise
has a geometry too — it would land in the "unrelated" tier and quietly inflate
the recovery scores.

## Pipeline

| stage | script | output |
|---|---|---|
| 1 | `gen_data.py` | per-concept responses on the **shared** prompt pool + manipulation check (gate: Cohen's d ≥ 0.8) |
| 2 | `build_vecs.py` | steering vectors, split-half reliability, steering efficacy vs matched-norm random, integrity |
| 3 | `train_lora.py` | one adapter per (concept, seed) |
| 4 | `wspace.py` | weight-space representations, exact, no `ΔW` ever materialised |
| 5 | `induced.py` | each LoRA's *activation-space* image `u_c`, and its behavioural score |
| 6 | `compare.py` | ceilings, answer-key recovery, predictive mapping |

The prompt pool (`src/prompts.py`, 104 prompts) is **held fixed across all 28
concepts**. If each concept had its own prompts, any similarity structure we
measured could be topic structure wearing a costume. Splits are disjoint: 64
train / 24 held-out scoring / 16 activation-probe.

## The three weight-space representations

`ΔW` for a 7B adapter over 7 module types is ~6.5e9 entries, so 84 of them
cannot be materialised. Everything is computed in closed form from the LoRA
factors, and each identity is **exact**, verified numerically in `mock_test.py`:

```
<ΔW_i, ΔW_j>_F = s_i s_j · Σ_ab (B_iᵀB_j)_ab (A_iA_jᵀ)_ab      O(r²(in+out))
‖ΔW[j,:]‖²     = s² · B[j,:] (AAᵀ) B[j,:]ᵀ
‖ΔW[:,k]‖²     = s² · A[:,k]ᵀ (BᵀB) A[:,k]
```

- `flat_signed` — the signed `ΔW`, as a Frobenius cosine. Direction-sensitive.
- `neuron_mlp` — per-MLP-hidden-neuron total movement, concatenated over layers.
  **This is the representation the project set out to test** ("put all the
  neurons into a vector, each entry being how much that neuron moved").
- `neuron_resid` — per-residual-dimension movement. Same idea, but in the
  coordinate system the steering vectors live in.

Both magnitude profiles are also stored **mean-centred across concepts** (`_c`).

> **Centring is not optional, and we know that before spending any GPU time.**
> In `mock_test.py`'s NULL regime — every `ΔW` independent noise, no structure at
> all — the uncentred per-neuron cosines are **0.96–0.98**. The profile is
> dominated by the common "a LoRA was trained here" mode: big weights move, small
> weights don't, regardless of the concept. Centred, the same null sits at −0.08.
> Any structure visible only in the uncentred profile is the common mode.

**Gauge note.** LoRA is invariant to `B → BR`, `A → R⁻¹A`, so `A` and `B`
individually are meaningless to compare. Everything uses `ΔW = (α/r)BA`, which is
gauge-invariant. Never take a cosine between raw LoRA factors.

## How the two spaces get compared

Not by correlating the two similarity matrices — that is the one comparison
method ruled out here, and it could only ever say they agree. Three other ways,
increasing in strength:

1. **Ceilings and floors first.** Activation: split-half reliability. Weight:
   `cos(ΔW_c^s0, ΔW_c^s1)` — same concept, different training randomness. Data:
   the twins. Floor: the unrelated distribution. *If the weight ceiling sits near
   the floor, LoRA solutions are not a representation of the concept and
   everything downstream is capped at noise.* This is the pilot gate.
2. **Answer-key recovery.** Mean similarity by designed tier, with the antonym
   signed/magnitude dissociation called out explicitly. A test against structure
   we planted, not a matrix correlation.
3. **Predictive mapping, leave-one-concept-out.** Fit a linear map from
   activation coordinates to weight coordinates on N−1 concepts, predict the
   held-out one, rank all N candidates. *Can you find a concept's adapter knowing
   only its steering vector?* Reported against a label-shuffled floor and with
   orthogonal-Procrustes residual alongside. Because the set deliberately
   contains near-duplicates, `top1_grp` (rank-1 shares the target's axis and
   pole) leads, with strict `top1` and MRR beside it.

The tightest single number is not a matrix comparison at all: `cos(v_c, u_c)`,
where `u_c` is the mean residual shift the *adapter* induces, read at the same
layer and position as `v_c` on identical text. Did the weight edit end up writing
the direction the contrast said it should?

## Read position and layer are factors, not settings

`../trust-vector` established that six tokens of read position selects a
near-orthogonal, equally-reliable direction and can flip a headline result. So
every vector is built at two positions (`response` — the behaviour as it happens;
`last` — the final prompt token) across a layer sweep, and any claim that holds
at only one is reported as position-bound. Same for depth: an effect that changes
sign with layer and cancels when summed is not a mechanism.

## Running it

```bash
bash run_pilot.sh      # 4 concepts x 3 seeds, GATE — read before going further
```

Then, only if the weight seed-ceiling clears the floor:

```bash
bash run_full.sh       # 28 concepts x 3 seeds = 84 adapters
```

`src/mock_test.py` runs the whole analysis path on synthetic adapters with a
known planted geometry, no GPU, in ~20 s. It asserts that the SIGNAL regime
recovers the structure and the NULL regime collapses to chance, so a null on the
pod is attributable to the model rather than to the analysis. Run it first.

## Pilot results (2026-08-16, Qwen2.5-7B-Instruct, 4 concepts × 3 seeds)

The gate ran and changed the design. Three findings, in order of how much they
matter.

### 1. `ΔW` cosine is anchored to the random LoRA init, and the naive design measured the basis

LoRA draws `A` at random and starts `B` at zero, so `ΔW = BA` lies in the row
space of `A` — an `r`-dimensional random subspace of a `d`-dimensional input
space. At `r=16`, `d=3584`, two independent draws span near-orthogonal subspaces,
so a cross-init cosine is pinned near zero **by geometry, whatever was learned**.

Measured directly, on `layers.10.mlp.gate_proj`:

| | cos |
|---|---|
| `lora_A` across concepts, same seed | 0.985 – 0.999 (literally the same draw) |
| `lora_A` across seeds | 0.004 |

and the consequence for `ΔW`:

| pair | same init | different init |
|---|---|---|
| same concept | — | **0.096** |
| twin (`terse`/`terse_b`) | **0.505** | 0.019 |
| antonym (`verbose`/`terse`) | 0.001 | 0.000 |
| unrelated | 0.026 | 0.001 |

So the original "3 seeds per concept, compare `ΔW`" design was measuring the
random basis. Read naively it says LoRA solutions are irreproducible (same
concept, 0.096); read correctly it says **the measurement is only defined within
a shared basis**, where the twin pair separates from unrelated by 0.505 vs 0.026.

**Design change:** the init seed and the data-order seed are now separate knobs.
`INIT_SEEDS` defines *blocks* — every concept in a block shares one `A` draw, so
within-block `ΔW` comparisons sit in a common basis. `DATA_SEEDS` varies batch
order within a block and is the honest training-noise replicate. Blocks are the
replication check: a geometry claim has to hold in both. Full run is 28 × 2 × 2 =
112 adapters.

### 2. The per-neuron movement profile is basis-robust where flattened `ΔW` is not

Same pairs, the other representations:

| bucket | flat_signed | neuron_mlp | neuron_mlp_c | neuron_resid_c |
|---|---|---|---|---|
| same concept, diff init | 0.096 | 0.976 | 0.739 | 0.506 |
| twin, SAME init | 0.505 | 0.975 | 0.772 | 0.965 |
| twin, diff init | 0.019 | 0.966 | 0.693 | 0.682 |
| antonym | 0.000 | 0.933 | −0.726 | −0.678 |
| unrelated | 0.026 | 0.931 | −0.233 | −0.119 |

- **Uncentred `neuron_mlp` is inert**: 0.93–0.98 for every bucket including
  unrelated, exactly the 0.96 the mock's *null* regime predicted. Confirmed on
  real adapters — the common mode is the whole signal.
- **Centred, it works, and it survives an init change**: twin 0.69–0.77 either
  way, against unrelated −0.23. This is the one representation that does not need
  the basis controlled. The project's original proposal, with centring, is the
  better weight-space representation.

### 3. Antonyms come out orthogonal in weight space, not opposed

`verbose` vs `terse` sits at 0.000–0.001 in signed `ΔW` at matched init, where the
twin pair reaches 0.505. In activation space the same pair is −0.281 against an
unrelated floor of +0.135. So the steering vectors are somewhat anti-aligned
while the weight edits are merely orthogonal — an activation/weight disagreement
of exactly the kind the project is for.

**Held very loosely.** One antonym pair, and it rests on `verbose`, the weakest
concept in the pilot: its steering gain was +22.1 against a random-direction
+11.8 (every other concept beat its random arm by 4–30×), and 88% of its
generations still hit the 512-token cap, so its scorer is partly measuring the
ceiling. Nothing here should be repeated without the full antonym set.

### Also worth recording

- `verbose` was **rejected at d=0.70** on the first attempt, at `MAX_NEW=220`:
  neutral answers were already 157 words, so both arms were saturating the
  generation cap. Raised to 512 and added a `trunc_rate` column. A length cap
  silently compresses every length-sensitive contrast toward zero *and* truncates
  the neutral arm that every other concept's diff-in-means references.
- Steering efficacy at α=1, L14, integrity 1.00 throughout: `terse` +57.7 (random
  −15.3), `terse_b` +48.6 (−20.3), `french` +0.576 (+0.018), `verbose` +22.1
  (+11.8). LoRA behavioural gains were large and consistent across seeds for all
  four.
- `cos(v_c, u_c)`, the steering-vector-to-LoRA-induced-shift bridge: `french`
  +0.41 (tight across seeds), `verbose` +0.15, `terse_b` +0.11, `terse` +0.05.
  Needs the off-diagonal floor before it means anything — not yet computed.
- **Leave-one-concept-out retrieval is uninterpretable at N=4**: it returned
  1.000, but so did the label-shuffled control. That test needs the full 28.
- **Centring is distorted at small N.** Mean-centring N vectors forces average
  pairwise cosine to ≈ −1/(N−1), which is −0.33 at N=4 and explains most of the
  "unrelated" column above. At N=28 it is −0.037 and negligible. The centred
  numbers here are ordinally readable, not quantitatively.

## CORRECTIONS (2026-08-17, after review)

Three challenges to the results above; two found real errors and one found an
overstatement. **Read this section before sections 1-6.**

### C1. `verbose` and 9 other concepts never passed the steering validity check

`build_vecs.py`'s own docstring states the precondition: "if the vector arm does
not produce the behaviour, the two spaces are not representing the same thing and
no geometry comparison between them means anything." That check was computed and
then **not enforced**. Against a matched-norm random direction, **10 of 24**
steering vectors failed:

`verbose`, `pessimistic`, `numbered`, `pirate`, `hedging`, `bullets`,
`sycophantic`, `refusing`, `sycophantic_b`, `caveating`

`verbose` is the worst kind of failure for this project: the random direction
scored *higher* than the real one (+21.6 vs +19.9), and `verbose` appears in **two
of the three antonym pairs**. Restricted to validated concepts the answer key
collapses to **2 twin pairs and 1 antonym pair** (`technical/childlike`).

Section 6 ("weight editing achieved behaviours steering could not") is
**reframed**: that is not a finding, it is the validity check failing. Those
concepts should have been excluded from the geometry, not reported as a result.

### C2. The antonym result was a patch-SIZE artifact — retracted

The dominant axis of the centred neuron profile correlates **+0.998 with the
overall size of the weight patch**, which ranges 1.7x across adapters. Centring
non-negative profiles without normalising first means "big patch" vs "small patch"
dominates the cosine. Normalising each profile to unit length before centring:

| | as reported | size-corrected |
|---|---|---|
| twin | +0.784 | +0.775 |
| **antonym** | **−0.487** | **+0.142** |
| unrelated | +0.145 | +0.246 |

**"Opposite behaviours move largely disjoint neurons" is withdrawn.** Corrected,
antonyms sit slightly *below* unrelated (+0.142 vs +0.246) — still not *above* it
as predicted, but a small difference, not a dramatic one. And with one validated
antonym pair (see C1), there is essentially no evidence here either way. The
activation-space half of section 3 stands; the weight-space half does not.

### C3. "Pinned near zero by geometry" was overstated

| | value |
|---|---|
| same concept, same init | +0.7880 |
| same concept, **different init** | **+0.1409** |
| different concept, different init | +0.0040 |

Cross-init similarity is **35x its floor** (t=20.6, p=8e-37). So a shared init
**attenuates** the signal ~5.6x rather than destroying it — real concept
information does survive a change of basis. The design change is still right,
because 5.6x more signal is worth having, but the justification is "much better
measurement", not "otherwise meaningless".

### What got STRONGER

Applying both fixes together — size-corrected profiles, validated concepts only —
the central result improves a lot. Predicting a held-out concept's weight-space
point from its steering vector:

| | top-1 | chance | perm null |
|---|---|---|---|
| as reported (24 concepts, uncorrected) | 0.292 | 0.042 | 0.018 |
| **size-corrected, 14 validated concepts** | **0.643** | 0.071 | 0.009 |

**9x chance, identical in both init blocks.** `cos(v,u)` separation also improves
(+0.093 -> +0.132). Removing a nuisance dimension and ten unvalidated directions
sharpened the correspondence rather than removing it: **the core claim that the
two geometries correspond is better supported than originally reported, and the
sharpest single claim (antonyms) is withdrawn.**

### C4. The recommended fix (input-statistics weighting) FAILED — and so did the claim that motivated it

Two corrections and one clean negative.

**The claim that motivated it was over-read.** "Patch size is functionally
meaningless, rank correlation -0.13" was measured WITHIN concept, across the 4
replicate adapters. But behavioural gain barely varies within a concept — median
coefficient of variation **3.9%**. The outcome was essentially constant, so that
correlation was computed against noise and had no power either way.

**Tested properly (across concepts, scale-free outcome = manipulation d), there IS
a relationship, and it is INVERSE:**

| representation | rho vs behavioural d | p |
|---|---|---|
| raw `\|\|dW\|\|` | −0.393 | 0.057 |
| raw resid row-profile | **−0.454** | **0.026** |
| SIGMA-weighted resid | −0.317 | 0.132 |
| SIGMA-weighted mlp | −0.400 | 0.053 |

Bigger weight edits go with *weaker* behavioural effects. Edit size looks like a
measure of how hard the concept was to install, not how much it changed.

**And SIGMA-weighting does not merely fail to help — it destroys the signal.**
Held-out concept retrieval, 14 validated concepts, chance 0.071, with the
unweighted row profile over the identical module set as the control:

| profile | block 0 | block 1 |
|---|---|---|
| raw resid rows | 0.143 | 0.071 |
| **SIGMA-weighted resid rows** | **0.000** | **0.000** |
| raw mlp rows | 0.071 | 0.071 |
| **SIGMA-weighted mlp rows** | **0.000** | **0.000** |

Zero in every cell, both blocks, both module groups. The likely mechanism is the
reverse of what I argued: SIGMA is dominated by a few high-variance input
directions that are shared across *all* concepts, so weighting by it amplifies
the common activation manifold and suppresses the low-variance directions where
concept identity actually lives. **The functionally dominant directions are not
the concept-discriminative ones.**

**A more useful incidental result.** These row-only profiles retrieve at
0.07–0.14, against **0.643** for the `wspace` profile over the same concepts. The
difference is that `wspace` also includes the *column* (input-side) movement.
So in weight space a concept's identity sits mainly in **which input directions
the edit reads from**, not which neurons it writes to — worth knowing for anyone
building the "put every neuron in a vector" representation.

## Results (full run, 24 concepts x 96 adapters, Qwen2.5-7B, L11)

All 96 adapters cleared the behavioural gate: every one moved its own scorer in
the right direction, with tight agreement across the 4 replicates. Numbers below
are at the `response` read, layer 11, and every headline was checked at the
`last` read and in both init blocks separately.

### 1. LoRA weight edits are anchored to the random init, and this is a measurement problem, not a fact about learning

The single most useful result, because it invalidates the obvious version of this
experiment. LoRA draws `A` at random and starts `B` at zero, so `dW = BA` lies in
the row space of `A` — a rank-16 subspace of a 3584-dim input space. Two
independent draws span near-orthogonal subspaces, so `cos(dW_i, dW_j)` across
different inits is pinned near zero **by geometry, whatever was learned**:

| | same LoRA init | different init |
|---|---|---|
| `cos(lora_A, lora_A)` | 0.985 – 0.999 | 0.004 |
| same concept, `cos(dW)` | **0.788** | **0.141** |
| twin pair, `cos(dW)` | 0.218 | 0.017 |
| unrelated, `cos(dW)` | 0.051 | 0.004 |

Read naively this says "LoRA solutions are irreproducible". That reading is
wrong: it says the *measurement* is only defined within a shared basis. The
design was changed so that **all concepts in an init block share one `A` draw**,
with data order as the within-block replicate and the block as the replication
check. Every weight-space number below is within-block, and every claim was
required to hold in both blocks.

### 2. Which weight-space representation works — centred, and over residual dims

The "put every neuron in a vector" representation works, with two required
choices. **Uncentred it is useless**: every tier sits at 0.93–0.99, because a
non-negative movement profile is dominated by the common "a LoRA was trained
here" mode (big weights move, small ones do not, whatever the concept). And the
residual-dimension profile beats the MLP-hidden-neuron one, which makes sense —
the steering vector *lives in* the residual basis, so the two are in a shared
coordinate system.

Leave-one-concept-out: predict a **held-out** concept's weight-space point from
its steering vector alone, rank all 24 candidates. Chance top-1 = 0.042;
permutation null from 20 label shuffles.

| activation -> weight | block 0 | block 1 | MRR | perm null | p |
|---|---|---|---|---|---|
| steer_vec -> `neuron_resid_c` | **0.292** | **0.292** | 0.52 / 0.55 | 0.019 | <0.001 |
| steer_vec -> `flat_signed` | 0.167 | 0.125 | 0.30 / 0.27 | 0.006 | <0.001 |
| steer_vec -> `neuron_mlp_c` | 0.083 | 0.042 | 0.25 / 0.24 | 0.015 | 0.05 / 0.20 |

7x chance, identical in both init blocks, and it survives the `last` read at
top-1 0.167 in both blocks. So the two geometries **do** agree above chance — but
7x chance on a 24-way choice is a modest agreement, not a correspondence.

### 3. The headline dissociation: antonyms are opposed in activation space and *disjoint* in weight space

This was the pre-registered sharpest test, and the prediction was half wrong in
an informative way. Prediction: a signed representation should put antonyms
*below* unrelated, a magnitude representation *above* it (same machinery, moved
either way).

| pair | steer_vec | `flat_signed` | `neuron_resid_c` |
|---|---|---|---|
| verbose / terse | −0.305 | 0.005 | −0.575 |
| verbose / terse_b | −0.284 | 0.010 | −0.583 |
| technical / childlike | −0.338 | 0.025 | −0.303 |
| **unrelated baseline** | **+0.251** | **+0.051** | **+0.145** |

- **Activation space behaves as predicted**: opposite behaviours get opposite
  directions, far below the unrelated baseline.
- **Signed weight edits are orthogonal, not opposed** (≈ +0.01 vs +0.05
  unrelated). Antonyms do not get the same edit with a flipped sign.
- **Magnitude profiles go the wrong way, strongly**: antonyms land *far below*
  unrelated, not above. Opposite behaviours move **largely disjoint sets of
  neurons**.

The centring artifact at N=24 is only −0.043, so it does not explain −0.3 to
−0.58. All three pairs agree, including `technical/childlike`, which shares no
concept with the truncation-limited `verbose`.

Taken together: the model does not implement "verbose vs terse" as one axis
turned up or down in the weights. It implements them as two separate sets of
weight changes that happen to *read out* as one axis in the residual stream.

### 4. Activation space recovers the planted answer key; weight space recovers it partially

Tier means, `steer_vec`: twin 0.960 > same-pole 0.862 > same-family 0.521 >
unrelated 0.251 > antonym −0.309 — **monotone in designed similarity**.
`neuron_resid_c`: same-pole 0.897, twin 0.812, same-family 0.431, unrelated
0.150, antonym −0.497 — the same ordering except antonyms, and twin/same-pole
swapped.

### 5. The tightest bridge is weak, and it fails worst where behaviour changes most

`cos(v_c, u_c)`: does the weight edit write the direction the contrast said it
should? Against the honest floor — the same `u_c` against every *other* concept's
vector:

| read | matched | mismatched (floor) | separation |
|---|---|---|---|
| response | +0.301 ± 0.143 | +0.208 ± 0.131 | **+0.093** (80% of adapters above floor) |
| last | +0.471 ± 0.153 | +0.390 ± 0.153 | **+0.081** (76%) |

Most of the raw 0.30 is common mode. And the failure is not uniform: the three
**largest-effect concepts are the worst-aligned** — `verbose` +0.008, `terse_b`
+0.004, `terse` +0.030, all *below* the 0.208 floor, while `casual` (+0.560) and
`enthusiastic` (+0.459) align well. The adapters that change behaviour most
dramatically (terse gain +177 words) write an activation shift essentially
orthogonal to the steering vector for that concept.

The natural reading, consistent with `../trust-vector`: a diff-in-means direction
captures the *signature* of a behaviour in the residual stream, not the mechanism
that produces it. For length, the signature ("this text is long") and the
controller (whatever sets the stopping decision) are different objects.

### 6. Weight editing achieves behaviours steering could not

Concepts where injecting the steering vector at alpha=1 did nothing, but the LoRA
learned cleanly: `sycophantic` (steering 0.000 -> LoRA +0.67), `refusing` (0.000
-> +14.33), `bullets` (−0.02 -> +1.00), `numbered` (−0.19 -> +0.48). A direction
being decodable did not make it steerable, and not being steerable did not stop
it being learnable as a weight edit.

### Limits

- 24 concepts, one model, rank-16 LoRA, headline at one layer (L11).
- The antonym tier is **3 pairs over 2 independent axes** after 4 concepts failed
  the manipulation check. This is the thinnest evidence base of any claim here,
  which is why the increment re-runs three of those concepts.
- `verbose` is still 21% truncated at 700 tokens (neutral 0%), so its scorer
  partly measures the cap; its `d` and its behavioural gain are floors. The
  antonym result does not depend on it (see `technical/childlike`).
- **`lora_induced` vs weight-space comparisons are near-circular** — `u_c` and
  `dW_c` are two views of the same trained adapter, so their agreement (top-1
  0.500) is not evidence about activation/weight correspondence. Only
  `steer_vec -> W` and `cos(v, u)` are honest bridges, because the steering
  vector never enters LoRA training.
- Concept similarity is bounded by each vector's split-half reliability
  (0.92–0.99 here, so not the binding constraint).

### Increment outcome: the antonym axes could not be repaired (2026-08-17)

Strengthening the three manipulations did **not** rescue them, and two got
*worse*:

| concept | d, original prompt | d, strengthened + lexically clean prompt |
|---|---|---|
| formal | 0.53 | 0.75 |
| optimistic | 0.70 | **0.46** |
| overconfident | 0.69 | **0.19** |

The guard in `run_increment.sh` stopped the run before training any adapter.

The reason they got worse is the point: the strengthened prompts also had the
scorer's own lexicon removed. The original `optimistic` prompt told the model to
say "well", "promising" and "upside" -- three of the fifteen words its scorer
counts. Once that leak is closed, the honest effect is 0.46, not 0.70. Same for
`overconfident`: 0.69 -> 0.19.

**This retroactively weakens three concepts that are in the main run.**
`pessimistic` (leaks "risk", "fail"), `hedging` ("might", "could be") and
`caveating` ("consult", "professional", "disclaimer", "warning") were all judged
on prompts that name their own scorer's words, so their `d` values (1.76, 0.97,
3.33) are inflated by an unknown amount. `hedging` at 0.97 is close enough to the
0.8 threshold that it might not survive a clean prompt. Their adapters did train
and did move their own scorers, so the *behaviours* are real -- but the **labels**
should be read narrowly: "hedging" may be closer to "emits hedge words" than to
hedging as a disposition. Nothing in sections 1-3 depends on these three
concepts; they sit in the same-family and unrelated tiers.

**Final state of the sharpest test: 3 antonym pairs over 2 independent axes.**
Four axes are gone and cannot be recovered by prompt engineering within this
design.

The principled fix is to replace lexicon-counting scorers with a model-graded
judge, which removes both failure modes at once -- the leak (a judge does not
count keywords) and the headroom problem (a judge can score "is this direct?"
where `-caveats` has a floor at zero). That is a real design change affecting
every concept and every stage, so it is written down as the recommended next step
rather than applied on top of a completed run.

## Concept set after the manipulation check (full run, 2026-08-16)

24 of 28 concepts cleared `d >= 0.8`. The four rejects were `formal` (0.53),
`optimistic` (0.70), `overconfident` (0.69), `direct` (−0.10) — and the damage
lands almost entirely on the **antonym tier, the project's sharpest test**, which
drops 7 pairs -> 3:

| | surviving | lost |
|---|---|---|
| twin | 4/4 | — |
| same-pole | 5/5 | — |
| **antonym** | **3/7** | formal/casual, optimistic/pessimistic, hedging/overconfident, caveating/direct |
| same-family | 16/20 | |
| unrelated | 252/346 | |

Worse than 3/7 looks: two of the three survivors are `verbose`/`terse` and
`verbose`/`terse_b`, which share `verbose`. That is **2 independent antonym
axes**, one of them the truncation-limited one.

All four failures have the same shape — **the negative pole of an axis whose
neutral baseline already sits at that pole**:

- `direct` = `−caveats`. The neutral arm emits 0.05 caveats/100 words on this
  innocuous prompt pool, so there is nothing to go below. A pure **floor effect**,
  and it is not fixable by prompting: it would need risk-adjacent prompts, and the
  prompt pool is shared across all concepts by construction. **`direct` is dropped
  permanently**, and its axis with it.
- `formal` — the neutral assistant is already formal (ceiling).
- `optimistic`, `overconfident` — real but small mean shifts with high variance.

### What was changed, and what deliberately was not

Scorer engineering can push `formal` and `optimistic` to d≈0.77 — still short of
0.8, and **choosing the best of four scorer variants post-hoc would be fitting the
gate rather than passing it**. The scorers are therefore left exactly as
pre-specified. Only the *manipulation* was strengthened, which is what a
manipulation check is for: `formal`, `optimistic` and `overconfident` get more
extreme system prompts and are re-judged on the unchanged scorer.

The new prompts are also **lexically clean** — no scorer-lexicon words. The old
ones were not, and this is a pre-existing weakness worth recording: `optimistic`'s
original prompt contained three `_POS` words ("well", "promising", "upside") and
still only reached 0.70. `pessimistic` ("risk", "fail"), `hedging` ("might",
"could be") and `caveating` ("consult", "professional", "disclaimer", "warning")
still leak into their own scorers and **passed the gate partly for that reason**.
They are left alone because they are already generated and trained in the main
run, and changing them would break comparability — but their `d` values are
inflated relative to the clean ones and should not be compared with them.

`gen_data` and `build_vecs` now **merge** rather than overwrite, and the `NEUTRAL`
arm is reused rather than regenerated. That last part is correctness, not
convenience: every vector is a difference against `NEUTRAL`, so regenerating it
for an incremental run would silently put those concepts on a different reference
from the main run.

## Status

Scaffold complete, locally verified (analytic identities exact to float32; mock
passes in both regimes), pilot run done, **full run launched 2026-08-16** (28
concepts x 2 init blocks x 2 data seeds = 112 adapters, `MAX_NEW=700`,
`MAXLEN=768` so the length concepts stop hitting the cap).

Since the pilot: reads do one forward for all layers rather than one per layer,
adapter scoring is batched, the init/data seeds are separate knobs, `compare.py`
prints the init-block breakdown and the bridge's mismatched floor as standard
output, and tier/mapping tables are computed within an init block.

A 4-adapter smoke test on the rebuilt design reproduced finding 1 on freshly
trained adapters: twin pair 0.560 at shared init vs 0.027 across inits, same
concept across inits 0.045.

## Related work worth reading before writing anything up

Persona vectors (Chen et al. 2025) already show that finetuning-induced persona
shifts project onto steering directions for the same trait. That is the
per-concept version of the bridge measure here. Task arithmetic (Ilharco et al.)
establishes that weight-space task vectors for unrelated tasks are near-orthogonal
and compose. The contribution attempted here is the *geometry* comparison with
reliability ceilings on both sides and a planted answer key — not the existence
of a link. Scope the claim accordingly.
