# DAS → Pile checklist (2026-07-23)

## ✅ POSITION-WRITING HEADS FOUND: L14H19 / L16H3 / L20H3 (2026-07-30, Sandra's design)

**⚠️ RETRACTS "no head-selection method finds a coordinate circuit."** That was true of every method I
tried, and all of them shared two errors Sandra identified: (a) DAS on the RESIDUAL stream, whose
coordinate subspace is ~56% (grid) / 73% (ring) MLP+embedding-written — attention cannot reconstruct what
it does not build; (b) working at L24, where attention carries almost none of the variable.

**The fix: DAS restricted to ATTENTION OUTPUTS (MODE=concat over all heads at one layer), swept over
EVERY layer, then greedy over that layer's 32 heads.** Single-layer makes the decomposition EXACT (heads
run in parallel from the same input, so sum-over-heads == target, verified: recon err ~4e-7) and offline
attribution equals causal intervention — versus the residual version's 0.972 -> 0.235 collapse.

**Attention-only DAS localises the writing sites, and grid and ring AGREE:**
| layer | grid flip | grid top head | ring flip | ring top head |
|---|---|---|---|---|
| 14 | **0.904** | **L14H19** | **0.875** | **L14H19** |
| 16 | **0.954** | **L16H3** | **0.863** | **L16H3** |
| 20 | **0.817** | **L20H3** | **0.779** | **L20H3** |
| 24 | 0.233 | — | 0.554 | — |
Same three layers, SAME dominant head at each, across a 2-D lattice (rot180) and a 1-D cycle (cyc1) —
different variables on different graphs, so not an artefact of the counterfactual. cos^2 excess over
random-k=1: grid +0.50/+0.44/+0.65, ring +0.64/+0.46/+0.77.

**Motifs (all-head means: same_tok 0.065, ind 0.030, prev 0.032):**
| head | same_token | rank | ind rank | prev rank |
|---|---|---|---|---|
| L14H19 | **0.971** | **1/1024** | 738 | 1003 |
| L10H1 | 0.684 | 7 | 190 | 722 |
| L20H3 | 0.522 | 12 | 278 | 756 |
| L0H2 | 0.036 | 513 | 937 | **2 (prev)** |
| **L16H3** | 0.060 | **343** | **184** | **830** |
=> 3 of 5 writers are top-12 DUPLICATE-TOKEN heads (aggregating previous visits to the current node is
how you build "where am I" from context — and why induction scoring misses them, ranks 190-738).
L0H2 is the #2 prev-token head. **But L16H3, the STRONGEST site on the grid, fits NO standard motif.**

**⚠️ Traps this run exposed, both flagged in advance and both fired:**
1. **cos^2 is high (0.60-0.94) at EVERY layer, including where DAS failed** (L28 0.598 at flip 0.042).
   Greedy always reconstructs whatever subspace the optimiser converged to. Only interpretable where
   flip is high, and only against rand@1.
2. **"Localised to L14" was wrong** — from a 3-layer smoke test (L13/L14/L24). L16 is HIGHER. The
   variable is written at >=3 sites; sparse WITHIN a layer, distributed ACROSS layers, which is exactly
   why every cross-layer method failed.

**⚠️ ALSO RETRACTED: the behavioural coordinate greedy (75.2% grid / 181% ring).** Selecting on a logit
margin is gameable: the ring keep-set reached 1.75x the INTACT model's margin. Diagnostic confirmed
collapse — entropy 0.809 (full) -> 0.124 (k=20), top-1 mass 0.62 -> 0.95, KL 0.141 -> 3.889, i.e. WORSE
than ablating everything (3.291). A margin larger than the full model is never recovery. **Check the
full-model baseline before reporting any recovery percentage.**

## ⚠️ RETRACTIONS: MOTIF MECHANISM DEAD, "GEOMETRY INERT" DEAD (2026-07-30, Sandra's designs)

**(1) "Induction heads compose with duplicate-token heads" — RETRACTED.** Sandra's control: split the
SAME 21 heads into a RANDOM 12/9 and re-run keep-set sufficiency.
| 12/9 partition | 12-half | 9-half |
|---|---|---|
| **motif** (induction12/duplicate9) | +0.031 | -0.022 |
| random 0 | -0.006 | -0.009 |
| random 1 | -0.052 | **+0.255** |
| random 2 | -0.049 | +0.036 |
| random 3 | -0.018 | -0.006 |
| all 21 | **+0.476** | |
EVERY partition leaves both halves at ~0; the motif split is not privileged and is among the WORSE ones
(a random 9-half beat both motif halves). **The superadditivity is a property of the SET, not of motif
complementarity.** Membership still matters (substituting 9 outside heads -> 2.8%), but the
writers-plus-readers story is gone.

**(2) The payload hypothesis — REFUTED.** Sandra's conflicting-transitions 2x2 (x->a vs x->b, frequency
and recency crossed orthogonally, corr = 0.0000 by construction):
| ablation | recency eff | d | frequency eff | d |
|---|---|---|---|---|
| baseline | 2.879 | — | 5.838 | — |
| induction12 | 2.151 | **-0.728** (~12 SD) | 7.768 | +1.930 |
| duplicate9 | 3.141 | +0.262 | 8.739 | **+2.901** (~27 SD) |
Induction heads DO carry recency. But duplicate heads do NOT carry frequency — ablating EITHER set makes
the model rely on frequency MORE. Reading: these heads do precise in-context retrieval and their removal
exposes a cruder base-rate fallback. Not "aggregation vs copying".

**(3) "The geometry is causally inert" — RETRACTED.** Sandra recalled the earlier steering result;
`steer_probe_square_grid.json` (L25, probe R2 0.454), dE = change in decoded output coordinate:
steer along ROW -> [+1.85, +0.15]; along COL -> [-0.01, +1.72]; RANDOM dir -> [+0.01, +0.16].
**The coordinate directions demonstrably drive behaviour.** Correct statement: geometry is READ but NOT
NECESSARY (projecting it out costs <=3%). Steering tests SUFFICIENCY, projection tests NECESSITY — I
conflated them all session. "Detectability != causal use" should be **"detectability != necessity"**.
Untested mechanism for the redundancy: later layers may REBUILD the plane after projection (script
`rebuild.py` deployed, not yet run).

**(4) Context sweep: the interaction is NOT metric nonlinearity, but does NOT emerge at a stage.**
⚠️ First two sweeps were VOID — window was 150 wide, so the "ctx=3" row averaged contexts 3..153 and read
acc 0.89, impossible from 3 tokens (chance ~0.156). **Tell: an accuracy far above chance at a context
that cannot support it.** Fixed to WINDOW=8:
| ctx | 4 | 8 | 32 | 70 | 100 | 250 | 400 |
|---|---|---|---|---|---|---|---|
| acc | 0.509 | 0.482 | 0.839 | 0.920 | 0.982 | 1.000 | 1.000 |
| inter_acc | 1.143 | 0.788 | 0.864 | 0.903 | 1.102 | 0.611 | 0.848 |
| inter_par | 2.624 | 2.789 | 2.231 | 1.623 | 0.991 | 0.607 | 0.397 |
Interaction is present at acc 0.48-0.51 => NOT a ceiling artefact. But it is FLAT across the whole
transition, and inter_par is LARGEST BEFORE the model can do the task (2.8 -> 0.40). Attention motifs
(0.053-0.074) and Dirichlet energy (3.16 -> 2.81) drift with no inflection. Nothing switches on.

**(5) R^2 caveat.** Negative out-of-sample R^2 is normal, but leave-one-NODE-out with 16 nodes biases it
negative: permutation null on real data gives **mean -0.166, sd 0.074** (not 0), matching the synthetic
-0.19/-0.21. So the floor is ~-0.17. I over-read post-projection values (-0.78 to -4.86) as a graded
"degree of removal"; once signal is gone the magnitude only reflects refit instability. Valid claim is
above-null -> below-null. **The projected-data matched null is still running** — the earlier null was
computed on UNPROJECTED activations, which is not a matched comparison.

## SUPERADDITIVITY IS SPECIFIC + GEOMETRY/CAUSALITY ARE INVERSELY RELATED (2026-07-30)

**(1) The 47.6% is NOT a keep-set size threshold.** Sandra's question: if induction and duplicate-token
heads "serve the same function" (IOI), why superadditive? First rule out the artifact:
| keep-set (1012 heads mean-ablated) | parity recovery |
|---|---|
| induction12 alone | 3.1% |
| induction12 + 9 RANDOM (3 draws) | 3.1% / 6.5% / -1.2% -> **mean 2.8%** |
| induction12 + the 9 duplicate-token heads | **47.6%** |
Padding with arbitrary heads does nothing; those specific nine give a 17x jump. Superadditivity is real.

**⚠️ TERMINOLOGY CORRECTION: "same-token aggregators" = DUPLICATE TOKEN HEADS**, a standard named class
(Wang et al. IOI: "given 'A B A', the duplicate token head attends from the second A to the first").
Measured on-task: neither9 same_token mass **0.580** vs 0.061 for all other heads (perm p=0.0000), with
induction mass 0.062 AT the all-heads baseline of 0.082. Five of the top-5 same_token heads are neither9.
=> They are genuinely NOT induction heads. But the 2026-07-26 "contribution" claim (that task-independent
scoring misses this motif) OVERSTATED it — the class is well known; only its role here is new.
Caveat: the label fits 6 of 9. L2H26 is the opposite profile (same_token 0.001, induction 0.271);
L1H21 (0.142) and L7H25 (0.193) are weak on everything.

**⚠️ TENSION WITH IOI, UNRESOLVED.** IOI frames induction heads as "serving the same function as
duplicate token heads, mediated via previous-token heads" — parallel routes, which predicts REDUNDANCY
(sub-additive). We measure strict complementarity. Hypothesis (untested): on a graph walk they carry
different payloads — an induction head attends to SUCCESSORS of previous visits (= u's neighbours, i.e.
candidate answers), a duplicate-token head attends to the visits THEMSELVES (= visit context). Same
trigger, different content. Directly testable via what each writes at the answer position.

**(2) GEOMETRY PEAKS AT L10-14; THE ANSWER IS NOT COMPUTED UNTIL L18-28.**
| layer | coord_r | rsa_pc2 | logit-lens nbr acc |
|---|---|---|---|
| 10 | **0.811** | 0.594 | 0.268 |
| 14 | 0.788 | **0.737** | 0.228 |
| 18 | 0.481 | 0.191 | 0.474 |
| 24 | 0.488 | 0.187 | 0.974 |
| 28 | 0.453 | 0.190 | **1.000** |
An 8-14 layer gap between peak structure and completed computation. Also: rsa_pc2 > rsa_full early
(0.737 vs 0.669 at L14) but collapses after L16 — the graph structure lives in the top-2 PCs early and
moves out of them later. **This explains the "bizarrely low" RSA baselines: full-dimensional Euclidean
RSA is diluted by ~14 non-coordinate dimensions. coord_r is the better metric.**

**(3) RANK-2 FAILS AT EVERY LAYER — the ring conclusion is NOT an L24 artefact.** I feared it was; the
opposite is true:
| layer | coord_r | r2 flip | r16 flip |
|---|---|---|---|
| 10 | **0.811** | **0.18** | 0.68 |
| 12 | 0.543 | 0.23 | 0.87 |
| 14 | 0.788 | 0.32 | 0.94 |
| 24 | 0.488 | 0.31 | **1.00** |
Baseline flip 0.16. **Geometry strength and causal patchability are INVERSELY related**: the layer with
the strongest circular structure (L10) patches worst. Rank-16 efficacy instead tracks logit-lens
readability (0.68->1.00 as acc goes 0.27->0.97). **DAS patchability follows the COMPUTATION, not the
REPRESENTATION.** L24 was picked for the wrong reason (upstream of writer L21H10) but is near-optimal for
the right one.
Transfer at L14 r16 replicates the L24 finding: swappairs +4.04 (flip 0.71) vs cyc2 -2.23 (0.22) vs
randbij -4.82 (0.09) — transfer follows DISPLACEMENT, not rotational structure.

**⚠️ METHOD BUG FOUND (affected the first geometry sweep).** A head at layer > LAYER has no causal path
to hidden_states[LAYER+1]. The original sweep ablated all 21 regardless and drew random controls from all
1024 heads, so (a) the effective set shrank as LAYER dropped and (b) controls were systematically weaker
at low LAYER. At L14, SIX of 21 were inert — including induction ranks 1, 3, 16. Fixed by restricting
both arms and the controls to the upstream pool. **The tell was in the output: nbr_valid repeated
IDENTICALLY down every layer row (it comes from logits) while rsa moved.**
After the fix, the "double dissociation" (induction carries geometry, duplicates carry task) holds ONLY
at L14 — at L10/L18/L22 ablating the duplicate-token heads LOWERS geometry. At L10, where geometry is
strongest, they carry BOTH geometry and task while induction/prev-token heads carry NEITHER
(task 0.9992, rsa 0.487 vs random 0.528) — contrary to Arditi's prev-token mechanism at that depth.

**Olsson procedure is Olsson-STYLE, not exact:** correct prefix-match offset (q-SEQLEN+1), but SEQLEN=60,
2 copies, tokens sampled 1000..30000, 12 sequences, attention-weight variant (not logit-attribution).
Prev-token = mean attention q -> q-1 over the same window. Robustness of the induction12/neither9 split
to these choices is UNTESTED.

## ⚠️⚠️ THE "PARITY-21 CIRCUIT" IS AN IN-CONTEXT-LEARNING CIRCUIT (2026-07-30)

Sandra asked two questions that broke the project's central claim open: (1) do the heads that hurt the
ring task hurt the Engels task? (2) what is the parity-21 circuit actually ablating?

**The name recorded how the heads were FOUND (interchange patching against a parity counterfactual on the
grid), not what they DO.** Probe with NO graph, NO parity, NO semantics — repeated random tokens
[BOS] t1..tL t1..tL, score next-token accuracy inside the second copy (pure prefix matching):

| head set | **induction** | ring | ring_lazy | edays | emonths | neutral |
|---|---|---|---|---|---|---|
| parity-21 (causal) | **0.987 -> 0.049 (-0.938)** | -0.787 | -0.562 | -0.200* | -0.046 | 0 |
| Olsson top-20 induction (score) | -0.356 | -0.123 | -0.026 | -0.004 | +0.008 | 0 |
| top-20 high-frequency | -0.014 | 0.000 | +0.005 | -0.024 | +0.022 | 0 |
| random 20-21 | ~0 (sd 0.002) | ~0 | ~0 | ~0 | ~0 | 0 |
\* confounded: the SAME ablation costs the neutral city/river task 2.5 nats of margin.

**The pure-induction probe is damaged MORE than the ring task it was supposedly about.** Damage ordering
is identical across all three ICL probes, and ring damage tracks induction damage set-for-set. The toy
graph tasks collapse because they need to bind arbitrary words to nodes FROM CONTEXT — that capability is
what gets removed. Parity was the readout used for discovery, never the variable removed. Confirmed by
ring_lazy: parity is nulled there and the damage persists (-0.562).

**Redundant, not localised.** LOO over the 21: best single head (L4H16) restores induction to 0.148
against a 0.049 floor and 0.987 baseline — ~10% of the deficit from removing 1 of 21.

=> **RE-READ the headline result.** "21-head circuit recovers 47.6% parity margin / 86% validity vs 0.3%
random" means: these heads restore IN-CONTEXT LEARNING, which restores everything about the toy task,
parity included. The rank-1 DAS parity DIRECTION (a claim about the representation) is unaffected; the
CIRCUIT is generic ICL machinery. Retire the name "parity circuit".

**Also explains the long-standing negatives**: inert on the Pile and on Engels day/month because those
run on pretrained knowledge, not in-context binding. Not a failure of transfer — a category error about
what the circuit was.

**Two lessons, both now seen 3+ times in this project:**
1. **Causal selection >> score selection.** The causally-derived set destroys induction (-0.938) 2.6x
   more than the top-20 heads ranked BY INDUCTION SCORE (-0.356), and overlaps them at only 4/21. Same
   shape as the RSA-derived coordinate circuit at -1.0%.
2. **Writing a mode != using it.** The top-20 high-frequency heads carry 69-90% of their write energy in
   the top modes and are causally inert everywhere (-0.014 induction, 0.000 ring). Cf. the ring geometry:
   PCA shows a convincing circle, rank-2 patching fails.

**Premise audit for the original question** ("mode ablation hurts ring, heads write modes, so why don't
heads hurt ring?"): premise 3 was FALSE (heads devastate the toy ring, -0.787); premise 1 was
CONTEXT-DEPENDENT (mode ablation hurt at CTXLO=100, but at CTXLO=800 the model is at 1.000 and the worst
single-mode projection costs 0.015); and the two "ring tasks" were different tasks (toy in-context walk
vs Engels calendar arithmetic).

## TRANSFER: PARITY CIRCUIT DOES NOTHING ON REAL PARITY TASKS (2026-07-29)

Sandra's idea: ablate the toy-task circuits and run tasks that genuinely need the same variable.
Better targeted than the Pile test (prose has no reason to invoke parity; "is 47 odd or even" does).
Mean-ablate the 21-head parity circuit vs 5 RANDOM same-size head sets. `transfer_tasks.py`.

**⚠️⚠️ TWO HARNESS BUGS FOUND — Sandra pushed back ("this makes no sense may I see the prompt"), and
BOTH the headline effects I had reported were artefacts. Show the prompt when a result is surprising.**

**BUG 1 — first-token scoring.** Llama-3 tokenises `" 1"` as `[220, "1"]`: the leading space is its own
token. So ALL TWELVE clock candidates shared first token 220, argmax always returned index 0, and clock
accuracy was pinned at **exactly 1/12 = 0.083 in every condition and every run**. I reported this as
"Llama cannot do clock arithmetic." After scoring the FULL candidate token sequence: **clock = 1.000**.
*Tell: a metric identical across all conditions is a constant prediction, not a failing model.*

**BUG 2 — memorised numbers.** First run used textbook composites (91, 121, 143, 169) and reported a
z=-15.09 collapse on `prime_odd` under ablation. With 4-digit non-textbook semiprimes, `prime_odd` CLEAN
accuracy is **0.250** — the model cannot do it at all, so there is nothing for ablation to break. Zero-
shot and few-shot both give prime=0.722 on hard numbers => it was the NUMBERS, not the framing.
**The z=-15.09 result is RETRACTED**: ablation was disrupting retrieval of memorised facts, not arithmetic.

**BUG 3 (minor) — ungrammatical few-shot.** Chess examples read "Answer: It is a dark" ("a dark" is not a
noun phrase). Fixed to "It is a dark square". Chess STILL 0.542 after the fix => genuine limitation of
Llama-3.1-8B *base*, not a harness artefact. Chess (= (file+rank) mod 2, the same variable as the toy
task) is therefore UNTESTABLE on this model.

**RESULT (all bugs fixed, few-shot framing, 4-digit numbers, n=5 random controls):**
| task | clean | ablated | random | excess |
|---|---|---|---|---|
| odd_even | 1.000 | 1.000 | 1.000 | **0.000** |
| prime_even (2*p) | 1.000 | 1.000 | 1.000 | **0.000** |
| clock | 1.000 | 1.000 | 1.000 | **0.000** |
| months | 1.000 | 1.000 | 1.000 | 0.000 |
| days | 0.964 | 1.000 | 0.979 | +0.021 |
| neutral (city/river) | 1.000 | 1.000 | 1.000 | 0.000 |

**The decisive line is `prime_even`.** The model scores **1.000** on 2*p semiprimes and **0.250** on
structurally identical ODD semiprimes (1006=2*503 vs 1003=17*59) — parity is the ONLY thing separating
the groups, so the model is provably solving them by parity. Ablating the parity circuit changes it by
**exactly zero**.
=> The model uses parity; the toy-task parity circuit contributes nothing to it. With the Pile result,
the circuit is **specific to the in-context graph task**.
Neutral task 1.000->1.000 confirms 21-head ablation does no free damage, so this is a real null, not a
too-gentle intervention.
Weak instruments, not conclusions: `div2` subgroups swing oppositely (even -0.097, odd +0.139) = residual
answer bias, clean only 0.833; `prime_odd` -0.117 sits on a 0.250 floor.

## RING: 2-D CIRCULAR POSITION CODE REJECTED (2026-07-28)

Sandra's idea: run the same DAS on a RING instead of a grid, counterfactual = local transposition
(ABCDE -> ABCED). The ring is a much sharper instrument — position collapses to ONE cyclic coordinate,
so "compact code" has an exact prediction: a 2-d (cos t, sin t) embedding, in which EVERY rotation is a
rotation within that same plane. So `cyc1` should work at **rank 2** and the same subspace should serve
cyc2/cyc4.

Perms added (ring): `cyc<k>` rotate by k (a genuine automorphism), `refl` i->-i, `swapadj` (the literal
ABCDE->ABCED, one adjacent pair), `swappairs` = the same transposition applied EVERYWHERE
(ABCDEF->BADCFE), `randbij` control.

**⚠️ LAZY SELF-LOOP CONFOUND — caught in smoke test, would have produced a false positive.** For any perm
where pi(u) is ADJACENT to u (cyc1, swappairs, swapadj — i.e. exactly the short-displacement ones asked
about), `nbrs(pi(u))` CONTAINS u, and on lazy walks the model puts ~p of its mass on the current node.
`cyc1` baseline was **+0.30 (flip 0.67) with NO patching**. Fix: drop {u, pi(u)} from both T and S.
After fix, cyc1 base = **-3.05 (flip 0.03)**. Also shifted the 4x4 GRID baseline (-3.28 -> -2.26), so my
claim that far perms were unaffected was too strong — but grid FLIP RATES were unchanged (r2 0.89, r8
1.00 both), and every rank/transfer conclusion rests on flip rate, so the grid results stand.

**(1) RESULT: rank 2 does NOT implement cyc1.** LAZY=0.5, L24, STEPS=120:
| ring n | r1 | **r2** | r4 | r8 | r16 | r32 | threshold (flip>=0.95) |
|---|---|---|---|---|---|---|---|
| 8  | 0.34 | **0.72** | 0.88 | 1.00 | 1.00 | 1.00 | r8 |
| 16 | 0.22 | **0.31** | 0.48 | 0.95 | 1.00 | 1.00 | r8 |
| 32 | 0.16 | **0.22** | 0.30 | 0.48 | 0.78 | 0.96 | r32 |
Required rank ~0.5n-1.0n, not the constant 2 a circular manifold predicts. A ring has ONE coordinate with
n values, so categorical predicts rank ~ n — observed, and consistent with the grid's rank ~ K_r+K_c.

**(2) Displacement-matched contrast (the point of swappairs).** cyc1 and swappairs move EVERY node by
exactly 1; only cyc1 is a rotation. Trained on cyc1, ring16 @ r16: **swappairs 0.91 vs cyc2 0.89** —
identical, where a rotational code predicts a sharp split. Both beat randbij (0.61).
**CAVEAT WAS REAL — and normalisation removed an apparent asymmetry.** Raw flips looked lopsided
(cyc1->swappairs 0.91 vs swappairs->cyc1 0.71). Normalising by what each perm achieves TRAINED ON ITSELF
(recovery of base->ceiling range, ring16 @ r16):
  swappairs->cyc1 = (3.90 - -2.37)/9.52  = **66%**
  cyc1->swappairs = (4.40 - -2.63)/11.51 = **61%**
Symmetric. The asymmetry was pure task difficulty (randbij base -5.23 vs swapadj -1.16 etc). *Any raw
flip comparison across perms in this project is confounded by baseline difficulty and must be normalised
this way.*
=> The shared structure is **NOT rotational**: swappairs (not a rotation) shares ~60-65% with cyc1 in
BOTH directions and is indistinguishable from cyc2 (which IS a rotation).
**Still open:** the randbij FLOOR was never measured — I compared against it without ever training on it,
so its ceiling was estimated, not known. randbij-TRAINED run queued to complete the 3x3 matrix
(cyc1/swappairs/randbij each trained, each evaluated on the others) at ring16/ring32.

**(3) Sanity check passed:** at ring8, r8 = full node-identity rank and EVERYTHING transfers, randbij
included (0.95). The degenerate regime shows up exactly where it should — and is why ring8 says nothing
about transfer.

**(4) FULL TRANSFER MATRIX, every cell normalised by that perm's own trained ceiling (ring16 @ r16):**
| trained \ eval | cyc1 | swappairs | randbij |
|---|---|---|---|
| cyc1      | 100% | **61%** | 46% |
| swappairs | **66%** | 100% | 26% |
| randbij   | 30% | 15% | 100% |
Local perms share 61-66% with each other vs 15-46% with a random bijection => the sharing is REAL (floor
now measured, not estimated). ring32 same pattern, compressed (57 / 45 / 26-35%).
Note the asymmetry: a randbij-trained subspace transfers poorly to EVERYTHING (15-35%); local-perm
subspaces generalise. Fitting a global scramble gives a specific solution; fitting local moves does not.

**(5) GEOMETRY — the decisive measurement (objective-free, no DAS).** Node-mean cloud at L24, lazy:
| | ring16 | ring32 |
|---|---|---|
| var in top-2 dims | **0.386** | **0.255** |
| RSA(repr dist, ring dist) all | +0.790 | +0.605 |
| RSA local (d<=2) | +0.847 | +0.741 |
| mean repr dist d=1 / d=2 / d=n/2 | 11.18 / 13.37 / 15.91 | 11.43 / 13.56 / 16.47 |
| dims for 90% var | 11 | 21 |
**Positions are NEARLY EQUIDISTANT, not circular.** A planar circle gives d(1)/d(n/2) = sin(pi/n) = 0.195
(n=16), 0.098 (n=32). Observed **0.70 / 0.69** — flatter by 3.6x and 7x. Eight steps around ring16 raises
representational distance only 42% over one step. High-dimensional near-orthogonal code + weak local
gradient, NOT a manifold.

**⚠️⚠️ THE HEADLINE — DETECTABILITY vs CAUSAL USE COME APART.** 38.6% of variance in the top 2 dims and
RSA +0.79 mean a **PCA plot of ring16 WOULD show a convincing ring, and RSA would confirm it**. Yet
**rank 2 gives flip 0.31**. The geometry is real and visible; it is not what the model uses. Any method
stopping at PCA/RSA reports a ring code the causal test rejects. This is the sharpest evidence in the
project for the visualisation-vs-usage distinction.
Corroboration: cyc1- and swappairs-trained subspaces have **ZERO principal angles above cos 0.9 at any
rank**, yet mean cos 0.42 and 61-66% behavioural transfer — they share much in aggregate and nothing in
particular. Diffuse partial overlap (weak smoothness bias on a high-dim code), not a shared low-dim code.
DAS subspaces capture 20-40% of node-mean variance, ~100x over the random-subspace floor (0.002-0.008).

**LIMIT:** measured at layer 24 on LAZY walks. PCA/RSA results in this literature are usually taken at
other depths on non-lazy walks, so this does NOT show the ring is illusory there — only that where it was
tested causally, the low-dimensional appearance overstates causal role by a large factor.

## NO COMPACT COORDINATE CODE: perm-counterfactual DAS across grid sizes (2026-07-28)

**Method change.** Every counterfactual until now came from D4 (rot90/rot180/transpose). Realised the
counterfactual enters ONLY as node-mean activations, `D[t] = z[pi(u)] - z[u]`, with the input sequence
untouched — so **pi need only be a BIJECTION, not a graph automorphism**. That unlocks arbitrary row/col
permutations, a far larger family than D4's 8 elements.
(Aside: parity-safe restriction of pi, which I first applied here, is only meaningful on NON-lazy walks.
On lazy walks parity is already null so it buys nothing, and it over-constrains badly — on 3x3 the only
parity-safe row perm is 0<->2, i.e. reversal, so `rowcolperm` collapsed to EXACTLY `rot180`, reproducing
the counterfactual we already had. Sandra caught this. Free perms used for the real run.)

Setup: residual DAS @ **L24** (the depth fix), **lazy walks p=0.5**, free row+col perm counterfactual,
grids 3x3..10x10, ranks 1..64, context scaled per node (`CTXLO_PER_N=14`).

**⚠️ (1) BELOW IS CONFOUNDED — see the non-square result further down. On SQUARE grids n = K^2 and
K_r+K_c = 2K move together, so this sweep CANNOT separate "rank tracks node count" from "rank tracks
coordinate-code size". I originally read it as node identity without noticing. The controlled test (n
fixed at 64, K_r+K_c varied) favours the coordinate reading.**

**(1) Required rank scales with NODE COUNT, not coordinate dimensionality.**
| n | 9 | 16 | 25 | 36 | 64 | 100 |
|---|---|---|---|---|---|---|
| rank for flip>=0.95 | 2 | 8 | 8 | 16 | 16 | 32 |
A true coordinate code needs **2** dims (row, col) at EVERY size; 2*log2(K) would be ~6.6 at K=10, where
r8 gives only 0.68 and r2 gives **0.07**.

**(2) No transfer to unseen remaps.** 10x10 @ r64: trained perm **+14.56**, vs `rowcolperm#1` -0.19,
`#2` +0.31, `rowperm#3` +1.76, `randbij#4` -0.15, `rot180` +1.25. The **random-bijection control is
indistinguishable from same-family draws**. The patch is linear with no learned map beyond the projection
`(D R^T) R`, so if R spanned the node-mean space every perm would pass through — it does not.

**(3) The representation itself is high-dimensional** (objective-free, no DAS involved). Node-mean cloud
at L24: n=16 → 11 dims for 90% var (PR_var 9.4); n=100 → **55 dims for 90% var** (PR_var 32.4). Grows
with n. Not a 2-D coordinate manifold.

**(4) The DAS subspace is a small slice of it.** Node-mean variance captured: 4x4 r8 = 0.013
(random-subspace floor 0.002); 10x10 r32 = 0.164 (floor 0.008). Enriched 6-20x over chance but nowhere
near containing the identity space — sufficient for ONE remap, useless for another.

→ **Fourth independent failure to find a coordinate circuit/code, and the most direct: it does not depend
on head selection at all.** What DAS finds is a per-counterfactual slice of a high-dimensional
node-identity representation.

**⚠️ RETRACTION: "coordinates need rank 4-8" was measured at 4x4 ONLY.** r8 at n=16 is half the node
count — squarely on the ~0.3n trend, never evidence of compactness. I had been treating it as though it
were. **Parity contrast survives and sharpens: parity is rank-1 regardless of grid size** — that is what
a compact code actually looks like.

**OPEN — scaling exponent not resolved.** With powers-of-2 ranks, `rank ~ 2K` (a CATEGORICAL row+col
code: 20 at K=10) fits nearly as well as `~0.3n` (30 at n=100). Transfer failure at r64 >> 2K=20 argues
against the categorical-coordinate reading (R had room for the whole code and still didn't transfer), but
that is an inference, not a measurement.

  **Fine-rank test at 10x10: INCONCLUSIVE — noise exceeds the effect.** r12=.81 r16=.88 r20=.86 r24=.96
  **r28=.74** r32=.97 r40=.99 — r28 falls BELOW r20. And r32 gave margin +5.81 here vs +10.36 for the
  same rank in the previous run: **run-to-run variance ~4.5 margin units at fixed rank**, larger than the
  20-vs-30 gap being tested. 45 Adam steps is not converged. *No scaling exponent should be read off any
  single-seed rank curve in this project* — including the earlier coarse sweep, whose THRESHOLDS are
  robust (2 -> 32 over 11x in n) but whose fine structure is not.
  Transfer stayed dead at every rank (-1.16 to -2.29) with randbij == rowcolperm#1 throughout — that
  result does not depend on the optimizer settling.

  **NON-SQUARE RESULT (seed 0; seed 1 pending) — REVERSES the (1) reading.** n=64 FIXED for all three,
  STEPS=120:
  | grid | K_r+K_c | r8 | r12 | r16 | rank for flip>=0.95 |
  |---|---|---|---|---|---|
  | 8x8  | 16 | **0.96** | 1.00 | 0.99 | **r8** |
  | 4x16 | 20 | **0.98** | 0.96 | 0.98 | **r8** |
  | 2x32 | 34 | 0.76 | 0.91 | **0.95** | **r16** |
  Same n, but 2x32 needs 2x the rank of 8x8; K_r+K_c ratio 2.1 vs observed rank ratio 2.0.
  **Fit: rank ~ 0.5*(K_r+K_c)** -> predicts 8, 10, 17 -> bins to 8, 8, 16. Exactly observed.
  => rank tracks a **CATEGORICAL row+column code** (which IS a coordinate code, just categorical rather
  than scalar), NOT node identity.
  Caveats: (a) **transfer is still dead at r40 > K_r+K_c=34** (rowcolperm#1 -1.61, randbij -1.28 on
  2x32) — a rank-40 subspace has room for the whole 34-dim code, so if it had found the code, transfer
  would work. Likely reconciliation: the rank REQUIREMENT reflects the representation, the transfer
  FAILURE reflects what the optimiser converges to (trained on one remap, nothing pushes it to the
  canonical code). (b) 2x32 is a ladder (mean degree 2.9 vs 3.5) with a harder baseline (-7.74 vs -4.82);
  4x16 sitting WITH 8x8 rather than in between argues against "elongated = harder" being the whole story,
  but it is not airtight. (c) STEPS matters: 8x8 threshold was r16 at STEPS=45, r8 at STEPS=120 — all
  three compared at STEPS=120, so internally consistent, but absolute thresholds are optimiser-dependent.

  **Non-square design.** das_multihead now takes non-square "KrxKc". Test holds n=64 FIXED
  while varying K_r+K_c: 8x8 (16), 4x16 (20), 2x32 (34). Rank ~ n predicts identical thresholds; rank ~
  K_r+K_c predicts 2x32 needing ~2x of 8x8 — a 2x effect instead of 1.5x, above the noise floor.
  STEPS=120 and 2 seeds. Caveat: 2x32 is a ladder (mean degree ~2.9 vs 3.5 for 8x8), so compare each
  grid's rank threshold relative to its OWN saturation, not absolute margins.

## COORDINATE CIRCUIT: higher-rank + correct depth (2026-07-26)

**Depth bug found (real, consequential).** Coordinate DAS had been trained at LAYER 14, but the main
coordinate writer L21H10 is at layer 21 — DOWNSTREAM. A layer-14 subspace cannot contain its
contribution, so every coordinate analysis built on it was structurally blind to the late writers.
Retrained at layer 24 on LAZY walks (rot180): base −4.27, **r1 flips 57%, r4 100%, r8 99%, r16/r32 100%**
— versus r1 ≈ 1% at layer 14. Depth mattered more than rank.

**Three coordinate circuits, same evaluation (coord margin, lazy walks, 5 random controls):**
| circuit | selection | recovery |
|---|---|---|
| parity-derived 21 heads | causal carriers, normal walks | **19.5%** |
| lazy-carrier 22 heads | causal carriers, lazy walks | 8.5% |
| RSA-attribution 22 heads | correlational, rank-8 @ L24 | **−1.0% (at/below floor)** |
| random same-size | — | −0.4% |

→ **Correlational selection FAILS for circuit building.** The RSA set was worse than keeping nothing
(KL 2.92 vs floor 2.83; lazy-aware validity 0.165 vs floor 0.177). Warning sign was visible in the
control: random projections already scored RSA 0.5–0.75, because nearly every head carries node
identity. RSA finds coordinate-CORRELATED heads, not heads the model USES. Causal criteria
(interchange / ablation) are required.
→ **Coordinates are genuinely distributed**, now method-independent: no 21–22 head set exceeds ~20%
across three selection methods, vs parity's 47.6% from the same size. Consistent with parity=rank-1 and
coordinates=rank-4–8.

**⚠️ THREE BROKEN METRICS in a row on this question — all the same root cause: fitted per-node measures
against only 16 distinct nodes.**
1. in-sample R² of an 8-dim subspace predicting (row,col) → ~0.85 for EVERY head incl. ones with
   write_norm 0.018 (measures degrees of freedom, not the head)
2. held-out-NODE CV → 9 params fit to 8 training nodes = exactly determined → R² −0.4 to −8.4, and the
   "lift" ranking was driven by which random control blew up worst
3. earlier `val~label` = 0.906 for coordinates → circular (direction fit against the same labels)
RSA (no fitted parameters, 120 node pairs) is well-conditioned and was the right tool for the
*measurement* — but see above, it is still the wrong SELECTOR.
**Rule: any per-node coordinate metric in this project needs re-checking for this. Behavioural
measurements with random-set controls (margins, flips, sufficiency) were reliable throughout.**

## VS ARDITI'S INDUCTION-CIRCUIT CRITIQUE (2026-07-26) — `olsson_head_scores_*`, `suff_olsson`

Post: lesswrong.com/posts/qtdSzLpQ8BXv6YANd (also ICLR Blogposts 2026; code github.com/andyrdt/iclr_induction).
Claim: Park et al.'s in-context geometry is explained by induction circuits; the grid PCA structure is a
byproduct of previous-token mixing, not functional.

**Validity assessment.** Positive claim (induction solves grid tracing) is well-supported: consecutive
walk tokens are always neighbours, so bigram recall gives valid steps; his ablations show top-2
prev-token → <25% accuracy vs ~93% random-head control. The byproduct claim is UNDER-DETERMINED: he shows
prev-token heads are NECESSARY FOR the geometry, not that the geometry is causally inert — that needs an
intervention ON the geometry, not on the heads producing it. Also uses ZERO ablation (off-distribution,
overstates importance) where mean-ablation is better controlled. His own flagged weak point stands:
prev-token heads are active from token 1 but geometry only appears after hundreds of tokens.

**Methodological difference.** His identification is TASK-INDEPENDENT (Olsson attention-pattern scores on
repeated random sequences). Ours is task- and variable-specific (mean-ablation necessity, interchange
carrier, write decomposition). Correlation of our parity-carrier score with his scores is weak:
+0.27 (prev-token), +0.19 (induction).

**Mapping.** Our MINUS family = BOTH his classes: it holds the #1 prev-token head (L14H26, 0.686) AND the
#1 and #3 induction heads (L15H30 0.980, L16H20 0.828). Reason: on a BIPARTITE walk both motifs land on
opposite-parity positions (prev token = lag 1; induction target = successor of the last occurrence = a
neighbour), so both need negative value-coupling to write current parity.
**Our PLUS family is INVISIBLE to his scores**: L14H19 (our #2 carrier, z=-12.9) ranks 1005/1024 on
prev-token and 750/1024 on induction; likewise L14H17 (1007, 391), L10H2 (1012, 551), L4H16, L7H25, L8H11.
These are SAME-TOKEN AGGREGATORS (attend to previous occurrences of the current token ITSELF), a different
motif from prefix-matching (which attends to the SUCCESSOR of the match).

**Head-to-head sufficiency (same task, same metric, 5 random controls):**
| keep-set | parity margin | nbr validity | parity coef |
|---|---|---|---|
| our 21-head circuit | **47.6%** | **86%** | 1.019 |
| his top-8 induction + top-8 prev-token (16) | **4.3%** | **40%** | 0.582 (BELOW the 0.598 floor) |
| random same-size | 0.1% | — | — |
His set contains 6 of our heads incl. the #1 prev-token head and still builds no parity.

**Synthesis: both partly right.** Task ACCURACY is largely induction-driven (40% from his heads alone,
task-independently identified). The PARITY FEATURE is built by a different, overlapping-but-distinct set
dominated by same-token aggregators, and it is not decorative (highest causal usage of any eigenmode,
flips behaviour under DAS interchange, removal costs long-horizon validity).
**Contribution: task-independent attention-pattern scoring systematically misses same-token aggregation
heads**, because prefix-matching rewards attending to the successor of a repeat, not the repeat itself.
⚠️ Fairness caveat: his claims were about accuracy and PCA geometry; the parity metric is ours, not his.

## LEXICAL CONFOUND + WRITE PATTERN (2026-07-26) — `direction_lexical_purity_*`, `write_pattern_shape_*`

**The confound.** The parity coefficient = residual@L14 · v, correlated with the CURRENT node's parity.
With the word→node assignment fixed, parity is a deterministic function of the current token, so a
direction separating 8 words from the other 8 scores as a perfect parity direction with no in-context
computation. Evidence: with EVERY attention head mean-ablated (no cross-position flow at all) the
coefficient still separates the classes.

**Measured size (corrected — I first mis-stated this as 61%).** Static/full separation is
0.598/1.537 = **39% on the training assignment**, but only 0.097/0.923 = **10% on a HELD-OUT word
assignment**, where |r_static|=0.23 sits *below* the random control's 0.25 (i.e. at noise). So ~29 points
were lexical and assignment-specific; the structural component transfers (|r_full| = 0.78 out-of-sample).
The parity direction is substantially real.

**The averaging fix FAILED, informatively.** Trained the residual r1 direction under 6 random word→node
assignments. Each flips parity 90–98% in its OWN context — but they are **mutually near-orthogonal
(mean pairwise cos = 0.089**, vs ~0.016 for random 4096-d vectors and ~0.43 across SEEDS with the
assignment fixed). Averaging therefore cancels signal, not lexical noise: the average scores |r|=0.35
out-of-sample vs 0.78 for the original single-assignment direction.
→ **Use held-out-assignment validation, not averaging.**
→ The rank-1 DAS solution is UNDER-DETERMINED: many directions work causally within a context and
  training picks an arbitrary one. This also explains the recurring seed variance (same config flipping
  21% in one run, 73% in another), and is the sharpest form of "ephemeral pointer" so far — a different
  word assignment counts as a different context.
**FIVE held-out assignments (2026-07-26, `lexical_purity5.log`) — supersedes the single-sample numbers:**
| direction | \|r\|_full | \|r\|_static | sep_full |
|---|---|---|---|
| seed_stable (4-seed avg, fixed assignment) | **0.725** | 0.293 | 0.902 |
| wp5 (best single) | 0.633 | 0.141 | 0.688 |
| AVERAGED over 6 assignments | 0.458 | 0.218 | 0.414 |
| individual wp0–wp4 | 0.16–0.45 | — | — |
| random | 0.097 | 0.167 | 0.015 |

- **The direction is structural**: transfers to unseen word assignments at |r|=0.725 vs random 0.097.
- **Correction to the single-assignment claim**: static is 0.293 vs a random floor of 0.167 — above
  noise, not "at noise" as the one-sample test suggested. Modest relative to 0.725, not fatal.
- **Assignment-averaging still loses** (0.458 < 0.725). RULE: average over SEEDS (reduces optimizer noise
  in an under-determined objective), validate on HELD-OUT ASSIGNMENTS, never average over assignments.
- Spread across individual runs 0.156–0.633 = under-determination showing up directly.
⚠️ Still only 16 distinct node values (effective n=16). 8×8 would be better but rank-1 is not a valid
  handle there (r1 flips ~1% at 8×8 vs 73–98% at 4×4), so the rank-1 test is inherently a 4×4 experiment.

**Write pattern shape (per-node write decomposed in the Laplacian eigenbasis; λ=2 is the checkerboard).**
Core parity heads write ~80–88% pure checkerboard: L14H19 0.878, L16H20 0.858, L14H17 0.807,
L14H26 0.795 (6–9% low-λ coordinate admixture). But the circuit is heterogeneous:
  - **L21H10 writes almost pure coordinates** (92.5% low-λ, col r=0.74) — independent confirmation of its
    label.
  - **L10H2 writes mostly ROW position** (48% low-λ, row r=0.65) despite sitting in the PLUS family by
    value sign.
  - **L2H26 writes 78% MID-frequency modes — neither parity nor coordinates.** New explanation for its
    necessary-but-not-a-carrier status: it contributes a different spatial pattern the computation needs.
  - L25H7 (reader) reads a mix: 40% parity + 52% coordinate.
Note the attention-ablated residual is itself 85% checkerboard-patterned, but that is CIRCULAR — v was fit
on this assignment, so projecting embeddings onto it reproduces the pattern by construction. The
in-context component (full − static) is 81% checkerboard, which is the meaningful number.

## ✅ CIRCUIT STRUCTURE (2026-07-26) — carriers, sign families, readers, sufficiency, laziness

New pod (port 12272). Files: `head_interchange_roles_*`, `circuit_sufficiency_*`, `write_mechanism_*`,
`head_composition_map_*`. Attribution graph artifact: claude.ai/code/artifact/b800dfb5.

**Carrier vs necessary (two different interventions).** Necessary = mean-ablation degrades the readout.
Carrier = splicing the head's output from a parity-inverted counterfactual makes the model follow the
counterfactual. Top carriers: L14H26 (−4.88, z=−17), L14H19 (−3.70), L16H20 (−3.06). **L2H26 is
necessary (abl z=+4.6) but NOT a carrier (+0.10)** — confirmed 3 ways (patch, DAS steering, and its
values contain no parity, r=0.03). Shuffle control: all effects collapse ~400× (sd 0.284 → 0.0048).

**Sufficiency: 21 heads (2% of all) recover 47.6% of parity margin, 86% of neighbour validity; random
same-size sets recover 0.3%** (8 draws). Coordinates 37.9% vs −0.0%. Caveats: 48% not 100% (core, not
whole computation); MLPs never ablated, so this is sufficiency of an ATTENTION circuit given intact MLPs.
Leave-one-out: dropping L14H26 alone collapses it (+2.15 → −0.11, below floor).

**Write mechanism (exact: write(t)=Σₛa(t,s)·val(s), reconstruction r=1.000).** Sign of a head's
value↔parity coupling is tied to where it attends (r=+0.66 across 21 heads):
  - PLUS family — broad attention to SAME-parity/same-node sources, + coupling: L14H19 (+0.84, 99.9%
    same, H=3.7), L14H17, L10H2, L21H2, L8H11, L4H16 … → aggregate parity over the node's history.
  - MINUS family — peaked attention to OPPOSITE-parity (previous-token) sources, − coupling: L14H26
    (−0.64, 11.5% same, H=0.79), L16H1, L16H20, L15H30, L4H12, L9H11 … → read previous node, invert.
  Both arrangements yield a correct current-parity write: (parity attended) × (value sign).
  Depth: early L0–7 peaked/weak encoders (|val| 0.33), mid L8–16 strongest (0.58), late L17+ broad
  (H 3.5) with L25H7 near-zero coupling (−0.11) = reader, not writer.

**Sign families are COMPLEMENTARY, not redundant.** PLUS-only (10+scaffold) recovers −2.4%; MINUS-only
(9+scaffold) +2.2%; PLUS+induction (14) −3.2% — all at floor — versus 47.6% for the union. Adding the
induction heads to PLUS does not rescue it, so MINUS's contribution is the local-flip computation, not
induction. Strongly super-additive, matching the earlier pair result (79% flip from L14H26+L14H19 vs
2%/22% alone).

**Reader-only heads are essential.** L2H26+L25H7 alone recover −0.5% (floor) — neither has parity in its
values. But removing them from the circuit drops recovery **47.6% → 26.9%** and validity 0.90 → 0.77:
two content-free heads account for ~43% of the circuit's performance. Classic read-out signature — no
content of their own, but without them the computed content never reaches the output.

**⚠️ LAZY WALKS SEPARATE THE VARIABLES — the automorphism confound's solution.** On a grid, parity is a
function of the coordinates, so NO automorphism changes one without the other; rot90 and rot180 carrier
rankings correlate r=+0.77 (both transplant whole node identity). Self-loops (p=0.5) break this:
  - parity carrier spread 0.284 → 0.021 (**7% retained**), profile stability r=+0.075, 4/15 overlap
  - coord carrier spread 0.386 → 0.304 (**79% retained**), profile stability r=+0.842, 8/15 overlap
  - L14H26 falls #2 → #16 on the coordinate test; L21H10 (coord writer) rises #4 → #2, L17H24 #10 → #3
  So heads that appeared to carry coordinates did so only via parity/identity. **Laziness is the tool
  that isolates the two circuits**, independently validating the original disjoint-circuits claim.
  The 21-head (parity-discovered) circuit recovers only ~19% of coordinates on lazy walks vs 37.9%
  non-lazy — a properly matched coordinate circuit should be re-derived FROM lazy walks.
  NOTE: neighbour-validity is meaningless on lazy walks (half the transitions are self-loops, which the
  metric scores as invalid); only the coordinate margin is interpretable there.

**Composition map** (ablate each of 512 early heads, measure every later head's output + attention):
top QK edge over all 385k pairs is **L14H26 → L16H20** (z=16.8 after removing adjacency/magnitude
gradients), plus L14H26 → L15H30 (z=13.4) — the parity head controls where the induction heads look.
L2H26's influence goes locally to L4–L6, never to the late machinery. The parity-WRITE matrix from that
run is broken (per-head normalization divides by near-zero early-layer variance; max 181 vs median 0.08)
— recompute with a global normalizer before using.

## ✅ FIRST SOLID NATURAL-TEXT RESULT (2026-07-25) — `pile_direction_effect_Llama.json`

Ablation at the RESIDUAL site (L14), 200 Pile docs, vs rank-matched random directions, normalized by
energy removed (dloss / coef_sd² — a direction aligned with high-variance residual directions damages
more for trivial reasons):

| direction | coef_sd | dloss | dloss/sd² | vs random |
|---|---|---|---|---|
| **par_r1 (residual DAS)** | 0.260 | +0.00462 | **0.0686** | **14.7× , z=+60** |
| coord_r1 (residual DAS) | 1.269 | +0.01001 | 0.0062 | 1.3× , z=+1.4 |
| sae_107994 | 0.461 | +0.00091 | 0.0043 | 0.9× , z=−0.4 |
| **head_das_r1** | 0.257 | +0.00012 | 0.0018 | **0.4× , z=−2.7 (INERT)** |
| random ×4 | 0.14–0.57 | — | 0.0047 ± 0.0011 | — |

- **The parity residual direction has a real, large causal effect on natural-language prediction** —
  ~15× the damage per unit energy of random directions. First natural-text result in this project that
  survives its own control. par_r8 is +0.0356 raw (4.2× random, z=+8.1; not energy-normalizable because
  only the first basis coefficient was recorded — fix in a rerun).
- **The coordinate result is an energy artifact.** Its raw 15.6× shrinks to 1.3× (n.s.) once normalized:
  coord_r1 simply sits on a high-variance residual direction (coef_sd 1.27 vs random ~0.37).
- **⚠️ RETROSPECTIVE: `head_das_r1` is causally INERT (0.4× random, below chance).** This is the direction
  used in ALL prior Pile work — share₁₆, das₁, `pile_das_norm`, `pile_das_transcripts*`, the transcript
  viewer highlights, and the SAE cosine match. Those analyses were measuring a direction that does
  nothing, which is the simplest explanation for why they produced only weak, unreliable patterns.
  **Any natural-text analysis worth keeping must be redone with the residual-trained parity direction.**
- **Logit readout: no resolution.** After normalizing by perturbation size (peak|Δlogit|/α), nothing
  separates from random — including the SAE positive control (z=+1.3). Eyeballed token lists are
  worthless here: random directions produced lists as thematic-looking as any real one
  (' occasionally'/' particularly'/' principally' from a random direction). Only head_das is
  significantly BELOW random (z=−2.4), consistent with inertness. We know THAT parity matters, not what
  it computes.

## ❌ NEGATIVE RESULT (2026-07-25, `direction_write_attribution_Llama.json`)

Exact additive decomposition of the coefficient h·v into 480 per-head + 15 per-MLP writes
(reconstruction r = 0.99998–1.00000, rel. err ~2e-3 — the decomposition is exact, not a probe).
Same direction decomposed on Pile text AND on grid walks.

- **On the GRID, the seed-stable parity direction is written overwhelmingly by L14H26** — contribution
  −0.205, **2.3× the next head** — the exact head this project identified as the parity writer by three
  independent earlier methods (head sweep, mean-ablation circuit, DAS). Strong internal validation that
  the direction is the parity direction and the method works. Writer profile is concentrated:
  participation ratio 123 vs 174 for random directions.
- **On natural text the SAME direction is written by a completely different, diffuse set of heads**
  (L12H12, L0H23, L13H18, L2H5, …). **Zero overlap between the top-10 grid writers and top-10 Pile
  writers.** Concentration is at the random level (PR 174 vs random 185).
- **Cross-setting profile correlation is NOT above chance**: par_stable_r1 abs-rank r = +0.221, *below*
  all three random directions (+0.287, +0.391, +0.464; mean +0.381). Pearson +0.050, inside the random
  range. The heads writing it in prose are no more related to its grid writers than for a random
  direction.

**CONCLUSION: the toy→natural-text bridge fails.** No shared circuit between the grid task and prose.

Do NOT dress this up as "one subspace, two circuits" — that description is true of essentially ANY residual
direction, including the random controls (which in fact had HIGHER cross-setting correlation than the
parity direction). Nothing here is a property specific to the parity direction. It is a null.

Also do NOT call this a demonstration of "stable subspace, ephemeral pointer": that claim was about
alignment across GRID SIZES, a different comparison entirely. This experiment says nothing about it.

The L14H26 grid result is a sanity check (it confirms the direction is the parity direction and that the
decomposition works), not a finding — the direction was trained by DAS on that task, so a known parity
head dominating is expected.

What this leaves: the natural-text thread has produced no supported positive claim. The one unexplained
residue is the 15× ablation damage per unit energy — and even that is not shown to be parity-SPECIFIC
(the obvious control, DAS directions trained on an unrelated task, has not been run; "DAS finds
high-leverage residual directions" would explain it equally well).

## ISOLATION ATTEMPT (2026-07-25, `pile_effect_spectrum_Llama.json`) — one real signature, no target

Fixed a real methodological error (the earlier logit readout AVERAGED per-position logit diffs, which
cancels opposing effects) by taking the SPECTRUM of the position×vocab logit-difference matrix; plus a
coefficient regression against context-global features; plus seed stability.

- **SEED STABILITY: only the rank-1 core is real.** Across 4 independently-seeded trainings, r1 shares
  mean cos² 0.431 vs 0.0003 random (1438×) — but the shared projector's eigenvalues are [0.72, 0.25,
  0.02, …], i.e. ~1–2 genuinely shared dimensions. **r8 shares only 0.0275 (14× random), eigenvalues
  [0.53, 0.38, 0.32, 0.30, …] — essentially arbitrary.** Explains the r1/r8 disagreement everywhere.
  ⚠️ The cross-SIZE rank-16 alignment result (0.53 vs 0.125) still lacks a same-size/different-seed
  control and may be substantially a "DAS finds similar subspaces on this head" effect.
- **CONCENTRATION is the one surviving distinction.** par_r1's effect has participation ratio 24.9
  (EVR1 0.190) vs 43.2/46.0 for random directions — about half the effective rank. NOT an SNR artifact:
  par_r8 has ~8× par_r1's damage yet PR 41.6, so concentration does not track effect size.
  par_stable_r1 is intermediate (34.2).
- **But the leading axis is GENERIC.** par_r1 comp1 promotes rare long code-identifier tokens
  ('OffsetTable', 'URLException') and suppresses short frequent ones (' S', ' in', ','). rand_r1's comp1
  *suppresses* the very same rare tokens — same axis, arbitrary sign. Random directions produce it too.
- **CONTEXT-GLOBAL HYPOTHESIS: NOT SUPPORTED.** Coefficient regression on entropy / surprisal / running
  surprisal / position / repeat / distance-since-last / category: par_r1 R²=0.089 but random r1 gets
  0.059 with log_pos of the same magnitude and OPPOSITE sign (generic position drift).
  **par_stable_r1 — the seed-stable core, the most meaningful object — has the LOWEST R² of all (0.018),
  below both randoms.** Effect size (KL) is even less predictable (R² ≤ 0.028, randoms highest).

**Verdict: the isolation methods can show the direction is SPECIAL (15× causal damage per unit energy,
half the effective rank of random) but cannot say WHAT it targets. No token feature and no context-global
quantity tested predicts it better than chance. Most consistent reading: it is real, low-rank,
causally-important machinery that does not correspond to a describable natural-text variable.**

## WHAT does the parity direction do? — diffuse (2026-07-25, `pile_parity_damage_profile_Llama.json`)

Per-token damage profile of ablating the residual parity subspace, 300 docs / 87,853 tokens, vs
rank-matched random directions. Pre-registered hypothesis (from the attention analysis: L14H19 sends 97%
of attention to previous occurrences of the current token) = ablation should hurt most on REPEATED tokens,
measured within baseline-loss quintiles.

- **Repeat hypothesis: NOT supported.** par_r1 is the only condition with a positive repeat effect
  (ratio +0.69 vs its own damage; all six randoms −0.74..−3.67) — but **par_r8 is −1.59, inside the
  random band**. Two independently-trained parity subspaces disagree in sign, so this is 1-of-2, not
  support.
- **Damage profile is NOT distinctive.** Split-half reliability: par_r1 0.396, par_r8 0.380 — *below*
  most randoms (up to 0.627). Cross-correlation of par_r1's profile with random profiles (+0.19..+0.37)
  is the same as random-vs-random (+0.14, up to +0.34). Third failed attempt at a token-profile claim.
- **One suggestive qualitative difference:** parity spares punctuation (+0.0016) where random directions
  damage it most (+0.0032); it hits digits (+0.0076) and continuations (+0.0051) hardest. Single
  condition, treat as a lead only.
- **Top-damaged tokens are incoherent** (' rather', 'c', '   ', ' in', ' route', '487'), 9 of 12 not
  repeats, base losses from 0.05 to 10.3.

**Conclusion: the residual parity direction is causally important (14.7× random damage per unit energy)
but functionally DIFFUSE in natural text** — it does not implement a localized, token-identifiable
function. This is consistent with the "stable subspace, ephemeral pointer" result: general-purpose
machinery whose natural-text role is not a single interpretable feature.

**Methodological warning for any future DAS interpretation:** r1 and r8 disagree, and the same resid-r1
config flipped 21% in one run and 73% in another (optimizer variance). These subspaces are not stable
objects across seeds — check seed stability BEFORE hanging interpretation on any one of them.

## ⚠️ CORRECTION (2026-07-25) — natural-text pattern claims retracted

`pile_random_subspace_control_Llama.json` (real DAS subspaces vs 8 random 16-dim subspaces in each of
L14H26 and L21H10, 1000 Pile docs, split-half by document):

- **RETRACTED: "parity and coordinates fire on different token populations."** Token-level correlation
  between the two real subspaces is −0.010 — but *every* subspace pair is uncorrelated: real-vs-random
  cross-head +0.001, random-vs-random cross-head −0.002, and real-vs-random **within the same head**
  +0.004. Energy shares of different subspaces of a high-dim vector are near-independent by construction,
  so decorrelation carries no information. This was the strongest natural-text claim and it is void.
- **RETRACTED: the token-category stories** (word-boundary pulse, boundary disengagement, technical-vs-prose
  domains). Category explains 0.7% (pair) / 0.25% (coord) of token-level variance; the headline
  word-initial-vs-continuation contrast is Cohen's d = +0.072. Real in the means, negligible in effect.
- **SURVIVES, modestly:** each subspace has a reproducible per-token profile, and the DAS subspaces'
  profiles are stronger than random slices of the same head — split-half reliability 0.71 (parity) /
  0.75 (coord) vs 0.58 / 0.60 for random, with ~1.8× the amplitude (token-profile sd 0.021/0.023 vs
  0.012). So the DAS subspaces are more token-selective than typical subspaces of the same head — but
  random subspaces are ALSO reliable (~0.6), so most of "token identity matters" is generic to the head.
- **SURVIVES:** above-chance energy (0.174/0.176 vs 0.125 random) — though see open control below.
- **OPEN CONTROL:** compare the DAS subspace's natural-text energy share against the top-16 PCA
  directions of the same head's natural-text output. If PCA-16 captures far more, the DAS subspace is a
  specific low-variance structure; if comparable, "fires above chance" reduces to "DAS found the head's
  dominant directions." Until run, the above-chance-firing result is not interpretable.

Unaffected by this correction: all TOY-TASK causal results (rank, flips, cross-model, lazy-walk
persistence), the SAE geometric match, and the reverse-arrow SAE ablation.

## DAS follow-ups a–d (2026-07-24, all done)

- [x] **(a) Subspace alignment across sizes** (`das_subspace_align_r16.json/.pdf`): the rank-16
      DAS subspaces ALIGN strongly across grid sizes — mean cos² 0.53 vs random 0.126, top
      principal cosines 0.96–0.99 everywhere — and CONVERGE with size (4×4-pairs 0.36–0.42,
      10×10 vs 12×12 = 0.72 with 14/16 dims at cos>0.5). Measured reconciliation: the
      mechanism/subspace is shared; the 1-D readout direction inside it is per-context.
- [x] **(b) Pile causal patch** (`pile_das_patch_Llama.json`): ablating the DAS subspace on
      150 Pile docs damages loss most on CONTINUATION-token predictions (das_r16 +0.0012
      vs rand16 +0.0003 nats, 4×; das_r1 +0.0006 vs rand1 +0.0002) — right category, tiny
      absolute size (one head's slice at one layer). Direction consistent, not headline.
- [x] **(c) Basis steering** (`das_basis_steer_Llama.json/.pdf`): ~4 of 16 dims pin the
      parity sublattice antisymmetrically (das10 ±0.12, das7 ∓0.07, das0 ±0.05, das4 ±0.03
      log-odds; random band ±0.016); NO single dim damages next-token validity (<0.001,
      base 0.999) — parity steering is distributed over a few dims, behaviour is redundant
      to any single one.
- [x] **(d) Identification** (`das_dir_identify_Llama.json`): the DAS direction is NOT the
      word-boundary direction — cos vs Pile word-boundary mean-diff 0.175 (random p99 0.22),
      word-boundary AUC only 0.60 (probe ceiling 0.97). WEAKEN the earlier claim to "has a
      modest word-boundary component". BUT at the correct hookpoint (LlamaScope L14
      residual SAEs) the direction matches a specific feature cluster: max |cos| 0.44 (8x,
      feat 9114 — fires on discourse/function tokens " that", ":", " online"; fragments on
      the other side) and 0.57 (32x, feat 107994) vs random-max ~0.08. So it IS a real,
      identifiable feature — a boundary/discourse-flavoured one, not purely word-start.
- [x] Viewer: ALL eigenmodes (up to 255) selectable for axes/colour/ground-truth panel
      (int8 recapture `grid_walk_ctx2000all_Llama.json`; per-occurrence cloud stays top-6
      with a caption note). Torus viewer still top-6 (needs torus recapture).

Plan: finish the DAS story on the toy grids, then run the learned subspaces on natural
text (The Pile) and find where the patterns arise. Items 1–7 from the review discussion;
item 0 and 8 added after auditing the n×n scale results (see "n×n audit" at bottom).

## 0. Fix the rank-tautology in the prototype DAS (blocks the headline claim)

The prototype-mode delta (`das_parity.py:157`, `das_parity_scale.py` PATCH=prototype) is
`±(proto[+] − proto[−])` — a single fixed vector. Any subspace containing that vector
reproduces the full patch exactly, so "r=1 captures ~116% of the full-head effect" and the
flat r-curves in `das_parity_scale_Llama.json` are guaranteed by construction and cannot
measure rank. The eff_rank values in that file (32→16→8→8→2) are jitter on a flat curve.

- [ ] Restate the rank claim from the **rotation/interchange** variant only
      (`das_parity_scale_rotation_Llama.json`): eff_rank 8 (4×4) → 16 (6×6) and saturates
      at 16 through 12×12. The honest headline is "parity intervention saturates at ~16 of
      128 head dims and does not keep growing with n", not "parity is 1-D".
- [x] Add a **held-out walk split** (2026-07-23: HOLDOUT env in `das_parity_scale.py`;
      rerun `das_parity_scale_rotation_ho3_Llama.json` — eval ≈ train margins at every
      size, eff_rank 8/16/16/16/32(borderline r16≈r32) → the rotation result is NOT
      overfitting).
- [x] Record the **unpatched baseline margin** (same rerun: baseline ≈ rand4 at every
      size, e.g. 4×4 base −1.37 vs rand4 −1.36 — rand4 was a fair proxy).
- [ ] Optional: disentangle parity from coord-remap in the rotation patch (π = rot90 moves
      coords too). E.g. patch delta = rotation-delta minus its projection on the coord
      subspace of L21H10, or use a parity-preserving automorphism as control.

## 1. Project the DAS subspace (not the mean-diff axis) on the Pile

- [x] Load learned subspace — using the interchange-trained `global_R1`/`global_R4` from
      `runs/axes/4_circuits/das/das_grid_patch_Llama_L14H26.npz` (non-tautological; the
      `das_parity` R_1 is prototype-trained and contains proto_delta by construction).
- [x] Extend `src/scripts/analysis/parity_on_pile.py` (v2, 2026-07-23): projects proto axis,
      DAS r=1 direction, and rank-4 subspace norm in one pass; queued on pod as
      `parity_on_pile_das_Llama.json` (OUTTAG=_das).
- [x] Read results (`parity_on_pile_das_Llama.json`, 2026-07-23): DAS r=1 ≈ proto axis —
      cos 0.927, token-level Pearson r=0.955, top-K context overlap ~0.55. So the
      interchange-trained DAS direction VALIDATES the mean-diff axis rather than changing
      it. Both fire + on clause-boundary function words (", which"/" or"/" and"/" of") and
      − on mid-word continuation subwords (non|conscious, S|3, 258|07, Ara|uc|aria). DAS
      slightly sharpens the word-boundary contrast (see item 3).

## 2. Causal validation on natural text

- [ ] Ablate / patch the rank-1 (and rank-16) subspace of L14H26 on Pile docs; measure
      per-token loss deltas (hook machinery from `das_grid_patch.py`).
- [ ] Check whether loss damage concentrates on the same clause-boundary / continuation
      tokens that the projection flags.
- [ ] Control: same-size random subspace of the same head, and a shuffled-walk-derived axis.

## 3. Nulls & stats for the Pile analysis

- [x] Null distribution done (`parity_on_pile_das_Llama.json`). VERDICT — two different
      answers, and it matters which metric you quote:
      * **Max-projection is NOT a signature.** Against the matched partition null (mean-diff
        axis of a random balanced 8/8 colouring of the same head's node means), proto's
        global |max| ranks 9/32 (z=0.82), das1 7/32 (z=0.94). The original headline
        "pile max 1.15–1.26× grid separation" is an artifact — any balanced mean-diff axis
        of this head peaks about that high on 100k Pile tokens. Drop that number.
      * **The word-boundary CONTRAST is a signature.** word_initial-minus-continuation mean
        projection: proto 0.084 (rank 0/32 partitions, z=1.74), das1 0.111 (rank 0/32,
        z=2.8). No random balanced partition of this head's node means separates
        word-start from continuation as well as the parity axis does, and DAS sharpens it.
        This is the defensible form of "grid parity reuses a word-boundary feature."
      * (Random-direction null is too weak to cite — any structured mean-diff axis beats it,
        proto max_z=4.25; use the partition null.)
- [ ] Shuffle-control axis (from existing shuffle runs) projected on the same Pile tokens.
- [ ] Error bars + frequency-matched comparison for category means (ordering is
      punct +0.073 > word_initial −0.026 > continuation −0.111 > digit −0.201; small but
      the contrast beats the partition null). Frequency/position still confounded.

## 4. Identify the feature against known references

- [ ] Correlate the parity direction with public Llama-3.1-8B SAE features (Goodfire /
      EleutherAI) — is it a known word-start/continuation feature?
- [ ] Hand-built probes: is-word-start, is-punctuation, token-position parity, prev-token
      boundary. Report which probe the direction best matches.
- [ ] Reverse direction: on top-firing Pile contexts, check *which heads write* the
      projection (is it L14H26/L2H26 or different heads?) — feature reuse vs circuit reuse.

## 5. General Pile capture script

- [ ] `src/scripts/capture/capture_pile.py`: cache head-output (and residual) activations
      for a fixed doc sample to npz, mirroring the graph-capture family, so items 1–4 reuse
      one capture instead of streaming per-analysis.
- [ ] Stratify docs (code / prose / math) and store doc metadata for per-category analysis.

## 6. Cross-model replication (DAS + Pile)

- [ ] Find Gemma-2-9B's parity-writer head (rerun head_eig sweep or reuse existing sweep
      results), run DAS there, project on Pile.
- [ ] Same for Qwen3-8B (mind the massive-activation dim — z-score first).
- [ ] Verdict: is "grid parity recycles a word-boundary feature" universal or Llama-specific?

## 7. Commit the work

- [ ] Commit modified tracked files + `runs/axes/` results + ~30 untracked analysis scripts
      (split by axis: decomposition / circuits / DAS+parity / pile).
- [ ] Include this checklist and updated SESSION_SUMMARY.

## 8. n×n scale sweep — fixes and expansions (from audit)

- [x] **Scale context with n** (2026-07-23, `das_parity_scale_rotation_ho3_ctx2_Llama.json`,
      CTXLO_PER_N=2 WLEN_CAP=1600): the large-grid decay was substantially CONTEXT
      STARVATION. With ctx ∝ n, baseline parity margin is ~size-invariant (4×4 −1.37,
      12×12 −1.31 vs −0.78 fixed-ctx), intervention gap recovers (12×12: 0.39 vs 0.22),
      and eff_rank is a clean 16 at every size ≥6×6 (the borderline-32 at 12×12 fixed-ctx
      was an artifact). Headline: parity feature + its ~16-dim intervention subspace are
      essentially size-invariant once context scales with n.
- [ ] **Chance-normalize parity_mode_power**: raw decay (0.415→0.012) looks like
      disappearance, but ÷(1/n) chance level it is a stable ~2–4× enrichment at every size
      (3×3: 3.7×, 8×8: 2.6×, 12×12: 1.9×, 16×16: 3.1×). Present it normalized.
- [ ] **Odd/even axis structure** (potentially a real finding): cross-size |cos| is high
      odd↔odd (7v9 0.84, 11v13 0.79, 13v15 0.85), low even↔even (≤0.45), and 4×4 aligns
      with nothing (max 0.36). Hypothesis: odd grids have unbalanced colour classes with
      all four corners in the majority class → the axis picks up a degree/visit-frequency
      component shared across odd sizes; even grids are balanced (and sign-ambiguous).
      Test: regress out the frequency/degree-correlated direction and recompute; also
      check sign convention. If odd-odd alignment survives the control, there is a shared
      absolute parity direction — exactly the reusable feature the Pile work needs, and
      the natural direction to project on the Pile in item 1.
      → `grid_parity_compare.py` computes visit-count- and degree-controlled axes
      (`grid_parity_compare_freqctrl_Llama.json`, 2026-07-23). VERDICT: the odd/even split
      is REAL and survives both controls — odd-odd mean cross-size |cos| 0.474 (raw) →
      0.445 (freq-ctrl) → 0.427 (deg-ctrl); even-even stays ~0.17; odd-even ~0.19. So it is
      NOT a visit-frequency/degree confound.
      → BUT it IS (at least partly) a word confound: with the legacy shared perm seed(0),
      word i is assigned to node j at EVERY size, and for odd k node j's colour is exactly
      j%2 at every size (verified), so word-specific components add coherently across odd
      sizes. Rerun with per-size perm seeds done →
      `grid_parity_compare_freqctrl_seedfix_Llama.json`. VERDICT: word artifact CONFIRMED.
      With independent word assignments per size, odd-odd cos collapses 0.474 → 0.202,
      indistinguishable from even-even (0.236) and odd-even (0.214). No privileged shared
      parity direction among odd grids; cross-size alignment is uniformly ~0.2, at the
      split-half noise ceiling. Implication: the parity axis is re-derived per context —
      do NOT claim "one size-invariant absolute axis". The Pile word-boundary contrast
      (item 3) stands on its own nulls and is unaffected. ctx-1000 rerun (2026-07-23,
      running) will show whether alignment rises once the representation is unstarved.
- [~] **Noise ceiling for cross-size cosines**: split-half (3+3 perms) ceiling per size
      added to the same rerun (`axis_splithalf_cos`); report cross-size cosines relative
      to it. (Within-size per-permutation consistency is only 0.15–0.25.)
- [x] **eff_rank robustness**: settled by the ctx-1000 sweep (below) — rank pattern
      8/16/16/16/16 reproduces identically at three context regimes with held-out eval.

### ctx-2000 sweep (2026-07-24, final — 16×16 = 256 nodes justifies the depth)

Files: `das_parity_scale_rotation_ho3_ctxf2000_Llama.json`,
`grid_parity_compare_freqctrl_seedfix_ctx2000_Llama.json`, `mode_var_vs_use_ctx2000_{odd,even}_Llama.json`,
`grid_walk_ctx2000_Llama.json`. Figures (regenerated): `das_parity_scale_ctx_regimes_Llama.pdf` (4 regimes),
`grid_parity_power_normalized_Llama.pdf` (now ctx-2000), `mode_var_vs_use_ctx2000_Llama.pdf`.
Viewer artifact republished with the ctx-2000 capture (same URL).

- **DAS**: eff_rank 8/16/16/16/16 for the FOURTH consecutive context regime; relative effect
  converges to ~0.33 at 10×10/12×12 (same as ctx 1000); 4×4 baseline −4.31 (≈ saturating).
- **Compare**: ctx 2000 ≈ ctx 1000 everywhere — sep_own converged (~1/n = sample statistics),
  cross-size cosine still ~0.2 all families, parity enrichment stable 1–4×. Axis metrics are
  context-saturated.
- **Var-vs-use**: family-level result slightly stronger — high-λ modes carry 54–74% of usage
  on 13–27% of variance at every grid ≥4×4 (3×3 exception again). Exact top-used mode
  identity varies run-to-run at 6×6/8×8/10×10; state the claim at family level.
- **Viewer capture** (`grid_walk` at ctx 2000): grid RSA 0.68–0.86 across ALL 14 sizes
  (16×16: 0.71 at 256 nodes) — the in-context grid persists at every size once context is
  adequate.

### ctx-1000 sweep (2026-07-23, all four experiments at CTXLO=1000)

Results files: `das_parity_scale_rotation_ho3_ctxf1000_Llama.json`,
`grid_parity_compare_freqctrl_seedfix_ctx1000_Llama.json`, `mode_var_vs_use_ctx1000_{odd,even}_Llama.json`.
Figures: `das_parity_scale_3ctx_Llama.pdf`, `mode_var_vs_use_ctx1000_Llama.pdf`.

- **DAS**: parity signal ~3× stronger at ctx 1000 even at 4×4 (baseline −3.96 vs −1.37);
  effect grows with context at every size; relative effect stabilizes ~0.31–0.33 at
  10×10/12×12; eff_rank exactly 8 (4×4) then 16 everywhere, at all three context regimes.
- **Axis alignment**: cross-size cosine stays ~0.21 at ctx 1000 (odd-odd 0.208 = even-even
  0.231) — the parity DIRECTION is context-specific at any depth; the invariants are the
  rank-16 subspace structure and write enrichment, not the direction. sep_own still ~1/n
  (per-node sample statistics in a fixed window, not starvation).
- **Variance vs causal use** (the "structure in low-variance directions" result, now clean):
  high-frequency modes (λ>1.5, parity family) carry 54–69% of causal usage at every grid
  ≥4×4 while holding only 13–26% of variance. The exact parity mode is the single
  most-used mode at 4×4/5×5/7×7/8×8 (u=0.31/0.34/0.17/0.12 on 2–10% variance); at
  6×6/9×9/10×10 usage spreads over near-parity λ≈2 neighbours instead — state the claim
  at the family level. Var-use Pearson falls to ~0.05 for n≥36. 3×3 is the one exception
  (coords most-used; var and use agree, r=0.78) — the grid is small enough that the
  low-frequency modes suffice.
