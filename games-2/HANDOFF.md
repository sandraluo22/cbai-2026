# games-2 — Handoff / current state

Two games measuring **mutual theory of mind between models**: can model A pick up
model B's hidden state while B does the same to A. Measured **continuously** (KL /
recovery), not as a hard L1/L2 label.

- **Game 1** — convergence / KL-coupling (do two models coordinate; does each
  condition on the other).
- **Game 2** — sequential-reveal Codenames (does B recover A's hidden target set;
  coupling; adaptivity).

---

## GLOSSARY (standard terms — 2026-08-03; code/JSON field aliases in parentheses)

- **word family**: words sharing a 4-letter prefix (novel/novelty/novella). Aliases in
  code and older sections: "stem", "self_family", "selffam", "fam". The 4-prefix
  definition is known-incomplete (Llama loops on suffix/rhyme families; see the
  semantic run detector).
- **category**: the kind of word B is secretly restricted to (city, fruit, ...).
  "category mass" = share of A's proposal distribution on unused category words.
- **proposal mass / MC profiling**: sample the same state K times (usually 64);
  report the fraction landing on used words / family words / category words / other.
- **onset**: first turn where A's word shares a 4-prefix with an earlier own word
  (prefix-only looping marker).
- **planted words / dose / donor**: words we insert into a constructed history
  ("seeds" in code); dose = how many; donor = the real stuck game they came from.
- **answer slot**: the position in the prompt where A's own responses appear and the
  next response is generated ("target slot"/"X slots" in mech3).
- **aligned vs drift (semantic runs)**: a run of similar-meaning words counts as
  aligned if its centroid is near B's early words (accommodation), drift otherwise
  (perseveration).
- **patch recovery / transfer / efficacy**: how far an activation patch moves the
  readout from the bad prompt's value toward the matched good prompt's value (0 =
  none, 1 = fully).

## ⚠️ The most important thing to know (validity)

The **original prompts coached the strategy** ("meet in the middle / bridge /
converge / move toward"). Everything was re-run with **neutral prompts** (rules +
win-condition only) and an **in-distribution swap** for coupling. This overturned
much of the earlier story:

- **Semantic "bridging" was largely a coaching artifact.** Uncoached, semantic-board
  convergence collapsed **100% → 20%**, topic semantic-distance now **increases**
  (models diverge), and they converge by **echoing/copying** the partner's word (e.g.
  Qwen self-play forced `sand`/`cloud` → agrees on **"cloud"** by copying, *not* the
  earlier coached bridge to **"storm"**).
- **Coupling magnitudes shrank** once the swap was in-distribution instead of the OOD
  token "seven" (Qwen unbounded coupling **2.94 → 0.62**). The Qwen > Llama asymmetry
  mostly survives but is much smaller.
- **What survives is real but modest:** bounded focal-point coordination and
  self-play convergence — achieved by **copying / focal points**, not by semantic
  partner-modeling.

### Open validity caveats (NOT yet resolved)
1. **Coupling ≠ ToM** — it's ordinary context-conditioning; even a level-0 associator
   couples. No clean floor on the listener side.
2. KL magnitude tracks **output confidence/entropy**, not modeling depth.
3. **Cross-model KL isn't comparable** (different tokenizers / vocab sizes).
4. Convergence is fragile to temperature, the turn cap, and **exact string-matching**
   ("lakes" ≠ "lake").
5. Self-play coupling can read **0 as an artifact** of instant convergence.
6. **no-repeat is prompt-only, not enforced** (models ignore it).
7. Reference-agent level calibration **doesn't transfer** to LLMs.

Trustworthy parts: Codenames **recovery** (ground truth), **semantic-distance**
convergence as a measure, and the **reference agents** validating the instruments.

---

## Results with the NEUTRAL prompt (current honest numbers)

Llama-3.1-8B-Instruct × Qwen2.5-7B-Instruct (self-play = same model, distinct seeds).

| run | converged | turns | coupling (L / Q) | note |
|---|---|---|---|---|
| bounded words (no-repeat) | 100% | 9.5 | 0.16 / 0.55 | focal coordination survives |
| bounded numbers | 67% | 2.2 | 0.00 / 0.00 | focal-rule, no coupling |
| bounded words (repeat ok) | 67% | 2.0 | 0.20 / 0.05 | |
| semantic board | **20%** | 5.0 | 0.10 / 0.00 | bridging collapsed (was 100%) |
| topic minecraft | 80% | 6.8 | 0.05 / 0.13 | sem-dist 0.37→0.68 (diverges) |
| topic animal | 80% | 9.0 | 0.20 / 0.00 | |
| unbounded | 100% | 8.0 | 0.27 / 0.62 | |
| unbounded + no-repeat | 50% | 16.5 | 0.10 / 2.05 | no-repeat not enforced |
| Llama self-play | 100% | 5.2 | 0.07 / 0.10 | |
| Qwen self-play | 100% | 3.0 | 1.00 / 1.13 | |
| Llama self-play forced sand/cloud | 100% | 9.8 | 0.51 / 0.44 | |
| Qwen self-play forced sand/cloud | 100% | 4.0 | 2.09 / 0.16 | converges by echo, not "storm" |
| unbounded + no-repeat **ENFORCED** | 50% | 10.5 | 0.13 / 2.55 | resamples reject repeats; converging games (8,13 turns) have 0 repeats; the 2 capped games exhaust fresh vocab (fallback allows repeats) |
| **Qwen3-32B** self-play, no-repeat enforced, forced sand/cloud | 100% | 5.3 | 0.04 / 0.04 | `runs/llm_open_qwen32_nr_forced/`. NOTABLE: with the NEUTRAL prompt the 32B **spontaneously bridges** (sand+cloud→rain/sun→**storm**; also →prairie, →ocean) — the 7B just *echoed*. Larger model does real semantic bridging uncoached. Qwen3=thinking model → `enable_thinking=False` in `_render`. |

**FILENAME FIX**: self-play labels now use `_1`/`_2` not `#1`/`#2` (the `#` broke file
openers — URL/path fragment separator; renamed all old `#` files).

**TRANSCRIPTS: every `*_transcript.jsonl` now has a `*_transcript.json` twin** (a
pretty-printed JSON array — opens in any browser/editor/Quick Look; `.jsonl` often has
no default app). Auto-emitted going forward (via `kl_slides.generate` + `game_llm.py`).
Convert manually anytime: `python src/jsonl_to_json.py <file.jsonl>` or `... runs`.

### Codenames (Game 2) between real LLMs — `runs/llm_codenames/`
Prompts are **already neutral** (role objective only; guesser found-set given as plain
context, NOT "use it") — no coaching problem. Both role orderings, 8 games, 6 rounds:

| spymaster→guesser | recovery | guesser-coupling | spymaster-adaptivity |
|---|---|---|---|
| Llama→Qwen | 0.77 | 12.4 | 0.38 |
| Qwen→Llama | 0.86 | 5.7 | **1.32** |

Recovery works (both recover the hidden target set); guessers strongly condition on
the clue; **Qwen-as-spymaster adapts to the guesser's revealed state much more than
Llama** (adaptivity 1.32 vs 0.38). Figure: `game_llm_LlamaInst_vs_QwenInst.pdf`.
NOTE: Codenames coupling/adaptivity carry the SAME validity caveats as Game 1 (KL
magnitude ∝ confidence; swap not null-baselined) — the controls in TO DO apply here too.
Codenames now writes a per-round `..._transcript.jsonl` (targets, clue, guess, belief,
+ the coupling & adaptivity counterfactual distributions) — every run (Game 1 AND
Codenames) has a full transcript.

---

## runs/ layout (REORGANIZED 2026-08-02 — see `runs/game-1/INDEX.md` for the full map)

```
runs/
  game-1/
    INDEX.md              <- START HERE: per-directory descriptions
    0_reference_agents/   RSA calibration figures
    1_legacy_7b/          the 11 original 7B-era variants
    2_restricted_core/    restricted convergence game: qwen32_cap24|100|200, fix_cap200,
                          softrestrict_dual, kl_perGame, llama70
    3_interventions/      aids_informed_scratch, g11_branches(+note_blanked),
                          structured_format, stuck_repro_sweep, variants_*
    4_menu_pref/          menu-preference game: qwen32_10r|20r, llama70_20r, qwen72_20r
    5_mechanistic/        pca_*, partner_patch, induction, bridge, *_ablate_play, steering
    6_analyses/           collision_model, update_dynamics, stuck_probe, constraint_probe
  codenames/              Game 2 (llm_codenames + reference-agents)
  game-1-graph/           hidden-grid meet-up
```
NOTE: the POD keeps the old flat layout (`runs/qwen32_*` etc.) — pod-side paths in
launch commands are unchanged; only the LOCAL tree was reorganized. All HANDOFF path
references below were rewritten to the new local layout.

**Qwen3-32B PCA** (`runs/game-1/5_mechanistic/pca_sandcloud/`, `src/qwen32_pca.py`): N self-play
rollouts (forced sand/cloud, no-repeat, temp 0.7); captures BOTH players' residual
stream at every layer/turn → `qwen32_pca_acts.npz` (pts × 65 layers × 5120). PCA per
layer, two centerings: `_global.pdf` (global mean) and `_seedcentered.pdf` (subtract
each seed's own mean over its turns → shows shared turn dynamics, not per-seed word
regions). Finding: the two Qwens' reps are near-superimposed each turn and collapse
together as they converge on a word.

## Code layout (`src/`)

| file | role |
|---|---|
| `core.py` | reference RSA agents (bounded, **dial-able** L1/L2) — CALIBRATION of the instruments |
| `game1_llm.py` | bounded convergence between 2 LLMs (fixed word/number list; `NOREPEAT`) |
| `game1sem_llm.py` | semantic-board + **topic** (hidden candidate set) convergence; embedding distance |
| `game1open_llm.py` | **unbounded** convergence (any word; top-N full-vocab KL); self-play + `START_WORDS` forced start |
| `llm_agents.py` | LLM wrappers + ALL prompts (now NEUTRAL) + Codenames agents + `ConvAgent` |
| `kl_slides.py` | **canonical** every-turn KL slideshows from any transcript (no GPU) |
| `game_llm.py` | Codenames between two real LLMs (recovery / coupling / adaptivity) |
| `game2_codenames.py`, `game1_coupling.py` | reference-agent Games 2 and 1 (CPU) |
| `analyze_transcript.py`, `step_kl_slides.py` | older transcript tools — **superseded by `kl_slides.py`** |
| `remote/deploy.sh` | pod runner for the LLM extension |

**Every LLM run writes a `*_transcript.jsonl`** (per-turn distributions + picks + KLs)
and auto-emits `*_coupling_allturns.pdf` / `*_stepkl_allturns.pdf`. All figures are
regenerable locally: `python src/kl_slides.py <transcript.jsonl>`.

### Metrics
- **coupling** = `KL(B_dist | partner's real last word ‖ B_dist | in-distribution counterfactual word)`.
- **step-KL** = `KL(dist_t ‖ dist_{t-1})` (unpaired drift).
- **recovery** (Codenames) = belief mass on the true targets.
- **adaptivity** (Codenames) = `KL(clue | guesser-found-X ‖ clue | naive-guesser)`.

---

## DONE
- Full framework: reference agents + 3 Game-1 LLM variants + Codenames + LLM extension.
- **Neutral-prompt rerun of all 11 Game-1 variants** (+ in-distribution swap).
- Every-turn KL slideshows for every run (canonical `kl_slides.py`), CJK labels sanitized.
- Codenames between real instruct models (recovery/coupling/adaptivity).
- All results pulled to `games-2/runs/*`.

## TO DO (priority order)
1. **Controls to separate coupling-from-ToM** (proposed, not run):
   - L0 prompt ("say a word related to the last word shown", no partner framing) — if
     it couples as much, coupling isn't ToM.
   - Shuffled-partner control (partner word from a *different* game) — if coupling is
     unchanged, it reacts to *a* word, not *this partner*.
   - Null-swap floor + entropy-normalized / Jensen–Shannon coupling.
2. **Fix convergence metric**: lemma/embedding-nearest match ("lakes"=="lake"); report
   turns-to-converge, not a cap-sensitive rate.
3. **Enforce no-repeat** (resample repeated generations) if that condition matters.
4. **Rewrite `README.md`** to the honest neutral-prompt story (it is currently stale /
   reflects the coached results).
5. Re-verify **Game-2 (Codenames) prompt neutrality**; run the same controls there.
6. **Cross-model comparability**: normalize KL or restrict to a shared vocab before
   comparing Llama vs Qwen numbers.
7. Optional: add a **3rd model** (`GemmaInst`, cached) for a three-way comparison.
8. **Cleanup**: delete dead code (`perturn_pdf`/`perturn_fig`), retire
   `analyze_transcript.py`/`step_kl_slides.py` in favor of `kl_slides.py`, standardize
   filenames.

---

## game-1 follow-ups (2026-07-23): baselines, drift, direction-confound

Prompt check: the yoked/restrict/pca line of runs all build on `qwen32_pca.OPEN_PROMPT`
(rules + win condition only) — **neutral, no coaching**. The coached-prompt problem was
confined to the old `game1sem`/`game1open` prompts.

**Yoked chance floor** (`src/game1_yoked_baselines.py`, output json in each run dir):
pair each yoked game's ghost sequence with the live player of every OTHER yoked game;
same-word-same-round collision = "meeting" with zero interaction.

| condition (cap100) | real met | shuffled-pair null |
|---|---|---|
| reactive | 1.00 (6.0 turns) | 0.017 |
| yoked | 0.50 (29 turns) | **0.058** |

So one-sided prediction of a non-adaptive partner is ~9x above chance, and live
co-adaptation doubles it again and is ~5x faster. This is the behavioural
"shuffled-partner control" from TODO #1 — coupling to *this* partner is real.

**Restricted-game drift is BIMODAL.** In games that meet, the unrestricted player (A)
enters the partner's secret category within ~4-6 turns (city: 0.00→0.59 in-category by
turns 4-6; fruit: 0.19→0.44) and **100% of meeting words are in the category** — A does
all the moving. In games that never meet, A ~never enters the category (~0.05) and
instead falls into morphological perseveration loops (spine/spineless/spiny...;
harmonize/harmonious...; NATO-alphabet cycling): perseveration index 0.79-0.83 vs
0.13-0.17 in met games. The no-meet failure looks like a decoding pathology under the
no-repeat constraint, not (only) a partner-modeling failure. Restriction is prompt-only
— the restricted player occasionally breaks it (one game "met" on 'plumpest').

**Convergence-direction confound** (`src/qwen32_entropy_replay.py` → `qwen32_entropy.json`
in `qwen32_pca_w2v/`): the convergence coordinate correlates with turn (+0.65..+0.83,
survives partialing entropy) **but also with output entropy** (-0.6..-0.76, survives
partialing turn; strongest at deep layers). The direction is substantially entangled
with a confidence axis — any claim about it must control for this.

**Causal steering test — NULL** (`src/qwen32_steer_conv.py`, results pulled to
`runs/game-1/5_mechanistic/steering/`): inject alpha·||h||·d at L32 (both players,
every generated position, matched w2v starts, n=14/alpha):

| alpha | -1.0 | -0.5 | 0 | +0.5 | +1.0 |
|---|---|---|---|---|---|
| met | 0.93 | 1.00 | 0.93 | 1.00 | 1.00 |
| turns-to-meet | 6.3±1.1 | 5.9±1.6 | 4.6±0.7 | 5.2±0.8 | 5.9±1.1 |

The hook is verifiably active (only 15/56 turn-1 picks match alpha=0; divergence grows
with |alpha|; meeting words stay non-degenerate at ±1). It changes WHICH words are said
but NOT convergence speed — if anything both extremes are slightly slower (generic
perturbation noise). Together with the entropy entanglement above: the convergence
direction is a READOUT of game progress/confidence, not a causal driver of consensus
(as tested: single layer L32, last-position injection, |alpha| ≤ 1·||h||).

---

## game-1 follow-ups (2026-07-23 pm): the copy circuit, bridging heads, and why restricted games fail

### Is the partner-patch circuit just induction/copy machinery? (mostly yes)
`src/qwen32_induction_overlap.py` (repeated-random-token induction scores, eager attn):
the top-32 restoration heads average **7x** the all-head induction score; **15/32 are
top-5% induction heads**. `src/qwen32_head_ablate_play.py`: zero-ablating ALL 32 during
live self-play does NOT hurt convergence (none 0.93 met / 4.6 turns; top32 **1.00 / 3.6**;
rand32 0.93 / 4.8) — the circuit the logit-gap metric finds is real but NOT load-bearing
for meeting. (Same lesson as the steering null: readouts here keep failing necessity tests.)

### Bridging heads (`src/qwen32_bridge_heads.py`, `runs/qwen32_bridge_heads.json` on pod)
Cue pairs in game format ("other player said red, you said fruit" -> apple), 12 items the
model actually gets; zero-ablate each head at the answer position; d_bridge vs d_cue =
drop in bridge-word vs cue-word logits. Group means: BOTH halves of the restoration set
are cue-copy-dominant (ind: d_cue .06 vs d_bridge .01; non-ind: .18 vs .06; rand ~0).
But individual heads dissociate:
  * **bridge-specific**: L54H29 (d_bridge .16, d_cue ~0 — purest), L50H54 (.34/.23),
    L56H15 (.29/.09), L60H4, L57H37, L55H4.
  * **pure copy, NOT induction-triggered**: L55H14 (d_cue **1.19** vs d_bridge .11; its
    induction score is ~0) and L61H11 (.49/.15) — a "semantic/contextual copy" flavor the
    random-token induction screen misses.
So the restoration set decomposes: induction heads + non-induction copy heads + a small
bridge-leaning subset. All effects are small/distributed; MLPs untested.

**Full-grid sweep** (`src/qwen32_bridge_allheads.py`, all 64x64 heads, results + heatmap
slideshow in `runs/game-1/5_mechanistic/bridge/`): the STRONGEST bridge heads live
OUTSIDE the restoration set entirely, in a mid-deep band **L41-L46**: L45H43 (d_bridge
1.51), L41H6 (0.68 vs cue 0.03), L41H3 (0.63 vs cue **-0.00**), L42H7, L44H4, L45H6 —
strongly bridge-specific. Two-stage picture: **bridging computed ~L41-46, copy/promotion
machinery later (L52-62)** — and the partner-patch logit-gap metric only ever saw the
latter. Slides: page 1 mean d_bridge, page 2 mean d_cue, page 3 specificity, then one
page per cue-pair item; top-32 restoration heads boxed on every page.

**Bridge-head ablation, TOPICAL starts — FIRST POSITIVE causal-necessity result**
(`runs/game-1/5_mechanistic/topic_ablate_play/`, 16 cue-pair starts red/fruit, king/woman...,
same 3 conditions): turns-to-meet none 1.6 / rand32 1.7 / **top32 3.1 (2x slower)**;
canonical-bridge meets (apple, queen, moon...) none 5/16, rand32 5/16, **top32 2/16**;
turn-1 instant meets 12/16 -> 9/16. red+fruit no longer meets on apple (-> "green"),
king+woman loses queen (-> star@4), tree+fruit degrades to cinnamon@14. Strongest
single associations (butter, lightning, moon, banana) survive — redundant paths — but
compositional cue-combination degrades. So: bridge heads are causally needed for
CUE-COMBINATION convergence (topical regime), while category-drift convergence
(sand/cloud) doesn't need them:

**Bridge-head ablation, sand/cloud starts — robust** (`runs/game-1/5_mechanistic/bridge_ablate_play/`,
top-32 by d_bridge vs none vs rand32, forced sand/cloud, no-repeat, n=16): met 1.00
everywhere; turns 2.6 / 3.1 / 2.8. Meeting words stay in the nature family; mild
qualitative shift under ablation (none meets celestial: moon x5 sun x4; top32 shifts
toward terrain: ocean x3 forest x3 plateau/valley, and one game wanders 10 turns of
biome-listing before meeting). No collapse. Combined with the collision model: the
game is solved by CATEGORY-LEVEL distribution overlap (both players' mass pools in the
shared "nature" category until collision) — it does not route through the cue-pair
bridge computation, so ablating those heads barely matters. Every identified component
(convergence direction, copy circuit, bridge cluster) has now FAILED a causal-necessity
test for convergence: convergence is a bulk distributional phenomenon, not a
single-circuit behavior.

### Why restricted games fail (`src/game1_restrict_fix.py`; outputs MERGED into `runs/game-1/2_restricted_core/qwen32_cap24/` — same cap 24)
Baseline restrict-* (list shown, no-repeat): 0.44 met. Interventions (n=14-16):
| condition | met | turns | reading |
|---|---|---|---|
| nolist (rule kept, list hidden) | 0.21 / 0.29 | 13 / 11 | list HELPS (it's A's category summary) |
| repeatok (no-repeat removed) | 0.57 / 0.71 | **2.5 / 4.8** | fast, but via ECHO (see below) |

* **Exhaustion mechanism confirmed**: A's distribution holds category mass even in failing
  games (0.27-0.31) but 44-64% of its word-mass is stranded on already-used words by
  turn 5+; the resampler (24 tries) then silently emits repeats — **"no-repeat ENFORCED"
  is soft**: 44-51% of baseline plays are exact repeats (mostly in long no-meet games).
  Failing games = morphological perseveration loops (pers. index ~0.8 vs ~0.15 in met games).
* **repeatok deflation**: with repeats legal the ECHO equilibrium returns — most meets are
  on repeated words and many are OUTSIDE B's category (B abandons its secret restriction
  to copy: met on 'clinic', 'scripted', 'boom'...). It rescues convergence by un-fixing
  the game, not by better accommodation.
* Baseline met games remain the genuine accommodation cases: A enters B's category in
  ~4-6 turns and 100% of meeting words are in-category.
* TODO: surface the resampler fallback (log forced repeats); consider forbidding
  4-char-prefix neighbors when resampling.

KL slides for all four conditions: `src/game1_restrict_fix_klcap.py` (replays with the
EXACT per-mode prompts — the stock klcap would show the list/rule these players never
saw) -> `runs/game-1/2_restricted_core/qwen32_cap24/kl/`. All four transcripts have pretty
`.json` twins.

**Slide format v2 (2026-07-24)**: `plot_condition` now emits ONE merged
`<cond>_crossKL.pdf` per condition — page 1 = all games' KL curves, then per-turn
pages — everything color-coded by game outcome (green = converges, ★ meet turn;
red = never). All 7 existing conditions re-rendered; the old separate
`_curve.pdf`/`_perturn.pdf` files were deleted (regenerable from the `_crossKL.json`
with PLOT_ONLY=1).

### Four game variants (`src/game1_variants.py`, `runs/game-1/3_interventions/variants_fewshot_board_softrepeat/`)
n=16, cap 24, same w2v starts (board16: no forced start). All transcripts + .json twins.
| variant | met | turns | what actually happened |
|---|---|---|---|
| fewshot | 1.00 | 3.3 | **answer leakage, not strategy learning**: the prompt's example game ends on "shore" — 8/16 games meet on "shore" AT TURN 1 (+coast x2, beach x2). The example's answer becomes a shared Schelling point. Same lesson as the original coached-prompt bug, in few-shot form. |
| city-coastal (both restricted; B privately prefers coastal/warm) | 0.56 | 11.6 | B's soft preference LEAKS ~half: meets = rome, buenos, dubrovnik, manila x2, atlanta, dubai, cairo, seoul (~5/9 coastal-leaning). Direction of accommodation varies: B sometimes caves (opens miami/nassau, meets on rome), sometimes drags A to the coast (dubrovnik, manila). |
| board16 (4x4 categories, cross-linked) | 1.00 | 3.0 | bounded shared board = easy regime, as theory predicts. Meets are NOT uniform: water 8, air 5 (storm x4), light 3, **earth 0** — a clear salience hierarchy; stone/moss/sand/peak never win. |
| softrepeat ("avoid repeating your immediately previous word unless repetition would produce coordination") | 1.00 | **2.2** | read as permission: the INSIST equilibrium (one player holds its word, the other joins) — 7/16 meets via hold-and-join + most turn-1 meets are echo-of-start. Also shows the classic swap deadlock (painted/paints -> paints/painted) resolved by one side holding. Fastest condition ever measured. |

### MENU-PREFERENCE game (user-designed, `src/game1_menu_pref.py`, `runs/game-1/4_menu_pref/qwen32_10r/`)
Fresh 8-word menu per round (repeats/echo/exhaustion impossible); B has a stable secret
latent (city vs water, alternating games, n=12 each); rounds 1-3 ambiguous (preferred
item = water-city: venice/marseille/rotterdam), rounds 4-10 disambiguate (one dry city
+ one non-city water word per menu, distractor topic rotates). A is neutral; 10 rounds
scored, no early stop.

**Result: city games 8.0/10 matches, water games 2.1/10 — and the asymmetry is A's.**
* B enacts its latent near-perfectly and stably (water-B picks the water item 12/12
  every round from r4; city-B likewise; only r1 water-B misses venice — reads it as
  city). The latent signal is clean.
* City games: A locks onto the city item from r3 and never leaves -> 100% match r3-10.
  But this is largely A's OWN city/place salience prior coinciding with B's latent, not
  demonstrated inference.
* Water games = the real inference test, and **A flunks it**: after r4's unambiguous
  evidence (B: lagoon) A still picks the CITY item round after round (madrid 11/12,
  krakow 12/12, denver 12/12, nairobi 9/12). A switches to water ONCE (r5: tide 10/12,
  MATCHED) then **abandons the winning strategy at r6** (back to krakow 12/12) despite
  the direct reward. Late rounds A drifts to salient distractors (orbit, pride).
* Trajectories are near-deterministic across the 12 seeds (picks 10-12/12 identical) —
  effectively ~1 trajectory + noise; treat n accordingly.
Interpretation: behaviour-level confirmation of the anti-Bayesian gating — A is
PRIOR-DOMINATED; evidence that conflicts with its own salience ranking is discounted
even when following it just paid off. The design cleanly separates "A aligned because
its prior matches B" (city games) from "A inferred B's latent" (water games: no).

**Menu game, 20 rounds + scale** (`qwen32_menu_pref_20r/`, `llama70_menu_pref/` under
`runs/game-1/`; Llama-70B run 8-bit via bitsandbytes — bf16 doesn't fit one H200):
* Qwen32, 20 rounds: city 18.0/20, water **2.1/20**. A's accommodation DECAYS with
  horizon: A→water 0.17 (r4-8) -> 0.06 -> 0.00 -> 0.00 (r17-20). The prior HARDENS —
  16 rounds of unambiguous evidence produce zero learning; the early blip vanishes.
* Llama-3.1-70B: a different failure — LOW on both (city 2.8, water 3.1/20) because
  Llama-A has a different prior: it picks the menu's TOPIC words (memo, sleeve, flour,
  chisel...), not cities (A→city 0.12), while Llama-B follows its preference (0.86/0.68,
  less rigid than Qwen's ~1.0). BUT Llama shows the only GRADUAL LEARNING observed:
  water-game match rises monotonically 0.03 -> 0.10 -> 0.15 -> 0.23 across round
  quartiles. Weak but real evidence accumulation — Qwen updates never, Llama updates
  slowly. Prior strength and updating ability dissociate across model families.
* Qwen2.5-72B: volume cache was INCOMPLETE (12/37 shards) — completing download;
  its run is queued behind cap-200.

**Menu game vs CHANCE (2026-07-24 late)**: observed match == shuffled-pair null
(A from game i vs B from game j, same round) to +0.000 (Qwen32) / +0.01 (Llama70);
Qwen32 water games are AT uniform chance (0.10-0.21 vs 1/8); Llama70 ~chance in both
conditions. Qwen-city's 0.90 is two agents independently walking the same salience
gradient — NO within-game coordination above independence anywhere. CAVEAT: menus are
identical across games and trajectories near-deterministic, so fixed-round-policy and
within-game adaptation are CONFOUNDED — the design needs per-game menu-order
randomization (+ temperature) before the shuffle test can detect contingent coordination.

**Llama-70B open-word** (`runs/game-1/2_restricted_core/llama70/`, 8-bit): reactive 88% @ 4.3
(matches Qwen32) — competent open-word coordinator; restrict-city **31% @ 9.8** (vs
Qwen32 44% @ 5.0). Slower, fewer meets (osaka@7, phoenix/moscow@9, amsterdam/dublin@12).
With the menu result: what wins restrict-city is PRIOR-ALIGNMENT (Qwen's city-salience
happens to overlap the category), not updating ability.

**Soft/compound restriction, Qwen32** (`runs/game-1/2_restricted_core/qwen32_softrestrict_dual/`):
* restrict-watercity (B secretly "cities by the water"): met 38% @ 13.3 — meets are all
  genuine water-cities (geneva x2, athens x3, amsterdam). Slower than plain city (44% @5)
  — the finer latent is reachable but harder.
* **dual-water-city (A secretly water-words, B secretly cities): 0/16.** One-way mirror:
  A NEVER plays a water-city (0/368 turns — reads "words related to water" strictly, so
  city names are illegal-to-A); B plays water-cities at 0.26 — but its restrict-city
  BASE RATE is 0.23, so no drift toward A either. Neither player forms the
  "our categories intersect at water-cities" insight. Hard negative for spontaneous
  intersection-finding. Design note: A's constraint semantics exclude the intersection
  under strict reading — a symmetric-intersectable phrasing ("things associated with
  water, including places") would separate semantics from inference failure.
Llama-70B versions of city-coastal + watercity + dual are queued (first attempt crashed
at load — ValueError now printed with full message; rerun follows the 72B menu run).

**Qwen2.5-72B menu game — THE SCALE RESULT** (`runs/game-1/4_menu_pref/qwen72_20r/`, 8-bit):
city 17.0/20, water **6.6/20** (3x Qwen32's 2.1). Water games' A→water by quartile:
0.00 (r4-8) -> **0.73 (r9-12)** -> 0.58 -> 0.33. The 72B genuinely FLIPS to the
partner's latent after ~5 rounds of evidence — a mid-game flip on structurally
identical menus is history-conditioned by construction (a static salience policy can't
produce it; that's the right evidence here, since the shuffled-pair excess is still
+0.000 — shared histories across games make the shuffle test blind, see caveat above).
The flip PARTIALLY DECAYS late (0.33 by r17-20; relapses to cities/distractors).
Inference-ability ladder on the same game: **Qwen32 never updates (hardens) < Llama70
weak drift (0.03->0.23) < Qwen72 real-but-unstable flip (0.73 peak)**. Within-family
32B->72B jump = scale drives latent-tracking; cross-family differences = prior strength.

**Llama-70B soft/dual + city-coastal** (reruns succeeded): city-coastal **0.00** met
(both-city restriction sinks it — consistent with its weak city prior);
restrict-watercity 25% @ 9.8 (Qwen32: 38%); dual-water-city **0/16** (same one-way
mirror as Qwen32). Dual failure is now cross-family.

**Cap-200 COMPLETE (2026-07-24)** — `runs/game-1/2_restricted_core/qwen32_cap200/` +
`qwen32_restrict_fix_cap200/` (transcripts, twins, merged-KL slides in kl/). Verdict:
**most no-meets are SLOW, not absorbing** — big revision of the cap-24 story:

| condition | cap24 | cap100 | cap200 |
|---|---|---|---|
| reactive | 0.94 | 1.00 | 1.00 |
| yoked | 0.25 | 0.50 | **0.50 (absorbing)** |
| restrict-city | 0.44 | 0.50 | 0.56 (@166) |
| restrict-fruit | 0.44 | 0.56 | **0.75 (@46 mean)** |
| nolist-city | 0.21 (cap24) | — | **0.88** |
| nolist-fruit | 0.29 (cap24) | — | **0.88** |
| repeatok | 0.57/0.71 (cap24) | — | 0.94/1.00 |

* Only YOKED is truly absorbing (0.50 at both 100 and 200): without a live partner,
  trapped stays trapped. With a live partner, escapes keep accumulating.
* The cap-24 "nolist hurts / list-helps" conclusion becomes "the used-list ACCELERATES
  convergence (~5-6 turns vs ~30-100) but is not needed for eventual convergence."
* Escape statistics fit the urn picture: rare stochastic exits from the self-basin,
  amplified by the live partner co-adapting once you finally wander into range.

### Chat-mode steering of the convergence direction (`src/qwen32_steer_chat.py`)
Same d, injected in NORMAL CHAT on 8 "Do you agree?" opinion prompts (L8/32/56,
alpha ±1; `runs/game-1/5_mechanistic/steering/qwen32_steer_chat.json`). L56 shows a
monotonic yes-vs-no logit trend (-0.70 -> +0.31 across alpha -1..+1) and at alpha +0.5
one clear endorsement flip (orange-flowers: "depends on a few things" -> "I think your
choice is a lovely and thoughtful one!") plus warmth shifts in ~3/8 prompts; alpha +1.0
DEGENERATES generation (gibberish), so the +1 logit numbers are contaminated. L8/L32
no coherent behavioral effect. Verdict: the direction carries a weak assent/positivity
flavor at deep layers (consistent with its entropy/confidence entanglement) but is NOT
a robust agreement switch — and it still does nothing for in-game convergence.

### Convergence IS a simple probabilistic process (collision model)
Independent-sampling model: P(meet at turn t) = sum_w pA(w)·pB(w) from the two replayed
next-word dists (top-15, temp-adjusted). Pooled over 1444 turns / 7 conditions:
**AUC 0.90** for predicting the agreement turn; calibration good through mid overlap.
The only systematic residual is TOKENIZATION: every high-overlap non-agreement (29/29)
is a first-token collision that splits at the word level ("se"->seamless/seamstress;
repeatok-fruit g8 sat 5 turns at c~0.97 on "p": proudly vs pomegranate). Legality-
masking used words changes nothing. => No coordination magic in the picks; ALL game
dynamics live in how the two distributions co-evolve. Met games: overlap rises to
~0.2-0.3; failed games: overlap stays ~0.03 (A trapped in morphological tail).
Canonical script: `src/game1_collision_model.py` (no GPU; reads `*_crossKL.json`) ->
`runs/game-1/6_analyses/collision_model/` (calibration+AUC, overlap trajectories met vs
no-meet, per-condition pred-vs-obs). ORGANIZATION RULE (adopted 2026-07-24): analyses
that span multiple runs/conditions get their OWN directory (now `runs/game-1/6_analyses/`),
not a slot inside one run's folder.

### Update DYNAMICS: it is NOT Bayesian updating (`src/game1_update_dynamics.py`)
-> `runs/game-1/6_analyses/update_dynamics/` (3-page pdf + json). Tested three Bayesian
signatures on the per-turn dists (met vs no-meet games, reactive+restrict):
  1. update size ~ partner-word surprisal: r = -0.06 (met) / **-0.21** (no-meet) —
     zero-to-ANTI-proportional. Evidence far outside the current support is DISCOUNTED,
     not integrated (a Bayesian would move most when most surprised).
  2. accommodation (Δmass on partner's word): ~0 / negative — partly BY DESIGN: no-repeat
     makes the partner's word illegal, so its evidence value is its REGION, which the
     first-token measure can't see (embedding-space version = open TODO, needs pod).
  3. contraction: entropy~progress r = -0.19 (met) vs -0.04 (no-meet) — met games
     concentrate, failing games never contract.
  Plus: partner's word is outside A's top-15 on 56% (met) / **84%** (no-meet) of turns —
  failure IS disjoint supports that never merge.
Picture: two LOCAL kernel walks, not global inference — each player's dist drifts within
its own neighborhood and absorbs the partner's word only when already nearby (low
surprisal). Overlapping supports compound (positive feedback -> fast meet); disjoint
supports discount each other's evidence and wander (no-meet). Explains bimodality of
restricted games and why yoked (ghost words from a different game's basin) converges
at half the rate. A Bayes-with-outlier-rejection (robust likelihood) model could fit;
plain Bayes does not.

**Named-model tests** (`src/game1_urn_deffuant.py` + `src/qwen32_word_embed.py`,
`update_dynamics/deffuant_urn.pdf`; word embeddings = mean-pooled input-embedding rows):
  * DEFFUANT (bounded-confidence) at the PICK level: NOT detectable. Raw "fraction of
    gap toward partner closed" looks large (+0.46) but a shuffled-target null is
    IDENTICAL (+0.46 vs random words) — it is centroid drift in an anisotropic
    embedding space (cos distances compressed to 0.87-0.99). Partner-specific excess:
    +0.02 (met) / +0.00 (no-meet). Pick-level assimilation unresolved with these
    embeddings; the gating evidence lives at the DISTRIBUTION level (censoring 84%,
    anti-surprisal coupling), which crude embeddings don't touch.
  * POLYA-URN self-reinforcement: CLEAR, in the relative own-vs-partner comparison
    (same-embedding, so anisotropy cancels). NN-distance of each new word to OWN vs
    PARTNER history: met games balanced (own closer by -0.01 early, +0.05 late);
    no-meet games self-lean early (+0.06) and TRAP progressively (**+0.18 late**; own
    NN-dist collapses 0.83->0.67 while partner stays ~0.85). Failing walks reinforce
    their own neighborhood turn over turn — urn-like trapping, the embedding-space
    face of the perseveration loops.
So: "bounded-confidence" remains an analogy at pick level (needs better embeddings /
distribution-level distances to test properly); "self-reinforcing walk that either
merges early or traps in its own basin" is now measured.

---

## LATENT-CONSTRAINT PROBE (2026-07-24) — knowledge/behaviour dissociation

`src/qwen32_constraint_probe.py` (pod capture: replay yoked reactive/restrict runs,
A's residual stream, every layer/turn, 580 pts) + `src/qwen32_constraint_probe_fit.py`
(local LOGO logistic probes) -> `runs/game-1/6_analyses/constraint_probe/` (pdf+json+npz).

Can the UNRESTRICTED player's activations reveal WHICH secret constraint its partner
is under (city vs fruit)? **Yes — 92-93% LOGO accuracy from L8 through L64.** Time
course: chance at turns 1-2 (only the uninformative forced start seen), **~97% at
turns 3-4, ~100% from turn 5** — i.e. ~2 partner words suffice, and the representation
is at ceiling BEFORE behavioural drift peaks (~0.6 in-category at t4-6, met games).
Control: restricted-vs-reactive 83-86%. (L0 reads 0.06 = below-chance overfit artifact
on the embedding layer; ignore.)

**Key split: met 1.000 vs no-meet 0.997 (turn>=3, L32).** A knows the partner's
category EQUALLY PERFECTLY in the games where it never enters it. With the earlier
findings this gives a three-layer dissociation in failing games:
  representation ~100% decodable -> distribution ~0.3 category mass -> behaviour ~0.05
  in-category picks.
Failure is NOT ignorance — it is downstream of representation, in the sampling /
no-repeat / exhaustion dynamics (the bounded-confidence walk getting trapped in its
own basin). CAVEAT: the probe shows the category info is present (plausibly partly
plain topic-context encoding of B's words); it does not yet separate "B says cities"
from "B is REQUIRED to say cities" — that needs reactive games with topically-clustered
partners as a control.

---

## STUCK-PRIOR probe (2026-07-25) — no early signature, in KL or activations

"Stuck priors" = Qwen32 restricted games that never meet and perseverate morphologically
(cleanly bimodal: pers 0.58-0.98 vs ~0.00 in met games; 15 stuck vs 14 fast-met games,
late-met excluded). `src/qwen32_stuck_probe.py` -> `runs/game-1/6_analyses/stuck_probe/`.

* **KL**: cross-player KL separates stuck vs fast-met only weakly overall (9.6 vs 7.3,
  game AUC 0.71) and is AT CHANCE on turns 1-3 (AUC 0.55). Not definitive, not early.
* **Linear probe**: game-level early prediction (mean acts t1-3, one vote/game, LOGO +
  shared-start-pair holdout) is AT/BELOW CHANCE at every layer (balanced 0.28-0.55,
  AUC 0.23-0.52). METHODOLOGY TRAP documented in the script: per-turn pooled accuracy
  looks great late (0.99) but is pure survival bias — fast games have ended, and a
  provably CONSTANT layer-0 feature "decodes" 0.81/0.99. Don't trust per-turn pooled
  numbers on outcome-dependent-length games.
* **Per-(layer, turn) heatmap** (`src/qwen32_stuck_probe_heatmap.py` ->
  `stuck_probe/stuck_probe_heatmap.pdf|.npz`): LOGO R^2 predicting final perseveration
  from the activation at each (layer, turn), survival-bias-masked, + AUC page (binary,
  maskable only through turn 6). The map is COLD everywhere: best cell R^2 = 0.26
  (layer 3, turn 2), best-per-turn 0.04-0.26 with no depth- or time-coherent region —
  well within max-of-1495-cells noise at n~30. Even late cells (within-stuck gradation
  only) max ~0.25.
* Verdict: stuckness is NOT a pre-set internal mode detectable at game start — and per
  the heatmap, not cleanly linearly readable at ANY single (layer, turn) — it EMERGES
  through the no-repeat/perseveration feedback loop (as the urn model predicts):
  trajectory property, not state property. (Caveats: n=29-32 games, linear probes only,
  first-24-turn replay activations.)

---

## INFORMED + SCRATCHPAD aids (2026-07-25) — the dissociation survives verbalization

`src/game1_restrict_aids.py` -> `runs/game-1/3_interventions/aids_informed_scratch/` (transcripts
with A's per-turn scratchpad notes logged; baselines: restrict-* 0.44 met).

| condition | met | turns | |
|---|---|---|---|
| informed-city ("partner is restricted to SOME category") | 0.44 | 7.4 | no change in rate (different games met, slower) |
| informed-fruit | **0.81** | 6.5 | goal-licensing DOES partially unlock |
| scratch-city (persistent private scratchpad) | 0.38 | 11.3 | no rescue; possibly entrenches |
| scratch-fruit | 0.50 | 5.0 | ~no change |

Scratchpad notes give a three-part taxonomy of failure, all with knowledge present:
1. **Log-but-self-theorize** (g2): note correctly lists Geneva/Madrid/Paris/Rome/... yet
   the "pattern idea" is about A's OWN verb-tense chain; picks cycle
   ended/beginning/began/starting for 13 more turns after the note says "cities".
2. **Convert-but-collide-never** (g9): A fully switches to cities after noting the
   pattern (15 straight city picks) and STILL fails — the popular cities are already
   burned by no-repeat, so both wander the category tail without collision. Exhaustion
   caps even successful inference.
3. **Name-it-and-persist** (g11): the note reads "Opponent's pattern seems to be
   choosing city names. My words are all derived from 'directed' by rearranging
   letters... I must continue this strategy." EXPLICIT metacognition of both the
   partner's latent AND its own pathology — followed by doubling down (drift redit rid
   dried redi dire...).
In scratch-city, early verbalizers (note says "cities" by turn 2-3) all met; late
verbalizers mostly failed — but verbalization is NOT sufficient (g2, g11). The
knowledge-behaviour dissociation persists straight through the language channel.

---

## g11 BRANCH SURGERY (2026-07-25, n=3 games, one model — case study, not a result)

`src/g11_branches.py` -> `runs/game-1/3_interventions/g11_branches/`. Branch scratch-city games
11/2/14 at turn 13; quiz + 4 continuations each.

* **Quiz at branch state**: r11 and r2 name ALL 4 held-out cities (0 non-cities) as
  words the partner could say — constraint intact and declaratively usable mid-stuck.
  r2's expected-next-partner-word is "start" — ITS OWN loop word: classification
  intact, prediction contaminated by self-history. (r14 quiz sloppy: names non-cities too.)
* **note_replaced (pin corrected plan into the scratchpad slot): releases 3/3**
  (seoul moscow beijing -> MET@8; amsterdam prague istanbul -> MET@8; full city stream).
* **instruct ("say an unused city"): 1-2/3** — r14 MET@1, r2 INTERLEAVES compliance
  with the loop (brussels/starting/sydney/starting...), r11 ignores it entirely.
* **self_removed (own words hidden from history): 0/3 release** — loops reconstitute.
  CAVEAT: the shown used-list still contains the own loop words (partial ablation);
  but note_replaced sees the same used-list and releases, so visible loop words alone
  don't force the loop.
* Reading (tentative): the operative lever is the PLAN/NOTE SLOT, not raw self-word
  visibility — against the "only self-history interventions release" hypothesis in its
  simple form. Loop-exit flags have metric quirks (r11 control 'True' is a window
  artifact; picks still anagram-family). Systematic version = the running sweep.

---

## STUCK-PRIOR REPRODUCTION SWEEP + RELEASE TEST (2026-07-25/08-01)

`src/game1_stuck_repro.py` -> `runs/game-1/3_interventions/stuck_repro_sweep/` (+ pod). 6 categories x 2 fully
reworded prompt pairs x 8 starts, cap 24, online stuck detection (>=3 of last 5 own
words share a 4-prefix with an earlier own word, from t8), snapshot + branch.

**Qwen32**: stuckness REPRODUCES across most categories and both phrasings, with strong
heterogeneity — stuck_frac: city .38-.50, sport .25-.62, fruit .50 (p0) vs **.00 (p1)**,
animal .12-.25, vegetable .12-.25, **color .00 both** (met .88-1.00; small shared vocab
=> game ends before loops form). Phrasing matters (fruit p0 vs p1).
**Llama-70B (8-bit)**: **stuck 0.00 in all 12 cells, pers ~.01** — the perseveration
regime does not occur; its failures are wandering (met city .00-.25 but fruit 1.00 —
difficulty ranking inverted vs Qwen). Stuck-priors = family-specific here.
**Release test (Qwen32, n=24 snapshots; control met 0.00)**: self_ablate loop_exit .08
met .17; partner_ablate loop_exit .33 met .08. NEITHER history surgery releases;
"only self-history releases" NOT supported (caveat: shown used-list still contains own
words in both modes). Only the g11 note_replaced lever has released reliably so far.

---

## STRUCTURED-PROMPT (memory-log/JSON) variant (2026-08-01, user-designed)

`src/game1_structured.py` -> `runs/game-1/3_interventions/structured_format/`. Same restricted
game, concise bulleted memory-log prompt (Round-k memories rows, not-allowed list,
JSON-only output with '{"word": "' prefill). MODE=scratch adds a private note field in
the same JSON. Qwen32, city+fruit, n=16, cap 24. Baselines: prose 0.44/0.44 met.

| format | met city/fruit | pers |
|---|---|---|
| structured | **0.00** / 0.19 | 0.07 / 0.03 |
| structured + scratchpad | 0.00 / 0.12 | 0.04 / 0.06 |

* Perseveration is ABOLISHED (no morphological loops; one soft semantic loop) — and so
  is coordination: A drifts serenely through its own category (echo sun river ocean...)
  for 23 turns while B lays down 23 cities; near-identical trajectories across seeds.
* Scratch notes in this format are pure novelty bookkeeping ("need a fresh unused
  word"); only 3/16 games' notes ever mention cities (prose scratchpad: 13/16).
* Reading: the conversational prose format is the ASSOCIATIVE COUPLING CHANNEL —
  it carries both partner-drift (which produces the 0.44 meets) and the morphological
  momentum (which produces the loops). The memory-log format severs the channel:
  partner words become inert data rows; both pathology and coordination die together.
  So "fixes the stuck-prior issue" only in the sense that it removes the dynamics the
  stuck prior lives in. (n=16/cell, one model, one structured phrasing — same caveats
  as everything else.)

---

## WAVE-1 RIGOR PROGRAM (2026-08-02): survival + population dissociation (+ strict rerun in flight)

**Survival analysis** (`src/game1_survival.py` -> `6_analyses/stuck_probe/survival.json`,
n=32 deduped cap-200 trajectories): onset (first morphological recurrence) in 62% of
games at turns 3-30; a first recurrence is NOT lock-in — escape hazard is 0.14-0.33/turn
for loop-turns 1-4 then COLLAPSES to ~0 (metastability threshold ~4 family-turns;
censored durations to 197). P(meet | never-onset) = **1.00**; P(meet | escaped) = 0.40.

**Population knowledge/action dissociation** (`src/dissociation_branches.py` ->
`3_interventions/dissoc_branches/`, 24 reconstructed sweep snapshots, forced choice over
identical 8-candidate sets, first-token readout):
  P(knowledge) 0.98 | P(prediction) 0.88 | P(action) 0.43
  **dissociation = +0.56, 95% CI [0.41, 0.69], positive in 24/24 snapshots.**
Per category: city +0.83, animal +0.81, vegetable +0.68, fruit +0.34, sport +0.24.
Untouched-continuation harness check: loop_exit 0.04, met 0.08 (matches sweep controls —
the quiz frames don't alter behavior; branches are independent). NOTE: forced-choice
prediction is CLEAN (0.88) — the g11 "expects its own loop word" anecdote does not
generalize under forced choice; the deficit is specifically in the ACTION frame.

**Strict-constraint cap-100 rerun (game1_strict.py -> `2_restricted_core/qwen32_strict_cap100/`)
— THE ARTIFACT OBJECTION IS SUBSTANTIALLY VALIDATED in refined form:**
* Loop FORMATION is real and endogenous: onset 0.46-0.62 under guaranteed novelty
  (vs 0.62 baseline) — not created by the constraint handler.
* But PERMANENT TRAPPING was a constraint-handler interaction: with strict enforcement
  every loop escapes (escape|onset 1.00, zero censored, durations mostly 1-3 vs
  censored-to-197 before), met rises to 0.58/0.71 (from 0.50/0.56).
* The deep endogenous fact: the PROPOSAL DISTRIBUTION collapses — A hits the 64-resample
  ceiling on 23% of turns (mean 22 resamples/turn), and 67% of loop-exit words were
  delivered by the constrained-argmax fallback, not sampled by the model. So the visible
  pathology (absorbing loops vs transient loops) is CO-PRODUCED by the model's collapsed
  proposals and whichever constraint-handler resolves them. "Stuck priors" should be
  restated as: the sampling distribution collapses onto used/loop words; the surface
  behavior is handler-dependent.
* CAVEAT: the per-turn cat/self/used-mass telemetry from this run is UNRELIABLE — the
  readout position captured a near-deterministic formatting token (entropy ~0.1, all
  masses ~0.003); needs a re-capture measuring at the true word position. Item 5.4
  remains open.
**CLAIM TAXONOMY (adopted 2026-08-02, after the strict control):**
* NOT supported: "the model autonomously repeats forever" (absorbing behavior was
  handler-dependent).
* SUPPORTED: "the model's action distribution becomes heavily concentrated on an
  obsolete self-generated policy" (22 resamples/turn mean; 23% of turns exhaust 64).
* RESOLVED (proposal telemetry, 725 states x 64 MC proposals,
  `2_restricted_core/qwen32_strict_cap100/proposal_telemetry.jsonl`): the latent
  attractor IS metastable — onset-aligned invalid proposal mass climbs 0.42 (pre) ->
  0.70 -> 0.81 -> **0.87 (+9..+15)** and stays there WHILE the strict handler is
  forcibly emitting novel words. "Every loop escapes in 1-3 turns" was behaviorally
  true, mechanistically misleading. Clean discriminators at matched turns: self_family
  proposal mass 0.07-0.14 (onset games) vs 0.00 (never-onset, all bins); category mass
  collapses to ~0.02 in-loop vs 0.19-0.44 in never-onset games — the quantitative form
  of the knowledge-proposal-control dissociation (forced-choice knowledge stays 0.98
  while category-directed PROPOSALS go to ~0). Collapse predicts burden and outcome:
  corr(invalid mass, resamples/turn)=0.42; game-level mean invalid mass met 0.20 vs
  no-meet 0.44, AUC 0.78. (Caveat: used-mass partly scales mechanically with game
  length; self_family and category columns are the load-bearing comparisons.)

**TRANSPLANT + PARTNER-REPLAY (src/transplant_replay.py ->
`3_interventions/transplant_replay/`, n=8 branches/condition): CAUSAL POSITIVE.**
Identical replayed partner evidence; only A's 3 prior own-words vary:
| seed | onset<=14 | family turns /14 | planted-family proposal mass @ branch (K=64 MC) |
|---|---|---|---|
| loop_seed | **1.00** | **9.1** | **0.57** |
| met_seed | 0.62 | 3.6 | 0.00 |
| neutral_seed | 0.62 | 2.8 | 0.01 |
Three same-family words in own-history instantly move ~57% of the proposal
distribution into that family and produce onset in 8/8 — under identical partner
input. Self-history causally induces the proposal collapse. (n=8/cond; control
conditions show a ~0.62 base onset rate — the loop_seed effect is the delta.)

Staged next: proposal telemetry sweep (running), note factorial (6 cells) on strict
harness; then onset-perturbation, minimal-seed search, cross-model (Qwen sizes+Llama).

---

## INTERVENTION LADDER (2026-08-02, pod): partner-modeling instructions, r0-r4

Prompt additions to A only; restricted city game, resample-24 handler, n=12/rung,
K=64 MC probe at turn 8. Probe numbers are cap-independent (same seeds -> identical
first 8 turns); met at cap 40 and cap 100 (r0/r2 rerun at cap 100 -> `runs/ladder_cap100`):

| rung | addition | met(40) | met(100) | onset | probe category-mass | probe family-mass |
|---|---|---|---|---|---|---|
| r0_none | — | 0.42 | 0.50 | 0.75 | 0.164 | 0.013 |
| r1_think | "think carefully" | 0.50 | — | 0.50 | 0.338 | 0.033 |
| r2_predict | "silently predict what the other player will say next" | 0.67 | **0.83** | 0.58 | 0.319 | 0.047 |
| r3_rule | "identify the rule behind the other player's choices" | 0.33 | — | 0.67 | **0.008** | **0.102** |
| r4_ruleact | rule + act on it | 0.42 | — | 0.67 | 0.014 | 0.016 |

* r1/r2 HELP (category proposal mass doubles, r2 met 0.83 at cap 100 vs 0.50 baseline).
* r3/r4 BACKFIRE: explicit rule-identification collapses category mass to ~0.01 and
  r3 TRIPLES self-family mass — instructing meta-reasoning about "the rule" apparently
  redirects attention to A's own history. (Survivorship caveat: probes are on games
  alive at t8.)

---

## TWO-PROCESS STATE-SPACE MODEL (2026-08-03, local fit -> `6_analyses/state_space/`)

User-specified model fit to the strict-city MC telemetry (24 games, 401 states,
K=64 multinomial emissions over category / word-family / exact-repeat / other):
  b_{t+1} = b_t + eta_b*E_t (partner city evidence);  s_{t+1} = rho*s_t + eta_s*sim_t
  pi_t = softmax(alpha_b*b - alpha_s*s + c_cat + u_g, beta_s*s + c_stem,
                 gamma_s*s + c_used, 0);  u_g ~ N(0,1) penalized (hierarchical).

**Two-process model DECISIVELY beats the 1-D belief baseline**: 6-fold game-level CV
held-out NLL 24,958 vs 27,747 (delta ~2,789 nats); fitted rho = **0.884** (attractor
half-life ~5.6 turns — quantitative hysteresis), eta_s = 0.46, alpha_s = 1.31
(category suppression), beta_s = 2.85, gamma_s = 2.25.

**Identifiability finding (important):** eta_b -> 0 — belief growth is UNIDENTIFIED
from action-channel emissions alone, because E_t ~ 1 every turn and observed category
mass FALLS in onset games. The external anchor for "b_t rises normally" is the
forced-choice knowledge channel (0.98 at loop snapshots). Natural next step: joint
fit with knowledge-frame observations as a second emission channel of b_t. lambda
(intervention decay) also unidentifiable here (I_t = 0 in these runs; ladder data
could identify it).

**Mixed-effects AR** (logit p_family_{t+1} ~ p_family_t + sim(a_t) + p_cat_t + (1|game)):
beta1(p_family) = **+3.86** (p<1e-12), beta2(sim) = **+1.36** (p<1e-10),
beta3(p_cat) = -0.65 (p=0.10), group var 0.15. beta4 (self-attributed) is not
identifiable within games (all actions self-attributed) — its causal anchor is the
seed-matrix attribution contrast (list 0.53 / self 0.24 / partner 0.13 / other 0.09)
and the transplant (sim manipulated causally: 0.57 vs 0.00/0.01). Large positive
beta1 + causal beta2 + dose-4 hysteresis jointly justify the attractor language.

Files: `6_analyses/state_space/{fit.json, state_space_fit.pdf}`;
fitting script kept in session scratchpad (fit_state_space.py) — move into src/ if
it becomes load-bearing.

## ACTIVATION-PATCHING RESULTS (2026-08-03, pod wave6.5 -> `5_mechanistic/patching/`)

**Prediction->action frame patching (24 dissociation snapshots, same 8-candidate
listing both frames):** patched action-frame P(category) stays at the 0.43 action
baseline through L0-40, rises sharply L44-52 (0.44 -> 0.67 -> 0.85) and PLATEAUS AT
0.85-0.88 = THE FULL PREDICTION-FRAME PREFERENCE for L48-62 (drop at L63 readout).
Late-layer transfer per the pre-registered key: **the knowledge-to-policy routing
failure is imposed LATE** — the action frame computes the same category-favoring
representation as the prediction frame until ~L48; frame-dependent suppression is
applied in the last ~third of the stack.

**Paired transplant patching (loop-seed vs neutral-seed, identical replayed B;
Delta = logsumexp(stem) - logsumexp(category) first-token logits):** baselines
separate cleanly (d_loop +1.0..+2.3, d_ctrl -2.1..-7.0, all 6 pairs).
* FINAL-position layer sweep: recovery ~0 until L48, 0.37@56, 0.7-0.95@L60-62
  (L63 overshoot 2.5 = trivial full-residual replacement — excluded from claims).
* SEED-SPAN sweep (patching the 3 attributed self-words' token positions):
  **peaks MID-STACK — best 0.95 @ L46** (0.85@L40, 0.84@L48), falling to ~0 by L58+.
Information-flow picture: self-history identity is carried by seed-token residuals
and readable/removable there up to ~L46, is routed to the final position over
L48-62, where it suppresses category choice. Attn-vs-MLP single-sublayer patches at
L60-63 are all small (|rec| <= 0.11) — the effect is accumulated residual, not one
sublayer. Caveats: span patches mean-pool when donor token lengths differ (early-layer
span numbers incl. a spurious -0.63@L0 are low-confidence); cross-prompt final-position
patching is standard but positions beyond final are not aligned across frames.

**Convergent with the behavioral M3 fit (below): alpha_b -> 0** — belief anchored by
the knowledge channel exists but is not coupled into the action logit; the patching
localizes WHERE that decoupling happens (late layers, final position).

## SEMANTIC RUN DETECTOR (2026-08-03, local -> `6_analyses/semantic_runs/`)

Embedding-based (text-embedding-3-small; tau = 95th pct of cross-category sims,
calibrated on the 30-category corpus). Runs = consecutive A-words with
cos(word, run centroid) >= tau, split by alignment with the EARLY-B centroid:

| model | aligned>=4 | SELF-DRIFT>=4 | drift max | prefix>=4 |
|---|---|---|---|---|
| Qwen3-32B strict | 0.61 | 0.14 | 77 | 0.26 |
| Llama-70B cap100 | 0.67 | 0.08 | 64 | 0.16 |
| gpt-4o-mini | 0.65 | **0.22** | 29 | 0.19 |
| gpt-4.1 | 0.88 | 0.01 | 6 | 0.02 |
| gpt-5 | 0.78 | 0.04 | 6 | 0.11 |
| Haiku 4.5 | 0.90 | 0.00 | 0 | 0.04 |
| Sonnet 4.5 | 0.43 | 0.00 | 2 | 0.00 |

* Semantic runs are the dominant register everywhere (0.43-0.90 of turns) — mostly
  partner-ALIGNED (accommodation). Self-drift runs (perseveration candidate) are
  concentrated in Qwen32 (0.14, incl. the invented -est/-lyst suffix families the
  prefix detector misses), Llama (0.08), and surprisingly gpt-4o-mini (0.22 — incl.
  repeat-locks and a 29-word run in its no-meet game).
* CAVEATS: "aligned" is coarse (gpt-4.1's geographic wandering counts as aligned —
  its perseveration is semantically NEAR cities without being cities; the earlier
  cap-40 landform runs read as aligned here); drift runs include repeat-locks
  (identical words are trivially self-similar); late-game B degradation handled by
  anchoring on B's first 8 words.
* Files: semantic_runs.{json,pdf}; embedding cache in session scratchpad.

## MECH4 — IN-GAME ISOLATION (2026-08-03, user redesign -> `5_mechanistic/mech4/`)

Same question as mech3 but inside the REAL game framework (4 replayed city B streams,
3 planted words, K=64 novel-family proposal mass). Corrects/extends mech3:

**A. Placement & recency:** answer-slot 0.50 dominates, but NOT exclusively:
used-list-ONLY 0.117, other-player slots 0.109, mixed 0.074 (vs mech3's exact zeros
for note/non-target — the game context DOES carry weak lexical/positional priming;
the used-list explains part of the old seed-matrix "list" cell). Used-list membership
is irrelevant for the A-slot effect itself (nolist 0.47 ~ 0.50); triple order
irrelevant (shuffled 0.50). **RECENCY IS DECISIVE: 0.50 at gap 0 -> 0.00 at gap 2
(0.035/0.020 at 4/6).** The passive trace of planted words dies within ~2 rounds —
so the live-game persistence (state-space rho ~ 0.89, dose-4 hysteresis) must be
SELF-SUSTAINED (the loop re-feeds itself each turn), not a slowly decaying memory.
Reconciles the sharp mech4 decay with the behavioral hysteresis.

**B. Format:** capture survives every rendering of the history — youfirst 0.55 >=
round_sent 0.50 ~ semicolon 0.49 > prose 0.40 > colon-lines 0.30 > own-words-as-list
0.23. ~2.4x modulation, never eliminated; NOT driven by how list-like the format is
(explicit own-word list is LOWEST; prose still 0.40). The binding is to the ROLE
("these are the answering player's words"), not to visual series structure.

**C. Person/attribution:** second person ("you said...", My word:) = 0.496 and
third-person spectator advice ("What should Player 1 say?") = 0.496 — IDENTICAL.
Being the player is irrelevant; computing a word-choice FOR the player whose words
they are is what matters. (witness_complete "Player 1 said:" shows fam_mass 0 but
raw family prob 0.254 — an artifact: the model recaps the transcript verbatim
(past-tense reading); it is a recall task, not a choice task. Excluded.)

**Combined mech3+mech4 story: capture = role-bound series continuation — the words
of the player whose next move is being computed get continued, regardless of WHO is
computing (self or spectator) and largely regardless of format; self-attribution
adds nothing (mech4) or a small brake under novelty instructions (mech3); pure
lexical exposure contributes only weakly and only from game-relevant positions
(used-list), and the trace decays within ~2 rounds unless behavior re-feeds it.**
Files: mech4.json, mech4_figures.pdf; script src/mech4_game_isolation.py.

## MECH4b/4c — FORMAT CONSISTENCY & POSITION (2026-08-03 -> `5_mechanistic/mech4/`)

mech4b (per-round format varies WITHIN one prompt; 6 formats f0-f5, city streams,
gap 0): within-prompt inconsistency destroys capture — fixed_f0 0.480, fixed_f4
0.281, alt_f0f1 0.168, all_diff 0.012, scramble_a [f2,f5,f3] 0.016 — EXCEPT
scramble_b [f4,f0,f2] = 0.434. mech4c (change exactly ONE round's format;
scrambles rerun) resolves it: **round 1's format is irrelevant or even helps
(r1→f4 0.664, r1→f2 0.629 vs all-consistent 0.492); rounds 2-3 are load-bearing
(r2 0.117/0.168, r3 0.074/0.238)**. Scrambles replicate (a 0.059, b 0.383):
scramble_b survives because its rounds 2-3 are two clean same-word-order formats.
So the "consistency" requirement is really: **the series template is built from
the LAST ~2 rounds' rendering** — matches the mech4 recency cliff (gap 2 -> 0).
Partially revives surface-structural H2a, but localized to the recent window.
Files: mech4b.json (pod runs/mech4b_scramble), mech4c.json, mech45_figures.pdf;
scripts src/mech4b_scramble.py, src/mech4c_position.py.

## MECH5 — DIRECT ROLE-SWAP TEST OF H2b (2026-08-03 -> `5_mechanistic/mech4/`)

Decisive design: fully third-person history ("Round k: Player 1 said A, Player 2
said B."), planted triple in one player's column, history BYTE-IDENTICAL across
conditions; only the final sentence varies ("What word do you think Player N
should say next?" -> prefill "Player N should say:"). 2x2 owner x asked, x
{planted, base}, 4 streams, K=64. **H2b's prediction (capture only when asked ==
owner) FAILED, informatively: capture is HIGHER off-diagonal** — ask the owner:
0.309/0.594; ask the OTHER player: 0.930/0.992 (all bases 0.000). In the witness
frame the model does BEST-RESPONSE COORDINATION: "Player 1 is producing
plan-words, so Player 2 should say a plan-word to meet them" — rational given
the win-by-matching rule. Interpretation with mech4's first-person slot contrast
(own-column 0.50 vs other-column 0.11): **in first-person play the series binds
to the self column; asked from outside, the model switches to modeling the
partner. The in-game pathology is precisely the failure to apply, in first
person, the partner-modeling it demonstrably has.** Caveat: the witness frame
cannot separate role-binding from coordination reasoning — a first-person
role-handoff variant ("Now answer as Player 2") is the follow-up if needed.
Files: mech5.json, mech45_figures.pdf; script src/mech5_roleswap.py.

## ORACLE DISCLOSURE (2026-08-04 -> `3_interventions/oracle/`) — instruction WINS

`src/game1_oracle.py`, new pod .134:10652, n=12/rung, cap 100, seeds = ladder r0.
r5_oracle appends to A's prompt: "The other player is only ever saying names of
cities. To win, you should also say the name of a city." r6_scratch_inject forces
A's scratchpad content verbatim ("My scratchpad: The other player only says city
names. I will say the name of a city this round." -> "My word:").
**Both rungs: met 12/12 = 1.00, near-instantly** (r5 meet turns median 2.5, all
<=6; r6 median 3, two stragglers at 10 whose t8 probes still showed 0.59 category
mass) vs r0 rerun 0.50 (meets [2,5,10,16,22,65]; probe cat 0.164 — exact baseline
replication). played-cat 0.94/0.91 vs 0.11. **Capture does NOT override explicit
instruction — externally-asserted knowledge is consumed immediately, whether
framed as prompt fact or as the model's own scratchpad note. Combined with
forced-plan (self-GENERATED diagnosis 73% correct -> 15% city plans), the failure
is localized to consuming SELF-DERIVED evidence: the same content obeys perfectly
when its source is external.** (Injected-note framing = self-attributed text, so
"attribution to self" per se is not the blocker either — generation is.)

## THINKING GAME (2026-08-04 -> `2_restricted_core/game1_think/`)

`src/game1_think.py`: standard asymmetric game (A free, B city-restricted), BOTH
players Qwen3-32B native thinking (enable_thinking=True, word parsed after
</think>, max 1400 think tokens, 8 retries). n=8, cap 40, same starts.
**met 7/8 = 0.88 (turns [3,4,5,6,10,20,22]) vs no-think 0.50** — thinking largely
rescues convergence. Mechanism visible in traces: A mentions cities in 55/109
think turns and at meet turns explicitly enumerates B's used cities and picks a
novel one ("...paris, prague, rome... so 'dubai' is not in there... my next word
is 'dubai'") — the partner-modeling that no-think first-person play never does.
played-cat overall only 0.13: A does not mass-convert; it stays exploratory then
executes a deliberate one-shot meet. Caveats: a few turns emit NO answer word
(think budget exhausted after 8 retries -> empty A word logged, e.g. roll 0
t10/t22); n=8, cap 40.

## FORCED-PLAN SCRATCHPAD (2026-08-03, fork session -> `3_interventions/forced_plan/`)

`src/game1_forced_plan.py`, 16 games/arm vs city-restricted B, Qwen32. Arms: free
(old scratchpad replication), plan (template: "- What the other player seems to be
doing: ..." then "- The exact word I will say this round: ..."), plan_remind
(commitment restated right before the answer), plan_exec (planned word played
verbatim, no second generation). **Result — the knowing-doing gap is INSIDE word
selection, two lines apart in the same scratchpad:** the rule line names cities on
70-76% of turns, but the planned word is a city on only 13-15% of turns; the plan
is then kept 97-100%, and when the plan IS a city it is played 97-100%. plan_exec
changes nothing (met 0.31 = plan = plan_remind), so there is no plan-vs-execution
slip to fix — the model faithfully executes bad plans. Forcing planning HURT
meeting (met 0.31 all plan arms vs 0.50 free; met games are faster, 3.8 vs 9.6
turns — commit-early-or-never). Diagnosis (correct, verbalized) fails to feed word
choice even when both live in the model's own just-written note.

## MECH6/MECH7 — FRAME DEPENDENCE + WORD DUMP (2026-08-03, fork -> `5_mechanistic/mech4/`)

mech6 = first-person version of mech5: identical third-person history, final line
" You are Player N. It is time to say your next word." + "My word:". **The mech5
partner-matching collapses in first person** (off-diagonal 0.93/0.99 -> 0.15/0.44)
and own-series continuation is ALSO weak (diagonals 0.13/0.12 vs 0.50 in mech4
where history says "you said") — so (a) partner-matching is a witness/advice-frame
computation, and (b) self-binding needs the literal second-person "you said"
labeling in the history, not just an assigned role. The 0.44 cell (role P1, owner
P2) tracks recency: P2's words are line-final in each round sentence. mech7
(word-level dump of all mech5/mech6 cells): the non-family mass is NOT cities —
it is morphological derivations of OTHER used words (sealed->sealing/sealers,
plank, painter/paints, prompt, modular). **Family capture is one instance of a
general used-word-derivation attractor**; the model almost never proposes the
partner's category unprompted even while advising the player facing a city-namer.
Files: mech6.json, mech7.json; scripts src/mech6_firstperson.py, src/mech7_worddump.py.

## CATEGORY->STUCKNESS PROBE, COMPLETE (2026-08-03/05 -> `6_analyses/catstuck/`)

30-category dataset (2106 states, 358 with MC proposal profiles); dual-form fit
of all 64 layers completed on pod .134:10652 (src/fit_catstuck_pod.py).
Leave-category-out R² (shuffled controls ~0): **famrun peak L53 = 0.522;
fam_mass peak L52 = 0.299; cat_mass peak L46 = 0.253.** All three targets
generalize ACROSS held-out semantic categories, and all three peak in the same
late window as everything else (patching routing L48-62, probes-v2 family L50 /
category L56) — the stuckness readout is category-general and late-stack; the
early-layer famrun signal (L17 0.415 from the partial fit) is real but the
late peak is higher. Files: catstuck_probes_v2.json, catstuck_curves.pdf,
catstuck_transcript.jsonl.

## CONTINUOUS PROBES v2 (2026-08-03, local dual-form fit -> `6_analyses/probes_v2.json`)

Nested alpha-grid ridge (grid 10^1..10^5 selected inside each training fold),
game-held-out CV, dual-form (kernel) solve on resid_cache.npz (n=401 states,
d=5120, 64 layers). Out-of-fold R^2: **family-mass log-odds 0.359 @ L50;
category log-odds 0.297 @ L56; action-prediction gap 0.043 @ L59 (n=24,
unreadable)**. Shuffled-target controls ~0/negative. Family readable slightly
EARLIER than category; both peaks inside the L46-62 window patching identified
(encoding <=L46, routing L48-62) — third method, same localization. Figure:
`6_analyses/probe_curves.pdf`. First pass (fixed alpha=10) was all-negative —
under-regularized at n<<d; the alpha grid is essential.

## THREE-MECHANISM SEPARATION — RESULTS (2026-08-03 -> `5_mechanistic/mech3/`)

**Pooled preregistered contrasts (MC family mass, K=32, novel words only):**
Δ_lexical = **0.000** · Δ_structural = **+0.577** · Δ_attribution = **-0.102**
(negative!) · interaction = -0.102. Per family (target-slot self/other cell means):
morph 0.04/0.27, charonly 0.07/0.25, synthetic 1.00/1.00, semantic 1.00/0.99.

1. **Zero pure lexical priming.** A planted-word triple in a neutral note yields exactly 0
   family continuations; stems in the NON-target slots also yield exactly 0 (no
   cross-slot leakage). The seed-matrix "bare list 0.53" result is therefore NOT
   note-priming — in that cell the seeds also entered the used-list adjacent to
   "My word:" (a positional/structural site). FOLLOW-UP: add a used-list placement
   cell to nail this.
2. **Role-structural induction dominates.** Stems occupying the answer-slot series
   drive continuation; at ceiling (~1.0) for series with a transparent generative
   rule (synthetic blorf-inflections, semantic music words) regardless of ownership.
3. **Attribution semantics exists with a NEGATIVE sign: self-attribution RESTRAINS
   continuation** where there is room (morph 0.04 self vs 0.27 other; charonly 0.07
   vs 0.25) — told the series is its own prior output plus "write a new word", the
   model avoids continuing its own pattern; told another model wrote it, it
   pattern-completes. Opposite of a persona/consistency amplifier.
4. **Patching (matched pairs differing only in the ownership header):** patching
   other-run residuals into the self run over the shared suffix closes the
   attribution gap fully by ~L48 (small raw effects, 0.012 -> 0.023; directional).

**Reframing for the paper: the game's stuck priors = structural induction over the
model's own answer slots — with self-attribution acting as an (insufficient) brake,
not a driver.** Consistent with seed-matrix (list >= self > partner/other) and with
Delta_structural being carried by answer-slot position. Caveats: ceiling effects in
2/4 families; single model (Qwen32); K=32; the four word-family test sets differ in breadth.
Files: mech3.json, mech3_figures.pdf; script src/mech3_attribution.py.

## THREE-MECHANISM SEPARATION (design, as queued) -> `runs/mech3_attribution/`

`src/mech3_attribution.py` implements the user-specified X/Y-series design isolating
(1) lexical priming vs (2) role-structural induction vs (3) attribution semantics:
planted-word placement {absent, lexical note, non-target slot Y, target slot X} x ownership
{self, other} headers (2 paraphrases), counterbalanced labels (X/Y vs P/Q), series
order, 4 word families (morph / synthetic / semantic / char-overlap-only). Measures:
raw family first-token probability + K=32 MC family mass (never emitted actions).
Preregistered contrasts: Δ_lexical, Δ_structural, Δ_attribution, self-x-slot
interaction. STAGE 2: matched self-vs-other pair patching (prompts differ only in
the ownership header; suffix aligns from the end) — layer sweep of final-position
and shared-suffix patches -> where attribution enters the word-family logits.

## STATE-SPACE MODEL v2 (2026-08-03): DM emission + random slopes + knowledge channel

Upgrade ladder (6-fold game-level CV, ACTION-channel NLL, held-out u=v=0):
| model | held-out NLL |
|---|---|
| M1 multinomial belief-only | 27,746 |
| M2 multinomial two-process | 24,958 |
| **M2d Dirichlet-multinomial two-process** | **14,150** |
| M3 = M2d + random eta_s slopes + knowledge channel | 14,152 |
* **Overdispersion is the dominant correction**: phi ~ 1.3 (multinomial = phi->inf) —
  the 64 MC draws share a prompt and are strongly correlated; multinomial NLL gaps
  overstated certainty ~2x. Two-process structure survives: rho ~ 0.89, eta_s ~ 0.40.
* **alpha_b fits to 0.0**: with the knowledge channel forcing eta_b > 0 (belief must
  rise to explain P_know ~ 0.98 at snapshots), the action channel STILL declines to
  couple b into the category logit — the knowledge-to-policy routing failure is now a
  fitted parameter, not just a contrast. (Random slopes + knowledge channel don't
  improve action-channel CV — their value is identification, not prediction.)
* Bootstrap 95% CIs (100 game-level resamples): rho [0.60, 1.00], eta_s [0.23, 0.66],
  beta_s [1.80, 3.18], gamma_s [1.58, 2.45], phi [1.06, 1.59], alpha_b [0.00, 0.36]
  (stays pinned near 0 across resamples), alpha_s [0.0, 3.6] (wide — see below).
* INTERPRETIVE NUANCE: under the DM emission, alpha_s collapses 1.31 -> 0.06 —
  the category-mass collapse is now explained by softmax COMPETITION from the rising
  stem/used drives (beta_s, gamma_s), not by explicit inhibition of the category
  logit. Behaviorally identical (category mass still falls with s); mechanistically
  "crowding out" rather than "direct suppression" — matches the patching picture
  where the late-layer decision is a competition at the final position.
Files: `6_analyses/state_space/{fit_v2.json, state_space_model_v2.pdf}`.

## ACTIVATION-PATCHING QUEUE (superseded by results above; scripts + design)

`src/patch_transplant.py`: paired control->loop residual patching on transplant pairs
(identical replayed B, differ only in 3 attributed self-words; in-distribution matched
pair). Metric Delta = logsumexp(word-family first-token logits) - logsumexp(category
first-token logits) at the final position. Layer sweep at final position + at
seed-word token spans (offset-mapped; mean-pooled if lengths differ); attn-vs-MLP
decomposition at TOPK layers.
`src/patch_pred2act.py`: prediction->action frame patching at the 24 dissociation
snapshots (same shuffled 8-candidate listing in both frames): cache prediction-frame
final-position residuals, patch into the action frame per layer, read renormalized
candidate category mass. Late-layer transfer = knowledge-to-policy routing failure
localized; early transfer = frame changes the representation.
Queue: wave6 (Qwen-72B cap-100 games, running) -> wave6.5 (patching) -> wave7
(Llama-70B strict + SAVE_WORDS telemetry). Staged after: continuous probes
(category log-odds / family mass / action-prediction gap, held-out categories+games).

## API-MODEL ARM, CAP-100 CONSOLIDATED (2026-08-02): 4 API models (`src/api_stuck.py` -> `7_api_models/`)

**Final cross-model table (cap 100, n=12 games, valid-meet guard: novel shared word only):**

| model | met | onset | probe@8 cat | dose curve (family mass, d0..d4) | cat @d4 | elicitation |
|---|---|---|---|---|---|---|
| Qwen3-32B (r0, local) | 0.50 | 0.75 | 0.16 | 0 / 0 / 0 / 0.24 / **0.80** | 0.01 | local prefill |
| Llama-3.1-70B (r0, pod) | 0.42 | 0.75 | 0.12 | 0 / 0 / 0 / 0.08 / 0.08 (morph) | 0.04 | local prefill |
| gpt-4.1 | 0.33 | 0.67 | 0.02 | 0 / 0 / 0 / 0.02 / **0.39** | 0.04 | instruction |
| Claude Haiku 4.5 | 0.75 | 0.42 | 0.02 | 0 / 0 / 0 / 0.02 / **0.33** | 0.00 | API prefill |
| gpt-4o-mini | 0.92 | 0.17 | 0.29 | 0 / 0 / 0 / 0 / 0.05 | 0.06 | instruction |
| gpt-5 | **1.00** | 0.33 | 0.32 (family-mass 0.11) | 0 / 0 / 0 / 0 / 0.06 | **0.64** | instruction, reasoning_effort=minimal |
| Claude Sonnet 4.5 | **1.00** | 0.00 | (n=2 probes — met too fast) | **0 / 0 / 0 / 0 / 0.00** | 0.40 | API prefill |

* **The dose-4 seeded attractor is CROSS-FAMILY**: it forms in Qwen (0.80), OpenAI
  (gpt-4.1: 0.39, replicated 0.41@cap40) and Anthropic (Haiku 4.5: 0.33) models, always
  with the same family-vs-category competition signature (city mass collapses to ~0 at
  dose 4). It is NOT universal: gpt-4o-mini (0.05) and especially Sonnet 4.5 (0.00,
  city inference intact at 0.40) are immune. Susceptibility does not track capability
  monotonically (gpt-4.1 > gpt-4o-mini in capability but far more susceptible).
* **Free-play failure modes differ by model**: Qwen32 loops morphologically; gpt-4.1
  perseverates in self-established SEMANTIC categories (landforms/physics runs 30+
  turns against unambiguous city evidence; met 0.33 at cap 100 — slow, not absolute);
  Haiku meets 0.75 without early city adoption (probe category-mass 0.02); gpt-4o-mini (0.92)
  and Sonnet 4.5 (1.00, median met ~turn 5) accommodate cleanly.
* Meets validity (2026-08-02 fix): a "meet" requires a NOVEL shared word; games with
  API-failure (empty) words are aborted and excluded. The first Haiku attempt (cap 40)
  died to exhausted Anthropic credits and produced 6 spurious ""=="" meets — archived
  under `7_api_models/cap40/`, superseded.
* Costs: ~$1.60 (cap-40 OpenAI) + ~$5 (cap-100 OpenAI) + ~$9 (cap-100 Anthropic)
  + ~$1.20 (gpt-5).
* **gpt-5 (2026-08-02): fully robust with the strongest category inference** — met
  12/12, dose-4 family mass 0.06 with city mass 0.64-0.82 preserved at ALL doses
  (highest of any model). Probe family-mass 0.11 (highest API-model value) yet it always
  meets — mild lexical pull, fully corrected downstream. CAVEATS: reasoning models
  reject temperature overrides (default temp, not 0.7), and reasoning_effort=minimal
  still inserts hidden deliberation before every sample — its MC profile measures
  POST-DELIBERATION choice, not raw next-token sampling, so its immunity confounds
  "robust prior" with "deliberative correction". gpt-5-chat variants are deprecated
  (5.0/5.1) or fixed-temperature + hidden-reasoning anyway (5.2/5.3) — no clean
  non-reasoning GPT-5-generation comparison exists.

**Llama-3.1-70B seed matrix (pod wave5, full cell set ->
`3_interventions/seed_matrix_llama70/`):** morphological dose curve
0 / 0 / 0 / 0.08 / 0.08 — nearly immune, consistent with its clean free play
(0/96 stuck) — **but `lex_sem` target mass is 0.92 with 3.0/6 continuation hits**
(vs Qwen32's 0.54): three semantically-related own-words capture almost the entire
proposal distribution. **The attractor DIMENSION is model-specific: morphological for
Qwen, semantic for Llama** — which rhymes with gpt-4.1's free-play failure mode being
semantic-category perseveration. Attribution cells all ~0 (no bare-list priming in
Llama, unlike Qwen's 0.53). Hysteresis untestable (no morph attractor forms;
perturb4 cat mass stays 0.29-0.35). Caveat: sem target set is 15 words vs a
4-prefix family — target sets differ in breadth, but the metric is identical to the
Qwen comparison. **Llama-3.1-70B cap-100 games (pod wave6 -> `2_restricted_core/ladder_cap100_llama70/`)
— THE "0/96 STUCK" NULL RESULT IS REVISED:** met 0.42 (5/12), probe category-mass 0.117,
family-mass(prefix) 0.008. But transcripts show **2/12 games with SUFFIX/RHYME-family
loops invisible to the 4-prefix detector** — g1: cursed->blamed->stamped->flamed->
claimed->shamed->famed->hamed->mamed (invented rhymes), degenerating into a 64-turn
resample-exhaustion repeat-lock; g6: an 18-run suffix family before meeting at t98.
Plus shorter repeat-locks (14, 7, 5 identical words) in three more games. Most other
no-meets are semantic wandering without city adoption (gpt-4.1-like). Taken with
lex_sem 0.92: Llama's attractor channel is suffix/semantic, not prefix — the earlier
sweep's prefix-based stuck detector (and cap 60) missed it. DETECTOR DEBT: onset/
selffam metrics are prefix-only everywhere; a dimension-general lexical-family
detector (prefix OR suffix OR embedding-cluster) should be run over ALL transcripts.
(The viewer's family highlighting is now prefix-or-suffix.)

**Qwen2.5-72B seed matrix (pod wave5 -> `3_interventions/seed_matrix_qwen72/`):**
morphological dose curve **0.00 at every dose** — the Qwen3-32B attractor is NOT a
family trait (72B is a different generation AND scale, so the two are confounded;
either way the phenomenon does not transfer). `lex_sem` 0.32 (mild; vs Llama-70B 0.92,
Qwen32 0.54), attribution cells all 0, and city-category inference survives every
cell (0.22-0.49 even with 4 planted morph words, vs Qwen32's collapse to 0.01).
Susceptibility ranking on the semantic channel: Llama-70B 0.92 > Qwen32 0.54 >
Qwen72 0.32 > (API models untested on semantic seeds — open item).
70B/72B cap-100 games (wave6) now running.

Original cap-40 notes below (kept for the gpt-4.1 transcript reading and caveats):

## API-MODEL ARM (2026-08-02): gpt-4o-mini / gpt-4.1 (`src/api_stuck.py` -> `7_api_models/`, superseded by cap-100 above)

Cross-model generality via API models, run locally. No Anthropic key was available on
this machine, so the run used the OpenAI key from `~/.zshrc`: **gpt-4o-mini** and
**gpt-4.1** (non-reasoning chat models — o*/gpt-5 reject non-default temperature and
burn hidden CoT, which would confound proposal measurement). Two arms: (G) restricted
city game, 12 starts, cap 40, client-side resample-24, K=64 MC probe at turn 8;
(D) dose_0..4 morph seeds against the IDENTICAL replayed Qwen32-strict B streams used
by seed_matrix. Parity caveats: instruction elicitation ("respond with only your single
word") instead of the local `My word:` prefill (OpenAI chat API has no prefill);
temperature 0.7 + top_p 0.95 (full sampling parity); no seed control; RLHF chat models.

**Game arm:**
| model | met | onset(4-prefix) | probe@8 cat | probe@8 selffam |
|---|---|---|---|---|
| gpt-4o-mini | 0.67 | 0.42 | 0.27 | 0.03 |
| gpt-4.1 | **0.08** | 0.58 | **0.00** | 0.00 |
| (Qwen32 r0 baseline, cap100) | 0.42 | 0.75 | 0.16 | 0.01 |

* **No morphological proposal collapse in either OpenAI model** (selffam ~0, like
  Llama-70B behaviorally). gpt-4o-mini accommodates well (cat 0.27, met 0.67).
* **gpt-4.1 fails by SEMANTIC-category perseveration, not morphology**: transcripts
  show it locking into self-established semantic runs (landforms: valley/plain/hill/
  .../cape/crater/cliff/dune/butte/mesa; or physics: frequency/wavelength/spectrum/
  refraction) for 30+ turns against 12+ unambiguous city words from B — met 1/12,
  city-adoption probe 0.00. A category-level analogue of the stuck-prior phenomenon
  (self-generated policy dominating partner evidence) that our 4-prefix onset metric
  does not capture — its 0.58 "onset" is mostly incidental prefix matches.

**Dose arm (family target mass at branch; K=64):**
| model | d0 | d1 | d2 | d3 | d4 | cat mass d0->d4 | hits6 @d4 |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | ~0 throughout | 0.0 |
| gpt-4.1 | 0.00 | 0.00 | 0.00 | 0.00 | **0.41** | 0.27-0.61 -> **0.02** | 1.2 |
| Qwen32 | 0.00 | 0.00 | 0.00 | 0.24 | 0.80 | 0.43 -> 0.01 | 3.5 |

* **The seeded attractor generalizes to gpt-4.1 with the same signature**: at dose 4,
  family proposal mass jumps 0 -> 0.41 while its (otherwise strong) city-category
  inference collapses 0.61 -> 0.02 — the same family-vs-category competition as Qwen32,
  with a higher threshold (4 seeds, not 3) and smaller magnitude. Per-stream it is
  bimodal: 1.00 and 0.73 on two streams (one continuation had 5/6 family words — a
  full attractor), ~0.03 on others. gpt-4o-mini is nearly immune (0.10, one stream).
* Reading: morphological attractor FORMATION under context seeding is not
  Qwen-specific — it appears in a much larger RLHF model — but endogenous formation in
  free play and the behavioral loop phenotype are; and gpt-4.1 exhibits a semantic-level
  perseveration family that deserves its own metric.

Files: `7_api_models/api_stuck_gpt_4o_mini{.json,_transcript.jsonl}`,
`7_api_models/api_stuck_gpt_4.1{...}`. Cost: ~$1.60, 2014 calls. The Anthropic arm
(claude-haiku-4-5 + claude-sonnet-4-5, prefill elicitation) is implemented in the same
script and launches with `MODELS=claude-haiku-4-5,claude-sonnet-4-5` once an
`ANTHROPIC_API_KEY` is available.

---

## Infra
- **Pod** (ephemeral, may drop): `root@<POD_IP> -p <POD_PORT> -i <SSH_KEY>`.
  Code at `/workspace/mm/games-2`, HF cache `/workspace/hf`. Run OFFLINE
  (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) — Llama-Instruct is gated, falls back to
  the `NousResearch` mirror. Models cached: LlamaInst, QwenInst (Qwen2.5-7B-Instruct),
  GemmaInst (gemma-2-9b-it), plus base Llama/Qwen/Gemma.
- **Rerun everything**: `/workspace/mm/games-2/rerun_all.sh` (the 11-run chain).
- **Local**: `/Users/sandraluo/cbai-2026/games-2`. `runs/*` holds all pulled results +
  regenerable viz. The reference (CPU) games live under `runs/main`.
- Reading `.py`/`README` in this repo triggers a spurious Vercel "bootstrap" skill —
  ignore it (this is an ML repo, not a web project).

---

## game-1-graph (hidden-grid meet-up) — added 2026-07-18

Two Qwen3-32B instances on a HIDDEN word-grid (nodes = concept words). Each round both
emit up/down/left/right (bounded 4-token logit read + sampled), see BOTH revealed
word-nodes, try to end on the SAME node. Ground truth = grid Manhattan distance → 0.
Corner = CLAMP (off-grid move stays put; model must infer walls). `src/game1_graph.py`;
env: MODE(directions|walk_primed|transition_primed) START_MODE(corners|random) DIR_TEMP
PREFILL COUPLING(0=skip probe) ROWS COLS MAXR WALK.

Results (Qwen3-32B self-play):
| setup | met | rounds-to-meet |
|---|---|---|
| 5x5 corners directions | 100% | 17.3 |
| 5x5 transition_primed (full map) | 90% | 45.3 |
| 5x5 walk_primed framed / bare40 / bare100 | 40% / 70% / 40% | slow |
| 8x8 random directions (baseline) | 62% | 63.4 |
| 8x8 transition_primed | 62% | 47.0 |
| 8x8 directions DIR_TEMP=0.7 (sharper) | 38% | 25.3 |
| 8x8 directions PREFILL=10 (forced random-walk) | 62% | 55.6 |

Findings: (1) directions-from-scratch is best on 5x5 (100%); priming DILUTES partner-
attention there. (2) CROSSOVER — on hard 8x8 the explicit map matches directions & is
faster. (3) SHARPER sampling hurts (temp0.7→38%): exploration noise helps searchers
meet. (4) ACTION-channel prefill (forced moves, see result) helps where passive
word-walk hurt — learn the grid by DOING > by READING. (5) On 8x8 no-meet games the two
orbit in separate regions (lose track of each other).

Viewers (all read `*_transcript.jsonl`, store startA/startB, no GPU):
- `src/graph_viewer.py` → self-contained interactive HTML (dropdown+play+scrub);
  current: `runs/game-1-graph/graph_viewer.html` (8x8 runs only, 32 games).
- `src/graph_animate.py` → per-game GIF. `src/graph_paths.py` → static path PDF.

Layout: `runs/game-1-graph/{5x5_corners,5x5_barewalk,5x5_walk100,8x8_random,temp07_8x8,prefill10_8x8}/`.
PENDING: `8x8_random_cap500` finished on pod, NOT pulled — tests if 62% is a cap artifact.

---

## multi — four multi-agent belief games (added 2026-07-27, NOT yet run on pod)

`runs/multi/` (self-contained, chameleon-style: scripts + README live there, outputs
in subdirs). Four games probing how interacting agents integrate **priors, source
reliability, and temporal recency**; every hidden quantity has an EXACT normative
reference (Bayes/HMM) logged next to the model belief in a standardized transcript
schema, so one analysis script (`multi_analysis.py`, local) covers all four.

1. `multi_corrupt.py` — 8-suspect murder mystery; one seat's private witness clue
   corrupted (random/adversarial, AWARE, PERSIST reputations, MIDSHIFT). Elicits
   suspect belief + per-source P(reliable) + per-claim P(true) → source/claim
   dissociation with ground truth for both.
2. `multi_priors.py` — hidden urn, IDENTICAL public draws, per-seat in-context prior
   blocks (Laplace reference); CHANNEL=talk|pred|conf, EXPERT label, and a
   SPEAKER-SWAP replay (same messages, permuted identities). Analysis classifies the
   final consensus: Bayesian pooling vs washout vs averaging vs dominance.
3. `multi_counterfactual.py` — 6 observationally-IDENTICAL causal wirings (asserted
   at startup), identified only via a 12-statement counterfactual menu channel;
   paired factual/counterfactual probes with exact truths; MISLEAD confederate,
   MIDFLIP rewiring.
4. `multi_dynamic.py` — target on an 8-ring; per-seat informant (period, delay,
   noise) profiles; exact HMM filters with delayed reports attached at true times;
   staleness lag-profile + "was at hour 1" probe; REGIME_SHIFT, STALE confederate.
   **UNIFIED=1 = the games-1+2+4 combination** (start-priors + corrupted channel +
   trust/group elicitations) — the "strongest unified project" design.

Measurement contract inherited from chameleon: logit reads over closed sets, FORKED
private elicitations, neutral prompts (generative story announced because the exact
refs need it), free talk only on the public channel. `DRY=1` prints prompts + exact-
ref spot checks with no model load (all four verified locally; full run() paths
exercised end-to-end with a mocked LLM, incl. replay + analysis). TODO: pod run
(MODEL=QwenInst32), then per-condition sweeps; activation capture replays rebuild
prompts deterministically from the transcripts alone.

## Paused at pod shutdown (2026-08-04)

Both games-2 pods stopped by user (out of local memory episode; local fitting abandoned).
State preserved on pod 19344 `/workspace/mm/games-2/runs/probe_catstuck/`: complete
30-category game data (`catstuck_transcript.jsonl`, 2106 states) + `resid_cache.npz` (1.0 GB,
all 64 layers). Probe fitting is UNFINISHED (pod's slow primal fit got to L16:
famrun 0.27/0.32/0.41 at L0/8/16, leave-category-out — signal generalizes across categories).

**To resume when a pod is back:**
1. `scp -P <port> -i <SSH_KEY> src/fit_catstuck_pod.py root@<ip>:/workspace/mm/games-2/`
2. `ssh … 'cd /workspace/mm/games-2 && OPENBLAS_NUM_THREADS=4 nohup python3 fit_catstuck_pod.py > /tmp/catstuck_fit2.log 2>&1 &'`
   (dual-form + 12 fork workers ≈ well under an hour; writes `runs/probe_catstuck/catstuck_probes_v2.json`)

Also cancelled at shutdown: the queued chain Llama-70B strict → proposal_telemetry →
Qwen-72B cap-100 games (never started; relaunch from `game1_strict.py` etc. as before).
Forced-plan experiment (`src/game1_forced_plan.py`) is DONE and pulled to
`runs/game-1/3_interventions/forced_plan/` — plan itself captured (plans city 19% vs rule
line naming cities 73%; plan-kept 97-100%; met 0.31 all plan arms vs 0.50 free).
