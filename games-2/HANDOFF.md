# games-2 — Handoff / current state

Two games measuring **mutual theory of mind between models**: can model A pick up
model B's hidden state while B does the same to A. Measured **continuously** (KL /
recovery), not as a hard L1/L2 label.

- **Game 1** — convergence / KL-coupling (do two models coordinate; does each
  condition on the other).
- **Game 2** — sequential-reveal Codenames (does B recover A's hidden target set;
  coupling; adaptivity).

---

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

## runs/ layout (reorganized 2026-07-17)

```
runs/
  game-1/
    non-qwen32/        11 Game-1 LLM experiment dirs (bounded / semantic / topic / unbounded / self-play / forced)
    qwen32/            llm_open_qwen32_nr_forced/  +  qwen32_pca/ (per-layer PCA follow-up)
    reference-agents/  reference-agent Game-1 figures
  codenames/
    llm_codenames/     Llama×Qwen Codenames (recovery/coupling/adaptivity + transcript)
    reference-agents/  reference-agent Game-2 figures
```

**Qwen3-32B PCA** (`runs/game-1/qwen32/qwen32_pca/`, `src/qwen32_pca.py`): N self-play
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

## Infra
- **Pod** (ephemeral, may drop): `root@213.181.111.140 -p 19344 -i ~/.ssh/id_ed25519`.
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
