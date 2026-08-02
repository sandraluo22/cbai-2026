# runs/game-1 — index (reorganized 2026-08-02)

Numbered by theme. Every `*_transcript.jsonl` has a pretty `.json` twin (regenerate:
`python src/jsonl_to_json.py runs`). The POD keeps the OLD flat layout — pod-side
paths in scripts/launch commands are unaffected by this reorganization.

## 0_reference_agents/ — RSA reference-agent figures (instrument calibration)

## 1_legacy_7b/ — the 11 original 7B-era Game-1 variants (bounded/semantic/topic/unbounded/self-play)

## 2_restricted_core/ — the restricted convergence game (B secretly category-locked)
- `qwen32_cap24|cap100|cap200/` — same 16 games at growing turn caps (matched seeds);
  `cap24/kl/` holds the merged crossKL slides for ALL conditions incl. nolist/repeatok
  + `game1_yoked_baselines.json` (yoked chance floor, drift, perseveration)
- `qwen32_fix_cap200/` — nolist/repeatok interventions at cap 200 (+ kl/ slides)
- `qwen32_softrestrict_dual/` — restrict-watercity + dual-water-city (A=water, B=city)
- `qwen32_kl_perGame/` — older per-game-folder KL slide layout (safety-24 capture)
- `llama70/` — Llama-3.1-70B reactive/restrict-city/watercity/dual (8-bit)

## 3_interventions/ — what releases (or doesn't release) the stuck-prior loops
- `aids_informed_scratch/` — informed ("partner has SOME category") + free scratchpad;
  scratch transcripts carry per-turn notes (incl. the famous g11)
- `g11_branches/` + `g11_branches_note_blanked/` — branch surgery on stuck games:
  control/instruct/self_removed/note_replaced (+5th branch note_blanked);
  `g11_branches_readable.md` = human-readable side-by-side
- `structured_format/` — memory-log/JSON prompt format (plain + scratch modes)
- `stuck_repro_sweep/` — 6 categories x 2 phrasings x 8 starts x {Qwen32, Llama70}
  + self/partner-history-ablation branch continuations
- `variants_fewshot_board_softrepeat/` (+ `variants_llama70/`) — fewshot / city-coastal /
  board16 / softrepeat variants

## 4_menu_pref/ — the menu-preference latent-inference game (fresh 8-word menus)
- `qwen32_10r/`, `qwen32_20r/`, `llama70_20r/`, `qwen72_20r/` — the scale ladder
  (Qwen32 never updates; Llama70 weak drift; Qwen72 flips at r9, decays)

## 5_mechanistic/ — activations, circuits, steering (Qwen3-32B self-play)
- `pca_sandcloud/`, `pca_w2v/` — residual-stream captures + per-layer PCA
  (`pca_w2v/start_words.txt` = the canonical w2v start pairs; also entropy replay,
  convergence/orthogonal direction npz)
- `open_forced_sandcloud/` — the original 32B bridging run (sand+cloud→storm)
- `fixed_partner/` — ghost-partner swing-by control capture
- `partner_patch/` — head-patching localization (copy circuit, L52-62)
- `induction/` — induction scores + overlap with patch heads
- `bridge/` — cue-pair bridge heads: 96-head test + ALL-heads heatmap slides (L41-46 band)
- `head_ablate_play/`, `bridge_ablate_play/`, `topic_ablate_play/` — live-play ablations
  (copy heads: null; bridge heads: null on sand/cloud, 2x slower on topical starts)
- `steering/` — convergence-direction steering (in-game null) + chat-mode steering json

## 6_analyses/ — cross-run analyses (no GPU; each has its script in src/)
- `collision_model/` — independent-sampling collision model (AUC 0.90) + calibration pdf
- `update_dynamics/` — Bayesian-signature tests (anti-surprisal gating), support
  rasters, Deffuant/urn tests + word-embedding npz
- `stuck_probe/` — stuck-prior probes: game-level null + per-(layer,turn) heatmap
- `constraint_probe/` — latent-constraint probe (city-vs-fruit ~100%, met==no-meet)

Elsewhere: `runs/codenames/` (Game 2), `runs/game-1-graph/` (grid meet-up), `runs/main`
(reference games). Full findings narrative: ../../HANDOFF.md (sections name these dirs).
