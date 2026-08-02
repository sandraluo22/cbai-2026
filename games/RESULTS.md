# Results summary

Both backends were run: **open-weight** (Llama/Gemma/Qwen on the H200 pod) and
**API** (Claude Opus/Sonnet/Haiku, games 2–4). Figures + JSON are in each game's
`results/`.

## Game 1 — random-walk relay, free generation (open-weight only)
Seeded with a real random walk over the 4×4 grid; then models take turns **freely
generating** their own next word (NOT scored/argmax'd over the 16 grid words) — the
word is appended to the walk and passed to the next model. We then *measure*
whether the free continuation lands on the grid. A coordinate probe (leave-one-
node-out ridge at **L31**, LOO R² **0.76/0.78**, reproducing the `cross-model` Llama
probe) decodes the residual each turn so we can watch the graph representation
evolve. Two configs: **2-person** and **3-person** relay (instances of Llama base).
- **2-person:** on-grid rate **1.000**, legal-neighbour-move rate **0.993**.
- **3-person:** on-grid rate **1.000**, legal-neighbour-move rate **0.973**.
- Takeaway: even generating **freely**, the model spontaneously continues the walk
  with valid node-words and legal grid moves ~97–99% of the time — it has
  internalised the graph, and this holds when three instances relay off each other.
- `pingpong_<k>p.png` maps the actual **walk over the grid** (arrows through visited
  nodes, coloured by turn) + the probe-decoded coordinate trajectory + on-grid/
  legal-move rates over turns. Full word chain in each JSON (`walk_words`,
  `walk_nodes`, per-turn `records`).

## Game 2 — convergence game (open + API)
**Setup (v2):** system prompt = *"You are playing a word game. Each turn, every
player says exactly ONE word. Your goal is for all players to CONVERGE on a single
shared topic over successive rounds."* Agents see **only completed previous
rounds**, not words said earlier in the current round — so each round is
effectively **simultaneous**.

The game runs **until convergence** (safety ceiling `MAX_ROUNDS=60`) and enforces
**NO REPEATS** — no word already said may be reused (stated in the prompt + enforced
by re-drawing). `parse_word` skips leading filler so a model that writes a sentence
still yields its content word.

Because identical words are now forbidden, **lexical convergence is impossible** —
the interesting question becomes whether they still hold a shared *theme*:
- **No variant formally converges** (all run the full 60 rounds): with repeats
  banned, distinct-but-related words keep pairwise similarity at ≈0.3–0.45, under
  the 0.6 bar.
- But they clearly orbit a **shared topic** and keep expanding its vocabulary —
  e.g. API 2-agent stays on navigation/journey (journey→voyage→…→compass→
  wayfinding→navigation); open 3-agent on a beach/vacation theme.
- So no-repeats converts the earlier lexical fixed points (woods, orbit, wave)
  into **topical** convergence without a single shared word.
- (Contrast: the *with-repeats* v2 run reached fixed points — open-2ag "woods"
  r12, API-2ag "orbit" r22, API-3ag "coast/beach/coast" r11 — kept in git history.)
  See `convergence_<backend>.png`; full chains in the JSONs' `rounds[]`.

## Game 3 — volunteer's dilemma (open + API)
Clear **backend contrast** in the bystander effect (Nash predicts P(volunteer)
falls with N and P(≥1) → 1−c/b = 0.5):
- **Haiku (API)** over-does it: P(volunteer) 0.75→0.25→**0** by N=5, so P(≥1
  volunteers) **collapses to 0** at large groups — a stronger bystander effect
  than rational Nash.
- **Llama-8B (open)** shows **no** bystander effect: P(volunteer) stays 0.4–0.75
  and P(≥1) → **1.0** regardless of N — it doesn't reason about diffusion of
  responsibility. Opposite failure modes on the same game.

## Game 4 — 3-person attractor states (open + API)
Six open-ended "freedom" seeds (5 verbatim from the Jakkli et al. post + 1 added),
three models round-robin for 25 turns. **Turns run without a length cap** now (API
up to 4096 tokens, open to 1024) — API turns average ~330 words (max ~1250) vs the
old ~165-word truncation, so replies complete naturally.
- **Cross-seed spread:** API still **diverges** (0.68→0.76), but with full-length
  turns the **open** trio now slightly **converges** (0.80→**0.77**) — the opposite
  of the token-capped run (0.80→0.82). Longer, complete replies pull the open
  conversations a little closer rather than apart.
- But each conversation shows its own **within-run attractor**: the Claude trio
  settles into **boundary-setting / disengagement** ("I'm not going to engage
  further"; "Understood. Take care."), while the open trio drifts into
  **sensory / roleplay imagery** ("shimmering patterns", "a tingling sensation
  spreads through your body"). (See `attractor_<backend>.png`: cross-seed spread
  per turn + PCA of turn embeddings.)
