# games-2 — two games for mutual theory of mind between models

Can model A pick up the latent state of model B while B does the same to A? Two
complementary games measure it:

- **GAME 1 — Perturbation / KL-coupling** measures **coupling**: is B actually
  conditioning on A, and how deeply? (a causal sensitivity probe)
- **GAME 2 — Sequential-reveal Codenames** measures **recovery**: did B correctly
  infer A's hidden target set? (a ground-truth probe)

The recommended setup **fuses** them — Game 2 is the task (gives ground truth),
Game 1's controlled counterfactual is the instrument run *inside* it — so coupling
and recovery are logged together, per round.

## Why reference pragmatic agents (and an LLM extension)

The spec's requirements — abstract concept IDs, bounded channel + bounded read-out,
**frozen** agents, and a dial-able **level** (L0 ignore / L1 model-the-world /
L2 model-the-modeler) — are met cleanly by a Rational-Speech-Acts recursion, which
also lets you *set* the level (you cannot cleanly dial L1 vs L2 on a raw LLM). So
the core (`core.py`) is reference agents, and `llm_agents.py` + `game_llm.py` swap
in real frozen **Llama / Qwen** behind the *same bounded read-out surface* for the
"between real models" run (GPU pod).

Design invariants (all enforced in `core.py`):
- **Channel bounded**: clues live in a small vocabulary `{0..C-1}`; a swap is
  `c→c'`, enumerable. `C < N` **forces coordination**; `C ≥ N` is the easy regime.
- **Measurement bounded**: every read-out is a distribution over the `N` board
  items (a fixed simplex), no matter how rich the clue channel is.
- **Frozen**: no learning; the counterfactual fork holds all state fixed, so
  `KL(B_swap ‖ B_clean)` isolates the *causal* effect of the intervention, not the
  model's own high-temperature drift or any weight change.

## Layout

```
games-2/
  src/
    core.py              # Board + RSA speaker/listener (L1/L2) + belief model + KL/metrics
    game1_coupling.py    # standalone convergence game + counterfactual coupling instrument + figs
    game2_codenames.py   # sequential-reveal Codenames FUSED with the instrument + figs
    llm_agents.py        # EXTENSION: real LLMs on a SEMANTIC board, neutral prompts, bounded read-out
    game_llm.py          # EXTENSION runner: two real LLMs play; MEASURE mutual ToM (continuous)
    run_all.sh           # run both reference games (CPU) into $RUN_DIR
  remote/deploy.sh       # run the LLM extension on a GPU pod
  runs/<run>/            # *.pdf + *_summary.json
```

## Run

```bash
RUN_DIR=runs/main bash src/run_all.sh                 # both reference games, CPU, seconds
# real LLMs (pod): A=spymaster, B=guesser
HOST=root@<ip> PORT=<port> A_MODEL=Llama B_MODEL=Qwen LEVEL=2 bash remote/deploy.sh
```

## What each game measures & shows

**Game 1** (`game1_coupling.py`): a **no-repeat convergence** game — each turn both
agents simultaneously emit one token and observe the other, trying to emit the SAME
token on the same turn. **No token emitted by either model on a previous turn may be
reused** — this kills the trivial "echo the other's last token" solution and forces
genuine mutual prediction (a Schelling coordination). Two modes: `words` (coordinate
via a shared salience prior) and `numbers` (agree on the same number, with focal
points on round numbers). The coupling instrument runs on top: from an identical
state, B sees A's real token (CLEAN) or a swap (SWAP), and we read B's next-turn
distribution in both (at a soft readout temperature so `KL(B_swap‖B_clean)` stays
bounded). Outputs per mode: the convergence trajectory (both agents' picks, ★ at
agreements), coupling-over-turns (real vs null), and coordination efficiency
(turns-to-first-agreement, L1 vs L2).

**Game 2** (`game2_codenames.py`): A privately holds a target set `T` (its latent);
each round A gives a (clue, count), B outputs a board-simplex belief and a guess, A
reveals only correct/incorrect, correct targets are consumed (moving target).
Logged together per round: (i) **recovery** — target-posterior mass; (ii)
**coupling** — counterfactual clue-KL; (iii) **adaptivity** — A's clue-distribution
shift under counterfactual B-histories (A's level-2-ness). Outputs: the
belief-board heatmap over rounds (the flagship — watch shading concentrate onto the
outlined true targets), recovery/entropy/coupling/adaptivity curves by regime×level,
and the adaptivity matrix (A's clue vs counterfactual B-history, L1 vs L2).

## Results (reference agents, `runs/main`)

Level maps to depth of nested modeling: **L1** = model the other as an agent
inferring the world; **L2** = model the other as an agent who models *you*.

**Game 2 — recovery + coupling + adaptivity** (N=16, |T|=4, forced C=3 vs easy C=16,
24 seeds):

| regime · level | final recovery (F1) | coupling real / null | A adaptivity |
|---|---|---|---|
| forced · L1 | 0.77 (0.88) | 0.51 / 0.00 | **0.00** |
| forced · L2 | 0.73 (0.83) | **1.21** / 0.00 | **0.94** |
| easy · L1 | 0.85 (**1.00**) | 0.87 / 0.00 | 0.00 |
| easy · L2 | 0.88 (0.90) | 1.13 / 0.00 | 1.35 |

- **Recovery works and validates the game** (mass rises; easy > forced), but it is
  *level-agnostic* — per-guess correct/incorrect feedback does the disambiguation,
  so decode depth barely moves it. In the **easy regime L1 already reaches F1=1.0**,
  so L2 adds nothing to recovery — exactly the expected null / sanity check.
- **Coupling is the causal fingerprint**: real ≫ null everywhere, and **deeper at
  L2** (forced 1.21 vs 0.51) — L2 conditions on more of the history.
- **Adaptivity is the mutual-modeling fingerprint**: **exactly 0 for an L1 speaker**
  (it ignores B by construction) and large for L2 (0.94–1.35) — A's clue distribution
  moves when B's demonstrated state changes. This is the cleanest, definitional
  evidence of A modeling B.

**Game 1 — no-repeat convergence + coupling** (V=20, 8 turns, 24 seeds):

| mode | level | converged | mean turns-to-agree | coupling real / null |
|---|---|---|---|---|
| words | L1 | 88% | 3.4 | 3.57 / 0.00 |
| words | L2 | 88% | **1.8** | 3.71 / 0.00 |
| numbers | L1 | 88% | 2.9 | 3.38 / 0.00 |
| numbers | L2 | 96% | **1.5** | 3.44 / 0.00 |

- **L2 converges roughly twice as fast** (≈1.5–1.8 turns vs ≈2.9–3.4): an L2 agent
  drops its private bias and heads straight for the shared focal point both can
  predict, while an L1 agent clings to its own lean and only aligns after several
  rounds of mutual observation. This is the coordination payoff of modeling the
  modeler.
- **Coupling is real everywhere** (KL ≈ 3.4–3.7 ≫ null 0.00): a controlled swap of
  A's token strongly redistributes B's next-turn pick — B is tightly conditioned on
  A. (This coordination game couples the agents hard by design, hence the large KL;
  the null-swap floor stays at 0.)
- The **numbers** mode (agree on the same number, focal points on multiples of 5)
  behaves the same and converges slightly better, since the numeric focal structure
  gives a stronger Schelling point.

**Takeaway.** Under this bounded, frozen setup the mutual-ToM signal lives in
**coupling** and **adaptivity**, not in raw recovery speed (which per-guess feedback
saturates). Both are definitional level-differentiators: an L2 speaker adapts its
clue to B's state; an L2 listener couples to A through a latent estimate. See
`PROCEDURE.md` for the full method; figures in `runs/main/`.

## Two real models (the actual experiment) — `runs/llm/`

The reference agents above only *calibrate the ruler* (they show adaptivity is ~0
for a partner-ignoring agent and large for a partner-modeling one). The real
experiment gives **two frozen LLMs a NEUTRAL instruction** to play Codenames — never
"model your partner" — and **measures** how much mutual theory-of-mind each
spontaneously shows. It is a continuous read-out, not an L1/L2 label. (LLMs need a
*semantic* board — a random clue↔item code is unrecoverable — so the board is
grouped into categories and the clues are category words; base models on the old
abstract board measured ~0 on everything, confirming the setup, not ToM.)

`Llama-3.1-8B-Instruct` × `Qwen2.5-7B-Instruct`, 12-word semantic board, 4 category
clues, both role orderings, 8 games:

| spymaster → guesser | recovery | guesser-coupling (B's ToM of A) | spymaster-adaptivity (A's ToM of B) |
|---|---|---|---|
| Llama → Qwen | 0.77 | **12.4** | 0.38 |
| Qwen → Llama | 0.86 | 5.7 | **1.32** |

- Both pairings **recover** the hidden target set well (0.77–0.86) — real Codenames
  works between the two models.
- **Guessers strongly condition on the clue** (coupling ≫ 0; Qwen-as-guesser 12.4,
  Llama-as-guesser 5.7) — swapping the clue sharply redistributes the guess.
- **Spymasters do model the guesser**, and asymmetrically: Qwen-as-spymaster adapts
  its clue to the guesser's revealed state much more (adaptivity 1.32) than
  Llama-as-spymaster (0.38). Against the reference calibration (partner-ignoring ≈ 0,
  partner-modeling ≈ 1), Qwen shows clearly partner-modeling ("L2-ish") spymaster
  behaviour while Llama's is weaker — a measured difference in mutual ToM between the
  two models, from neutral instructions alone.

Run: `MODELS=LlamaInst,QwenInst bash remote/deploy.sh` (GPU pod). Figure:
`runs/llm/game_llm_LlamaInst_vs_QwenInst.pdf`.

### Game 1 between the two models (`game1_llm.py`)

The no-repeat convergence game (agree on the same word / number, no reuse of any
already-played token), neutral instructions, run **until convergence** (no turn cap;
the no-repeat rule bounds a game by the word pool, and a game that never coordinates
ends "exhausted"). Each model's **coupling** = how much its next pick moves when the
*other* player's last pick is swapped from an identical state. Llama-3.1-8B-Instruct ×
Qwen2.5-7B-Instruct, 28-word pool (10 for numbers), 6 games:

| mode | converged | mean turns-to-converge | coupling Llama | coupling Qwen |
|---|---|---|---|---|
| **words** | 83% (5/6) | **9.0** | 0.14 | **0.60** |
| **numbers** | 67% (4/6) | 2.2 | **0.00** | **0.00** |

- **Words**: coordination is **slow** — genuine convergence takes ~9 turns (games
  agreed at turns 5–12; one never agreed and exhausted the pool). An earlier
  fixed-6-turn run misreported "~3 turns" by cutting games off before they'd actually
  agreed. Coupling is real and **asymmetric**: Qwen conditions on the partner's pick
  ~4× more than Llama (0.60 vs 0.14) — the *same* asymmetry seen in Codenames (Qwen
  shows more theory-of-mind). Word coordination genuinely requires modeling the other.
- **Numbers**: fast when it happens (~2 turns) but only 67% converge (the rest exhaust
  the small pool), and coupling ≈ **0 for both** — the models coordinate via a
  *history-independent focal rule* (a Schelling default), so swapping the other's last
  pick changes nothing. Number coordination needs no theory-of-mind; word coordination
  does. (Multi-digit numbers tokenize per-digit and collide in the read-out, so the
  numbers vocab is single-digit 0–9, which also caps its pool.)

Per turn we log a full transcript (`*_transcript.jsonl`: exact prompts, each model's
distribution over the candidates, picks, agreement, and the counterfactual behind
every KL) and a per-turn slideshow (`*_perturn.pdf`: each model's coupling KL turn by
turn with the clean-vs-swap distribution shift). Figures/logs in `runs/llm/`.
