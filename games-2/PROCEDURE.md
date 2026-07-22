# games-2 — full procedure

## Stage 0 — the shared code (`core.Board`)
An abstract board of `N` concept-IDs, a clue vocabulary of size `C`, and a fixed
(seeded) association matrix `assoc` (C×N ≥ 0) — the shared "language". The literal
listener is `L0[i|c] = assoc[c,i] / Σ_i assoc[c,i]`. `C < N` forces coordination
(no clue names one item); `C ≥ N` (with an identity-biased `assoc`) is the easy
regime where one clue can name an item. Targets `T` (|T|=`M`) are the speaker's
private latent, sampled per seed.

## Stage 1 — agents (`core.SpeakerAgent`, `core.ListenerAgent`)
RSA recursion, frozen (no learning).

**Speaker** privately holds `T` and emits `(clue, count)`.
- `S1(c|T) ∝ exp(α · mean_{i∈T} log L0[i|c])` — informativeness toward the targets.
- **L1 speaker**: clue ~ `S1(·|remaining T)`; ignores B.
- **L2 speaker**: for each clue, simulate B's belief update and score the *gain in
  B's target-mass on the targets B is still MISSING*; clue ~ `softmax(α · gain)`.
  This conditions on B's demonstrated belief → adaptive (skip what B has, reinforce
  what B misses). `tau` raises clue-sampling temperature ("make it non-obvious").

**Listener** keeps a per-item log-odds belief (prior `M/N`), updated each round.
- Evidence that clue `c` points at item `i`:
  - **L1**: `log L0[i|c]` (literal association).
  - **L2**: `log S1(c | K∪{i}) − log S1(c | K)` — the *marginal* boost to a
    pragmatic speaker's preference for `c` if `i` were a target on top of the
    already-known set `K`. Conditions on B's own state ("what would A bother to say
    given what I've shown I know").
- Feedback: a correct guess pins the item (log-odds +∞, added to `K`, consumed by
  the speaker → moving target); a wrong guess pins it out (−∞).
- Read-outs (bounded): `belief()` = full board posterior; `guess_dist()` =
  belief renormalized over unguessed items (the N-simplex).

## Stage 2 — GAME 1: no-repeat convergence + instrument (`game1_coupling.py`)
Two agents try to CONVERGE over a `V`-token vocabulary: each turn both emit (sampled
at pick-temperature `TEMP`) and observe the other, aiming to emit the SAME token on
the same turn. **No-repeat rule**: any token emitted by either agent on a previous
turn is removed from the allowed set, so the trivial "echo the other" fixed point is
impossible and coordination must come from mutual prediction. A shared public
salience prior `sal` gives a focal structure; each agent has a hidden private bias
`bias`; each keeps a running estimate `m` of the other's next pick.
- **L1** scores tokens by `log(own-preference) + WM·log(m)` — clings to its private
  lean, aligns only after several rounds.
- **L2** scores by `log(sal) + WM·log(m)` — drops private bias for the shared focal
  point both can predict → converges faster.
Modes: `words` (random shared salience) and `numbers` (salience peaked on round
numbers → agree on the same number).
The **controlled counterfactual**: from an identical B-state, `B.observe(a_real)`
(CLEAN) vs `B.observe(a_swap)` (SWAP), and `KL(B_next_swap ‖ B_next_clean)`, read at
a softer `READ_TEMP` so the KL is bounded/interpretable. Null swap = same token (≈0).
Recorded per turn: real/null KL, and per run the convergence trajectory + first
agreement turn.

## Stage 3 — GAME 2 fused (`game2_codenames.py`)
Each round: A gives a clue (with the coupling + adaptivity probes taken first from
the identical pre-clue state), B updates and guesses, A reveals correct/incorrect,
correct targets are consumed. Logged together per round:
- **recovery**: `target_mass = Σ_{i∈T} belief_norm[i]`, and top-|T| F1.
- **coupling**: `KL(guess_dist_swap ‖ guess_dist_clean)` for `c→c+1`, plus the null.
- **adaptivity**: `KL(S.clue_dist(B_real) ‖ S.clue_dist(B_naive))` — how much A's
  clue distribution moves when B's demonstrated state changes (0 for L1 by
  construction).
Swept over {L1 pair, L2 pair} × {forced C, easy C} × `SEEDS`.

## Stage 4 — figures
Game 1: coupling-over-turns (real/null/reversed), perturbation heatmap,
distribution-shift bars. Game 2: **belief-board heatmap over rounds** (N-cell grid
per round, true targets outlined, shading = B's belief — the flagship recovery
story), recovery/entropy/coupling/adaptivity curves by regime×level, and the
**adaptivity matrix** (rows = counterfactual B-history, cols = clue token, cell =
A's clue prob; identical rows ⇒ A ignores B (L1), differing rows ⇒ A adapts (L2)).

## Stage 5 — LLM extension (`llm_agents.py`, `game_llm.py`, `remote/deploy.sh`)
Swap in frozen Llama/Qwen behind the same bounded surface: the clue distribution is
`softmax` over the `C` clue-word token logits; B's board belief is `softmax` over
the `N` item-word token logits at the answer position. Levels via system prompt
(L1 "decode the clue"; L2 "reason about a spymaster watching your guesses"). Because
read-out is bounded, recovery / coupling / adaptivity are computed identically.
Runs on the GPU pod (gated 8–9B models with ungated mirrors).

## Design decisions (per the spec)
- **Abstract vs semantic board**: start abstract (concept-IDs + a random/identity
  association) so any recovery failure is *inferential*, not lexical. `game_llm.py`
  uses real single-token words only because an LLM needs a token surface.
- **Trained vs fixed**: agents are fixed/frozen during measurement, so a changing B
  cannot confound the perturbation probe (B's shift is inference, not weight change).

## Minimal parameters
`N=16`, `|T|=3–4`, forced clue vocab `C=3` (forces coordination) vs easy `C=N`,
`ROUNDS=6–8`, raised `tau`, levels L1 and L2, multiple seeds for error bars.
