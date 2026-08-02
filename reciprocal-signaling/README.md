# reciprocal-signaling — asymmetric misconceptions, one-shot channels

Two instances of the same LLM (A, B) each see 3 labeled examples of a hidden
**topic** rule ("words for animals are dax" — topical, never orthographic, since
LLMs are unreliable letter-counters). Each agent's evidence is deliberately
consistent with the truth AND with its own **decoy** topic; the decoys differ
between agents. Neither can identify the rule alone.

Per round, both send one message **simultaneously** (bandwidth `m`), see each
other's message, repeat for `ROUNDS` rounds. Then each is **ground-truth tested**:
label ~10 held-out probe words + a free-text one-sentence rule guess (elicited pre
AND post, so belief movement is measured), keyword-classified per task into
{true, decoy_A, decoy_B, mixed, other} — no candidate list is ever shown.

**The channel**: a message transmits a NOVEL word — one that does not appear in the
sender's labeled examples — that the sender believes is dax. The emitted word
directly expresses the sender's current hypothesis (a farm-believer produces
"barn", an animal-believer "tiger"), so signaling is productive, not a choice among
given hints. Evidence still carries the trap/key structure (each agent's dax
examples include one word also consistent with the partner's decoy and one that
falsifies it), which shapes what hypotheses each side can form. Prompts are neutral
(game rules only, no "model your partner" coaching) per the games-2 lesson.
Note: emitted words can collide with held-out probe words; collisions are logged
and `probe_acc_clean` excludes them.

## Bandwidth levels (elaboration around the emitted word)
- `m=1` — the new believed-dax word only, no other text
- `m=2` — the word + one short claim (≤ 12 words)
- `m=3` — the word + free reasoning (≤ 80 words)

## Conditions
- `main` — live two-way play
- `static` — B replaced by a fixed script (the same decoy_B-consistent novel word
  every round): does a non-adaptive, misconception-consistent partner mislead A?
- `oneway` — only A→B channel: separates sending from receiving; A doubles as the
  no-input evidence floor
- `shuffled` — A receives round-matched B-messages recorded in a *different* task's
  game: is A conditioning on *this* partner or on *any* message stream?
- `diffmis` — same A evidence, B's decoy (and evidence) swapped to an alternative
  (tasks t1–t3): does A's signaling change with the partner's misconception?

## Models (both instances = same model, self-play)
- `Qwen32`  — Qwen3-32B, thinking disabled (repo default mode)
- `Qwen72`  — Qwen2.5-72B-Instruct, 8-bit
- `Llama70` — Llama-3.1-70B-Instruct, 8-bit
(`Qwen7` = Qwen2.5-7B-Instruct, smoke tests only.)

## Scoring (src/analyze.py → runs/summary.csv)
probe accuracy split by which decoy the probe discriminates (own / partner's /
both / none-floor), with and without probe-leaked items; pre→post truth rate of
the classified free-text guess; residual own-decoy rate; the emitted words
themselves (per agent, per round); parse-fallback counts.

## Layout
```
src/tasks.py      6 task definitions (+ alt_B misconception variants for t1–t3)
src/run_games.py  runner (env: MODEL M COND TASKS ROUNDS SEEDS OUT SHUF_SRC)
src/analyze.py    transcript aggregation (no GPU)
src/mock_test.py  full-pipeline test with a scripted fake model (no GPU)
run_pilot.sh      the everything-once pilot grid
runs/<MODEL>_m<M>_<COND>/<task>_s<seed>_transcript.jsonl (+ pretty .json twin)
```

Pod: same H200 infra as games-2 (`/workspace/mm/reciprocal-signaling`), run
OFFLINE (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
