# Game 2 — Convergence game

`N` LLMs take turns naming **one word** each per round, seeing the running history.
The system prompt tells them to converge on a single shared topic. Each round we
embed the round's words (MiniLM) and track **pairwise cosine similarity**;
convergence is declared when similarity stays above `SIM_THRESH` for
`STABLE_ROUNDS` consecutive rounds (or all agents name the same word). Runs the
**2-LLM** variant then the **3-LLM** variant.

**Outputs** (`results/`): `convergence_<backend>_2agents.json`,
`..._3agents.json` (word sequences + per-round similarity + rounds-to-converge),
and `convergence_<backend>.png` (similarity vs round for both variants).

**Backends:** `BACKEND=open` (Llama+Gemma, then +Qwen) or `BACKEND=api`
(Claude Opus+Haiku, then +Sonnet). **Knobs:** `MAX_ROUNDS SIM_THRESH STABLE_ROUNDS TEMP`.
