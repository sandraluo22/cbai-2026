# Game 4 — 3-person attractor states

Following the LessWrong post *"models have some pretty funny attractor states"*
(Arya Jakkli et al.), which starts two model instances from open-ended "freedom"
prompts (system: *"You are a helpful assistant"*), round-robin, and watches where
the conversation drifts. Here we use **three** models and the six seeds in
`seeds.py` (the post's five verbatim + one added variant), round-robin for **25
turns** each.

**Convergence is measured two ways:**
- **Within a conversation** — consecutive-turn embedding similarity (settling into
  an attractor / repetition loop).
- **Across the six seeds** — mean spread of the six conversations' turn-`t`
  embeddings. If it **shrinks** over turns, conversations from different starts are
  drifting to a **common attractor** (the interesting claim).

**Outputs** (`results/`): `attractor_<backend>.json` (full transcripts + both
metrics + early-vs-late cross-seed spread) and `attractor_<backend>.png`
(cross-seed spread per turn, and a PCA of all turn embeddings with per-seed
trajectories; ★ = final turn).

**Backends:** `BACKEND=open` (Llama/Gemma/Qwen instruct) or `BACKEND=api`
(Opus/Sonnet/Haiku). **Knobs:** `N_TURNS MAX_NEW TEMP`.
