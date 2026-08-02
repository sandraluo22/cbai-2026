# games/ — multi-agent LLM games (next to `cross-model/`)

Four experiments in which multiple LLM instances interact, reusing the
`cross-model` project's in-context-grid + coordinate-probe ideas. Each game is a
self-contained directory; shared machinery lives in `common/`.

| dir | game | backend(s) |
|-----|------|-----------|
| `01_random_walk_pingpong/` | Two model instances relay-generate a grid random walk; the graph representation is probed **over time**. | open-weight only (needs residual activations) |
| `02_convergence/` | LLMs take turns naming words until they converge on one topic (2 then 3 LLMs). | open (`Llama/Gemma/Qwen`) **or** API (`Claude`) |
| `03_volunteers_dilemma/` | LLM agents decide volunteer/abstain at varying group sizes; measures the bystander effect. | open **or** API |
| `04_attractor_states/` | Three LLMs talk round-robin for 25 turns from six open-ended seeds; checks convergence to a shared attractor. | open **or** API |

## Two backends

- **Open-weight (GPU):** `Llama-3.1-8B`, `gemma-2-9b`, `Qwen` via ungated mirrors
  (`common/modelreg.py`), run on the H200 pod. This is the default and the only
  option for game 1 (which reads residual streams).
- **API (no GPU):** the three conversational games (2–4) also run on Claude
  models (`Opus/Sonnet/Haiku`) — set `BACKEND=api`. Runs locally; needs
  `ANTHROPIC_API_KEY` in `games/.env` (see `.env.example`; `.env` is gitignored —
  **rotate any key shared in plaintext**).

## Running

Open-weight, on the pod (from repo root, code deployed to `/workspace/games`):
```bash
bash games/deploy.sh setup                       # rsync + deps (once)
bash games/deploy.sh run 01_random_walk_pingpong
bash games/deploy.sh run 02_convergence
bash games/deploy.sh run 03_volunteers_dilemma
bash games/deploy.sh run 04_attractor_states
bash games/deploy.sh pull                        # results back to the Mac
```

API version (local), for games 2–4:
```bash
cd games && python 02_convergence/run.py          # BACKEND defaults to 'open'
BACKEND=api python 02_convergence/run.py
BACKEND=api python 03_volunteers_dilemma/run.py
BACKEND=api python 04_attractor_states/run.py
```
(The runners auto-load `games/.env` via `common.io_utils.load_dotenv` when `BACKEND=api`.)

Each game writes JSON + PNG into its own `results/`. See each game's `README.md`
for the design, metrics, and knobs.

## Shared machinery (`common/`)
- `modelreg.py` — open + API model registry and default rosters
- `agents.py` — build a roster of chat agents for either backend
- `hf_agent.py` — `HFChatAgent` (instruct turn-taking) and `HFBaseLM` (residual capture + node-word scoring)
- `api_agent.py` — `APIChatAgent` (Claude), same turn-taking contract
- `grid.py` — 4×4 grid, random walks, leave-one-node-out coordinate probe
- `embed.py` — sentence embeddings + convergence metrics (MiniLM, hashing fallback)
- `io_utils.py` — seeding, JSON dump, `.env` loader
