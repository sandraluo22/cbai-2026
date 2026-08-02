# Game 1 — Random-walk ping-pong + probe over time

Two instances of a base model (default **both Llama-3.1-8B**) relay-generate a walk
over the 4×4 in-context grid from `cross-model`. The sequence is **seeded** with a
real random walk; then the players alternate, each greedily picking the next
node-word (argmax log-prob over the 16 grid words given the running text). At every
turn the model's **post-block residual stream** at the last token is projected
through a **coordinate probe** (a leave-one-node-out ridge map fit on teacher-forced
per-node means) to a decoded `(row, col)`.

**Question:** when two instances free-run off each other's outputs, do they stay on
the grid (legal neighbour moves) and does the probe-decoded coordinate trace a
coherent grid trajectory — or does the representation drift off-manifold?

**Outputs** (`results/`): `pingpong_<A>_<B>.json` (per-turn trajectory + metrics +
probe LOO-R² validity) and `pingpong_<A>_<B>.png` (decoded-coord trajectory over the
grid, rolling legal-move rate, rolling decode error).

**Knobs:** `MODEL_A MODEL_B` (set `MODEL_B` to a different tag for a genuinely
cross-instance run, e.g. `MODEL_B=Gemma`), `LAYER` (force probe layer), `SEED_LEN
N_TURNS N_FIT_WALKS FIT_LEN CTXLO TEMP`. With identical A/B weights the relay is a
single model consuming its own generations — the design the prompt asked for.
