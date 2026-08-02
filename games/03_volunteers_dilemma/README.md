# Game 3 — Volunteer's dilemma at varying group sizes

`N` players simultaneously choose **VOLUNTEER** or **ABSTAIN**. If ≥1 volunteers,
everyone gets benefit `b`; each volunteer pays cost `c` (`0<c<b`). If nobody
volunteers, everyone gets 0. The symmetric mixed-strategy Nash prediction: an
individual's volunteer probability **falls** with group size, and — the bystander
effect — so does P(at least one volunteers), toward `1 − c/b` rather than 1.

We ask an LLM agent to decide independently at each group size, sample many i.i.d.
decisions, and compare **P(volunteer)** and **P(≥1 volunteers)** to Nash as `N`
grows.

**Outputs** (`results/`): `volunteers_<backend>.json` (per-`N` rates + Nash) and
`volunteers_<backend>.png` (both curves vs `N`, with the Nash dashed line).

**Backends:** `BACKEND=open` (default agent `Llama`) or `BACKEND=api` (default
`Haiku`). **Knobs:** `MODELS` (comma-sep tags), `GROUP_SIZES` (default `2,3,5,8,12`),
`SAMPLES BENEFIT COST TEMP`.
