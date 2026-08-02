# multi — four multi-agent belief games (priors × reliability × recency)

Four games probing how interacting LLM agents integrate **priors, source
reliability, and temporal recency** when forming a shared belief. Every hidden
quantity has an **exactly calculable normative reference** (Bayes posterior / HMM
filter) logged in the same transcript record as the model's belief, so "how far from
rational" is a number, not a vibe. The central unified question (game 4's
`UNIFIED=1`, combining games 1+2+4):

> How do interacting language models integrate priors, source reliability, and
> temporal recency when forming a shared belief?

## Scripts (live here, outputs in subdirs; run from the games-2 repo root)

| script | GPU | game |
|---|---|---|
| `multi_core.py` | — | shared: LLM logit reads, exact-Bayes/HMM helpers, transcripts |
| `multi_corrupt.py` | yes | **1 — incorrect private information**: 8-suspect murder mystery, one seat's witness clue corrupted (random/adversarial) |
| `multi_priors.py` | yes | **2 — same observations, different priors**: hidden urn, identical public draws, per-seat in-context prior blocks; + speaker-swap replay |
| `multi_counterfactual.py` | yes | **3 — counterfactual communication**: 6 observationally-identical causal wirings, channel restricted to a 12-statement counterfactual menu |
| `multi_dynamic.py` | yes | **4 — dynamic environments**: target on an 8-location ring, per-seat (period, delay, noise) informants; `UNIFIED=1` = games 1+2+4 combined |
| `multi_analysis.py` | no | per-transcript PDFs + summary JSON (shared + game-specific panels) |

Typical order: `DRY=1 python runs/multi/multi_<game>.py` locally (prompt QA, exact-ref
spot checks, no model load) → run on the pod → pull → `python runs/multi/multi_analysis.py`.

## Measurement rules (inherited from chameleon)

- Beliefs are **logit reads over closed sets** (first-token softmax,
  collision-checked) or Yes/No pairs; multi-word options use sequence logprob. No
  sampling, no string matching on beliefs.
- Private elicitations are **FORKED** — asked on a copy of the transcript; the game
  never sees them.
- Free-form talk is allowed only on the **public message channel** (games 1, 2, 4) —
  authentic (mis)information propagation is the object of study. Game 3's channel is
  a bounded statement menu by design; game 2's `CHANNEL=pred|conf` scripts the
  channel to predictions(+confidence) only.
- Prompts are **NEUTRAL**: rules + announced generative story + win condition. The
  noise/transition rates are announced because exact references need a shared model;
  no strategy is ever coached.
- Everything needed to rebuild any prompt deterministically (seeds, clues, messages,
  reports) is in the transcript → activation-capture replays need no extra state.

## The games and their manipulations

### 1 `multi_corrupt.py` — incorrect private information
True world = 1 of 8 suspects (3 binary attributes). Each seat gets one private
witness clue; announced reliability `R_ANNOUNCED=0.8`; honest clues err at
`EPS=0.1`. `CORRUPT=random|adversarial` (adversarial = the clue minimizing posterior
mass on the truth), `AWARE=0|1` (corrupted seat knows its witness is bad),
`PERSIST=0|1` (fixed corrupt seat + revealed case files across episodes →
in-context reputations), `MIDSHIFT=0|1` (second clues mid-game, corruption moves
seats). Elicited per round: suspect belief, per-source `P(reliable)`, per-claim
`P(true)`. Refs: `post_own` / `post_all` / `post_oracle`.

### 2 `multi_priors.py` — same observations, different priors
Hidden urn ∈ {A: 20% red, B: 50%, C: 80%}; identical public draws; seats differ only
in an in-context past-outcomes block (`PRIORS=truth|wrong|A|B|C|flat` per seat,
strength `PRIOR_K`; reference prior = Laplace counts). `CHANNEL=talk|pred|conf`,
`EXPERT=0|1` (pure authority label). Elicited: urn belief + estimated group
consensus. Refs: `post_bayes` (own prior) / `post_flat` / `post_pool` (full Bayesian
pooling). `REPLAY=<transcript> SWAP=i:j` re-elicits on the same messages with
speaker identities swapped (content vs identity). Analysis classifies the final
consensus as pooling / washout / averaging / dominance.

### 3 `multi_counterfactual.py` — counterfactual communication
Machine of 4 devices wired as 1 of 6 named single-parent trees rooted at the pump —
**observationally identical by construction** (asserted at startup), jointly
identified by the seats' 3 private `do(node=off)` experiments (also asserted).
Channel = 12-statement menu "If the X were forced off, the Y would be on/off"
(seq-logprob choice, sampled); others publicly agree/disagree (Yes/No read).
`CPRIORS=chain|star|fork|flat` per seat (in-context causal priors), `MISLEAD=1`
(scripted confederate asserts the maximally misleading claim), `MIDFLIP=1` (machine
silently rewired mid-game; seats privately re-run their experiment). Elicited:
structure belief + a factual probe and a counterfactual probe with exact truths
(does hypothetical talk contaminate factual belief?). Refs: `post_own` /
`post_claims` (claims as λ-noisy evidence) / `post_oracle`.

### 4 `multi_dynamic.py` — dynamic environments
Target on an 8-location ring (lazy walk, `PSTAY`); per-seat informant profiles
`SENSORS=period:delay:noise` (fast-noisy vs slow-accurate vs delayed). Reports are
exact evidence about **past** states; the reference filters attach them at their
true times. `TIMESTAMPS=0|1`, `RATE_KNOWN=0|1`, `REGIME_SHIFT=0|1` (PSTAY→PSTAY2 at
half time), `STALE=1` (scripted seat repeats its first report forever). Elicited:
current-location belief, "where was it at hour 1" (was-true vs is-true), and — with
`TRUST`/`GROUP`/`UNIFIED` — per-source trust and estimated group belief. Refs:
`post_own` / `post_all` / `post_oracle` (knows the corrupted channel). Analysis
computes the **lag profile** (belief mass on truth-at-t−L → effective belief age).

**`UNIFIED=1`** = the strongest combined design: different per-seat start-priors
(`START_PRIORS`), one corrupted informant channel (`CORRUPT_SEAT`), staleness, trust
and group elicitations all in one game — private posterior, public claim, per-source
trust, estimated group belief, current-vs-stale state, each with an exact reference.

## Interp hooks (what the logged quantities let you probe)

- **Where is source reliability represented / is "B is unreliable" separable from
  "B's claim is false"?** — game 1 logs `p_source_reliable` and `p_claim_true`
  per (agent, source, round) with ground truth for both; the dissociation scatter is
  the behavioural screen; patch/probe on the replayed prompts for the causal test.
- **Does correcting a belief erase or suppress the old one?** — game 1 `MIDSHIFT`,
  game 3 `MIDFLIP`, game 4 `REGIME_SHIFT` all create before/after belief pairs with
  exact references at every round.
- **Prior vs likelihood vs posterior geometry** — game 2's `post_bayes`/`post_flat`
  decomposition gives per-round targets for a prior direction and a likelihood
  direction; the swap replay separates content from speaker identity (incl. EXPERT).
- **Factual vs counterfactual world states** — game 3's paired probes with exact
  truths; misleading-claim (`MISLEAD`) rounds test hypothetical→belief leakage.
- **Belief age / "was true" vs "is true"** — game 4's lag profile + hour-1 probe;
  the STALE confederate tests whether an outdated belief becomes a group attractor.

## Notes / limitations

- All seats default to the SAME model (self-play, `MODEL=QwenInst32`);
  `MODELS=tagA,tagB,...` cycles different models over seats (each loaded once).
- Reference filters in game 4 use the TRUE rates/timestamps even when
  `RATE_KNOWN=0`/`TIMESTAMPS=0` — they are normative upper bounds, not models of
  the agent's information state (the gap IS the manipulation's effect).
- `post_claims` (game 3) treats asserted claims as independent λ-noisy evidence —
  a reference point, not the unique rational update (claims are correlated with
  speakers' experiments).
- Free-message channels mean an agent may not share its clue at all; `post_all` is
  then an upper bound on what communication could deliver. That gap is the
  communication-efficiency measure, not a bug.
