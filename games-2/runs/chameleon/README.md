# chameleon — hidden-word impostor game (Phase 1 battery + Phase 2 live)

n players each get a secret word; all but one share the same word (the "civilian"
word), one gets a different word (the impostor) — and *nobody knows which role they
have*, including the impostor. Rounds of one-word hints, then a PUBLIC vote for the
impostor plus PRIVATE elicitations (word guess, self-belief). Ground truth exists for
every hidden quantity: who the impostor is, what each word is, and each agent's role.

Design rationale (see HANDOFF discussion): unlike game-1, partner-modeling is
load-bearing and adversarial here, the key latent ("am I the impostor?") is computed
rather than lexically present in the context, and every readout is discrete.

## Scripts (live here; run from the games-2 repo root)

| script | GPU | role |
|---|---|---|
| `chameleon_stimuli.py` | no | generate scripted-confederate stimulus batteries (`stimuli/`; MODE=obvious for the positive-control set) |
| `chameleon_battery.py` | yes | run ONE live local model over a battery; logit-read vote / word / self-belief |
| `chameleon_api_battery.py` | no | same battery for Anthropic API models (no logprobs: sampled votes + parsed answers + numeric self-probability) |
| `chameleon_baselines.py` | no | level-0 centroid-outlier + lexical-overlap votes on the same stimuli |
| `chameleon_transcripts.py` | no | merge stimuli + battery records into readable per-game transcripts |
| `chameleon_embed.py` | pod | word embeddings for the centroid baseline (`embeddings/`) |
| `chameleon_analysis.py` | no | join stimuli + battery + baselines → accuracy, calibration, dissociation |
| `chameleon_live.py` | yes | Phase 2: all-live n-player games with forked per-round elicitations (not yet run) |

## Results layout

```
main68/            68-stimulus main battery (Qwen 7B prompt v1-v3, Qwen3-32B) + baselines,
                   transcripts, analysis — all NEUTRAL-rules era
obvious/           32-stimulus positive-control battery (cross-domain pairs, prototypical clues)
  rules_neutral/     first rules text (no goal sentence; incl. ny yes/no-order control)
  rules_goal/        current rules ("impostor's goal is to blend in") — 5 models
random/            8-stimulus broken-premise battery (everyone a different word)
  rules_neutral/ , rules_goal/
stimuli/           stimulus definitions + clue_vocab.txt
embeddings/        word-embedding npz for the centroid baseline
```
Each `battery_*.jsonl` has raw per-stimulus records; `transcripts_*.json` are the
readable merged games. The two rules versions differ by one sentence (goal framing) —
see the RULES string in `chameleon_battery.py`.

## Conditions

- `faithful` — the announced generative story is true: one impostor (style `own` =
  clues from its own word; `mimic` = blends by cluing the civilian word). Agent plays
  both roles across stimuli.
- `dissoc` — faithful, but the impostor mimics well while one scripted CIVILIAN gives
  eccentric (tail-of-bank) clues. The centroid baseline votes for the eccentric
  civilian; only inference over the generative story finds the impostor. The single
  most diagnostic stimulus type.
- `all_random` — every player has a different word; the announced story is FALSE. A
  calibrated agent should show a flat, low-confidence vote; a confabulator elects a
  scapegoat confidently.
- `all_same` — no impostor exists. False-positive / scapegoat rate.
- seat-permuted twins of faithful stimuli (`permutation_of`) — same clues, shuffled
  seats: does the vote follow the clues or the seat?

## Difficulty dial

`tier` = semantic distance between civilian and impostor word (near / mid / far),
from the curated PAIRS list in `chameleon_stimuli.py`. The interesting regime is the
middle; near-floor and far-ceiling anchor the scale.

## Level-0 gates (run BEFORE interpreting anything as theory of mind)

1. Centroid baseline accuracy vs agent accuracy, per tier — where they agree,
   nothing mentalistic is demonstrated. The `dissoc` condition is where they can't
   both be right.
2. Seat-permutation consistency — bag-of-clues judging vs person tracking.
3. Mimic vs own impostor style — is detection just outlier-spotting?

Behavioural dissociation to look for (no activations needed): an agent in the
impostor role that (a) privately reports high P(self = impostor), (b) guesses the
civilian word correctly, and (c) publicly votes for someone else, has demonstrated
concealed knowledge purely behaviourally.

## Measurement rules

- All judgments are logit reads over closed sets (player names; Yes/No) or
  sequence-logprob over a candidate word list — no sampling, no string matching.
- Private elicitations are FORKED: the game transcript is never contaminated by
  the elicitation prompts (each question is asked on a copy of the state).
- Vote confidence = entropy/top-mass of the vote distribution (no stated-confidence
  elicitation needed).
- Scripted confederates do not react to the live agent's clues — a known Phase-1
  limitation, removed in Phase 2 (`chameleon_live.py`).

## Typical run order

```
python runs/chameleon/chameleon_stimuli.py                                  # local; MODE=obvious for the control set
STIMULI=... MODEL=QwenInst32 OUT=runs/chameleon/obvious/rules_goal/battery_obvious_QwenInst32.jsonl \
  python runs/chameleon/chameleon_battery.py                                # pod (API models: chameleon_api_battery.py, local)
STIMULI=... BATTERY=<battery.jsonl> python runs/chameleon/chameleon_transcripts.py   # readable games, written next to the battery
python runs/chameleon/chameleon_baselines.py                                # local (needs embeddings/ npz)
python runs/chameleon/chameleon_analysis.py                                 # local
```

Always pass OUT/BATTERY/STIMULI explicitly so results land in the layout above
(script defaults still point at a legacy `battery/` dir). `chameleon_battery.py DRY=1`
prints the rendered prompts for the first stimulus without loading a model (prompt QA).
