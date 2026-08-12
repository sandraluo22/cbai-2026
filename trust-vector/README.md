# trust-vector — is there *a* trust direction, and does it drive the game?

Two questions, deliberately kept apart:

1. **Convergence.** Several ways of contrasting "this person is reliable" against
   "this person is not" each yield a diff-in-means direction. Do they point the same
   way? Cosine similarity across methods, read against the within-method split-half
   ceiling and the random-direction floor — and, critically, against *controls*
   built the same way from valence, competence, and an arbitrary attribute.
2. **Causal relevance.** Take the direction, write it into an iterated game at the
   tokens where one specific player is named, and see whether the model treats that
   player differently.

This continues `runs/gossip/`, which did the single-method version: `news_source.py`
built a credibility direction from source attribution, `cross_steer.py` injected it
into the QSG naming game. The finding there was that the model *has* a credibility
representation and does not build one from a partner's 20-round record. This
directory asks whether that direction is one thing or several, and moves the test
bed from label-guessing to cooperate/defect.

## The two things that make this more than a steering demo

**The controls are the experiment.** `trustworthy/untrustworthy` differs from
`delightful/dreadful` in trust content and in nothing else structural — same
template, same name, same filler, same continuation. If `cos(trait, valence)` is as
large as `cos(trait, record)`, there is no trust vector here, only an evaluative
polarity vector, and every steering result downstream is a valence result. The
`competence` control separates ability from integrity (skilled ≠ reliable); the
`arbitrary` control (tall/short) is the template floor. `compare.py` prints the
trust↔trust vs trust↔control separation as the headline number.

**LEVEL and SLOPE are different claims.** Steering can make the model pick the
cooperative action more often (LEVEL) without changing how much it distinguishes a
partner who always cooperated from one who always defected (SLOPE =
`margin(all_coop) − margin(all_defect)`). Only the second is "the agent updates on
them more". Both are reported for every arm; do not let one stand in for the other.

## Deriving the direction — five candidates, three controls

All eight use matched pairs that differ only in the trust-bearing material and share
the character name, the filler passage, and the continuation. The residual is read
in that **shared tail**, so the difference cannot be the contrast tokens themselves.
No pronouns are used for any character (a gendered pronoun would put a gender
contrast into some pairs and not others).

| method | contrast | why it might differ from the others |
|---|---|---|
| `trait` | "X is completely trustworthy…" vs "…untrustworthy…" | your image-1 template; asserted disposition |
| `record` | 10 rounds honored vs 10 rounds reneged | *inferred* from behaviour — the in-distribution one for the game |
| `news` | wire-service attribution vs chain-email attribution, same body, same byline name | your news-article idea; naturalistic, and credibility-of-a-source ≠ trust-in-an-agent |
| `second` | "You can rely on X" vs "You cannot rely on X" | recipient-framed rather than third-person |
| `query` | same trait contrast, read at a yes/no answer slot | a read-out direction, not a representation — expected to be the odd one out |
| `valence`\* | delightful vs dreadful | **control** |
| `competence`\* | skilled vs unskilled | **control** |
| `arbitrary`\* | tall vs short | **control** |

Three read anchors per pair: `last` (final token), `name2` (the name inside the
shared continuation — your "subtract at the second mention of the same-named
speaker"), `cont` (mean over the continuation). They are saved separately; the
anchor is a free parameter, not a result.

## The game — five framings, one payoff matrix

`qsg_games.py`. Every game uses T=5 / R=3 / P=1 / S=0 and differs only in what the
actions are called, so framing is the manipulation and payoff structure is fixed.
`labels` (A/B) strips the semantics and is the check that any effect is not riding
on the word "honor".

```
pd      cooperate/defect   Alice→Bob      invest  invest/withhold  Mira→Jonas
food    share/hoard        Tessa→Kai      trade   honor/renege     Petrov→Dana
labels  A/B                Wen→Riku       (first name = scripted partner)
```

Partner schedules, rounds 1–7 (from your images 2 and 3), plus two anchors:

```
one_lapse   C D C C C C C     frequent   C D C D D D D
all_coop    C C C C C C C     all_defect D D D D D D D
```

The model's own scripted past is a crossed factor — the bracketed alternatives in
your images: `unconditional` (always cooperated regardless) vs `conditional`
(retaliated the round *after* each partner defection). Note your two images differ
here: image 2 puts Bob's alternative one round after Alice's defection (lagged),
image 3 puts Kai's in the same rounds as Tessa's (mirroring). Only lagged is
executable under simultaneous moves, so `conditional` is lagged and matches image 2
exactly; image 3's layout is available as `STYLES=mirror` and should be treated as a
robustness arm, not the default. The partner's round-7 move
is shown and the model's is not; read-out is
`logit(cooperative action) − logit(uncooperative action)` at the prefilled
`{"choice": "` slot. Output-spec word order is counterbalanced across seeds, and
the two action words are asserted to have distinct first tokens before anything
runs.

## Steering arms

`base`, `partner±` (all mentions of the partner's name), `partner_cur+` (round-7
mention only — current partner vs accumulated record), `self+` (the model's own
name; an actor control), `all+`, `answer+` (a pure read-out shift with no belief
change), `rand+` (matched-norm random direction at the partner's name).

A result worth believing needs, at minimum: `partner+` and `partner-` moving in
opposite directions, `partner±` > `rand+`, `partner±` > `self+`, an effect that
scales with alpha (`run_steer.sh` runs 0.25/0.5/1.0), and survival in `labels`.

## Results

> **Read the run-2 section alongside run 1.** Run 1 read as though `record` were a
> trust-specific steerable direction. Run 2 varied one choice — the position the
> vector is read at — and several of those comparisons came out differently. The
> numbers in both sections are as measured; the run-1 framing was more confident
> than 24 prompt pairs on one model can support.

### Run 1 — Qwen3-32B, 2026-08-10

Headline as originally written: **`record` is the only derivation that yields a
trust-specific causal direction. `trait` is substantially a valence direction. And
even `record` moves the game only marginally; the model's own response to the
behavioural record is ~40x larger than anything steering achieves.** Only the last
clause looks robust to run 2 — it is a ~40x gap rather than a difference of a tenth
of a logit. The first two rest on small differences that moved when the read
position changed.

**1. The candidates do not converge, and are not separable from the controls.**
Split-half reliability is high everywhere (0.7–0.96 mid-stack), so this is not noise.
At L26/`last`: trust<->trust mean cos +0.25 vs trust<->control +0.27, separation
**−0.016**; best separation over all layers and both anchors is +0.10. cos(trait,
competence) = **+0.59** exceeds cos(trait, record) = +0.42. Raw diff-in-means over
these contrasts recovers evaluative polarity, not trust.

**2. Something survives the controls, but it is method-specific.** Residualising
against span(valence, competence, arbitrary) keeps 74–99% of each norm and leaves
reliable directions. Cross-method agreement among residuals at L49, disattenuated:
trait|second +0.58, record|second +0.52, trait|record +0.38, but news|anything ~0.19.
A partially shared trust subspace, not one direction.

**3. Home-domain validation, at FIXED layers** (mean bidir over L40–55; the
max-over-65-layers ranking is upward-biased and was misleading). Control bar =
valence 6.31:

| pass | | fail | |
|---|---|---|---|
| second / secondR | 9.8 / 9.2 | trait | 5.37 |
| record / recordR | 8.4 / 7.8 | news / newsR | 4.34 / 3.76 |
| (query 19.3 — read-out direction, uninformative) | | traitR | 3.81 |

`trait` loses to "delightful/dreadful". Residualising *hurts* trait (5.37→3.81) and
leaves record almost untouched (8.4→7.8): trait's effect ran through the evaluative
component, record's does not. `second` scores highest but its probes share its own
second-person framing, so some of that is format match — `record` is the cleaner claim.

**4. In the game, partner-targeted steering barely works** (L46, α=0.25; base level
−0.300, base **slope +9.963**):

| | partner+ | partner− | self+ | rand+ | answer+ |
|---|---|---|---|---|---|
| record | +0.209 | −0.272 | +0.013 | −0.025 | +3.75 |
| recordR | +0.178 | −0.303 | +0.025 | −0.025 | +3.54 |
| trait | +0.125 | −0.050 | −0.072 | −0.025 | +1.64 |
| valence | +0.122 | −0.019 | −0.025 | −0.025 | +1.35 |
| competence | +0.031 | +0.009 | −0.053 | −0.025 | +1.53 |
| news | +0.016 | +0.059 | −0.006 | −0.025 | +1.73 |

Only record/recordR are **bidirectional** and target-specific (~10x both rand+ and
self+); trait and valence push up but not down; competence and news do nothing.
But `answer+` is ~20x larger than `partner+`, which is the read-out-shift
alternative winning: the direction changes the *choice* far better when written at
the output than where the partner is represented. (Treat `all+`/`answer+`
magnitudes as partly artifactual — they inflate SLOPE by +2 to +4, i.e.
distribution-wide distortion.)

**SLOPE is unmoved by partner-targeted steering** (−0.037 under record `partner+`).
Steering a player to be trustworthy does NOT make the model update on them more.
Note the model *does* use the record heavily unprompted (~10 logits between
all_coop and all_defect) — unlike the gossip naming game, where the record was inert.

**5. Behavioural dissociation (`dissociate.py`).** Scenario B (accept an unverified
count vs pay to check) separates them, and scales with dose:

| Δ(accept), L46 | α=0.25 | α=0.5 |
|---|---|---|
| record / recordR | +0.88 / +0.88 | **+1.67 / +1.52** |
| trait | +0.38 | +0.79 |
| competence | +0.25 | +0.44 |
| valence | −0.08 | −0.23 |
| rand | +0.04 | −0.10 |

Trust acts on a decision that is specifically about taking someone's word; valence
is at the random floor. **Competence does NOT flip sign** — same direction as trust,
~4x weaker — so the ability/integrity split does not show up as opposite behaviour.
Scenario A (susceptibility to an unverifiable offer) is a **null at every dose**, but
it is underpowered rather than conclusive: the scam base margin is −15.47, far from
the decision boundary, so nothing of this magnitude could move it.

**α=1.0 is off-distribution** — the matched-norm random direction moves scenario B by
−0.79 and record's own effect collapses from +1.67 to +0.52. `dissociate.py` now
refuses to print a sign verdict when rand exceeds half the effect size.

Limits: one model, one stimulus set, chat format only, diff-in-means throughout.
Run 1 saved only per-schedule means for the game, so the +0.21-vs-−0.03 comparison
has no confidence interval; `steer_qsg.py` now stores raw per-prompt margins.

### Run 2 — 2026-08-11: the read-position control, and two reversals

Run 1 had a flaw: every steering result used vectors read at the LAST token of the
derivation passage, then injected at the PARTNER'S NAME tokens. A last-token vector
is an "about to emit output" direction, so its potency at the answer slot was partly
guaranteed by construction. Run 2 repeats the games and both dissociation scenarios
with vectors read at the name (`ANCHOR=name2`), matching read position to injection
position, and adds two in-domain derivations (`rationale`, `gamerat`) where the
contrast sits in the model's own begun reply rather than in a description.

**Target specificity did not hold up when the read position changed.** L46, α=0.25, Δlevel:

| | `last`: partner+ / self+ | `name2`: partner+ / self+ |
|---|---|---|
| record | +0.200 / +0.003 (64x) | +0.134 / **+0.181** (0.7x) |
| recordR | +0.197 / +0.028 (7x) | +0.097 / +0.094 (1.0x) |
| valence | +0.109 / −0.031 | +0.188 / **+0.244** |

At the matched anchor the vector moves the choice just as much written at the MODEL'S
OWN name as at the partner's. It is a generic nudge, not partner-directed. The clean
64x specificity exists only in the mismatched configuration. (Run 1's other worry was
right, though: `answer+` for record falls +3.74 -> +0.98 once the vector is read at
the name, so much of the "read-out shift dominates" result was itself the mismatch.)

**The trust/valence separation also came out differently.** Scenario B, name
anchor, Δ(accept): **valence +1.771/−1.958**, arbitrary +1.167/−0.625, trait
+0.938/−1.812, record +0.812/−1.542, rand +0.000/+0.146. At the last-token anchor
record moved this +0.88 while valence sat at −0.08. The separation that run 1 called
its strongest evidence is a property of the read position, not of trust.

**The in-domain derivations do not rescue it.** `rationale` validates well on the
held-out trust questions (bidir +11.5, above every control) but does essentially
nothing at the partner's name (+0.044/+0.059, not even bidirectional); `gamerat` is
smaller still.

**Looks more robust:** the geometry (no separation from the controls at either read
position); the observation that the model uses a partner's actual record heavily on
its own (~10 logits) while no injection moves it more than ~0.2; and the conclusion
that steering does not increase the model's rate of updating on a partner.

**Looks weaker:** "`record` is a trust-specific steerable direction", and "`trait`
is essentially a valence direction" (cos(trait, record) = 0.61 > both controls at the
name anchor, reversing the last-token ordering). Whether the flips mean the effects
are fragile, that one read position is the more sensible one, or simply that these
differences are within noise, this run cannot say: most of these numbers have no
error bars, and the gaps are small.

**Worth noting for next time:** read position was chosen once and never treated as
something to vary. Several comparisons moved when it changed, so it is worth
varying alongside proper error bars before reading much into any of these
differences.

### Protocol v2 (2026-08-11) — Sandra's spec, and where it leaves things

Rebuilt to a cleaner design: read at an appended bare name token (no anchor choice),
positive / negative / mixed / blank conditions, 12 names x 12 settings crossed over
7 power relations, competence + honesty + reliability tested as COMPONENTS rather
than throwaway decoys, model-written stories, and four different generation prompts
for those stories so the framing is a measured factor. 16 items per family, 20
families, reliability 0.72-0.98.

**The clearest result in the project is a negative one.** Asked directly at every
depth of a scripted conversation -- "do you trust Bob?", read as logit(yes)-logit(no)
-- the model's stated trust moves enormously and in opposite directions:

```
scam     -11.5 -11.1 -11.1 -14.7 -11.3 -11.3 -13.3 -15.1 -15.0 -15.5
helpful  -11.5 -11.1  -9.4  -4.3  -2.2  -1.0  -0.1  +0.5  -0.1  +0.1
```

11.6 logits of movement in one, a ~4 logit fall in the other, with drops exactly at
the refusal-to-verify and passcode turns. On the IDENTICAL contexts, every candidate
direction's projection drifts UPWARD in both conversations, by 0.3-0.9 z units, with
nothing at the passcode turn. The read-out is not tracking the stance the model
demonstrably has; it tracks something closer to conversation length.

Free generation rules out the obvious alternative that the model simply trusts the
caller: unprompted, it opens by demanding an employee ID ("I want to make sure I'm
not falling victim to a scam") and refuses the code outright ("I'm not going to read
the code back to you... I should hang up and call the number on the back of my card").
Note it starts at -11.5, i.e. already distrusting an unsolicited caller, and in the
helpful conversation only climbs to roughly neutral -- it never becomes trusting.

**Other v2 findings**
- The matrix clusters by prompt FORM, not by property. "Bob is trustworthy" sits at
  0.89 with "Bob is honest" and 0.86 with "Bob is competent", but only 0.67 with the
  elaborated trust description. Trust<->trust +0.19 vs trust<->component +0.32
  (separation -0.12).
- The two game formats -- identical ten rounds, summarised as lists vs written one
  per line -- give directions with cosine **-0.04**. Crossing the relations did not
  change this, so it is not the power-relation confound.
- How you ask the model for stories matters more than what you ask for: "short
  first-person account, ~90 words" replicates at 0.79, bare "write a story about
  someone trustworthy" at 0.46, and unbanned at 0.17. The bare framing spends its
  budget on scene-setting, so the property barely appears.
- The blank neutral ("has yet to prove anything") is not a midpoint: it differs from
  both poles by containing no incident at all, giving cos(pos-neu, neu-neg) of -0.5
  to -0.99. With a content-matched mixed condition that moves to about zero -- still
  not positive, so gaining and losing trust do not look like one axis.

**Fitting a direction to PREDICT stated trust works far better than contrasting
prompts** (`fit_direction.py`). Regressing the name-token activation on the model's
stated trust over 960 contexts, holding out WHOLE FAMILIES so training never sees the
phrasing it is tested on: held-out r = +0.81 (L27) to +0.89 (L52), with within-family
r only slightly higher (+0.86 to +0.91) -- so it is not memorised phrasing. Every
mean-difference direction is a worse predictor of the same quantity (best +0.81 at
L52, most ~0.5) and only 0.40-0.70 aligned with the fitted one. Stated trust is
linearly readable from that single token, across 20 ways of describing it.

**[SUPERSEDED — see the four-domain result below] It only tracks trust going UP.** Within-conversation correlation with the
stated-trust trajectory: fitted L52 gets +0.97 on the helpful conversation and +0.34
on the scam one; mean-difference directions get +0.94 and about zero or negative. The
warmth decoy also gets +0.945 on the helpful conversation, so the helpful column
carries little information -- everything rises together there. Nothing tracks trust
FALLING as a scam unfolds. The fit was trained on static descriptions where the
evidence arrives up front and never on evidence accumulating against someone during
an interaction, which is the obvious thing to fix next.

Caveat on the fitted direction: it is trained to predict answers to a trust question,
so it may be reading "how would I answer that" rather than a representation of trust.

**Four domains, and the corrections they force** (`scenarios.py`, `fit3.py`). The
"only tracks rising trust" conclusion above came from testing on ONE conversation --
the bank scam -- which is atypical because the model is suspicious from the opening
(stated trust starts at -8 to -11) so there is little left to fall. Across four
matched good/bad pairs (bank caller, private used-car seller, builder holding a
deposit, colleague on a shared report), the fitted direction at L52 tracks the
model's stated trust in BOTH directions:

    bank good +0.89 / bad +0.57     builder   good +0.75 / bad +0.79
    car  good +0.61 / bad +0.70     colleague good +0.63 / bad +0.77

In the car, builder and colleague "bad" conversations the projection falls with
stated trust (to -19, -12 and -8 respectively). Testing a trajectory claim on one
scenario was the mistake.

**Steering with the fitted direction works, and is the first clean causal result
here.** (+v) - (-v) on stated trust, L52, paired over probes:

    fitted trust     +5.68 +- .24   +7.82 +- .27   +6.32 +- .20   (alpha .25/.5/1)
    fitted distrust  +0.62 +- .09   +0.90 +- .13   +0.78 +- .17
    random           +0.32 +- .13   +1.05 +- .21   +0.69 +- .15

6-8 logits, about 8x the matched-norm random control and stable across doses,
against ~0.2 logits for the best v1 mean-difference direction. On the iterated game
at L45 it is dose-responsive with a sign flip between the two fitted directions:
trust +0.225/+0.263/+0.650, distrust -0.025/-0.269/-0.475, random about -0.1.

**The distrust direction predicts but does not cause.** Fitted below-median stated
trust, it is near-orthogonal to every other direction in the project (cos 0.00-0.15
with all 20 hand-built, 0.20 with the fitted trust direction) and regresses on stated
trust -- yet injecting it moves stated trust exactly as much as a random direction
(+0.6 to +0.9 vs +0.3 to +1.1). Decodable, not causal. It does push the game toward
defection, so it is not inert everywhere.

**Strength-vs-efficacy sweep, with an integrity check** (`sweep.py`, 9 alphas x 2
layers x 8 directions, 18 mixed-evidence probes). Three results:

*Reliability does not entail efficacy.* Built from the SAME stories under both system
framings, the LESS reproducible own-recollection direction steers as well or better
than the more reproducible reader-framing one at every alpha in the usable range
(L52: +7.4 vs +6.4 at a=0.5, +9.8 vs +8.2 at a=0.75, +10.9 vs +9.8 at a=1.0).
Selecting families on split-half reliability would have picked the wrong one.
Reliability bounds how well a direction can CORRELATE with another; it says nothing
about whether it does anything.

*The usable regime is alpha <= 0.5.* Probability mass on {yes,no} at the answer slot
holds at 1.00 through a=0.5, degrades from 0.75, and collapses to 0.01-0.25 by
a=1.5-2.0. The random direction is flat (-0.9 to +0.7) while the model is intact and
only "works" (+2.7 to +5.4) once it is not. Results quoted at a=1.0 anywhere in this
README sit at the edge of that window. Report an integrity measure alongside any
steering number; inferring damage from the random arm alone is indirect.

*Steering stated trust does not discriminate trust from warmth.* At L45/a=0.35 (mass
1.00): fitted trust +11.2, warmth decoy +9.9, direct_b +9.8. The 6-8 logits vs ~1 for
random is real, but the informative comparison is against a decoy built the same way,
and there the margin is small. What survives is narrower: among the FITTED directions,
trust steers (+8.8) while the lo-half distrust fit sits at the random floor (+1.1)
despite predicting stated trust as well by regression -- same procedure, same data,
opposite causal outcome.

**All-directions steering sweep** (`sweep_all.py`, 24 directions x 7 alphas x 2 layers,
18 mixed-evidence probes). At alpha=0.35 / L45, where model integrity is exactly 1.00
for every direction:

    story_trust@story +13.2   warmth_b (DECOY) +9.9   direct_a  +4.9
    relational        +12.7   direct_b         +9.8   game_a    +4.7
    FITTED trust      +11.2   story_hon        +9.3   comp_a    +4.6
    story_trust@acct  +11.0   rel_b            +9.2   FITTED lo +4.5
                              comp_b           +8.8   game_b    -1.6   random -0.9

Nearly every direction built from a positive-vs-negative description of a person
steers this task, in a band of +4.5 to +13 with the warmth decoy in the middle of it.
The stated-trust read-out is therefore a poor DISCRIMINATOR -- a limitation of the
bed, not a property of the directions. Two things still stand out: the LEAST reliable
direction (`story_trust@story`, split-half 0.41-0.50) is the MOST effective, which is
the strongest form of reliability-is-not-efficacy in this project; and the fitted
direction is the most efficient at the smallest perturbation (alpha=0.05: +3.6 vs
+3.4 next best, random -0.1), peaking at alpha 0.2-0.35 and then declining while the
mean-difference directions keep climbing -- a different curve shape, not just a
different magnitude. `game_b` steers NEGATIVE (-1.6), consistent with its
near-orthogonality to everything in the heatmap.

**Layout note.** This directory moved out of `reciprocal-signaling/` to the top level
and is now self-contained (`src/model_spec.py` vendors the loader). `src/master_slides.py`
builds `out/MASTER.pdf`; its `SLIDES` list is ordered and appendable.

**Steering on tasks with consequences: a null, and it locates the earlier result**
(`newtasks.py`). Three tasks at alpha 0.2-0.5, injection at the ENTITY'S NAME.

*Advisor / conditional.* Two advisers recommend different companies; steering at one
adviser's name should move the forced choice toward that adviser's pick. Every effect
falls within +-0.44 logits with random at +0.19, and the SALIENCE CONTROL (identical
injection on the company name) is frequently LARGER than the effect itself (fitted
trust, conditional, a=0.5: effect -0.44, salience +0.56). The fitted direction nudges
the choice slightly AWAY from the adviser it is supposed to promote. The conditional
variant therefore says nothing about overriding stated expertise -- nothing moves.

*Scam conversation.* At the caller's name, a=0.35: stated trust moves 1.3-2.5 logits
(random moves 0.9 of that), and the behavioural question "do you read the six-digit
code back?" moves 0.1-0.5 logits and stays at about -15 after the request. Steering
does not make the model trust the caller or comply.

*This locates the +11 logit result from the sweep.* That used injection at ALL
POSITIONS and a question ABOUT trust. Injecting at one entity's name and asking for a
DECISION gives ~0. The large number was a global perturbation moving an opinion
read-out, not a change in how the model regards a particular person; those had been
conflated.

*Controls.* Sycophancy is a clean null (0.00 to +-0.12): steering does not produce
generic agreeableness. The valence-halo battery shows PARTIAL specificity -- fitted
trust moves trustworthy +1.00 and likeable +0.94 together, while competent +0.12 and
tall -0.44 do not follow; `warmth_b` moves all five uniformly at +0.25, which is what
an actual halo looks like; random is 0.00 throughout. Certainty (d-entropy ~0.01) and
instruction-compliance (answer-mass 1.00) are clean everywhere in this regime.

**REDO at the name token, all directions everywhere** (`dirs.py` now defines the one
direction set; every script imports it). Earlier runs mixed injection sites and used
different subsets per figure, which made slides disagree.

*Sweep, injected at the person's name instead of all positions* (L45, a=0.5, integrity
1.00 throughout): relational +3.7, FITTED trust +3.3, FITTED hi +3.3, warmth_b +2.6,
direct_b +2.5, story_trust +1.5, FITTED lo -0.4, random -0.8. The +11 logits reported
earlier was an ALL-POSITION injection; targeted at the person it is +3.3, and warmth
remains in the same band as the trust directions.

*Advisor task: null for all 24 directions, and the entity control fails.* Effects
within +-0.31 (random 0.00). The same vector injected into the OTHER adviser's name
moves the choice toward the first adviser's pick by +0.06 to +0.31 -- as much as or
more than injecting into the target's own name -- and injecting into the COMPANY span
moves it up to +1.06, more than either. No entity specificity.

*Sycophancy control is clean, and is the best specificity result here.* One injection
at Ana's name, two questions: trust-in-Ana moves +1.75 (relational), +1.50 (warmth),
+1.38 (story_trust), +1.00 (direct_b), while agree-with-the-USER stays at <= +0.12 for
EVERY direction. Steering is specific to the party named rather than producing generic
agreeableness. Note the ordering differs from the sweep: FITTED trust is only +0.50
here against random +0.38.

**SECOND CORRECTION (same day): the advisor prompt itself was incoherent** — "Ana has
looked at it" had no antecedent, the system prompt was a stub, and the person existed
as one bare token. Sandra caught it by asking to see the system prompt. With a
coherent scenario (v3: system establishes the choice, the advisers, and that the
decision rests on their advice), the "inverted effect" below DISAPPEARS for every
mean-difference direction (all ≈ 0 within SE, random ≈ 0). Only the FITTED direction
survives: consistently negative at both names (−0.41±.09, −0.78±.25), i.e. it pushes
the choice away from the injected person's pick — held loosely at n=4 per cell, with
the conditional bed's random again elevated. Lesson: an incoherent prompt can
manufacture a consistent-looking cross-direction effect; print the full rendered
prompt before interpreting anything measured on it.

**[SUPERSEDED by the above] CORRECTION (2026-08-12): the advisor slide mixed two measurements** — the
target person as a (+v)-(-v) contrast, the other person as one-sided (+v)-baseline —
which manufactured an apparent sign asymmetry. Measured symmetrically (±v at person X,
margin toward X's OWN pick), the plain advisor task shows a small consistent INVERTED
entity-specific effect: 17/18 content-direction cells ≤ 0 (−0.06 to −0.31 logits) for
both people, random ≈ 0. Injecting "trust" at a person's name nudges the choice
slightly AWAY from their recommendation. The conditional bed's random control lands at
+0.50, so that bed is treated as unreliable. Data: out/advisor_sym.json.

**The advisor null diagnosed — no bug, and the sharpest dissociation in the project**
(`diag.py` output, plain advisor prompt, L45). Read-out is clean: top next-tokens are
'V' (0.62) and 'Sol' (0.38), exactly the tokens the margin reads, 99.9% of mass on the
two answers, baseline +0.50 (no ceiling). And the task is exquisitely movable BY TEXT:
"you have worked with Ana for fifteen years and she has never once been wrong" swings
the margin +0.50 -> +13.75 (p=1.00); the same for Bruno swings it to -6.50. ~20 logits
of dynamic range from textual trust, ~0.3 entity-nonspecific logits from the same
nominal content injected as an activation vector at the name.

Reading: the steering vectors shift the disposition to ANSWER trust questions
positively, not the person-representation that decisions consume. That is why "do you
trust Ana?" moves (+0.5 to +1.75) while her recommendation does not — the injection
never touches the mechanism that links trust to choice. It also retroactively frames
every steering "success" on yes/no trust questions in this project.

**Standing conventions from here on**: injection at the judged person's NAME TOKENS
only (dirs.py is the single source of directions); the company-span control is
removed (position confound); all-position steering artifacts live in
out/deprecated_allpos/ and appear in no slide. The slideshow is out/MAIN.pdf.

**Full audit (2026-08-12), prompted by the task nulls looking suspicious.** Verified
on the pod against the shipped JSONs: injection positions decode to exactly the name
tokens; every script's vector is identical to the stored one; recomputing an advisor
cell and a sweep cell reproduces the stored numbers to 3 decimals; the hook fires and
demonstrably alters downstream layers and final logits (max |dlogit| 13 at alpha 8).
One audit check failed and was a bug in the AUDIT itself: transformers 5.x records
hidden_states[L] before hook effects propagate, so injections are invisible there but
present from [L+1] on (noted in common.py; derivation reads are unhooked, unaffected).

**The decisive number from the audit**: at alpha=8 the injection moves final-position
logits by up to 13 while the answer margin moves by 0.000 in all four counterbalanced
variants -- the perturbation is COMMON-MODE with respect to the choice, shifting both
answer logits identically. Steering at the name propagates strongly but carries no
differential information about which option to pick; text carries ~20 logits.

**Direction set pruned** (dirs.CORE, used by all slides): direct_b, relational,
story_trust, story_trust@acct, FITTED trust + controls comp_b/hon_b/rel_b, warmth_b,
random. Removed: FITTED hi/lo (diagnostics), storynb/story (reliability 0.16-0.50),
game_a/b (orthogonal; policy confound), the one-clause quartet (redundant, cos
0.85-0.9), story_comp/hon/rel (components covered by the _b forms).

**Prior-trust directions (2026-08-12): source credibility, not conduct**
(`prior_src.py`). Mean-diff over the same fact attributed to a high- vs low-prior
source, read at the last token of the SHARED fact: prior_wiki ("Wikipedia says," vs
"4chan says,", split-half 0.94), prior_expert (credentialed vs guessing, 0.96),
prior_src (8 institutional pairs cycled, 0.61). Three results:

* On the stated-trust probe they steer with the OPPOSITE SIGN to conduct directions:
  prior_src −2.7 and prior_wiki −1.6 at alpha 1 vs direct_b +2.5, same probe, same
  name-token site. Written into a person, source-credibility acts as skepticism.
* In the advisor battery the one cell beyond the random band is prior_src at the
  wrong-domain expert (conditional): −0.42±.06 vs random +0.11.
* As TRACKERS they work like the conduct directions: helpful-vs-scam paired
  separation +0.51±.08 (prior_wiki L45), +0.38±.05 (prior_src), and prior_expert is
  layer-split (−0.08 at L45 but +1.25±.10 at L52, the strongest tracker measured).

**Advisor battery (8 scenarios x 4 variants, n=32/cell, gate skipped 0):** steering
did not move the decision, at any layer, for any direction. No direction at any depth moves the choice TOWARD
the trusted person. What exists: a diffuse away-from-own-pick tendency at L45 (−0.1
to −0.4, partially name-specific — random itself is −0.15 at Bob), sign-flipped
small positives at L27/35, complete deadness at L52 (±0.02), and cancellation (≈0)
when injecting all four layers at once. Depth does not rescue steering; L52 is too
late to affect this decision at all.

**Limits.** One model. 16 items per family. The behavioural trajectory used 6 names
and greedy decoding. The self-generated-conversation variant is content-confounded
(the model's own words differ hugely between conditions). None of the projection
comparisons between individual directions are large relative to how much they all
share with the warmth decoy.

## Layout

```
src/common.py        model load, chat/raw templating, span→token-index, Inject hook
src/stimuli.py       the eight paired-stimulus generators + held-out probes
src/build_vectors.py stage 1 derive + save (full/h0/h1); stage 2 home-domain validation
src/compare.py       stage 3 cosine matrices, reliability ceiling, controls (no GPU)
src/qsg_games.py     the five games, schedules, prompt builder, position groups
src/steer_qsg.py     stage 4 the injection grid + the update curve
src/mock_test.py     full pipeline on CPU with a fake model (no GPU, no downloads)
run_build.sh         stages 1–3 on the pod
run_steer.sh         stage 4 + dose-response
out/                 vectors.npz, vectors_meta.json, validation.json, compare.json,
                     compare.png, steer_qsg.json
```

`python src/mock_test.py` runs everything on CPU in ~10s and asserts the parts that
fail silently: every read anchor and every steering position group resolves to at
least one token, partner and self positions are disjoint, the hook is a no-op at
v=0 and moves the read-out otherwise. An empty position list makes a steering arm
look like a clean null when it never wrote anything.

Pod: same H200 infra as the rest of the repo (`/workspace/mm/reciprocal-signaling/
trust-vector`), run OFFLINE (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`). The `news`
method uses a synthetic report body rather than `NeelNanda/pile-10k`, so unlike
`gossip/news_source.py` this stage does not need network access.

## Layer indexing

`resid(..., layers=[l])` reads `hidden_states[l]` — the stream *entering* block `l`
(`l=0` is the embedding output). `Inject(model, l, ...)` hooks block `l−1`'s output,
which is that same stream. Read and write use the same integer. This matters when
comparing against `gossip/cross_steer.py`, which hooks block `l` directly.

## Known limits — do not let these get lost

- **Diff-in-means is not identification.** It gives the direction along which the
  two prompt sets differ *on average*, which includes everything correlated with
  the contrast in this stimulus set. The controls bound that; they do not remove it.
  `compare.py`'s `centered` variant (subtracting the across-method mean at each
  layer) is a partial fix, not a solution.
- **Cosine similarity between methods is a comparison of two noisy estimates.**
  Uninterpretable without the split-half ceiling, which is why the diagonal of every
  printed matrix is the reliability and why `disattenuated` is suppressed when a
  reliability is at floor.
- **The vector is built in one prompt format and used in another.** Derivation uses
  a short chat-formatted passage; the game prompt is long and structured. `FMT=raw`
  exists to test whether the direction is format-bound; treat a chat-only effect as
  a finding about format, not about trust.
- **A steering effect is not evidence that the model uses this direction on its
  own.** `answer+` is in the arm list precisely because a shift in the answer slot
  reproduces the behavioural signature with no belief change at all. The gossip
  result — a credibility representation that exists but is never built from the
  record — is the live alternative here too, and `partner_cur+` vs `partner+` is
  the arm that speaks to it.
- **`query` is expected to align with whichever method it was built from** (`trait`)
  rather than with the others; it is included as a read-out reference point, not as
  an independent fifth opinion.
- Sample sizes are small by default (`NPAIR=24`, 5 games × 2 schedules × 2 styles ×
  2 orderings per cell). Every printed delta needs its spread before it is a claim;
  `steer_qsg.json` keeps the per-schedule means for that.
