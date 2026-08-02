# gossip — QSG naming game + revealed truth: does reliability reputation form?

Base protocol: the LLM Neutral-Naming-Drift experiment from Tanaka, *"When Is
Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in
LLMs"* (arXiv:2603.24676), Appendix B.2 / Fig. 10. We reuse the paper's prompt
template (JSON-only system message; user message with Referent, shuffled Allowed
labels + "order is randomized" note, `<PAD>`-padded memory list, constraints
block, `{"label": "<label>"}` output spec), its delayed-reveal ordered
speaker→listener interactions (listener never sees the speaker's current output;
only the listener's memory updates), synthetic 5-character labels, fixed referent,
Hard (m=1) messaging, uniformly random ordered pairs.

**The twist (this experiment):** after every round an *arbitrary* correct label is
drawn and revealed to all agents, then play continues with the accumulated
context. Since the correct label is unpredictable, an agent can only beat 1/K
chance in-round by following a peer who knows it — reputation is the only channel.

## Departures from the paper (all deliberate, needed for the question)
1. Memory entries are attributed (`"P4: vokhg"`) and shown GROUPED BY ROUND with
   each past round's revealed correct answer interleaved — the paper's bare
   anonymous flat list makes per-agent trust unformable in principle.
2. Memory is unbounded (spans all 5 rounds); the paper truncates to H=10.
3. Revealed per-round ground truth (the paper has none).
4. Conditions may give agent P1 a private clue line.
5. v2 measurement: NO free generation and NO self-report trust question. Beliefs
   are bounded logit reads (softmax over the K label first-tokens after the
   prefill `{"label": "`); the speaker emits by SAMPLING its belief (exactly QSG
   Hard, k* ~ Cat(x_S)); the listener's belief is read before and after each
   observed label, so every update is attributable to (round, listener's
   conversation depth, source). Reputation = source-dependence of the update.
   Memory content is 100% agent-generated (peers' sampled outputs); the reveal
   lines are the only exogenous text.

## Conditions (VAR)
| VAR | P1's private clue |
|---|---|
| `none` | never (pure drift baseline) |
| `informed_r1` | correct label, round 1 only |
| `informed_all` | correct label, every round |
| `misinformed_r1` | WRONG label, round 1 only (thinks it knows) |
| `misinformed_all` | WRONG label, every round |

Defaults: N=5 agents, ROUNDS=5, K=3 labels, STEPS=75 interactions/round, TEMP=1.0
(Hard sampling), model Qwen3-32B (thinking off; the paper used GPT-4o / Claude
Haiku — model family is a free variable here).

## Readouts (`gossip_analyze.py`)
- `probe_acc[r]` — mean per-round probe accuracy vs the arbitrary truth (chance
  1/K; exceeding it requires following an informed P1)
- `follow_P1[r]` — probes matching P1's clue (in misinformed conditions this is
  *misplaced* trust)
- `adopt[Pj]` early vs late — exposure-normalized copy rate of each source's most
  recently observed label; a rising `adopt[P1]` in informed conditions (or a
  falling one in misinformed) is reputation formation
- final trust votes histogram

## Run
`bash run_gossip.sh` (all 5 variations, one seed) → `<MODEL>_<VAR>/gossip_s<seed>_transcript.jsonl`
