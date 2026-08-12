# recursive-ft — reciprocal fine-tuning on ESConv strategy priors

Two LoRA adapters A, B on one frozen base (Qwen2.5-1.5B-Instruct), initially
fine-tuned on covariate-matched ESConv supporter turns that differ only in
strategy mix (grouping fixed in `data_prep.py`: E = questions/restatement/
reflection, I = information/suggestions, M = affirmation/reassurance/
self-disclosure). Then recursive updates on a fixed unlabeled context pool P
(gold responses never used):

    reciprocal  A_{t+1} <- FT(A_t, {x, B_t(x)})   B_{t+1} <- FT(B_t, {x, A_t(x)})
    self        A_{t+1} <- A_t(x)                 B_{t+1} <- B_t(x)
    frozen      A_{t+1} <- B_0(x)                 B_{t+1} <- A_0(x)
    static      both    <- fixed 50/50 mix of A_0(x) and B_0(x)

Both teacher datasets are generated before either model updates. Constant
examples/steps per generation; LoRAs continued, never stacked; per-generation
checkpoints in `ckpt_runs/`. Measurement on a fixed held-out 200-context eval
set: strategy distribution via a TF-IDF+logreg classifier trained on reserved
dialogues (held-out acc 0.64, 3-class; confusion matrix kept for prevalence
correction), JSD(A,B), per-context label agreement, length, distinct-2, and
base-model mean per-token logprob as a fluency proxy.

## Pipeline

    ESCONV=/workspace/esconv/ESConv.json python data_prep.py   # splits + classifier
    MIX_A=90,5,5 MIX_B=5,90,5 NFT=1200 python init_ft.py       # A_0, B_0 + manipulation check
    bash run_conditions.sh                                     # 4 conditions, seed 0
    bash run_seeds.sh                                          # reciprocal/self, seeds 1-2
    python analyze_traj.py                                     # traj_summary.pdf (no GPU)

## Results (2026-08-09; 10 generations, 400-context pool, 60 steps/gen;
## reciprocal & self replicated over seeds 0-2, controls seed 0 only)

Manipulation check: A_0 E/I/M = .65/.20/.15, B_0 = .225/.58/.195 (JSD 0.149),
fluency −2.76 vs −2.71, lengths 17.5 vs 23.1 — priors differ, quality matched.
(Covariate note: B_0's training contexts are longer, 113 vs 79 words — strategy
correlates with dialogue position in ESConv.)

1. **Reciprocal training does not homogenize: it is an amplifying anti-phase
   oscillator — replicated 3/3 seeds.** A and B swap strategy identities every
   generation (period 2 — each generation the student adopts its teacher's
   distribution), and the swing amplifies: JSD(A,B) rises .15 → .31/.44/.56
   (seeds 0/1/2), with dominant-strategy mass reaching .80-.90. Pooled
   pre-saturation per-hop gain on the teacher's dominant strategy:
   +0.027 ± 0.043 SD, positive on 27/33 hops — one FT generation slightly
   *overshoots* the teacher's modal strategy, and the overshoot compounds.
2. **Self-recursion sharpens in place — replicated 3/3 seeds** (JSD drifts up
   .15 → .20/.29/.34 without identity swaps). Distributions keep their
   identity but the dominant mode grows (e.g. seed 0 B: I .58 → .82, M
   .20 → .07; seed 1 A: E .68 → .84) — majority-mode amplification without any
   partner. Sharpening strength varies by seed/agent.
3. **Frozen cross-teacher is stable.** One swap onto the teacher distribution,
   then flat (JSD ~.12–.17), with *no* quality drift — so repeated ordinary
   distillation is benign; the drift below is specific to recursion.
4. **Static mixture converges immediately** to ~.48/.33/.19 with JSD ≈ 0 by
   gen 1 and stays there. Reciprocal dynamics are therefore not "approaching
   the mixture" — the co-evolving loop behaves qualitatively differently from
   every control. Residual conditional structure in static is small
   (conditional JSD ~.01–.02) though per-context agreement (.55) stays above
   the marginal-chance level (.38).
5. **Quality drift only under recursion** (all recursive seeds): fluency
   degrades monotonically (−2.7 → −3.5..−4.2), lengths inflate (18 → ~30
   words), and distinct-2 falls (.74 → .43–.60); all three stay flat in frozen
   and static.
6. **Minority strategy (M):** declines under recursion in most seed×agent
   cells (final .05–.17 vs initial .15–.22; strongest under reciprocal,
   .05–.10 by gen 10 in seeds 1–2) but is not extinct in 10 generations;
   stable in the controls.

Caveats: controls at 1 seed (recursive conditions at 3); classifier-based
measurement (raw distributions reported; corrected prevalences stored in the
trajectories); one model size, one initial-mix pair; gen-0 remeasurement across
seeds puts sampling noise at about ±.03 on any single distribution entry.

Next per proposal: dynamical map over initial mixes
(90/5/5, 70/15/15, 55/30/15 grid) → minority-extinction thresholds (M at
1/5/10/20%) → mech interp on the per-generation checkpoints.

## Files

    data_prep.py      ESConv dialogue-level splits + strategy classifier
    engine.py         LoRA/generation/training/measurement shared machinery
    init_ft.py        generation-0 fine-tuning + manipulation check
    run_recursion.py  one condition x seed (env: COND GENS NPOOL STEPS SEED)
    run_conditions.sh / run_seeds.sh   pod batch drivers (nohup, DONE markers)
    analyze_traj.py   trajectory panels -> traj_summary.pdf
    traj_<cond>[_s<seed>].json         per-generation measurements
