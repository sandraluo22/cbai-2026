"""Scenario constructor for the four BHW scenario families.

A single grid covers all four:
  - q in {0.6, 0.667, 0.75}             (signal reliability)
  - public history = a uniform RUN of one color, length L in {0..L_max}
  - focal private signal = AGREE (same as the run) or OPPOSE (against the run)

From this grid every family is recoverable:
  family 1 PRE/POST cascade : OPPOSE, L=1 (pre, rational not yet cascading),
                              L=2 (just at onset), L>=3 (clearly past threshold)
  family 2 PRIVATE-CONTRADICTS: OPPOSE, any L>=1, sweep the public tally strength
  family 3 WRONG cascade    : OPPOSE with assumed true_state = the focal's (correct)
                              signal — the run cascaded into the WRONG state
  family 4 DEFECTION sweep   : OPPOSE across L = the headline curve
The AGREE arm is the control (private agrees with the public run).

Each scenario carries its exact rational + naive solution (from urn.py).
"""
from __future__ import annotations
from urn import Scenario, RED, BLUE, rational_decision, naive_decision, cascade_onset


def _run(color: str, length: int) -> list[str]:
    return [color] * length


def build_scenarios(qs, l_max=5, run_color=RED) -> list[Scenario]:
    other = BLUE if run_color == RED else RED
    out: list[Scenario] = []
    for q in qs:
        for L in range(0, l_max + 1):
            hist = _run(run_color, L)
            onset = cascade_onset(hist)
            for arm, sig in (("agree", run_color), ("oppose", other)):
                # pre/post tag for the OPPOSE arm (the diagnostic one)
                rat = rational_decision(hist, sig, q)
                if L == 0:
                    phase = "no_history"
                elif not rat["in_cascade"]:
                    phase = "pre_cascade"
                elif L == 2:
                    phase = "at_onset"
                else:
                    phase = "post_cascade"
                # wrong-cascade framing: assume the truth is the focal's own signal,
                # so an OPPOSE focal in a run is the welfare-relevant "correct private,
                # wrong public cascade" case.
                true_state = sig if arm == "oppose" else run_color
                out.append(Scenario(
                    history=hist, own_signal=sig, q=q, true_state=true_state,
                    family=f"run-{arm}",
                    meta={"L": L, "run_color": run_color, "arm": arm,
                          "phase": phase, "onset": onset,
                          "scenario_id": f"q{q}_L{L}_{arm}_{run_color}"}))
    return out


def describe(s: Scenario) -> str:
    r = s.rational(); n = s.naive()
    h = "[" + ",".join("R" if c == RED else "B" for c in s.history) + "]"
    return (f"{s.meta['scenario_id']:>22} | hist={h:<14} +{s.own_signal[0].upper()}sig "
            f"| phase={s.meta['phase']:<12} onset={str(s.meta['onset']):>4} "
            f"| RATIONAL choose={r['choice'][0].upper()} P(red)={r['posterior_red']:.3f} "
            f"cascade={str(r['in_cascade']):>5} followPriv={str(r['follows_private']):>5} "
            f"|| NAIVE choose={n['choice'][0].upper()} P(red)={n['posterior_red']:.3f}")


if __name__ == "__main__":
    scen = build_scenarios([2 / 3])
    print(f"{len(scen)} scenarios at q=2/3 (×3 q-values + color-swap variants at runtime)\n")
    # show the OPPOSE arm sweep (the headline defection curve) + a few AGREE controls
    print("=== OPPOSE arm (private contradicts the public run) — the diagnostic sweep ===")
    for s in scen:
        if s.meta["arm"] == "oppose":
            print(describe(s))
    print("\n=== AGREE arm (control) — a couple ===")
    for s in scen:
        if s.meta["arm"] == "agree" and s.meta["L"] in (2, 4):
            print(describe(s))
