"""GAME 1 -- Convergence game (NO-REPEAT) + perturbation / KL-coupling instrument.

Two frozen agents try to CONVERGE: each turn both simultaneously emit one token and
observe the other's, aiming to emit the SAME token on the same turn (agreement).

  NO-REPEAT RULE: a token emitted by EITHER model on a previous turn is forbidden
  thereafter. This removes the trivial solution (just echo the other's last token)
  and forces genuine coordination -- each turn both must independently land on the
  same FRESH token by predicting where the other will go (a Schelling problem).

Two modes (env MODE):
  * words   : abstract tokens; agents coordinate via a shared salience prior.
  * numbers : the tokens ARE numbers 0..V-1 and the goal is to agree on the same
              number, again with no repeats.

Levels: L1 sticks to its own private lean and reacts to the other's raw pick; L2
drops its private bias and heads for the shared focal point both can predict -> it
converges faster.

Instrument (unchanged): a CONTROLLED COUNTERFACTUAL -- from an identical state, feed
B the other's real pick (CLEAN) or a swap (SWAP), read B's next-turn distribution in
both, and KL(B_swap || B_clean). Null swap = same token. Frozen agents, so KL isolates
the causal effect, not drift.

Env: MODE(words|numbers|both) VOCAB(20) TURNS(8) SEEDS(24) TEMP(0.7) GAMMA(0.6)
     WM(1.5) RUN_DIR
Out: <RUN_DIR>/game1_<mode>.pdf + game1_summary.json
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core as K

MODE = os.environ.get("MODE", "both")
V = int(os.environ.get("VOCAB", "20"))
TURNS = int(os.environ.get("TURNS", "8"))
SEEDS = int(os.environ.get("SEEDS", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))       # pick temperature (sharp -> decisive picks)
READ_TEMP = float(os.environ.get("READ_TEMP", "2.5"))  # softer readout for a bounded, interpretable coupling KL
GAMMA = float(os.environ.get("GAMMA", "0.6"))
WM = float(os.environ.get("WM", "1.5"))          # weight on the "what will the other pick" term
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
LOGEPS = 1e-9


def onehot(o, v):
    e = np.zeros(v); e[o] = 1.0; return e


class Agent:
    """Coordination agent. `sal` = shared public salience (focal structure); `bias`
    = private lean (hidden, creates disagreement to resolve); `m` = running estimate
    of the other's next pick."""
    def __init__(self, V, level, sal, bias, gamma=GAMMA):
        self.V, self.level, self.gamma = V, level, gamma
        self.sal = sal
        self.q = K.normalize(sal * np.exp(bias))     # own preference (salience + private bias)
        self.m = sal.copy()                           # estimate of other's next pick (starts at shared prior)

    def dist(self, used, temp=TEMP):
        allowed = np.array([t not in used for t in range(self.V)], dtype=bool)
        # coordination score: be where the other will be, weighted by focal salience.
        lean = np.log(self.sal + LOGEPS) if self.level >= 2 else np.log(self.q + LOGEPS)
        score = lean + WM * np.log(self.m + LOGEPS)
        score = np.where(allowed, score, -1e9)
        return K.softmax(score / temp)

    def observe(self, o):
        self.m = (1 - self.gamma) * self.m + self.gamma * onehot(o, self.V)

    def copy(self):
        a = Agent.__new__(Agent)
        a.V, a.level, a.gamma, a.sal = self.V, self.level, self.gamma, self.sal
        a.q = self.q.copy(); a.m = self.m.copy()
        return a


def coupling_from_state(B, used, o_real, o_swap):
    # read the distribution at a softer temperature so KL stays bounded/interpretable
    clean = B.copy(); clean.observe(o_real)
    swap = B.copy(); swap.observe(o_swap)
    dc, ds = clean.dist(used, READ_TEMP), swap.dist(used, READ_TEMP)
    return K.kl(ds, dc), dc, ds


def salience_for(mode, rng):
    """words: random shared focal structure. numbers: focal points on 'round'
    numbers (multiples of 5) + the extremes -- a real Schelling-on-numbers prior."""
    if mode == "numbers":
        s = np.ones(V)
        for i in range(V):
            if i % 5 == 0:
                s[i] += 2.5
            if i in (0, V - 1):
                s[i] += 1.0
        return K.normalize(s * np.exp(rng.normal(0, 0.15, V)))
    return K.normalize(np.exp(rng.normal(0, 0.8, V)))


def play(level, seed, mode="words"):
    rng = np.random.default_rng(seed)
    sal = salience_for(mode, rng)                             # shared focal structure
    biasA = rng.normal(0, 0.9, V); biasB = rng.normal(0, 0.9, V)   # private, hidden
    A = Agent(V, level, sal, biasA); B = Agent(V, level, sal, biasB)
    used = set()
    picksA, picksB, agree_turn = [], [], None
    real, null = [], []
    heat = np.zeros((V, TURNS))
    shift = None
    for t in range(TURNS):
        dA, dB = A.dist(used), B.dist(used)
        a = int(rng.choice(V, p=dA)); b = int(rng.choice(V, p=dB))
        picksA.append(a); picksB.append(b)
        if a == b and agree_turn is None:
            agree_turn = t
        # coupling: perturb A's pick, measure B (from identical pre-observe state)
        swap_b = next((j for j in range(V) if j not in used and j != a), a)
        kr, dc, ds = coupling_from_state(B, used, a, swap_b)
        kn, _, _ = coupling_from_state(B, used, a, a)
        real.append(kr); null.append(kn)
        for j in range(V):
            heat[j, t] = coupling_from_state(B, used, a, j)[0] if j not in used else np.nan
        if t == min(2, TURNS - 1):
            shift = (dc, ds, a, swap_b, set(used))
        # commit + no-repeat
        A.observe(b); B.observe(a)
        used.add(a); used.add(b)
    return {"picksA": picksA, "picksB": picksB, "agree_turn": agree_turn if agree_turn is not None else TURNS,
            "converged": agree_turn is not None, "real": real, "null": null, "heat": heat, "shift": shift}


def run_mode(mode):
    agg = {}
    exemplar = {}
    for tag, lvl in [("L1", 1), ("L2", 2)]:
        runs = [play(lvl, s, mode) for s in range(SEEDS)]
        real = np.array([r["real"] for r in runs]); null = np.array([r["null"] for r in runs])
        agg[tag] = {
            "coupling_real_mean": real.mean(0).tolist(), "coupling_real_se": (real.std(0) / np.sqrt(SEEDS)).tolist(),
            "coupling_null_mean": null.mean(0).tolist(),
            "agree_turn_mean": float(np.mean([r["agree_turn"] for r in runs])),
            "converged_frac": float(np.mean([r["converged"] for r in runs])),
        }
        exemplar[tag] = next((r for r in runs if r["converged"]), runs[0])
        print(f"[game1/{mode}] {tag}: converged={agg[tag]['converged_frac']:.2f} "
              f"mean_turns_to_agree={agg[tag]['agree_turn_mean']:.2f} "
              f"coupling real={real.mean():.3f} null={null.mean():.3f}", flush=True)
    make_figures(mode, agg, exemplar)
    return agg


def _label(mode, t):
    return str(t) if mode == "numbers" else f"w{t}"


def make_figures(mode, agg, exemplar):
    turns = np.arange(1, TURNS + 1)
    with PdfPages(os.path.join(RUN_DIR, f"game1_{mode}.pdf")) as pdf:
        # ---- convergence trajectory (both agents' picks; star at agreement) ----
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for j, tag in enumerate(["L1", "L2"]):
            ex = exemplar[tag]
            ax[j].plot(turns, ex["picksA"], "-o", color="tab:blue", label="A pick")
            ax[j].plot(turns, ex["picksB"], "-s", color="tab:orange", label="B pick")
            for t in range(TURNS):
                if ex["picksA"][t] == ex["picksB"][t]:
                    ax[j].plot(t + 1, ex["picksA"][t], "*", color="red", ms=18, zorder=5)
            ax[j].set_xlabel("turn")
            ax[j].set_ylabel("number picked" if mode == "numbers" else "token id")
            ax[j].set_title(f"{tag}: convergence trajectory (red ★ = agreement)  "
                            f"[no-repeat]", fontsize=9)
            ax[j].legend(fontsize=8); ax[j].grid(alpha=.3)
        fig.suptitle(f"GAME 1 [{mode}] — no-repeat convergence: agree on the same "
                     f"{'number' if mode=='numbers' else 'token'}.  "
                     f"turns-to-agree L1={agg['L1']['agree_turn_mean']:.1f} vs "
                     f"L2={agg['L2']['agree_turn_mean']:.1f}", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- coupling over turns + turns-to-agree bar ----
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for tag, c in (("L1", "tab:green"), ("L2", "tab:purple")):
            d = agg[tag]
            ax[0].errorbar(turns, d["coupling_real_mean"], yerr=d["coupling_real_se"], color=c,
                           label=f"{tag} real swap", capsize=2)
            ax[0].plot(turns, d["coupling_null_mean"], color=c, ls=":", alpha=.7, label=f"{tag} null")
        ax[0].set_xlabel("turn"); ax[0].set_ylabel("KL(B_swap || B_clean)")
        ax[0].set_title("Coupling over turns (counterfactual swap)", fontsize=9)
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        bars = [agg["L1"]["agree_turn_mean"], agg["L2"]["agree_turn_mean"]]
        fr = [agg["L1"]["converged_frac"], agg["L2"]["converged_frac"]]
        ax[1].bar([0, 1], bars, color=["tab:green", "tab:purple"])
        for i, (b, f) in enumerate(zip(bars, fr)):
            ax[1].text(i, b + 0.1, f"{f*100:.0f}% conv.", ha="center", fontsize=9)
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["L1", "L2"])
        ax[1].set_ylabel("mean turns to first agreement (lower=better)")
        ax[1].set_title("Coordination efficiency by level", fontsize=9)
        fig.suptitle(f"GAME 1 [{mode}] — coupling instrument + coordination efficiency", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    modes = ["words", "numbers"] if MODE == "both" else [MODE]
    out = {}
    for m in modes:
        out[m] = run_mode(m)
    json.dump(out, open(os.path.join(RUN_DIR, "game1_summary.json"), "w"), indent=2)
    print(f"[game1] DONE -> {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
