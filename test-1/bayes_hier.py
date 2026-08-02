"""Hierarchical (latent-source) upgrade of the Dirichlet-Markov surrogate.

Three learner classes, all with the fitted Llama constants (gamma=0.96, alpha0=0.05),
all processing the EXACT token streams of the saved runs (reconstructed from gen_logs):

  merged : one discounted count matrix C;      predictive from C[x_{t-1}]
  parity : per-phase matrices C_0, C_1 (phase = joint-step parity, the generator
           schedule); bigram updates C_phi[x_{t-1}, x_t]; predictive from
           C_{phi(t)}[x_{t-1}]
  lag2   : interleaved-Markov / latent-source learner: per-phase matrices over
           SAME-PHASE consecutive tokens, C_phi[x_{t-2}, x_t]; predictive from
           C_{phi(t)}[x_{t-2}]  -- each phase is its own Markov chain.

Prefix handling: prefix transitions update all matrices (the prior applies to every
source); with gamma=0.96 the prefix has decayed to ~0 by the late window anyway.

Evaluation mirrors probe_phase.py on late joint steps [302, 600):
  real (ctrl_real)  : mass on ring-nbrs(x_{t-2}) / ring-nbrs(x_{t-1}) /
                      grid-nbrs(x_{t-1}) before R-turns (mirror for G-turns)
  coupled (k2_fix)  : phase contrast of ring/grid neighbour mass
Then fit the mixture weight lam: predictive = lam * lag2 + (1-lam) * merged that best
reproduces the LLM's six real-condition numbers.

Also: a coupled-generation simulation (grid+ring priors, top-2, 6 pairs, T=400)
comparing merged vs lag2 learner PAIRS: do hierarchical learners preserve both
cultures (per-phase native validity) while still reaching predictive consensus?

Out: runs/bayes_hier.json, figs/fig_phase_model.png
"""
from __future__ import annotations
import json, os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cross-model", "src"))
from config import get_config
import graph as G

GAMMA, A0 = 0.96, 0.05
N = 16
LATE0 = 302


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def load_run(rundir, wl):
    log = json.load(open(os.path.join(HERE, rundir, "gen_log.json")))
    z = np.load(os.path.join(HERE, rundir, "nodemeans_dueling.npz"), allow_pickle=False)
    words = [str(w) for w in z["words"]]
    P, T, CTX = log["npairs"], log["tgen"], log["ctx"]
    joint = np.array([[s["node"] for s in log["steps"][f"pair{p}"]][:T]
                      for p in range(P)])
    cfg = replace(get_config("gemma_qwen"), n_walks=P, walk_length=wl,
                  seed=log.get("seed", 0))
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    gw = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    rw = G.generate_walks(ring, replace(cfg, graph_type="ring"))
    return dict(P=P, T=T, CTX=CTX, joint=joint, A_g=adjacency(grid),
                A_r=adjacency(ring),
                prefixes={"grid": [w.nodes[:CTX] for w in gw],
                          "ring": [w.nodes[:CTX] for w in rw]})


class Learner:
    def __init__(self, kind):
        self.kind = kind
        self.C = np.zeros((N, N))
        self.C0 = np.zeros((N, N))
        self.C1 = np.zeros((N, N))

    def decay(self):
        self.C *= GAMMA
        self.C0 *= GAMMA
        self.C1 *= GAMMA

    def prefix_step(self, prev, x):
        self.decay()
        self.C[prev, x] += 1
        self.C0[prev, x] += 1
        self.C1[prev, x] += 1

    def joint_step(self, t, x, x1, x2):
        """x appended at joint step t; x1 = token at t-1, x2 = token at t-2."""
        self.decay()
        self.C[x1, x] += 1
        Cp = self.C0 if t % 2 == 0 else self.C1
        if self.kind == "parity":
            Cp[x1, x] += 1
        else:                                   # lag2 (and merged keeps copies unused)
            Cp[x2, x] += 1

    def predictive(self, t, x1, x2):
        if self.kind == "merged":
            row = A0 + self.C[x1]
        elif self.kind == "parity":
            row = A0 + (self.C0 if t % 2 == 0 else self.C1)[x1]
        else:
            row = A0 + (self.C0 if t % 2 == 0 else self.C1)[x2]
        return row / row.sum()


def stream_eval(run, ctxname):
    """Run each learner over prefix+joint of every pair; collect late predictive
    masses in the probe's format."""
    out = {}
    for kind in ("merged", "parity", "lag2"):
        masses = {k: [] for k in ("even_ring_lag2", "even_ring_lag1", "even_grid_lag1",
                                  "odd_grid_lag2", "odd_grid_lag1", "odd_ring_lag1",
                                  "par0_ring", "par0_grid", "par1_ring", "par1_grid")}
        preds_by_t = {}
        for p in range(run["P"]):
            L = Learner(kind)
            pref = run["prefixes"][ctxname][p]
            for a, b in zip(pref, pref[1:]):
                L.prefix_step(a, b)
            seq = list(run["joint"][p])
            for t in range(run["T"]):
                x1 = seq[t - 1] if t >= 1 else pref[-1]
                x2 = seq[t - 2] if t >= 2 else pref[-1]
                if t >= LATE0:
                    pr = L.predictive(t, x1, x2)
                    preds_by_t.setdefault((p, t), pr)
                    par = t % 2
                    masses[f"par{par}_ring"].append(pr[run["A_r"][x1]].sum())
                    masses[f"par{par}_grid"].append(pr[run["A_g"][x1]].sum())
                    if par == 0:
                        masses["even_ring_lag2"].append(pr[run["A_r"][x2]].sum())
                        masses["even_ring_lag1"].append(pr[run["A_r"][x1]].sum())
                        masses["even_grid_lag1"].append(pr[run["A_g"][x1]].sum())
                    else:
                        masses["odd_grid_lag2"].append(pr[run["A_g"][x2]].sum())
                        masses["odd_grid_lag1"].append(pr[run["A_g"][x1]].sum())
                        masses["odd_ring_lag1"].append(pr[run["A_r"][x1]].sum())
                L.joint_step(t, seq[t], x1, x2)
        out[kind] = {k: float(np.mean(v)) for k, v in masses.items() if v}
        out.setdefault("_preds", {})[kind] = preds_by_t
    return out


def coupled_sim(kind, run, seed=3, topk=2, T=400):
    """Two learners of one class, primed grid/ring, alternating (ring even turns)."""
    rng = np.random.default_rng(seed)
    A_r, A_g = run["A_r"], run["A_g"]
    ph_valid = {0: [], 1: []}
    jss = []
    for p in range(6):
        La, Lb = Learner(kind), Learner(kind)          # a = grid-primed, b = ring
        for L2, pref in ((La, run["prefixes"]["grid"][p]),
                         (Lb, run["prefixes"]["ring"][p])):
            for a, b in zip(pref, pref[1:]):
                L2.prefix_step(a, b)
        seq = []
        pa, pb = run["prefixes"]["grid"][p][-1], run["prefixes"]["ring"][p][-1]
        for t in range(T):
            x1 = seq[t - 1] if t >= 1 else (pb if t % 2 == 0 else pa)
            x2 = seq[t - 2] if t >= 2 else x1
            gen = Lb if t % 2 == 0 else La
            pr = gen.predictive(t, x1, x2)
            if t >= T // 2:
                jss.append(_js(La.predictive(t, x1, x2), Lb.predictive(t, x1, x2)))
            pp = pr.copy()
            pp[np.argsort(pp)[:-topk]] = 0
            x = int(rng.choice(N, p=pp / pp.sum()))
            seq.append(x)
            for L2 in (La, Lb):
                L2.joint_step(t, x, x1, x2)
        for t in range(T // 2 + 2, T):                  # late per-phase lag-2 validity
            par = t % 2
            valid = (A_r if par == 0 else A_g)[seq[t - 2], seq[t]]
            ph_valid[par].append(bool(valid))
    return {"phase0_ring_lag2_valid": float(np.mean(ph_valid[0])),
            "phase1_grid_lag2_valid": float(np.mean(ph_valid[1])),
            "late_js": float(np.mean(jss))}


def _js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mk = a > 0
        return float((a[mk] * np.log(a[mk] / np.maximum(b[mk], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    probe = json.load(open(os.path.join(HERE, "runs", "out_probe", "phase_probe.json")))
    real = load_run("runs/out_ctrl_real", 1300)
    coup = load_run("runs/out_k2_fix", 1000)

    ev_real = stream_eval(real, "ring")
    ev_coup = stream_eval(coup, "ring")
    KEYS = ("even_ring_lag2", "even_ring_lag1", "even_grid_lag1",
            "odd_grid_lag2", "odd_grid_lag1", "odd_ring_lag1")
    llm_real = {k: np.mean([probe["real"]["lag"][c][k] for c in ("grid", "ring")])
                for k in KEYS}

    # mixture fit: masses are linear in the predictive, so mix the masses
    lams = np.linspace(0, 1, 101)
    errs = [np.mean([(lam * ev_real["lag2"][k] + (1 - lam) * ev_real["merged"][k]
                      - llm_real[k]) ** 2 for k in KEYS]) for lam in lams]
    lam_hat = float(lams[int(np.argmin(errs))])

    sims = {kind: coupled_sim(kind, coup) for kind in ("merged", "lag2")}

    out = {"lambda_hat": lam_hat,
           "real_masses": {"LLM": {k: float(llm_real[k]) for k in KEYS},
                           **{kind: {k: ev_real[kind][k] for k in KEYS}
                              for kind in ("merged", "parity", "lag2")},
                           "mixture": {k: lam_hat * ev_real["lag2"][k] +
                                       (1 - lam_hat) * ev_real["merged"][k]
                                       for k in KEYS}},
           "coupled_phase_contrast": {
               "LLM": float(np.mean([probe["coupled"]["phase_pred"][c]["par0_ringnbrs"]
                                     - probe["coupled"]["phase_pred"][c]["par1_ringnbrs"]
                                     for c in ("grid", "ring")])),
               **{kind: ev_coup[kind]["par0_ring"] - ev_coup[kind]["par1_ring"]
                  for kind in ("merged", "parity", "lag2")}},
           "coupled_sim": sims}
    json.dump(out, open(os.path.join(HERE, "runs", "bayes_hier.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    labels = ["ring-nbrs\n(t-2)", "ring-nbrs\n(t-1)", "grid-nbrs\n(t-1)"]
    series = [("LLM", "#111111"), ("merged", "#999999"), ("parity", "#8ecae6"),
              ("lag2", "#0e7c86"), ("mixture", "#fb8500")]
    ax = axes[0]
    xs = np.arange(3)
    for i, (nm, col) in enumerate(series):
        vals = [out["real_masses"][nm][k] for k in KEYS[:3]]
        ax.bar(xs + (i - 2) * 0.16, vals, 0.15, color=col,
               label=nm + (f" (λ={lam_hat:.2f})" if nm == "mixture" else ""))
    ax.axhline(2 / 16, color="crimson", ls=":", lw=1, label="ring chance")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("evidence source for the prediction")
    ax.set_ylabel("predictive mass before R-turns (real stream, late)")
    ax.set_title("lag-2 de-interleaving: LLM vs learner classes")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    names = [s[0] for s in series if s[0] != "mixture"]
    vals = [out["coupled_phase_contrast"][nm] for nm in names]
    ax.bar(names, vals, color=[s[1] for s in series if s[0] != "mixture"])
    ax.set_xlabel("learner")
    ax.set_ylabel("ring-mass contrast: before R-turns $-$ before G-turns")
    ax.set_title("phase awareness in the coupled stream")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[2]
    m = sims["merged"]; h = sims["lag2"]
    xs = np.arange(3)
    ax.bar(xs - 0.18, [m["phase0_ring_lag2_valid"], m["phase1_grid_lag2_valid"],
                       m["late_js"]], 0.35, color="#999999", label="merged pair")
    ax.bar(xs + 0.18, [h["phase0_ring_lag2_valid"], h["phase1_grid_lag2_valid"],
                       h["late_js"]], 0.35, color="#0e7c86", label="hierarchical pair")
    ax.set_xticks(xs)
    ax.set_xticklabels(["phase-0 stream\nring-valid (lag2)",
                        "phase-1 stream\ngrid-valid (lag2)", "late JS\ndivergence"],
                       fontsize=8)
    ax.set_ylabel("fraction / divergence")
    ax.set_title("coupled simulation: do hierarchical learners preserve both cultures?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Hierarchical (latent-source) Dirichlet-Markov upgrade "
                 f"(γ={GAMMA}, α₀={A0})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_phase_model.{ext}"), dpi=160)
    print("DONE -> figs/fig_phase_model.png")


if __name__ == "__main__":
    main()
