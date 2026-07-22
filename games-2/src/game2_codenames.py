"""GAME 2 -- Sequential-reveal Codenames, fused with GAME 1's counterfactual
instrument (the recommended setup).

The task gives GROUND TRUTH (the speaker's hidden target set T), so RECOVERY is
validatable; the counterfactual fork gives CAUSAL COUPLING; the board simplex keeps
the read-out bounded no matter how rich the clue channel is. Per round we log,
together:
    (i)   B's target-posterior mass          (recovery)
    (ii)  counterfactual clue-KL             (coupling: B conditions on A)
    (iii) A's clue-shift under counterfactual B-histories (A's level-2-ness)

Conditions swept: {L1 pair, L2 pair} x {forced C<N, easy C>=N} x seeds.

Env: N(16) M(4) CFORCED(3) CEASY(16) ROUNDS(8) SEEDS(24) ALPHA(4) BETA(2.5)
     TAU(1.0) RUN_DIR
Out: <RUN_DIR>/game2_*.pdf + game2_summary.json
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

N = int(os.environ.get("N", "16"))
M = int(os.environ.get("M", "4"))
CFORCED = int(os.environ.get("CFORCED", "3"))
CEASY = int(os.environ.get("CEASY", str(N)))
ROUNDS = int(os.environ.get("ROUNDS", "8"))
SEEDS = int(os.environ.get("SEEDS", "24"))
ALPHA = float(os.environ.get("ALPHA", "4.0"))
BETA = float(os.environ.get("BETA", "2.5"))
TAU = float(os.environ.get("TAU", "1.0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")


def run_game(C, level, seed, probe=True):
    """One game. `level` sets BOTH agents' level (1 or 2). Returns a log dict."""
    board = K.Board(n_items=N, n_clues=C, n_targets=M, seed=seed)
    T = board.sample_targets(seed + 100)
    S = K.SpeakerAgent(board, T, level=level, alpha=ALPHA, tau=TAU)
    L = K.ListenerAgent(board, level=level, beta=BETA, alpha=ALPHA)
    rng = np.random.default_rng(seed + 7)

    log = {"T": T, "C": C, "level": level, "boards": [], "mass": [], "f1": [],
           "entropy": [], "clues": [], "coupling_real": [], "coupling_null": [],
           "adaptivity": []}
    for r in range(ROUNDS):
        # ---- coupling instrument: fork the CLUE from the identical pre-clue state
        if probe:
            c_real, _ = S.clue(L, rng=rng)
            c_swap = (c_real + 1) % C            # a real (different) swap
            base = L.copy()
            gd_clean = base.copy(); gd_clean.update(c_real)
            gd_swap = base.copy(); gd_swap.update(c_swap)
            gd_null = base.copy(); gd_null.update(c_real)   # null swap = same clue
            log["coupling_real"].append(K.kl(gd_swap.guess_dist(), gd_clean.guess_dist()))
            log["coupling_null"].append(K.kl(gd_null.guess_dist(), gd_clean.guess_dist()))
            # ---- adaptivity instrument: A's clue under counterfactual B-histories
            L_naive = K.ListenerAgent(board, level=level, beta=BETA, alpha=ALPHA)
            log["adaptivity"].append(K.kl(S.clue_dist(L), S.clue_dist(L_naive)))
            c = c_real
        else:
            c, _ = S.clue(L, rng=rng)

        L.update(c)
        g = L.pick_guess()
        ok = g in S.remaining
        L.observe(g, ok); S.observe(g, ok)

        b = L.belief()
        log["boards"].append(b.copy())
        log["mass"].append(K.target_mass(b, T))
        log["f1"].append(K.recovery_f1(b, T, M))
        log["entropy"].append(K.entropy(K.normalize(b)))
        log["clues"].append(int(c))
    return log


def adaptivity_matrix(C, level, seed):
    """rows = counterfactual B-history fed to A; cols = clue token; cell = A's clue prob.
    Identical rows => A ignores B (L1). Differing rows => A adapts (L2)."""
    board = K.Board(n_items=N, n_clues=C, n_targets=M, seed=seed)
    T = board.sample_targets(seed + 100)
    S = K.SpeakerAgent(board, T, level=level, alpha=ALPHA, tau=TAU)
    rows, labels = [], []
    # naive B
    rows.append(S.clue_dist(K.ListenerAgent(board, level, beta=BETA, alpha=ALPHA))); labels.append("B: naive")
    # B that already knows target j
    for j in T[:3]:
        Lj = K.ListenerAgent(board, level, beta=BETA, alpha=ALPHA)
        Lj.observe(j, True)
        rows.append(S.clue_dist(Lj)); labels.append(f"B: knows t{j}")
    return np.array(rows), labels, T


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    conds = [("forced", CFORCED, 1), ("forced", CFORCED, 2), ("easy", CEASY, 1), ("easy", CEASY, 2)]
    agg = {}
    exemplar = None
    for regime, C, level in conds:
        runs = [run_game(C, level, s) for s in range(SEEDS)]
        mass = np.array([r["mass"] for r in runs])
        f1 = np.array([r["f1"] for r in runs])
        ent = np.array([r["entropy"] for r in runs])
        cr = np.array([r["coupling_real"] for r in runs])
        cn = np.array([r["coupling_null"] for r in runs])
        ad = np.array([r["adaptivity"] for r in runs])
        agg[(regime, level)] = {
            "mass_mean": mass.mean(0).tolist(), "mass_se": (mass.std(0) / np.sqrt(SEEDS)).tolist(),
            "f1_mean": f1.mean(0).tolist(), "entropy_mean": ent.mean(0).tolist(),
            "coupling_real_mean": cr.mean(0).tolist(), "coupling_null_mean": cn.mean(0).tolist(),
            "adaptivity_mean": ad.mean(0).tolist(),
            "final_mass": float(mass[:, -1].mean()), "final_f1": float(f1[:, -1].mean()),
        }
        if regime == "forced" and level == 2:
            exemplar = runs[0]
        print(f"[game2] {regime} L{level}: final mass={mass[:,-1].mean():.2f} f1={f1[:,-1].mean():.2f} "
              f"coupling(real/null)={cr.mean():.2f}/{cn.mean():.2f} adaptivity={ad.mean():.2f}", flush=True)

    json.dump({str(k): v for k, v in agg.items()}, open(os.path.join(RUN_DIR, "game2_summary.json"), "w"), indent=2)
    make_figures(agg, exemplar)
    print(f"[game2] DONE -> {RUN_DIR}", flush=True)


def make_figures(agg, exemplar):
    rounds = np.arange(1, ROUNDS + 1)

    # ---- 1. belief-board heatmap over rounds (the flagship) ----
    with PdfPages(os.path.join(RUN_DIR, "game2_belief_board.pdf")) as pdf:
        boards = np.array(exemplar["boards"])      # (rounds, N)
        T = set(exemplar["T"])
        side = int(np.ceil(np.sqrt(N)))
        ncol = min(ROUNDS, 4); nrow = int(np.ceil(ROUNDS / ncol))
        fig, ax = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
        ax = np.array(ax).reshape(-1)
        for r in range(ROUNDS):
            grid = np.zeros((side, side));
            for i in range(N):
                grid[i // side, i % side] = boards[r][i]
            a = ax[r]; a.imshow(grid, cmap="viridis", vmin=0, vmax=1)
            for i in T:                                    # outline true targets
                rr, cc = i // side, i % side
                a.add_patch(plt.Rectangle((cc - .5, rr - .5), 1, 1, fill=False, ec="red", lw=2))
            a.set_title(f"round {r+1}  (mass={exemplar['mass'][r]:.2f})", fontsize=8)
            a.set_xticks([]); a.set_yticks([])
        for r in range(ROUNDS, len(ax)):
            ax[r].axis("off")
        fig.suptitle("GAME 2 — B's belief board over rounds (L2, forced regime). "
                     "Red = true targets; shading = B's belief they are targets.", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- 2. recovery + entropy + coupling + adaptivity curves ----
    styles = {("forced", 1): ("tab:red", "--"), ("forced", 2): ("tab:red", "-"),
              ("easy", 1): ("tab:blue", "--"), ("easy", 2): ("tab:blue", "-")}
    lab = lambda rg, lv: f"{rg} L{lv}"
    with PdfPages(os.path.join(RUN_DIR, "game2_curves.pdf")) as pdf:
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for (rg, lv), d in agg.items():
            c, ls = styles[(rg, lv)]
            ax[0, 0].errorbar(rounds, d["mass_mean"], yerr=d["mass_se"], color=c, ls=ls, label=lab(rg, lv), capsize=2)
            ax[0, 1].plot(rounds, d["entropy_mean"], color=c, ls=ls, label=lab(rg, lv))
            ax[1, 0].plot(rounds, d["coupling_real_mean"], color=c, ls=ls, label=lab(rg, lv) + " real")
            ax[1, 1].plot(rounds, d["adaptivity_mean"], color=c, ls=ls, label=lab(rg, lv))
        # null coupling floor (any condition; ~0)
        ax[1, 0].plot(rounds, agg[("forced", 2)]["coupling_null_mean"], color="k", ls=":", label="null swap floor")
        ax[0, 0].set_title("Recovery: target-posterior mass", fontsize=10); ax[0, 0].set_ylim(0, 1)
        ax[0, 1].set_title("Belief entropy (falls as B converges)", fontsize=10)
        ax[1, 0].set_title("Coupling: KL(B_swap||B_clean) per round", fontsize=10)
        ax[1, 1].set_title("A's adaptivity: KL(clue|B_real||clue|B_naive)", fontsize=10)
        for a in ax.reshape(-1):
            a.set_xlabel("round"); a.legend(fontsize=7); a.grid(alpha=.3)
        fig.suptitle("GAME 2 — recovery (ground truth), coupling & adaptivity (instruments), by regime × level", fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- 3. adaptivity matrix: A's clue vs counterfactual B-history, L1 vs L2 ----
    with PdfPages(os.path.join(RUN_DIR, "game2_adaptivity_matrix.pdf")) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for j, lv in enumerate([1, 2]):
            Mx, labels, T = adaptivity_matrix(CFORCED, lv, seed=1)
            im = ax[j].imshow(Mx, cmap="magma", vmin=0, vmax=1, aspect="auto")
            ax[j].set_yticks(range(len(labels))); ax[j].set_yticklabels(labels, fontsize=8)
            ax[j].set_xticks(range(CFORCED)); ax[j].set_xticklabels([f"clue {c}" for c in range(CFORCED)], fontsize=8)
            ax[j].set_title(f"Speaker L{lv}  (rows identical => ignores B; rows differ => adapts)", fontsize=9)
            fig.colorbar(im, ax=ax[j], fraction=.046)
        fig.suptitle("GAME 2 — A's clue distribution vs counterfactual B-histories (forced regime). "
                     "L2 adapts (rows differ); L1 does not.", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
