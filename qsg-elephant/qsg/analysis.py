"""Convergence analysis: pairwise similarity heatmaps + order-parameter curves.

All functions take belief *trajectories* of shape ``(rounds+1, N, K)`` (the same
shape produced by both ``qsg_reference.run_reference`` and ``engine.py``), so the
analysis is identical for the numeric null model and the LLM runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Headless backend so plots render on a GPU box without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = p + eps
    q = q + eps
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Jensen-Shannon divergence (base-2, in [0, 1]) along the last axis."""
    p = p + eps
    q = q + eps
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    mmix = 0.5 * (p + q)
    js = 0.5 * _kl(p, mmix) + 0.5 * _kl(q, mmix)
    return js / np.log(2.0)


def l1_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(p - q), axis=-1)


def population_states(beliefs: np.ndarray) -> np.ndarray:
    """Population mean belief per round: (rounds+1, K)."""
    return beliefs.mean(axis=1)


# --------------------------------------------------------------------------- #
# 5a. Pairwise similarity matrices                                            #
# --------------------------------------------------------------------------- #
def pairwise_cosine(states: np.ndarray) -> np.ndarray:
    """Round x Round cosine-similarity matrix of population belief states."""
    norm = np.linalg.norm(states, axis=1, keepdims=True)
    unit = states / np.clip(norm, 1e-12, None)
    return unit @ unit.T


def pairwise_js_similarity(states: np.ndarray) -> np.ndarray:
    """Round x Round (1 - JS divergence) similarity matrix."""
    R = states.shape[0]
    sim = np.empty((R, R))
    for i in range(R):
        sim[i] = 1.0 - js_divergence(states[i][None, :], states)
    return sim


def save_similarity_heatmaps(
    beliefs: np.ndarray, out_dir: Path, prefix: str = "similarity"
) -> dict[str, Path]:
    """Build + save both pairwise similarity heatmaps (PNG) and raw matrices (.npy)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = population_states(beliefs)

    cos = pairwise_cosine(states)
    jss = pairwise_js_similarity(states)

    paths: dict[str, Path] = {}
    for name, mat, label in (
        ("cosine", cos, "cosine similarity"),
        ("js", jss, "1 - JS divergence"),
    ):
        npy = out_dir / f"{prefix}_{name}.npy"
        np.save(npy, mat)
        paths[f"{name}_npy"] = npy

        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"Pairwise population-belief similarity\n({label})")
        ax.set_xlabel("round j")
        ax.set_ylabel("round i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        png = out_dir / f"{prefix}_{name}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        paths[f"{name}_png"] = png
    return paths


# --------------------------------------------------------------------------- #
# 5b. Convergence diagnostics                                                 #
# --------------------------------------------------------------------------- #
def order_parameters(beliefs: np.ndarray) -> dict[str, np.ndarray]:
    """U (polarization) and V (disagreement energy) per round."""
    x_bar = beliefs.mean(axis=1)                       # (T, K)
    U = np.sum(x_bar * x_bar, axis=1)                  # (T,)
    d = beliefs - x_bar[:, None, :]                    # (T, N, K)
    V = np.sum(d * d, axis=(1, 2))                     # (T,)
    return {"U": U, "V": V}


def accuracy_curves(beliefs: np.ndarray, ground_truth: int) -> dict[str, np.ndarray]:
    """Group accuracy (argmax of mean) and fraction of agents correct, per round."""
    x_bar = beliefs.mean(axis=1)
    group_correct = (np.argmax(x_bar, axis=1) == ground_truth).astype(float)
    agent_argmax = np.argmax(beliefs, axis=2)          # (T, N)
    frac_correct = (agent_argmax == ground_truth).mean(axis=1)
    return {"group_correct": group_correct, "frac_correct": frac_correct}


def consensus_time(U: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(U >= threshold)[0]
    return int(idx[0]) if idx.size else None


def save_diagnostic_curves(
    beliefs: np.ndarray,
    out_dir: Path,
    ground_truth: Optional[int] = None,
    consensus_threshold: float = 0.95,
    prefix: str = "diagnostics",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    op = order_parameters(beliefs)
    rounds = np.arange(beliefs.shape[0])

    ncols = 2 if ground_truth is None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(rounds, op["U"], color="C0")
    ct = consensus_time(op["U"], consensus_threshold)
    if ct is not None:
        axes[0].axvline(ct, ls="--", color="grey", label=f"consensus t={ct}")
        axes[0].legend()
    axes[0].set(title="Polarization U = ||x̄||²", xlabel="round", ylabel="U")

    axes[1].plot(rounds, op["V"], color="C3")
    axes[1].set(title="Disagreement energy V", xlabel="round", ylabel="V")

    if ground_truth is not None:
        acc = accuracy_curves(beliefs, ground_truth)
        axes[2].plot(rounds, acc["group_correct"], label="group argmax correct")
        axes[2].plot(rounds, acc["frac_correct"], label="fraction agents correct")
        axes[2].set(title="Accuracy vs ground truth", xlabel="round", ylabel="accuracy")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend()

    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# 3c. Soft-vs-hard readout agreement                                         #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Drift-vs-selection comparison plots (the measurements that actually mean      #
# something — null vs selection, and the N-dependence of getting it right)      #
# --------------------------------------------------------------------------- #
def save_accuracy_overlay(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    chance: Optional[float] = None,
    title: str = "Accuracy vs round: neutral drift vs ground-truth selection",
    prefix: str = "accuracy_overlay",
) -> Path:
    """Overlay accuracy-vs-round for several conditions.

    ``curves[label] = (rounds, mean, sem)``; a shaded ±sem band is drawn.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for label, (rounds, mean, sem) in curves.items():
        line, = ax.plot(rounds, mean, label=label)
        ax.fill_between(rounds, mean - sem, mean + sem, alpha=0.2, color=line.get_color())
    if chance is not None:
        ax.axhline(chance, ls=":", color="grey", label=f"chance = 1/K = {chance:.2f}")
    ax.set(title=title, xlabel="round", ylabel="P(group answer correct)", ylim=(-0.05, 1.05))
    ax.legend()
    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def save_vs_N(
    Ns: np.ndarray,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    ylabel: str,
    title: str,
    chance: Optional[float] = None,
    prefix: str = "vs_N",
    logx: bool = True,
) -> Path:
    """Plot a summary statistic vs population size N for several conditions.

    ``series[label] = (mean_over_N, sem_over_N)``. This is the figure that can
    actually reveal the drift->selection crossover (does larger N select truth?).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for label, (mean, sem) in series.items():
        ax.errorbar(Ns, mean, yerr=sem, marker="o", capsize=3, label=label)
    if chance is not None:
        ax.axhline(chance, ls=":", color="grey", label=f"chance = 1/K = {chance:.2f}")
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set(title=title, xlabel="population size N", ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def save_soft_hard_agreement(
    soft: np.ndarray, hard: np.ndarray, out_dir: Path, prefix: str = "soft_vs_hard"
) -> Path:
    """soft, hard: (rounds+1, N, K). Plot mean KL and L1 over the run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kl = _kl(soft, hard)            # (T, N)
    l1 = l1_distance(soft, hard)    # (T, N)
    rounds = np.arange(soft.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rounds, kl.mean(1), color="C2")
    axes[0].fill_between(rounds, kl.min(1), kl.max(1), alpha=0.2, color="C2")
    axes[0].set(title="Soft-vs-Hard KL (mean, min–max)", xlabel="round", ylabel="KL")

    axes[1].plot(rounds, l1.mean(1), color="C4")
    axes[1].fill_between(rounds, l1.min(1), l1.max(1), alpha=0.2, color="C4")
    axes[1].set(title="Soft-vs-Hard L1 (mean, min–max)", xlabel="round", ylabel="L1")

    fig.tight_layout()
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png
