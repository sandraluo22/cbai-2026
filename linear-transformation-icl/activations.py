"""Activation capture via forward hooks (no API calls), with the anchor rule.

ANCHOR RULE (documented loudly — the whole analysis depends on it):
    The anchor is the position that carries the belief readout, defined by the SAME
    rule across every rollout, source, checkpoint, and run. It is NOT inherently a
    ":" token.
      * Balls & Urns (raw sequence): the anchor for prefix length t is position
        t-1 (0-indexed) — the last token of the prefix, whose NEXT-token prediction
        encodes the belief after t tokens.
      * Templated/LLM path: the anchor is the fixed template terminator.
    The belief readout and the activation capture use the IDENTICAL position.

We hook each residual-stream block, capture the FULL (seq, hidden) stream in one
forward, then select the log-spaced anchor positions. Stored as
[n_sources, n_rollouts, n_positions, n_layers, hidden] with a JSON metadata sidecar.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Hooks: capture full residual stream per layer in one forward                 #
# --------------------------------------------------------------------------- #
class ResidualHooks:
    def __init__(self, model):
        self.blocks = model.blocks()
        self.n_layers = len(self.blocks)
        self._buf: dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, i):
        def fn(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out      # (B, C, hidden)
            self._buf[i] = h.detach()
        return fn

    def __enter__(self):
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> torch.Tensor:
        """(n_layers, B, C, hidden) for the most recent forward."""
        return torch.stack([self._buf[i] for i in range(self.n_layers)], dim=0)


def anchor_indices(positions_t: list[int]) -> list[int]:
    """Map prefix-lengths t -> 0-indexed anchor positions (t-1)."""
    return [t - 1 for t in positions_t]


# --------------------------------------------------------------------------- #
# Store                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class ActMeta:
    run_id: str
    model_name: str
    n_sources: int
    n_rollouts: int
    positions_t: list[int]
    n_layers: int
    hidden_dim: int
    dtype: str
    anchor_rule: str
    template_hash: str
    shape: list[int]
    store_format: str


def estimate_disk_gb(n_sources, n_rollouts, n_positions, n_layers, hidden, dtype) -> float:
    return (n_sources * n_rollouts * n_positions * n_layers * hidden
            * np.dtype(dtype).itemsize) / 1e9


class ActStore:
    """[n_sources, n_rollouts, n_positions, n_layers, hidden] memmap/array + sidecar."""

    def __init__(self, out_dir: Path, meta: ActMeta, max_disk_gb: float = 20.0):
        gb = estimate_disk_gb(meta.n_sources, meta.n_rollouts, len(meta.positions_t),
                              meta.n_layers, meta.hidden_dim, meta.dtype)
        print(f"  [activations] store ~{gb:.3f} GB, shape {meta.shape}, dtype {meta.dtype}")
        if gb > max_disk_gb:
            raise RuntimeError(f"activation store {gb:.2f} GB exceeds cap {max_disk_gb} GB")
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        shape = tuple(meta.shape)
        if meta.store_format == "memmap":
            self.arr = np.memmap(self.dir / "acts.dat", mode="w+", dtype=meta.dtype, shape=shape)
        else:
            self.arr = np.zeros(shape, dtype=meta.dtype)
        (self.dir / "acts_meta.json").write_text(json.dumps(asdict(meta), indent=2))

    def put(self, source_id: int, rollout_id: int, anchor_acts: np.ndarray) -> None:
        """anchor_acts: (n_positions, n_layers, hidden)."""
        self.arr[source_id, rollout_id] = anchor_acts.astype(self.meta.dtype)

    def flush(self):
        if self.meta.store_format == "memmap":
            self.arr.flush()
        else:
            np.savez_compressed(self.dir / "acts.npz", acts=self.arr)


# --------------------------------------------------------------------------- #
# Loader + the slice the ruler consumes                                        #
# --------------------------------------------------------------------------- #
def load_activations(run_dir: str | Path):
    run_dir = Path(run_dir)
    d = run_dir / "activations"
    meta = json.loads((d / "acts_meta.json").read_text())
    shape = tuple(meta["shape"])
    if meta["store_format"] == "memmap":
        arr = np.memmap(d / "acts.dat", mode="r", dtype=meta["dtype"], shape=shape)
    else:
        arr = np.load(d / "acts.npz")["acts"]
    return arr, meta


def rollout_matrix(arr, meta, source_id: int, position_t: int, layer: int) -> np.ndarray:
    """The [n_rollouts, hidden] matrix the ruler consumes, for fixed (source, t, layer)."""
    p = meta["positions_t"].index(position_t)
    return np.asarray(arr[source_id, :, p, layer, :], dtype=np.float64)


def source_matrix(arr, meta, position_t: int, layer: int) -> np.ndarray:
    """[n_sources, hidden]: each row a source's MEAN activation over its rollouts.
    NOTE: averaging does the convergence itself (means separate early regardless of
    difficulty), so this axis is D-insensitive. Prefer `pooled_matrix`."""
    p = meta["positions_t"].index(position_t)
    return np.asarray(arr[:, :, p, layer, :].mean(axis=1), dtype=np.float64)


def pooled_matrix(arr, meta, position_t: int, layer: int) -> np.ndarray:
    """[n_sources*n_rollouts, hidden]: all rollouts pooled, NOT averaged, rows
    matched by (source, rollout) across positions. Early t -> unstructured blob;
    late t -> source-clusters emerge (later for harder/overlapping sources). Avoids
    both the rollout-axis collapse and the source-mean averaging confound — this is
    the primary convergence/phase instrument."""
    p = meta["positions_t"].index(position_t)
    S, R = meta["n_sources"], meta["n_rollouts"]
    return np.asarray(arr[:, :, p, layer, :].reshape(S * R, -1), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Capture driver for the tiny model                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_rollouts(model, seqs: np.ndarray, positions_t: list[int], device="cpu"):
    """Run a batch of sequences, capture anchor activations + next-token belief.

    seqs: (R, C) int. Returns:
      acts:    (R, n_positions, n_layers, hidden)
      beliefs: (R, n_positions, vocab)  softmax over symbols at each anchor
    """
    idx = torch.from_numpy(seqs).long().to(device)
    a_idx = anchor_indices(positions_t)
    with ResidualHooks(model) as hooks:
        logits = model(idx)                          # (R, C, vocab)
        stacked = hooks.stacked()                    # (n_layers, R, C, hidden)
    acts = stacked[:, :, a_idx, :].permute(1, 2, 0, 3).contiguous().cpu().numpy()  # (R,P,L,H)
    bel = torch.softmax(logits[:, a_idx, :], dim=-1).cpu().numpy()                 # (R,P,vocab)
    return acts, bel
