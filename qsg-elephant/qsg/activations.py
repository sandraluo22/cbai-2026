"""First-class activation capture.

INVARIANT (documented loudly because the whole later linear-transformation
project depends on it):

    Activations are captured at the LAST token of the soft-readout prompt — i.e.
    the *anchor position* defined in §3a (the token right after "Answer:").  This
    is the SAME position whose next-token logits define the soft simplex belief
    x_i.  Therefore, for every (round, agent), the stored residual-stream vector
    and the belief x_i are positionally aligned and directly comparable across
    agents / rounds / runs.

We hook the residual stream at the OUTPUT of every transformer block (the stream
*after* block L), at all layers, every round.  Storage is an
``(rounds+1, N, n_layers, hidden_dim)`` array (memmap or npz) with a JSON
metadata sidecar so downstream code can load without guessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Locating transformer blocks across architectures                           #
# --------------------------------------------------------------------------- #
def find_decoder_layers(model):
    """Return the ``nn.ModuleList`` of transformer blocks for common HF models."""
    candidates = [
        "model.layers",            # Llama, Qwen2, Mistral
        "model.model.layers",      # some wrapped CausalLM
        "transformer.h",           # GPT-2 (smoke model)
        "model.language_model.layers",  # some VLMs
        "language_model.model.layers",  # Qwen2-VL / Llama-Vision text stack
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return obj, path
    raise RuntimeError(
        "Could not locate decoder layers; add this architecture to "
        "find_decoder_layers()."
    )


# --------------------------------------------------------------------------- #
# Hook manager                                                                 #
# --------------------------------------------------------------------------- #
class ActivationHooks:
    """Registers forward hooks that grab the last-token residual stream per layer.

    Usage::

        hooks = ActivationHooks(model)
        with hooks.capture():
            model(**inputs)              # the anchor forward pass
        vec = hooks.last_token_per_layer()   # (n_layers, hidden_dim) float array
    """

    def __init__(self, model, layers: str = "all"):
        self.model, self.layer_path = find_decoder_layers(model)
        self.n_total = len(self.model)
        if layers == "all":
            self.layer_idx = list(range(self.n_total))
        else:
            self.layer_idx = [int(i) for i in str(layers).split(",")]
        self._buffers: dict[int, "np.ndarray"] = {}
        self._handles: list = []
        self._active = False

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            if not self._active:
                return
            hs = output[0] if isinstance(output, tuple) else output
            # hs: (batch, seq, hidden) — grab last token of first sequence.
            self._buffers[idx] = hs[0, -1, :].detach().float().cpu().numpy()
        return hook

    def __enter__(self):
        for i in self.layer_idx:
            self._handles.append(self.model[i].register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    class _Capture:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            self.owner._buffers = {}
            self.owner._active = True
            return self.owner

        def __exit__(self, *exc):
            self.owner._active = False

    def capture(self):
        return ActivationHooks._Capture(self)

    @property
    def hidden_dim(self) -> int:
        any_vec = next(iter(self._buffers.values()))
        return int(any_vec.shape[0])

    def last_token_per_layer(self) -> np.ndarray:
        """Stack captured layers -> (n_captured_layers, hidden_dim)."""
        return np.stack([self._buffers[i] for i in self.layer_idx], axis=0)


# --------------------------------------------------------------------------- #
# On-disk store                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ActivationMeta:
    run_id: str
    model_name: str
    n_agents: int
    n_rounds: int                 # inclusive of round 0 -> array dim is n_rounds
    layer_indices: list[int]
    n_layers: int
    hidden_dim: int
    dtype: str
    anchor_token_id: Optional[int]
    template_hash: str
    layer_path: str
    shape: list[int]
    store_format: str


def estimate_disk_bytes(
    n_rounds_incl: int, n_agents: int, n_layers: int, hidden_dim: int, dtype: str
) -> int:
    itemsize = np.dtype(dtype).itemsize
    return n_rounds_incl * n_agents * n_layers * hidden_dim * itemsize


class ActivationStore:
    """``(rounds+1, N, n_layers, hidden_dim)`` memmap/npz keyed by (round, agent, layer)."""

    def __init__(
        self,
        out_dir: Path,
        meta: ActivationMeta,
    ):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.shape = tuple(meta.shape)
        self.dtype = meta.dtype
        self._path = self.dir / "activations.dat"
        if meta.store_format == "memmap":
            self._arr = np.memmap(self._path, mode="w+", dtype=self.dtype, shape=self.shape)
        else:
            self._arr = np.zeros(self.shape, dtype=self.dtype)
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "activations_meta.json").write_text(json.dumps(asdict(self.meta), indent=2))

    def put(self, round_idx: int, agent_id: int, layers_hidden: np.ndarray) -> None:
        """layers_hidden: (n_layers, hidden_dim)."""
        self._arr[round_idx, agent_id] = layers_hidden.astype(self.dtype)

    def flush(self) -> None:
        if self.meta.store_format == "memmap":
            self._arr.flush()
        else:
            np.savez_compressed(self.dir / "activations.npz", activations=self._arr)


# --------------------------------------------------------------------------- #
# Loader helper (for the downstream linear-transformation project)            #
# --------------------------------------------------------------------------- #
def load_activations(run_dir: str | Path):
    """Return ``(array, meta_dict)`` for a run's activation store.

    ``array`` has shape (rounds+1, N, n_layers, hidden_dim).
    """
    run_dir = Path(run_dir)
    act_dir = run_dir / "activations"
    meta = json.loads((act_dir / "activations_meta.json").read_text())
    shape = tuple(meta["shape"])
    if meta["store_format"] == "memmap":
        arr = np.memmap(act_dir / "activations.dat", mode="r", dtype=meta["dtype"], shape=shape)
    else:
        arr = np.load(act_dir / "activations.npz")["activations"]
    return arr, meta


def layer_matrix_for_round(arr, meta: dict, round_idx: int, layer: int) -> np.ndarray:
    """Pull all agents' layer-``layer`` activations for ``round_idx`` -> (N, hidden_dim).

    ``layer`` is an absolute layer index; it is mapped through the stored
    ``layer_indices`` to the correct slot.
    """
    slot = meta["layer_indices"].index(layer)
    return np.asarray(arr[round_idx, :, slot, :])
