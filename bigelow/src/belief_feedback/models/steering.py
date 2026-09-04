"""Steering specifications and residual-stream interventions.

Two intervention modes:

* ``add``: h' = h + magnitude * vector (CAA steering).
* ``project_set``: replace only the component of h along unit(vector) with
  a target value (belief-component patching).

Scopes:

* ``final_token_and_generation``: during prefill, modify only the final
  non-padding prompt token; during autoregressive generation, modify each
  newly generated token; during sequence scoring, modify the final
  non-padding *prompt* token.
* ``all_tokens``: modify every non-padding token (robustness condition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class SteeringSpec:
    vector: np.ndarray
    layer: int
    magnitude: float = 0.0
    scope: Literal["final_token_and_generation", "all_tokens"] = "final_token_and_generation"
    mode: Literal["add", "project_set"] = "add"
    project_value: float = 0.0  # target projection for project_set

    @property
    def active(self) -> bool:
        return self.mode == "project_set" or self.magnitude != 0.0

    def unit(self) -> np.ndarray:
        n = float(np.linalg.norm(self.vector))
        return self.vector / n if n > 0 else self.vector


def apply_to_hidden_numpy(h: np.ndarray, spec: SteeringSpec) -> np.ndarray:
    """Apply the intervention to a single hidden-state vector (mock backend)."""
    if spec.mode == "add":
        return h + spec.magnitude * spec.vector
    u = spec.unit()
    return h + (spec.project_value - float(h @ u)) * u


def apply_to_hidden_torch(hidden, spec: SteeringSpec, token_mask=None):
    """Apply the intervention to a torch hidden-state tensor in place.

    ``hidden``: [batch, seq, dim]. ``token_mask``: boolean [batch, seq] of
    positions to modify; None means all positions.
    """
    import torch

    vec = torch.as_tensor(spec.vector, dtype=hidden.dtype, device=hidden.device)
    if token_mask is None:
        token_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if spec.mode == "add":
        hidden[token_mask] = hidden[token_mask] + spec.magnitude * vec
    else:
        u = vec / vec.norm().clamp_min(1e-8)
        sel = hidden[token_mask]
        proj = sel @ u
        hidden[token_mask] = sel + (spec.project_value - proj).unsqueeze(-1) * u
    return hidden


@dataclass
class SteeringHookState:
    """Bookkeeping shared between a forward pass and its layer hook.

    ``row_specs`` optionally overrides ``spec`` per batch row (None rows are
    untouched), so a mixed batch can steer only some agents. All non-None
    row specs must target the same layer.
    """

    spec: SteeringSpec
    prompt_lengths: list[int] = field(default_factory=list)  # non-padding prompt length per row
    prefill_done: bool = False
    row_specs: list[SteeringSpec | None] | None = None


def make_layer_hook(state: SteeringHookState):
    """Forward hook for a decoder block; edits the residual-stream output.

    During prefill (seq_len > 1) with the final-token scope, only the final
    non-padding prompt token per row is edited (rows are left-padded during
    batched generation, so that is position seq-1; during right-padded
    scoring it is prompt_lengths[i]-1). Every later call (seq_len == 1
    under KV caching) is a newly generated token and is edited fully.
    """

    def hook(module, args, output):
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        bsz, seq, _ = hidden.shape

        def row_spec(i: int) -> SteeringSpec | None:
            if state.row_specs is not None:
                return state.row_specs[i]
            return state.spec

        if state.row_specs is None and not state.spec.active:
            return output
        prefill = seq > 1 and not state.prefill_done
        if prefill:
            state.prefill_done = True
        for i in range(bsz):
            spec = row_spec(i)
            if spec is None or not spec.active:
                continue
            mask = torch.zeros(bsz, seq, dtype=torch.bool, device=hidden.device)
            if spec.scope == "all_tokens":
                mask[i, :] = True
            elif prefill:
                last = (state.prompt_lengths[i] if state.prompt_lengths else seq) - 1
                mask[i, min(last, seq - 1)] = True
            elif seq == 1:
                mask[i, 0] = True
            else:  # later multi-token call without cache bookkeeping: skip
                continue
            hidden = apply_to_hidden_torch(hidden, spec, mask)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    return hook
