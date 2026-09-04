"""Activation storage: NPZ stores keyed by identifying tuple.

One store per run; each entry key encodes (world, branch, agent, time).
Rows in belief_states.parquet carry ``selected_layer_activation_pointer`` =
``"<store file>::<key>"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ActivationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as z:
                self._data = {k: z[k] for k in z.files}

    @staticmethod
    def key(world_id: str, branch: str, agent_id: int, t: int) -> str:
        return f"{world_id}|{branch}|{agent_id}|{t}"

    def put(self, key: str, arr: np.ndarray) -> str:
        self._data[key] = np.asarray(arr, dtype=np.float32)
        return f"{self.path}::{key}"

    def get(self, key: str) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self._data)
