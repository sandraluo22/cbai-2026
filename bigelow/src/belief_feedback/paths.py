"""Artifact path layout.

All artifacts are namespaced by configuration name (smoke / pilot / full /
low_memory / second_model) so that mock, pilot, and full results can never
be mixed: e.g. ``artifacts/data/smoke/worlds.parquet``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ArtifactPaths:
    """Resolved artifact directories for one configuration."""

    config_name: str
    root: Path = field(default_factory=lambda: REPO_ROOT / "artifacts")

    def _sub(self, kind: str) -> Path:
        p = self.root / kind / self.config_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data(self) -> Path:
        return self._sub("data")

    @property
    def rendered_documents(self) -> Path:
        p = self.data / "rendered_documents"
        p.mkdir(exist_ok=True)
        return p

    @property
    def vectors(self) -> Path:
        return self._sub("vectors")

    @property
    def activations(self) -> Path:
        return self._sub("activations")

    @property
    def runs(self) -> Path:
        return self._sub("runs")

    @property
    def models(self) -> Path:
        return self._sub("models")

    @property
    def figures(self) -> Path:
        return self._sub("figures")

    @property
    def figure_data(self) -> Path:
        return self._sub("figure_data")

    @property
    def tables(self) -> Path:
        return self._sub("tables")

    @property
    def reports(self) -> Path:
        return self._sub("reports")

    @property
    def manifests(self) -> Path:
        return self._sub("manifests")
