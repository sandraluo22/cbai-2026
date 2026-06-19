"""Graph construction and random-walk generation.

Plain-random-walk condition of Park, Lee, Lubana et al. (ICLR 2025): nodes are
*semantically unrelated* concept words, edges are grid adjacencies, and we emit
the word at each visited node while taking a uniform random walk.

Nothing here touches a model or a tokenizer -- walks are materialized once as
word sequences (and strings) so that BOTH models consume the EXACT same
sequences. Activations are later paired by (walk_id, step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from config import Config


@dataclass
class Graph:
    n_nodes: int
    words: List[str]                       # node_id -> concept word
    adjacency: List[List[int]]             # node_id -> sorted neighbor ids
    coords: List[Tuple[int, int]]          # node_id -> (row, col) for grid recovery

    def neighbors(self, node: int) -> List[int]:
        return self.adjacency[node]

    def grid_distance_matrix(self) -> np.ndarray:
        """Manhattan distance between node coords -- ground truth for the
        paper-reproduction check (PCA of high-context node means should recover
        this geometry)."""
        c = np.array(self.coords)
        d = np.abs(c[:, None, :] - c[None, :, :]).sum(-1)
        return d.astype(float)


def build_grid_graph(cfg: Config) -> Graph:
    """rows x cols grid; orthogonal (4-)neighbor edges. Node count is configurable
    via cfg.grid_rows/grid_cols."""
    rows, cols = cfg.grid_rows, cfg.grid_cols
    words = cfg.words()
    coords: List[Tuple[int, int]] = []
    adjacency: List[List[int]] = []

    def nid(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            coords.append((r, c))
            nbrs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    nbrs.append(nid(rr, cc))
            adjacency.append(sorted(nbrs))

    return Graph(n_nodes=rows * cols, words=words, adjacency=adjacency, coords=coords)


@dataclass
class Walk:
    walk_id: int
    nodes: List[int]          # visited node ids, length == walk_length
    words: List[str]          # corresponding concept words

    @property
    def text(self) -> str:
        # Single-space join. Per-word character spans are recovered in models.py
        # via the tokenizer offset mapping, so the exact join only needs to be
        # consistent and deterministic.
        return " ".join(self.words)

    def char_spans(self) -> List[Tuple[int, int]]:
        """(start, end) char offset of each emitted word in `text`, matching the
        single-space join above."""
        spans = []
        pos = 0
        for i, w in enumerate(self.words):
            if i > 0:
                pos += 1  # the joining space
            spans.append((pos, pos + len(w)))
            pos += len(w)
        return spans


def generate_walks(graph: Graph, cfg: Config) -> List[Walk]:
    """Uniform random walks. Walk i starts at node (i mod n_nodes) so that with
    n_walks >= n_nodes every node is a start node and appears early."""
    rng = np.random.default_rng(cfg.seed)
    walks: List[Walk] = []
    for w in range(cfg.n_walks):
        start = w % graph.n_nodes
        nodes = [start]
        cur = start
        for _ in range(cfg.walk_length - 1):
            nbrs = graph.neighbors(cur)
            cur = int(rng.choice(nbrs))
            nodes.append(cur)
        walks.append(Walk(walk_id=w, nodes=nodes, words=[graph.words[n] for n in nodes]))
    return walks


def occurrence_table(walks: List[Walk]) -> Dict[str, np.ndarray]:
    """Flat per-occurrence index over all walks, in capture order.

    Returns parallel arrays; `context_length` is the 1-based word step (nodes
    emitted up to and including this occurrence) -- identical across models.
    """
    walk_id, step, node, ctx = [], [], [], []
    for wk in walks:
        for s, n in enumerate(wk.nodes):
            walk_id.append(wk.walk_id)
            step.append(s)
            node.append(n)
            ctx.append(s + 1)
    return {
        "walk_id": np.array(walk_id, dtype=np.int32),
        "step": np.array(step, dtype=np.int32),
        "node": np.array(node, dtype=np.int32),
        "context_length": np.array(ctx, dtype=np.int32),
    }
