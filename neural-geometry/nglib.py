"""Shared utilities for the neural-geometry experiments (shuffled-weekday, isometry).

Reuses the cross-model pipeline (graph construction, walk generation, hooked
activation capture) by putting cross-model/src on sys.path. Adds the pieces
those experiments need on top:

  - ring/manifold fitting (PCA plane + circle fit + circular ordering)
  - subspace comparison (principal angles)
  - isometry metrics between an activation manifold and a behavior manifold
    (distance-matrix Spearman + 2D Procrustes disparity)
  - behavioral read-out: next-word posterior restricted to the node vocabulary,
    embedded with the Hellinger map sqrt(p) so distances live in a proper
    (Fisher-information) geometry rather than raw probability space
  - patch-based readout/steering: replace the final-token residual at layer L
    and rerun the forward pass, so the "readout map" includes everything the
    real model does downstream of L (not just logit-lens norm+unembed)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

NG_DIR = Path(__file__).resolve().parent
CROSS_SRC = NG_DIR.parent / "cross-model" / "src"
if str(CROSS_SRC) not in sys.path:
    sys.path.insert(0, str(CROSS_SRC))

# re-exports from the cross-model pipeline (import after path setup)
from config import Config, DAYS, DAYS_PERMUTED, PRESETS          # noqa: E402
from graph import Graph, Walk, build_graph, generate_walks       # noqa: E402
import models as cm_models                                       # noqa: E402


# ---------------------------------------------------------------------------
# Manifold fitting
# ---------------------------------------------------------------------------
@dataclass
class RingFit:
    """A 2D-circle fit to a set of points in hidden space.

    plane   : [d, 2] orthonormal basis of the best-fit 2D plane (top-2 PCA)
    center  : [d] mean point (plane origin)
    angles  : [n] angular position of each input point on the fitted circle
    radii   : [n] distance of each projected point from the 2D centroid
    circularity : std(radii)/mean(radii); 0 = perfect circle
    var_explained : fraction of (centered) variance captured by the plane
    """
    plane: np.ndarray
    center: np.ndarray
    angles: np.ndarray
    radii: np.ndarray
    circularity: float
    var_explained: float

    def tangent(self, i: int) -> np.ndarray:
        """Unit tangent (counter-angle direction) at point i, in full d-dim space."""
        t2 = np.array([-np.sin(self.angles[i]), np.cos(self.angles[i])])
        t = self.plane @ t2
        return t / np.linalg.norm(t)

    def radial(self, i: int) -> np.ndarray:
        """Unit outward radial direction at point i, in full d-dim space."""
        r2 = np.array([np.cos(self.angles[i]), np.sin(self.angles[i])])
        r = self.plane @ r2
        return r / np.linalg.norm(r)


def fit_ring(points: np.ndarray) -> RingFit:
    """Fit a circle to [n, d] points: top-2 PCA plane, then angles about the
    projected centroid. Works for any n >= 3; for the 7-day ring n == 7."""
    X = np.asarray(points, dtype=np.float64)
    center = X.mean(axis=0)
    Xc = X - center
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    plane = Vt[:2].T                                   # [d, 2]
    proj = Xc @ plane                                  # [n, 2]
    proj_c = proj - proj.mean(axis=0)
    angles = np.arctan2(proj_c[:, 1], proj_c[:, 0])
    radii = np.linalg.norm(proj_c, axis=1)
    total_var = float((S ** 2).sum())
    var2 = float((S[:2] ** 2).sum() / total_var) if total_var > 0 else 0.0
    circ = float(radii.std() / radii.mean()) if radii.mean() > 0 else np.inf
    return RingFit(plane=plane, center=center, angles=angles, radii=radii,
                   circularity=circ, var_explained=var2)


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spans of A and B."""
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def cyclic_order_agreement(angles: np.ndarray, ref_cycle: Sequence[int]) -> float:
    """Fraction of edges of the reference cycle whose endpoints are angularly
    adjacent on the fitted circle. 1.0 = the fitted ring realizes the reference
    cyclic order exactly (up to rotation/reflection); ~2/n = chance-ish.

    angles    : [n] fitted angle of node i
    ref_cycle : node ids in reference cyclic order, e.g. [0..n-1] for the
                in-context ring, or the semantic weekday order.
    """
    n = len(angles)
    order = list(np.argsort(angles))                   # nodes by angular position
    pos = {node: k for k, node in enumerate(order)}
    hits = 0
    for a, b in zip(ref_cycle, list(ref_cycle[1:]) + [ref_cycle[0]]):
        if (pos[a] - pos[b]) % n in (1, n - 1):
            hits += 1
    return hits / n


# ---------------------------------------------------------------------------
# Isometry metrics
# ---------------------------------------------------------------------------
def pairwise_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def isometry_scores(act_points: np.ndarray, beh_points: np.ndarray) -> Dict[str, float]:
    """How isometric are two point sets (same n, different ambient dims)?

    - dist_spearman : rank correlation of pairwise distances (scale-free; the
      weakest, most robust notion of 'same geometry')
    - dist_pearson  : linear correlation of pairwise distances (isometry up to
      a single global scale predicts ~1.0)
    - procrustes_2d : Procrustes disparity between the top-2 PCA projections
      after centering + unit-norming (0 = identical shapes)
    """
    from scipy.stats import spearmanr, pearsonr
    from scipy.spatial import procrustes

    Da, Db = pairwise_dists(act_points), pairwise_dists(beh_points)
    iu = np.triu_indices(Da.shape[0], k=1)
    sr = float(spearmanr(Da[iu], Db[iu]).statistic)
    pr = float(pearsonr(Da[iu], Db[iu]).statistic)

    def top2(X):
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    _, _, disparity = procrustes(top2(act_points), top2(beh_points))
    return {"dist_spearman": sr, "dist_pearson": pr, "procrustes_2d": float(disparity)}


# ---------------------------------------------------------------------------
# Behavioral read-out: next-word posterior over the node vocabulary
# ---------------------------------------------------------------------------
def node_first_token_ids(tokenizer, words: List[str]) -> List[int]:
    """Token id of the FIRST subword of ' word' (leading space: mid-sequence
    continuation form). Asserts the ids are distinct across nodes so the
    restricted posterior is well-defined."""
    ids = []
    for w in words:
        toks = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        assert len(toks) >= 1, f"word {w!r} produced no tokens"
        ids.append(int(toks[0]))
    assert len(set(ids)) == len(ids), (
        f"first-subword collision across node words {words}; "
        "restricted posterior would be ambiguous"
    )
    return ids


def next_word_posterior(model, tokenizer, text: str, node_token_ids: List[int],
                        device: str) -> Tuple[np.ndarray, float]:
    """P(next word = node w | text), restricted to the node vocabulary and
    renormalized. Also returns the unrestricted probability mass on the node
    words (diagnostic: low mass means the model isn't 'playing the game')."""
    import torch
    enc = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device))
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p = probs[node_token_ids]
    mass = float(p.sum())
    return (p / mass if mass > 0 else np.full(len(p), 1 / len(p))), mass


def hellinger_embed(posteriors: np.ndarray) -> np.ndarray:
    """sqrt(p): embeds distributions on the unit sphere, where Euclidean distance
    is (up to a factor) the Hellinger distance -- the natural output-space
    geometry, matching the paper's use of distribution-space manifolds."""
    p = np.asarray(posteriors, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.sqrt(p)


# ---------------------------------------------------------------------------
# Patch-based readout / steering
# ---------------------------------------------------------------------------
def patched_forward_posterior(model, tokenizer, text: str, layer: int,
                              delta_or_h: np.ndarray, mode: str,
                              node_token_ids: List[int], device: str,
                              ) -> Tuple[np.ndarray, float]:
    """Rerun `text` with the FINAL token's post-block residual at `layer`
    modified, and read out the restricted next-word posterior.

    mode = "add"     : residual += delta_or_h   (steering)
    mode = "replace" : residual  = delta_or_h   (Jacobian probing)

    This is the honest readout map for a manifold fit at `layer`: the
    perturbation flows through every downstream block, not just norm+unembed.
    """
    import torch
    blocks = cm_models._decoder_blocks(model)
    vec = torch.tensor(np.asarray(delta_or_h), dtype=next(model.parameters()).dtype,
                       device=device)

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if mode == "add":
            hs[0, -1, :] = hs[0, -1, :] + vec
        elif mode == "replace":
            hs[0, -1, :] = vec
        else:
            raise ValueError(mode)
        return out

    handle = blocks[layer].register_forward_hook(hook)
    try:
        enc = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(device))
    finally:
        handle.remove()
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p = probs[node_token_ids]
    mass = float(p.sum())
    return (p / mass if mass > 0 else np.full(len(p), 1 / len(p))), mass


def logit_lens_posterior(model, hidden: np.ndarray, node_token_ids: List[int]
                         ) -> np.ndarray:
    """Cheap readout variant: final-norm + unembed applied directly to a hidden
    state (no downstream blocks). Restricted + renormalized posterior."""
    import torch
    dtype = next(model.parameters()).dtype
    h = torch.tensor(np.asarray(hidden), dtype=dtype,
                     device=next(model.parameters()).device).unsqueeze(0)
    norm = model.model.norm if hasattr(model, "model") else model.transformer.ln_f
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        logits = lm_head(norm(h))[0].float()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p = probs[node_token_ids]
    return p / p.sum()


# ---------------------------------------------------------------------------
# Misc conveniences
# ---------------------------------------------------------------------------
def node_means(acts: np.ndarray, nodes: np.ndarray, mask: np.ndarray,
               n_nodes: int) -> np.ndarray:
    """Mean activation per node over masked occurrences. Returns [n_nodes, d];
    rows with no occurrences are NaN (caller should check)."""
    d = acts.shape[1]
    out = np.full((n_nodes, d), np.nan)
    for n in range(n_nodes):
        sel = mask & (nodes == n)
        if sel.any():
            out[n] = acts[sel].astype(np.float64).mean(axis=0)
    return out


def circular_mean_position(posterior: np.ndarray, ring_order: Sequence[int]
                           ) -> float:
    """Expected angular position (radians) of a posterior over nodes, under the
    embedding node -> angle implied by `ring_order` (node ids in cyclic order).
    Used to ask 'which way did behavior rotate?' under steering."""
    n = len(ring_order)
    theta = np.zeros(n)
    for k, node in enumerate(ring_order):
        theta[node] = 2 * np.pi * k / n
    z = (posterior * np.exp(1j * theta)).sum()
    return float(np.angle(z))


def semantic_day_cycle() -> List[int]:
    """The pretrained weekday cyclic order expressed in NODE ids of the permuted
    ring. Node i carries word DAYS_PERMUTED[i]; the semantic cycle visits words
    in DAYS order, so we return the node ids in that order."""
    word_to_node = {w: i for i, w in enumerate(DAYS_PERMUTED)}
    return [word_to_node[w] for w in DAYS]
