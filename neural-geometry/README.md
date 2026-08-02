# neural-geometry

Experiments at the open edges of Goodfire's manifold-steering paper
([arXiv 2605.05115](https://arxiv.org/abs/2605.05115)), building on the
`cross-model/` pipeline (graph builders, walk generation, hooked capture are
imported from `cross-model/src` via `nglib.py`).

- **[shuffled-weekday/](shuffled-weekday/)** — manifold *dynamics* during
  in-context learning: weekdays on a permuted ring (in-context structure vs
  pretrained cycle), tracking how the activation manifold shifts with context
  depth, whether the activation↔behavior isometry holds *through* the
  transition, and which ring steering causally moves. The paper only fits
  manifolds after learning is complete; the during is unstudied.

- **[isometry/](isometry/)** — *why* are the activation and behavior manifolds
  isometric? The paper poses this as an open question (§7) with no follow-up
  in flight. Three attacks: (1) measure the readout map's conformality
  directly (lens vs full-patch localizes where isometry is enforced),
  (2) break the task's symmetry and see if the isometry survives,
  (3) test the spectral/Markov account — both manifolds as spectral embeddings
  of the same transition operator (connects to `cross-model`'s eigenmode and
  `markov_families` work).

Run order: `shuffled-weekday/capture_ctx.py` first (the isometry attack 1
reuses its capture). Every script takes `--smoke` (distilgpt2, CPU) to test
plumbing; real runs default to Llama-3.1-8B, layer 26, cuda.
