# test-1: dueling-context experiments (mutual in-context learning of graph geometry)

Two (or N) mirrored or cross-model LLM instances are primed on random walks over
different 16-node graphs (same word vocabulary; Park et al. plain-walk condition,
reusing `../cross-model`), then coupled: they alternate generating the next node and
every sampled word is appended to the coupled contexts. Per-node mean residuals are
captured at every layer in context windows and analyzed with PCA, coordinate
regressions, Laplacian eigenmode spectra, and Procrustes shape comparison (never RSA).

## Layout

- `*.py` -- all scripts (GPU runners + local analyses), see catalog below.
- `runs/` -- one directory per experiment (`nodemeans_dueling.npz`, `gen_log.json`,
  analysis outputs). `runs/out` is the original T=1 grid+ring run.
- `figs/` -- cross-run figures and summaries.
- `viewer/viewer.html` -- self-contained interactive 3D PCA viewer (published as an
  artifact; rebuild with `build_viewer.py`).

## Script catalog

GPU runners (ssh box; env-configurable, see each docstring):
- `run_experiment.py` -- original grid+ring dueling pair (Llama, mirrored contexts).
- `run_pair.py` -- generalized 2-graph pair (grid|ring|hex|prism|antiprism|ring3).
- `run_multi.py` -- N-way cycling generation (e.g. ring+grid+ring3).
- `run_xmodel.py` -- cross-MODEL duels (Llama / Qwen3-8B / Gemma-2-9b instances).
- `run_controls.py` -- free-generation, one-way influence, real interleaved steps.
- `run_sweep.py` -- degree-controlled 4-regular sweep (see `graphs16.py`).
- `capture_fresh.py` -- fresh-context control: blank context fed only the joint stream.
- `bigram_prior.py` -- measure Llama bigram priors over the word pool.
- `probe_update.py` -- mechanistic update-kernel probe (regression + attention).

Local tooling:
- `graphs16.py` -- 16-node 4-regular graph families (torus, circulants, degree-
  preserving interpolants, small-world, SBM, expanders) + sweep spec generator.
- `pick_words.py` -- choose low-bigram-prior word assignments per graph union.
- `analyze.py`, `analyze_hybrid.py`, `analyze_union_shape.py` -- grid+ring analyses.
- `analyze_pair.py`, `analyze_multi.py`, `analyze_xmodel.py`, `analyze_controls.py`,
  `analyze_sweep.py` -- per-family analyses (all take `RUN_OUT=<runs/dir>`).
- `bayes_model.py` -- coupled Dirichlet-Markov surrogate, fitted to a run.
- `bayes_sweep.py` -- surrogate run over the sweep cells (hyperparameter mapping).
- `bayes_hier.py` -- hierarchical (latent-source / interleaved-Markov) upgrade +
  mixture fit against the phase probe; `probe_phase.py` is the GPU-side probe.
- `plot_galpha.py`, `fig_synthesis.py` -- ground-truth family + synthesis figures.
- `build_viewer.py` -- rebuild `viewer/viewer.html` from all runs.

## Run naming

`runs/out*` grid+ring (T=1, topk4, k2, k2_fix); `out_hex*`, `out_prismring`,
`out_ringring3`, `out_antigrid` graph pairs; `out_tri` 3-way; `out_x_*` cross-model;
`out_ctrl_*` controls; `out_probe` update-kernel probe; `runs/sweep/*` the 4-regular
controlled sweep.
