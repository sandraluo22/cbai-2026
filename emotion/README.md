# Emotion activations — Qwen3-8B on GoEmotions

For each GoEmotions example, Qwen3-8B (non-thinking) generates a response. We
record residual-stream activations at **every layer** at three anchors, then
visualize them.

## What gets extracted

Let **Q** = the GoEmotions example (the user turn) and **A** = the model's
generated response.

| Tag | Where it's hooked |
|-----|-------------------|
| **Q**  | residual stream at the assistant anchor of Q's prompt — the token right after the `<\|im_start\|>assistant\n` header (Qwen has no literal `Assistant:` colon; this is its equivalent). The model's state *about to answer Q*. |
| **A1** | residual stream at the **end of the model's response** (last non-eos token of A), in the full Q+A context. |
| **A2** | feed the response text **A only** back in as a fresh user turn, hook the residual stream at the same assistant anchor as Q. |

Each is saved at every layer `L = num_hidden_layers + 1` (embeddings + each
block = 37 for Qwen3-8B), hidden size `H = 4096`.

## Run it (on a runpod GPU)

The Mac has no CUDA, so extraction runs on a runpod box and results are pulled
back automatically (per the standing "always pull GPU results" rule).

```bash
# smoke test first (stratified ~200 examples), then the full dataset
remote/run.sh <ssh-host>                                   # smoke
remote/run.sh <ssh-host> --split all --limit 0 --batch-size 32   # full ~58k
```

`<ssh-host>` is the alias you'll add to `~/.ssh/config` for the runpod box
(e.g. `runpod-qwen`), or `root@<ip>`. `run.sh` rsyncs the code up, installs
deps, runs `extract_activations.py`, and rsyncs `results/` back.

To run extraction directly (e.g. on the pod, or locally on a small model):

```bash
python extract_activations.py --limit 200                  # smoke
python extract_activations.py --split all --limit 0        # full
```

## Make the plots (locally, after pulling results)

```bash
python make_plots.py results/train_200          # Ekman (7-color) — default
python make_plots.py results/train_200 --fine   # 28 fine emotions
```

Outputs under `results/<run>/plots/`:

- `pca_Q_slideshow.pdf`, `pca_A1_slideshow.pdf`, `pca_A2_slideshow.pdf`
  — one PCA scatter per layer, color-coded by emotion (+ per-layer PNGs + GIF).
- `cos_a1_a2_hist_slideshow.pdf` — per-layer histogram of cosine similarity
  between matching A1/A2 activations (x: cos sim, y: frequency) (+ PNGs + GIF).
- `cos_a1_a2_mean_by_layer.png` — mean cos(A1, A2) vs layer summary.

## Storage note

Activations are float16 memmaps of shape `(N, L, H)` × 3. The full GoEmotions
(~58k) is ≈ `58000 × 37 × 4096 × 2 bytes × 3` ≈ **52 GB**. The smoke run
(200) is ≈ 180 MB. Plan disk/transfer accordingly for the full run.

## Files

- `extract_activations.py` — generate + hook + save activations.
- `make_plots.py` — PCA slideshows + cos-sim histograms.
- `goemotions_utils.py` — label names, Ekman grouping, stratified subset.
- `remote/run.sh` — one-command sync → run → pull on a runpod host.
- `remote/setup.sh` — pip install on the pod.
