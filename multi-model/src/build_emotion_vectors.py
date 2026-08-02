"""Extract per-emotion 'emotion vectors' for a BASE model from GoEmotions.

This is the emotion/ recipe (Anthropic emotion-vectors, adapted to GoEmotions in
`emotion/extract_emotion_vectors.py`) re-expressed for a *base* LM that has no
chat template:

  1. Labeled data ... GoEmotions examples, grouped by primary emotion.
  2. Per-example act. feed the example text as plain text; take the residual
                      stream at the LAST token, at EVERY decoder block. (The
                      chat-model recipe used the assistant-anchor token; a base
                      model has none, so the last content token is the natural
                      analogue -- the state having just read the emotional text.)
  3. Mean + center .. average per emotion, subtract the mean across the 27
                      non-neutral emotions -> strips generic "text-ness".
  4. Deconfound ..... project out the top neutral-activation PCs (>=50% var) per
                      layer.

Done per layer. Emotion axis order == GOEMOTIONS_LABELS minus 'neutral' (27),
so `emotion_names.index('sadness')` is the row Exp2 steers / measures.

Output (one file per model, under <RUN_DIR>/):
  emotion_vectors_<TAG>.npz   { clean:(E,L,H), raw:(E,L,H), emotion_names, counts, npcs }

Env: PRESET MODEL(Llama|Qwen) LIMIT(3000) MAXTOK(64) RUN_DIR DEVICE VAR_TARGET(0.5)
Usage:
  PRESET=gemma_qwen MODEL=Llama RUN_DIR=runs/main python src/build_emotion_vectors.py
  PRESET=gemma_qwen MODEL=Qwen  RUN_DIR=runs/main python src/build_emotion_vectors.py
"""
from __future__ import annotations

import os
import json

import numpy as np

import common as C  # noqa: E402  (sets up cross-model / emotion paths)
from goemotions_utils import GOEMOTIONS_LABELS, primary_label, load_goemotions, stratified_indices  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

MODEL = os.environ.get("MODEL", "Llama")
LIMIT = int(os.environ.get("LIMIT", "3000" if C.PRESET != "smoke" else "120"))
MAXTOK = int(os.environ.get("MAXTOK", "64"))
VAR_TARGET = float(os.environ.get("VAR_TARGET", "0.5"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")

NEUTRAL = GOEMOTIONS_LABELS.index("neutral")
EMO_IDS = [i for i in range(len(GOEMOTIONS_LABELS)) if i != NEUTRAL]      # 27
EMO_NAMES = [GOEMOTIONS_LABELS[i] for i in EMO_IDS]


def neutral_pcs(neutral_layer: np.ndarray, var_target: float):
    from sklearn.decomposition import PCA
    n, H = neutral_layer.shape
    max_c = min(n, H)
    pca = PCA(n_components=max_c, svd_solver="full")
    pca.fit(neutral_layer.astype(np.float32))
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_target) + 1)
    return pca.components_[:k], k


@torch.no_grad() if torch is not None else (lambda f: f)
def collect(model, tok, texts, labels, dev, nL, H):
    """One forward per example; accumulate per-emotion last-token residual sums
    (all layers) and stash raw neutral activations for the deconfound PCA."""
    blocks = C.decoder_blocks(model)
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    n_lab = len(GOEMOTIONS_LABELS)
    sums = np.zeros((n_lab, nL, H), dtype=np.float64)
    cnts = np.zeros(n_lab, dtype=np.int64)
    neutral_rows = []                                     # list of (nL,H) fp16
    try:
        for i, (txt, lab) in enumerate(zip(texts, labels)):
            if not txt.strip():
                continue
            ids = tok(txt, add_special_tokens=True, truncation=True, max_length=MAXTOK,
                      return_tensors="pt")["input_ids"].to(dev)
            grabbed.clear()
            model(input_ids=ids)
            vec = np.stack([grabbed[L][0, -1].float().cpu().numpy() for L in range(nL)])  # (nL,H)
            sums[lab] += vec
            cnts[lab] += 1
            if lab == NEUTRAL:
                neutral_rows.append(vec.astype(np.float16))
            if i % 200 == 0:
                print(f"  [scan] {i}/{len(texts)}", flush=True)
    finally:
        for h in hs:
            h.remove()
    neutral_acts = (np.stack(neutral_rows) if neutral_rows
                    else np.zeros((0, nL, H), dtype=np.float16))
    return sums, cnts, neutral_acts


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg()                                    # only needs dtype/device
    print(f"[emo/{MODEL}] loading (preset={C.PRESET}, dev={dev})", flush=True)
    model, tok = C.load_model(MODEL, cfg)
    nL, H = C.n_layers(model), model.config.hidden_size

    ds = load_goemotions("all" if C.PRESET != "smoke" else "validation")
    primary = np.array([primary_label(r) for r in ds["labels"]], dtype=int)
    idx = stratified_indices(primary, LIMIT, seed=0)
    texts = [ds[int(i)]["text"] for i in idx]
    labels = primary[idx]
    print(f"[emo/{MODEL}] N={len(texts)} nL={nL} H={H}; neutral={(labels==NEUTRAL).sum()}", flush=True)

    sums, cnts, neutral_acts = collect(model, tok, texts, labels, dev, nL, H)
    C.free(model, tok)

    present = [i for i in EMO_IDS if cnts[i] > 0]
    if len(present) < len(EMO_IDS):
        missing = [GOEMOTIONS_LABELS[i] for i in EMO_IDS if cnts[i] == 0]
        print(f"[emo/{MODEL}] WARNING: {len(missing)} emotions unseen (smoke?): {missing}", flush=True)

    # per-emotion means (only present rows), center across the present emotions
    means = np.zeros((len(EMO_IDS), nL, H), dtype=np.float64)
    for r, i in enumerate(EMO_IDS):
        if cnts[i] > 0:
            means[r] = sums[i] / cnts[i]
    grand = means[[EMO_IDS.index(i) for i in present]].mean(axis=0, keepdims=True)
    raw = (means - grand).astype(np.float32)

    clean = raw.copy()
    npcs = []
    if len(neutral_acts) >= 3:
        for L in range(nL):
            V, k = neutral_pcs(neutral_acts[:, L, :], VAR_TARGET)
            npcs.append(k)
            proj = raw[:, L, :] @ V.T
            clean[:, L, :] = raw[:, L, :] - proj @ V
    else:
        print(f"[emo/{MODEL}] too few neutral examples ({len(neutral_acts)}); skipping deconfound", flush=True)
        npcs = [0] * nL

    out_path = C.emotion_vec_path(MODEL, RUN_DIR)
    np.savez_compressed(
        out_path, clean=clean, raw=raw,
        emotion_names=np.array(EMO_NAMES),
        counts=np.array([int(cnts[i]) for i in EMO_IDS]),
        npcs=np.array(npcs),
    )
    sidx = EMO_NAMES.index("sadness")
    sad_norms = [float(np.linalg.norm(clean[sidx, L])) for L in range(nL)]
    peak = int(np.argmax(sad_norms))
    meta = {"model": MODEL, "nL": nL, "H": H, "n_examples": len(texts),
            "var_target": VAR_TARGET, "npcs_per_layer": npcs,
            "sadness_idx": sidx, "sadness_norm_by_layer": sad_norms,
            "sadness_peak_layer": peak}
    json.dump(meta, open(os.path.join(RUN_DIR, f"emotion_vectors_{MODEL}_meta.json"), "w"), indent=2)
    print(f"[emo/{MODEL}] DONE -> {out_path}  (sadness ‖v‖ peaks at L{peak}={sad_norms[peak]:.2f})", flush=True)


if __name__ == "__main__":
    main()
