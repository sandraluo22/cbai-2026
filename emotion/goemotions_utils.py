"""GoEmotions label metadata, Ekman grouping, and dataset helpers.

GoEmotions (simplified config) is multi-label over 28 classes (27 emotions +
neutral). For coloring activation plots we collapse each example to a single
*primary* label (its first annotated label), and also provide the standard
Ekman 7-way grouping which is far more legible than 28 colors.
"""
from __future__ import annotations

import numpy as np

# Canonical GoEmotions label order (matches the HF `go_emotions` features).
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
LABEL_TO_ID = {name: i for i, name in enumerate(GOEMOTIONS_LABELS)}

# Standard GoEmotions -> Ekman mapping (+ neutral), as published by the authors.
EKMAN_GROUPS = {
    "anger":    ["anger", "annoyance", "disapproval"],
    "disgust":  ["disgust"],
    "fear":     ["fear", "nervousness"],
    "joy":      ["admiration", "amusement", "approval", "caring", "desire",
                 "excitement", "gratitude", "joy", "love", "optimism", "pride",
                 "relief"],
    "sadness":  ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    "neutral":  ["neutral"],
}
EKMAN_NAMES = list(EKMAN_GROUPS.keys())
EKMAN_NAME_TO_ID = {name: i for i, name in enumerate(EKMAN_NAMES)}

# fine label id -> ekman id
FINE_TO_EKMAN = np.empty(len(GOEMOTIONS_LABELS), dtype=np.int16)
for _ek, _members in EKMAN_GROUPS.items():
    for _m in _members:
        FINE_TO_EKMAN[LABEL_TO_ID[_m]] = EKMAN_NAME_TO_ID[_ek]


def primary_label(label_list) -> int:
    """First annotated label as the example's primary class (neutral if empty)."""
    if label_list is None or len(label_list) == 0:
        return LABEL_TO_ID["neutral"]
    return int(label_list[0])


def stratified_indices(primary_ids: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Roughly even sample of `n` indices across the present primary classes.

    Deterministic given `seed`. Round-robins across shuffled per-class index
    pools so every emotion is represented before any class is over-sampled.
    """
    if n <= 0 or n >= len(primary_ids):
        return np.arange(len(primary_ids))
    rng = np.random.default_rng(seed)
    pools = {}
    for cls in np.unique(primary_ids):
        idx = np.where(primary_ids == cls)[0]
        rng.shuffle(idx)
        pools[int(cls)] = list(idx)
    chosen: list[int] = []
    classes = sorted(pools)
    while len(chosen) < n and any(pools.values()):
        for cls in classes:
            if pools[cls]:
                chosen.append(pools[cls].pop())
                if len(chosen) >= n:
                    break
    return np.array(sorted(chosen), dtype=np.int64)


def load_goemotions(split: str):
    """Load the simplified GoEmotions split(s). `split` may be train/validation/
    test or 'all' (concatenation of the three)."""
    from datasets import load_dataset, concatenate_datasets

    if split == "all":
        parts = [load_dataset("google-research-datasets/go_emotions", "simplified", split=s)
                 for s in ("train", "validation", "test")]
        return concatenate_datasets(parts)
    return load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
