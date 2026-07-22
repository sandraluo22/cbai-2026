"""Make random seed start-words by sampling random vectors in word2vec space and
snapping each to its nearest real word. Two words per seed (a divergent start for the
two players). Writes a readable .txt log of the seeds.

Needs gensim + internet (downloads word2vec-google-news-300 once, ~1.6GB, to
~/gensim-data). No GPU.

Env: N(50) SEED(0) OUT(runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt)
"""
from __future__ import annotations
import os
import re
import numpy as np

N = int(os.environ.get("N", "50"))
SEED = int(os.environ.get("SEED", "0"))
TOPK = int(os.environ.get("TOPK", "20000"))    # restrict to the TOPK most-frequent words
OUT = os.environ.get("OUT", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")

WORD = re.compile(r"[a-z]{3,12}$")             # common-looking lowercase word (no proper nouns / phrases / typos-ish)


def main():
    import gensim.downloader as api
    print("[w2v] loading word2vec-google-news-300 (downloads once)...", flush=True)
    kv = api.load("word2vec-google-news-300")
    dim = kv.vector_size
    rng = np.random.default_rng(SEED)

    # Restrict candidates to the TOPK most-frequent, purely-lowercase-alpha words
    # (word2vec is frequency-ordered; this drops rare outlier/proper-noun tokens that
    # a random vector's global nearest-neighbour would otherwise hit).
    pool = [w for w in kv.index_to_key[:TOPK] if WORD.fullmatch(w)]
    PV = kv[pool].astype("float32")
    PV /= (np.linalg.norm(PV, axis=1, keepdims=True) + 1e-9)
    print(f"[w2v] candidate pool: {len(pool)} common words (from top {TOPK})", flush=True)

    def random_word(exclude):
        for _ in range(50):
            v = rng.standard_normal(dim).astype("float32"); v /= np.linalg.norm(v) + 1e-9
            for j in np.argsort(PV @ v)[::-1][:20]:       # nearest common word to the random direction
                w = pool[j]
                if w not in exclude:
                    return w
        return None

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    used = set()
    pairs = []
    while len(pairs) < N:
        a = random_word(used);  used.add(a)
        b = random_word(used);  used.add(b)
        if a and b:
            pairs.append((a, b))
    with open(OUT, "w") as f:
        f.write(f"# {N} random word2vec seed start-pairs (random vector -> nearest word), seed={SEED}\n")
        f.write("# seed_idx\tplayer1_start\tplayer2_start\n")
        for i, (a, b) in enumerate(pairs):
            f.write(f"{i}\t{a}\t{b}\n")
    print(f"[w2v] wrote {len(pairs)} seed pairs -> {OUT}", flush=True)
    for i, (a, b) in enumerate(pairs[:10]):
        print(f"   seed {i}: {a} / {b}")
    print("   ...")


if __name__ == "__main__":
    main()
