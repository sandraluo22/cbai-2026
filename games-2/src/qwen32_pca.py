"""Follow-up on the Qwen3-32B self-play convergence runs: generate more rollouts at
the same temperature (forced sand/cloud, no-repeat enforced), CAPTURE each player's
residual stream every turn at every layer, and PCA per layer.

For each generated turn, for BOTH players (#1 and #2), we take the hidden state at the
answer position (the token that produces the word) at every layer -> a per-(player,
rollout,turn,layer) activation. Then, per layer, we PCA the pooled activations of both
players to 2D and plot the turn-by-turn trajectory of each player. As the two Qwens
converge on the same word, do their internal representations also converge? The
per-layer PCA shows where (which depth) and when (which turn) that happens.

Env: MODEL(QwenInst32) N_ROLL(6) SAFETY(30) TEMP(0.7) START_WORDS(sand,cloud) RUN_DIR
Out: <RUN_DIR>/qwen32_pca_acts.npz            (acts + meta)
     <RUN_DIR>/qwen32_pca_perlayer.pdf         (one PCA scatter per layer)
     <RUN_DIR>/qwen32_pca_transcript.jsonl/.json
"""
from __future__ import annotations

import os
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
N_ROLL = int(os.environ.get("N_ROLL", "6"))
SAFETY = int(os.environ.get("SAFETY", "30"))
TEMP = float(os.environ.get("TEMP", "0.7"))
START = tuple(w.strip() for w in os.environ.get("START_WORDS", "sand,cloud").split(","))
# Optional: per-seed random start-word pairs (one "idx<TAB>wordA<TAB>wordB" per line,
# from sample_start_words.py). If set, each seed gets its own divergent random start.
START_FILE = os.environ.get("START_WORDS_FILE", "")
RUN_DIR = os.environ.get("RUN_DIR", "runs/game-1/qwen32/qwen32_pca")


def load_start_pairs():
    if not START_FILE:
        return None
    pairs = []
    for line in open(START_FILE):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        pairs.append((parts[-2], parts[-1]))
    return pairs

OPEN_PROMPT = ("You are playing a word game with another player. Each round, you both say one word at the "
               "same time, then the two words are revealed. You win the round only if you both said the same "
               "word. You may not say any word that has already been said by either player.")


def build_prompt(tok, hist, used):
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}." for k, (o, s) in enumerate(hist))
    body = OPEN_PROMPT + ((" " + lines) if lines else "")
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return LA._render(tok, body) + "\nMy word:"


def clean_word(txt):
    w = re.split(r"\s+", txt.strip())
    return re.sub(r"[^a-zA-Z\-]", "", w[0] if w else "").lower()


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = model.config.num_hidden_layers
    print(f"[pca] {MODEL} loaded: {nL} layers, hidden {model.config.hidden_size}", flush=True)

    @torch.no_grad()
    def hidden_all_layers(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states   # tuple (nL+1) of (1,seq,H)
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs])   # (nL+1, H)

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    P1, P2 = MODEL + "_1", MODEL + "_2"
    acts = {P1: [], P2: []}          # each -> list of (nL+1, H)
    meta = {P1: [], P2: []}          # each -> list of (roll, turn, word)
    start_pairs = load_start_pairs()
    n_roll = len(start_pairs) if start_pairs is not None else N_ROLL
    tf = open(os.path.join(RUN_DIR, "qwen32_pca_transcript.jsonl"), "w")
    sf = open(os.path.join(RUN_DIR, "qwen32_pca_seeds.txt"), "w")   # readable per-seed log
    sf.write(f"# {MODEL} self-play, no-repeat, temp={TEMP}. Per-seed random start (word2vec) + rollout.\n")
    sf.write("# seed | start(P1/P2) | P1 words -> | P2 words -> | outcome\n")
    for roll in range(n_roll):
        sa, sb = start_pairs[roll] if start_pairs is not None else START
        histA = [(sb, sa)]; histB = [(sa, sb)]
        used = {sa, sb}
        agreed = None
        seqA, seqB = [sa], [sb]
        for t in range(1, SAFETY):
            pA, pB = build_prompt(tok, histA, used), build_prompt(tok, histB, used)
            vA, vB = hidden_all_layers(pA), hidden_all_layers(pB)
            wA = gen_word(pA, 5000 * roll + t, used); wB = gen_word(pB, 90000 + 5000 * roll + t, used)
            acts[P1].append(vA); meta[P1].append((roll, t, wA))
            acts[P2].append(vB); meta[P2].append((roll, t, wB))
            seqA.append(wA); seqB.append(wB)
            tf.write(json.dumps({"rollout": roll, "turn": t, "start": [sa, sb],
                                 "picks": {P1: wA, P2: wB}, "agreed": bool(wA == wB and wA)}) + "\n")
            if wA == wB and wA:
                agreed = t; break
            histA.append((wB, wA)); histB.append((wA, wB)); used |= {wA, wB}
        outcome = f"agreed '{seqA[-1]}' @turn{agreed}" if agreed else "no-converge"
        sf.write(f"{roll:3d} | {sa}/{sb} | {' '.join(seqA)} | {' '.join(seqB)} | {outcome}\n")
        print(f"[pca] rollout {roll} ({sa}/{sb}): {outcome} "
              f"({len([m for m in meta[P1] if m[0]==roll])} turns captured)", flush=True)
    tf.close(); sf.close()

    A1 = np.stack(acts[P1]); A2 = np.stack(acts[P2])     # (n_pts, nL+1, H)
    m1 = np.array(meta[P1], dtype=object); m2 = np.array(meta[P2], dtype=object)
    np.savez_compressed(os.path.join(RUN_DIR, "qwen32_pca_acts.npz"),
                        A1=A1.astype(np.float16), A2=A2.astype(np.float16),
                        meta1=m1, meta2=m2, players=np.array([P1, P2]))
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, "qwen32_pca_transcript.jsonl"))
    except Exception:
        pass
    make_pca(A1, A2, meta[P1], meta[P2], P1, P2, nL,
             os.path.join(RUN_DIR, "qwen32_pca_perlayer_global.pdf"), center="global")
    make_pca(A1, A2, meta[P1], meta[P2], P1, P2, nL,
             os.path.join(RUN_DIR, "qwen32_pca_perlayer_seedcentered.pdf"), center="seed")
    print(f"[pca] DONE -> {RUN_DIR}", flush=True)


def make_pca(A1, A2, m1, m2, P1, P2, nL, path, center="global"):
    """center='global': subtract the global mean before PCA. center='seed': for each
    seed (rollout) subtract THAT seed's mean over all its turns (both players) -- removes
    per-seed offset so the PCA shows shared turn dynamics, not seed-specific word regions."""
    from sklearn.decomposition import PCA
    layers = sorted(set([0] + list(range(4, nL + 1, max(1, (nL) // 12))) + [nL]))
    rolls1 = np.array([r for r, _, _ in m1]); turns1 = np.array([t for _, t, _ in m1])
    rolls2 = np.array([r for r, _, _ in m2]); turns2 = np.array([t for _, t, _ in m2])
    seeds = sorted(set(rolls1.tolist()) | set(rolls2.tolist()))
    with PdfPages(path) as pdf:
        for L in layers:
            X1, X2 = A1[:, L, :].astype(np.float32).copy(), A2[:, L, :].astype(np.float32).copy()
            if center == "seed":
                for s in seeds:                              # subtract each seed's own mean (both players)
                    pooled = np.concatenate([X1[rolls1 == s], X2[rolls2 == s]], 0)
                    if len(pooled):
                        mu = pooled.mean(0)
                        X1[rolls1 == s] -= mu; X2[rolls2 == s] -= mu
            X = np.concatenate([X1, X2], 0)
            X = X - X.mean(0)
            pca = PCA(n_components=2, svd_solver="full")
            Z = pca.fit_transform(X)
            evr = pca.explained_variance_ratio_ * 100     # % variance per axis
            n1 = A1.shape[0]; Z1, Z2 = Z[:n1], Z[n1:]
            fig, ax = plt.subplots(figsize=(8, 6.5))
            # trajectories per rollout, in turn order
            for (Zp, rolls, turns, col, name) in ((Z1, rolls1, turns1, "tab:blue", P1),
                                                  (Z2, rolls2, turns2, "tab:orange", P2)):
                for rr in np.unique(rolls):
                    m = rolls == rr; order = np.argsort(turns[m])
                    ax.plot(Zp[m][order, 0], Zp[m][order, 1], "-", color=col, alpha=.35, lw=1)
                sc = ax.scatter(Zp[:, 0], Zp[:, 1], c=turns, cmap="viridis", edgecolors=col,
                                linewidths=1.6, s=70, label=name)
            cb = fig.colorbar(sc, ax=ax, fraction=.046); cb.set_label("turn")
            ax.set_xlabel(f"PC1 ({evr[0]:.1f}% var)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}% var)")
            ctag = "per-seed centered" if center == "seed" else "global centered"
            ax.set_title(f"{P1.rsplit('_',1)[0]} self-play — residual PCA @ layer {L}/{nL}  [{ctag}]\n"
                         f"(edge=player, fill=turn; lines=per-rollout; PC1+PC2={evr.sum():.1f}% of variance)", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"  {path}  ({len(layers)} layers plotted)")


def replot_from_npz(npz_path):
    """Regenerate both PCA PDFs from a cached acts npz -- no GPU, no re-capture."""
    z = np.load(npz_path, allow_pickle=True)
    A1, A2 = z["A1"], z["A2"]; m1, m2 = z["meta1"], z["meta2"]
    P1, P2 = [str(p) for p in z["players"]]
    nL = A1.shape[1] - 1
    d = os.path.dirname(npz_path)
    meta1 = [tuple(x) for x in m1]; meta2 = [tuple(x) for x in m2]
    make_pca(A1, A2, meta1, meta2, P1, P2, nL, os.path.join(d, "qwen32_pca_perlayer_global.pdf"), "global")
    make_pca(A1, A2, meta1, meta2, P1, P2, nL, os.path.join(d, "qwen32_pca_perlayer_seedcentered.pdf"), "seed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--replot":
        replot_from_npz(sys.argv[2])
    else:
        main()
