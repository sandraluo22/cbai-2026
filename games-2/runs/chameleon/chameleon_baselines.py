"""CHAMELEON level-0 baselines (no GPU) — the gate every mentalistic claim must pass.

For each stimulus (+ the live agent's own clues, if a battery file is given), computes
non-ToM votes:

  centroid — vote for the seat whose clues are farthest (mean cosine distance) from
             the leave-one-seat-out centroid of ALL clues. Pure outlier detection;
             needs word embeddings (words/vecs npz — regenerate for missing clue words
             with src/qwen32_word_embed.py on the pod, WORDS_FILE=<vocab>).
  lexical  — embedding-free fallback: vote for the seat with the least 4-char-prefix
             overlap between its clues and everyone else's clues.

Agent performance == baseline performance means nothing mentalistic is demonstrated;
the dissoc stimuli are where truth and centroid disagree by construction.

Env: STIMULI(runs/chameleon/stimuli/stimuli.jsonl) BATTERY(optional battery jsonl)
     EMB_NPZ(runs/game-1/qwen32/update_dynamics/qwen32_word_embed.npz)
     OUT(runs/chameleon/battery/baselines.jsonl)
"""
from __future__ import annotations
import os
import json
import numpy as np

STIMULI = os.environ.get("STIMULI", "runs/chameleon/stimuli/stimuli.jsonl")
BATTERY = os.environ.get("BATTERY", "")
EMB_NPZ = os.environ.get("EMB_NPZ", "runs/game-1/qwen32/update_dynamics/qwen32_word_embed.npz")
OUT = os.environ.get("OUT", "runs/chameleon/battery/baselines.jsonl")


def load_emb():
    z = np.load(EMB_NPZ, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    vecs = z["vecs"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return dict(zip(words, vecs))


def seat_clues(stim, agent_clues):
    """seat -> list of clue words (scripted; agent's from the battery if available)."""
    out = {s: [] for s in range(stim["n_players"])}
    for r in range(stim["n_rounds"]):
        for s in range(stim["n_players"]):
            c = stim["clues"][r][s]
            if c is None and agent_clues is not None:
                c = agent_clues[r]
            if c is not None:
                out[s].append(c)
    return out


def centroid_vote(clues_by_seat, emb):
    vecs, missing = {}, []
    for s, cs in clues_by_seat.items():
        vs = [emb[c] for c in cs if c in emb]
        missing += [c for c in cs if c not in emb]
        if not vs:
            return None, None, missing
        vecs[s] = np.stack(vs)
    seats = sorted(vecs)
    dists = {}
    for s in seats:
        rest = np.concatenate([vecs[t] for t in seats if t != s])
        cen = rest.mean(0)
        cen /= np.linalg.norm(cen) + 1e-8
        dists[s] = float(1.0 - (vecs[s] @ cen).mean())
    vote = max(dists, key=dists.get)
    ranked = sorted(dists.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    return vote, {"dists": dists, "margin": margin}, missing


def lexical_vote(clues_by_seat):
    pref = {s: {c[:4] for c in cs} for s, cs in clues_by_seat.items()}
    ov = {}
    for s in pref:
        others = set().union(*(pref[t] for t in pref if t != s))
        ov[s] = len(pref[s] & others) / max(1, len(pref[s]))
    return min(ov, key=ov.get), ov


def main():
    stims = [json.loads(l) for l in open(STIMULI)]
    agent = {}
    if BATTERY and os.path.exists(BATTERY):
        agent = {r["id"]: r.get("agent_clues") for r in map(json.loads, open(BATTERY))}
    emb = load_emb()
    all_missing, recs = set(), []
    for stim in stims:
        cbs = seat_clues(stim, agent.get(stim["id"]))
        cv, cinfo, missing = centroid_vote(cbs, emb)
        all_missing |= set(missing)
        lv, lo = lexical_vote(cbs)
        truth = stim["true_impostor_seat"]
        recs.append({"id": stim["id"], "condition": stim["condition"], "tier": stim["tier"],
                     "true_impostor_seat": truth,
                     "centroid_seat": cv, "centroid_info": cinfo,
                     "centroid_correct": (cv == truth) if truth is not None and cv is not None else None,
                     "lexical_seat": lv,
                     "lexical_correct": (lv == truth) if truth is not None else None,
                     "used_agent_clues": stim["id"] in agent})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    scored = [r for r in recs if r["centroid_correct"] is not None]
    if scored:
        acc = np.mean([r["centroid_correct"] for r in scored])
        print(f"[baselines] centroid acc {acc:.2f} on {len(scored)} scored stimuli")
    if all_missing:
        print(f"[baselines] WARNING {len(all_missing)} clue words missing from {EMB_NPZ}: "
              f"{sorted(all_missing)[:20]}\n  -> regenerate embeddings for the chameleon "
              f"vocab with src/qwen32_word_embed.py (pod) before trusting centroid votes")
    print(f"[baselines] wrote {len(recs)} -> {OUT}")


if __name__ == "__main__":
    main()
