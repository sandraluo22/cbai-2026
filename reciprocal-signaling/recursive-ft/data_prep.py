"""ESConv preparation + strategy classifier for the reciprocal fine-tuning experiment.

Splits ESConv BY DIALOGUE into four disjoint parts so nothing leaks:
  ft      supervised pool for the initial (generation-0) fine-tuning of A_0 / B_0
  pool    contexts only — the recursion pool P; gold responses are never used
  eval    held-out contexts for measurement, never trained on
  clf     dialogues reserved for training/testing the strategy classifier

Strategy grouping (FIXED before any experiment):
  exploratory   Question, Restatement or Paraphrasing, Reflection of feelings
  informational Information, Providing Suggestions
  emotional     Affirmation and Reassurance, Self-disclosure
  ("Others" is dropped.)

Writes data.json (examples per split) and clf.joblib (+ its held-out report).
"""
from __future__ import annotations

import json
import os
import random
import sys

ESCONV = os.environ.get("ESCONV", "/workspace/esconv/ESConv.json")
OUT = os.path.dirname(os.path.abspath(__file__))

GROUP = {
    "Question": "E", "Restatement or Paraphrasing": "E", "Reflection of feelings": "E",
    "Information": "I", "Providing Suggestions": "I",
    "Affirmation and Reassurance": "M", "Self-disclosure": "M",
}
GROUPS = ["E", "I", "M"]
MAX_CTX_TURNS = 6


def build_examples(dialogues):
    """One example per annotated supporter turn: (context, response, group, problem)."""
    out = []
    for di, d in enumerate(dialogues):
        hist = []
        for turn in d["dialog"]:
            content = " ".join(turn["content"].split())
            if turn["speaker"] == "supporter":
                g = GROUP.get((turn.get("annotation") or {}).get("strategy", ""))
                if g and hist and len(content.split()) >= 4:
                    ctx = hist[-MAX_CTX_TURNS:]
                    out.append(dict(dialogue=di, context=ctx, response=content, group=g,
                                    problem=d.get("problem_type", "unknown"),
                                    emotion=d.get("emotion_type", "unknown")))
                hist.append(f"Supporter: {content}")
            else:
                hist.append(f"Seeker: {content}")
    return out


def main():
    rng = random.Random(0)
    dial = json.load(open(ESCONV))
    idx = list(range(len(dial)))
    rng.shuffle(idx)
    n = len(idx)
    parts = dict(ft=idx[:int(.40 * n)], pool=idx[int(.40 * n):int(.65 * n)],
                 eval=idx[int(.65 * n):int(.78 * n)], clf=idx[int(.78 * n):])
    data = {}
    for k, ids in parts.items():
        ex = build_examples([dial[i] for i in ids])
        data[k] = ex
        dist = {g: round(sum(e["group"] == g for e in ex) / len(ex), 3) for g in GROUPS}
        print(f"{k:<5} {len(ids):4d} dialogues -> {len(ex):5d} examples  {dist}", flush=True)
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"))

    # ---- strategy classifier, trained on held-out dialogues only ----------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    ex = data["clf"]
    rng.shuffle(ex)
    cut = int(.8 * len(ex))
    tr, te = ex[:cut], ex[cut:]
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    clf.fit([e["response"] for e in tr], [e["group"] for e in tr])
    pred = clf.predict([e["response"] for e in te])
    gold = [e["group"] for e in te]
    print("\nstrategy classifier, held-out dialogues (n=%d):" % len(te), flush=True)
    print(classification_report(gold, pred, digits=3))
    print("confusion (rows=gold E,I,M):")
    print(confusion_matrix(gold, pred, labels=GROUPS))
    joblib.dump(clf, os.path.join(OUT, "clf.joblib"))
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    cm = confusion_matrix(gold, pred, labels=GROUPS).tolist()   # rows gold, cols predicted
    json.dump(dict(acc=acc, n_test=len(te), confusion=cm, groups=GROUPS),
              open(os.path.join(OUT, "clf_report.json"), "w"))
    print(f"\nclassifier accuracy {acc:.3f} (chance ~{max(sum(g == x for g in gold) for x in GROUPS)/len(gold):.3f})",
          flush=True)
    print("DATA_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
