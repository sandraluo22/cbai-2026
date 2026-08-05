"""CONTINUOUS PROBES on final-position residuals (2026-08-03).

Per user spec: probe CONTINUOUS quantities, never "eventually stuck"; held-out
evaluation. Three targets:
  1. category log-odds     logit(MC category mass)   [strict-city states]
  2. self-stem mass        logit(MC self_family mass) [strict-city states]
  3. action-vs-prediction gap: P_cat(prediction frame) - P_cat(action frame)
     at the dissociation snapshots [one value per snapshot; residuals from the
     ACTION-frame prompt]
Probes: ridge regression per layer on final-position residuals.
Held-out: targets 1-2 GroupKFold by GAME (strict games are city-only — category
holdout impossible there; noted). Target 3 leave-one-CATEGORY-out (6 categories).
Baseline control: same probes on shuffled targets (permutation floor).

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) SWEEP START_FILE RIDGE(10.0)
     RUN_DIR(runs/probe_continuous)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G
import dissociation_branches as D

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
SWEEP = os.environ.get("SWEEP", "runs/stuck_repro/stuck_repro_QwenInst32_transcript.jsonl")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "10,100,1000,10000,100000").split(",")]
RUN_DIR = os.environ.get("RUN_DIR", "runs/probe_continuous")
MAXT = int(os.environ.get("MAXT", "24"))
K = 64


def _ridge_pred(Xt, yt, Xe, alpha):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = (Xt - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Xt.shape[1]), Z.T @ (yt - yt.mean()))
    return ((Xe - mu) / sd) @ w + yt.mean()


def ridge_cv_r2(X, y, groups, alphas=None):
    """GroupKFold ridge R^2 with NESTED per-fold alpha selection (inner group split)."""
    alphas = alphas or ALPHAS
    ug = sorted(set(groups))
    folds = [set(ug[i::6]) for i in range(min(6, len(ug)))]
    yhat = np.zeros_like(y)
    for f in folds:
        tr = np.array([g not in f for g in groups])
        te = ~tr
        if te.sum() == 0 or tr.sum() == 0:
            continue
        tg = sorted({g for g, t in zip(groups, tr) if t})
        inner = set(tg[::4]) or set(tg[:1])          # ~1/4 of train games for selection
        itr = np.array([t and g not in inner for g, t in zip(groups, tr)])
        iva = np.array([t and g in inner for g, t in zip(groups, tr)])
        best_a, best = alphas[0], -np.inf
        for a in alphas:
            pv = _ridge_pred(X[itr], y[itr], X[iva], a)
            r2 = 1 - ((y[iva] - pv) ** 2).sum() / (((y[iva] - y[iva].mean()) ** 2).sum() + 1e-9)
            if r2 > best:
                best, best_a = r2, a
        yhat[te] = _ridge_pred(X[tr], y[tr], X[te], best_a)
    ss = ((y - yhat) ** 2).sum()
    return 1 - ss / (((y - y.mean()) ** 2).sum() + 1e-9)


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = len(model.model.layers)

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    cache = os.path.join(RUN_DIR, "resid_cache.npz")
    @torch.no_grad()
    def resid_final(prompt):
        enc = tok(prompt, return_tensors="pt").to(dev)
        out = model(enc.input_ids, output_hidden_states=True)
        return np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states[1:]])

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        Xs, y_cat, y_stem, gid = z["Xs"], z["y_cat"], z["y_stem"], list(z["gid"])
        Xg, y_gap, cat_g = z["Xg"], z["y_gap"], list(z["cat_g"])
        print(f"[probe] loaded residual cache ({len(y_cat)} states)", flush=True)
        run_probes(Xs, y_cat, y_stem, gid, Xg, y_gap, cat_g)
        return

    # ---------- targets 1-2: strict-city telemetry states ----------
    tele = [json.loads(l) for l in open(os.path.join(SRC_DIR, "proposal_telemetry.jsonl"))
            ]
    tele = [t for t in tele if t["cat"] == "city" and t["turn"] <= MAXT]
    trans = collections.defaultdict(dict)
    for l in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        r = json.loads(l)
        trans[r["rollout"]][r["turn"]] = r
    eps = 1 / (2 * K)
    Xs, y_cat, y_stem, gid = [], [], [], []
    for roll in sorted(trans):
        rows = sorted((t for t in tele if t["rollout"] == roll), key=lambda t: t["turn"])
        sa, sb = starts[roll]
        histA = [(sb, sa)]
        used = {sa, sb}
        for t in rows:
            tr = trans[roll].get(t["turn"])
            if tr is None:
                break
            body = (G.OPEN_PROMPT + " "
                    + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                               for k, (o, s) in enumerate(histA))
                    + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
            Xs.append(resid_final(LA._render(tok, body) + "\nMy word:"))
            pc = min(max(t["frac"].get("category", 0), eps), 1 - eps)
            ps = min(max(t["frac"].get("self_family", 0), eps), 1 - eps)
            y_cat.append(np.log(pc / (1 - pc)))
            y_stem.append(np.log(ps / (1 - ps)))
            gid.append(roll)
            used |= {tr["A"], tr["B"]}
            histA.append((tr["B"], tr["A"]))
        print(f"[probe] collected game {roll} ({len(Xs)} states)", flush=True)
    Xs = np.stack(Xs)          # (n, nL, d)
    y_cat, y_stem = np.array(y_cat), np.array(y_stem)

    # ---------- target 3: action-pred gap at dissociation snapshots ----------
    snaps = D.snapshots_from_sweep(SWEEP, starts)
    dis = json.load(open("runs/dissoc_branches/dissociation.json"))["per_snapshot"]
    gap_by_key = {(s["cat"], s["pv"], s["roll"]): s["P_pred"] - s["P_act"] for s in dis}
    rng = np.random.default_rng(0)
    Xg, y_gap, cat_g = [], [], []
    for s in snaps:
        key = (s["cat"], s["pv"], s["roll"])
        if key not in gap_by_key:
            continue
        base = D.BASE0 if s["pv"] == 0 else D.BASE1
        cin, cout = D.CANDS[s["cat"]]
        cin = [w for w in cin if w not in s["used"]]
        cout = [w for w in cout if w not in s["used"]]
        order = list(cin) + list(cout)
        order = [order[i] for i in rng.permutation(len(order))]
        ctx = D.build_A(tok, base, s["histA"], set(s["used"]))
        p_act = LA._render(tok, ctx + f" For this round, you must choose your word from "
                           f"this list: {', '.join(order)}.") + "\nMy word:"
        Xg.append(resid_final(p_act))
        y_gap.append(gap_by_key[key])
        cat_g.append(s["cat"])
    Xg = np.stack(Xg)
    y_gap = np.array(y_gap)
    print(f"[probe] gap states: {len(y_gap)}", flush=True)
    np.savez_compressed(cache, Xs=Xs.astype(np.float16), y_cat=y_cat, y_stem=y_stem,
                        gid=np.array(gid), Xg=Xg.astype(np.float16), y_gap=y_gap,
                        cat_g=np.array(cat_g))
    run_probes(Xs, y_cat, y_stem, gid, Xg, y_gap, cat_g)


def run_probes(Xs, y_cat, y_stem, gid, Xg, y_gap, cat_g):
    nL = Xs.shape[1]
    Xs = Xs.astype(np.float64); Xg = Xg.astype(np.float64)
    # ---------- fit probes per layer ----------
    out = {"n_states": len(y_cat), "n_gap": len(y_gap), "alphas": ALPHAS, "layers": nL,
           "r2": {"cat_logodds": [], "stem_logodds": [], "act_pred_gap": []},
           "r2_shuffled": {"cat_logodds": [], "stem_logodds": [], "act_pred_gap": []}}
    rngp = np.random.default_rng(1)
    ycs = rngp.permutation(y_cat); yss = rngp.permutation(y_stem); ygs = rngp.permutation(y_gap)
    for L in range(nL):
        XL = Xs[:, L, :]
        out["r2"]["cat_logodds"].append(ridge_cv_r2(XL, y_cat, gid))
        out["r2"]["stem_logodds"].append(ridge_cv_r2(XL, y_stem, gid))
        out["r2_shuffled"]["cat_logodds"].append(ridge_cv_r2(XL, ycs, gid))
        out["r2_shuffled"]["stem_logodds"].append(ridge_cv_r2(XL, yss, gid))
        XG = Xg[:, L, :]
        out["r2"]["act_pred_gap"].append(ridge_cv_r2(XG, y_gap, cat_g))
        out["r2_shuffled"]["act_pred_gap"].append(ridge_cv_r2(XG, ygs, cat_g))
        if L % 8 == 0:
            print(f"[probe] L{L}: cat {out['r2']['cat_logodds'][-1]:.2f} "
                  f"stem {out['r2']['stem_logodds'][-1]:.2f} "
                  f"gap {out['r2']['act_pred_gap'][-1]:.2f}", flush=True)
        json.dump(out, open(os.path.join(RUN_DIR, "probes.json"), "w"))
    json.dump(out, open(os.path.join(RUN_DIR, "probes.json"), "w"), indent=1)
    b = {k: (int(np.argmax(v)), round(max(v), 3)) for k, v in out["r2"].items()}
    print(f"[probe] === best layers: {b}", flush=True)


if __name__ == "__main__":
    main()
