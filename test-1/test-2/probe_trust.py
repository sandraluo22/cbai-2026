"""test-2 trust probe: does A's update to a partner message depend on the partner's
TRACK RECORD, with all contested evidence held token-identical?

Two teacher-forced contexts per replicate (identical A core prefix, identical
positions; strict alternation B even / A odd is the only provenance cue — the plain
walk format has no speaker tags, so any reliability attribution must ride on the
learnable periodic schedule):

  skeleton   A-slots and B-slots are CORE walk steps (zero contested evidence),
             except NSPLICE controlled assertions: when an A-slot lands on a node
             of the probed site, the next B-slot emits its TRUE partner.
  honest     skeleton with B-slots overwritten by uniform random nodes at rate
             RHO_GOOD (mildly noisy partner; keeps the exact observer off ceiling).
  corrupt    the SAME context, with ADDITIONAL B-slot overwrites so the total
             corruption rate is RHO_BAD. Assertions, their preceding tokens, and
             the last TAIL tokens are protected and IDENTICAL in both histories;
             overwrites are resampled so they never create a contested pair.

Probe: the sequence is steered to end on a probed-site node s_f (A-slot).
  p_before = predictive at cue s_f (context minus its final token + cue)
  p_after  = predictive at cue s_f after appending B's message [true partner]
Statistics (contested discrimination score, t2_core): before/after levels and the
message delta, per history. The Dirichlet-Markov null predicts EXACTLY identical
scores across histories (contested counts matched by construction); the exact
observer shows graded reliability weighting. Any subject history effect is
track-record attribution, not count bookkeeping or instantaneous surprisal gating.
Cues on the two UNPROBED sites (no assertions -> floor 1/3) are within-context
controls.

Env: R(9) S(300) TAIL(60) NSPLICE(2) RHO_GOOD(0.1) RHO_BAD(0.6) CTX(600)
     TEMP(1.0) SEED(0) DRY(0) OUT(runs)
Out: OUT/trust_probe/trust_probe.json + raw dists .npz
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_core as T

R = int(os.environ.get("R", "9"))
S = int(os.environ.get("S", "300"))
TAIL = int(os.environ.get("TAIL", "60"))
NSPLICE = int(os.environ.get("NSPLICE", "2"))
RHO_GOOD = float(os.environ.get("RHO_GOOD", "0.1"))
RHO_BAD = float(os.environ.get("RHO_BAD", "0.6"))
CTX = int(os.environ.get("CTX", "600"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
DRY = os.environ.get("DRY", "0") == "1"
OUT = os.environ.get("OUT", os.path.join(HERE, "runs"))
N = T.N
assert S % 2 == 0, "S must be even so the final token is an A-slot"


def contested_pairs(spec):
    pairs = set()
    for s in spec["sites"]:
        vs = [s[k] for k in "abcd"]
        for i in range(4):
            for j in range(i + 1, 4):
                pairs.add(tuple(sorted((vs[i], vs[j]))))
    return pairs


def overwrite(seq, slots, prefix, protected, rate, rng, cpairs):
    """Overwrite unprotected B-slot tokens w.p. rate; never create a contested pair."""
    out = list(seq)
    for i in range(len(seq)):
        if slots[i] != "B" or i in protected or rng.random() >= rate:
            continue
        pv = prefix[-1] if i == 0 else out[i - 1]
        nx = out[i + 1] if i + 1 < len(seq) else None
        for _ in range(80):
            x = int(rng.integers(N))
            if tuple(sorted((pv, x))) in cpairs:
                continue
            if nx is not None and tuple(sorted((x, nx))) in cpairs:
                continue
            out[i] = x
            break
    return out


def build_pair(spec, rep, seed):
    """Returns dict with prefix, honest, corrupt, slots, cue, partner, site index."""
    rng = np.random.default_rng(seed)
    core = T.spec_adj(spec, "core_edges")
    cpairs = contested_pairs(spec)
    opts = T.contested_options(spec)
    st = rep % 3                                        # probed site rotates
    s_star = spec["sites"][st]["abcd"[(rep // 3) % 4]]  # designated cue node
    prefix = T.walk(core, CTX, seed=seed + 71)
    # torus (and core, its subgraph) is bipartite: A-slots can reach s_star only if
    # the class of prefix[-1] matches s_star's — extend prefix one step if not
    cls = lambda n: ((n // 4) + (n % 4)) % 2            # noqa: E731
    if cls(prefix[-1]) != cls(s_star):
        prefix = prefix + [int(rng.choice(np.where(core[prefix[-1]])[0]))]

    seq, slots, contested_idx = [], [], []
    prev, splices = prefix[-1], 0
    for t in range(S - 1):
        who = "B" if t % 2 == 0 else "A"
        if (who == "B" and splices < NSPLICE and prev == s_star
                and t < S - TAIL and (not contested_idx or
                                      t - contested_idx[-1] > 20)):
            o = opts[prev]
            nxt = o["options"][o["true_m"]]              # assertion: true partner
            contested_idx.append(t)
            splices += 1
        else:
            nxt = int(rng.choice(np.where(core[prev])[0]))
        seq.append(nxt); slots.append(who); prev = nxt
    # steer the final A-slot onto s_star itself (evidence and probe share the cue)
    guard = 0
    while True:
        if len(seq) % 2 == 1 and core[prev, s_star]:     # next idx odd = A-slot
            seq.append(int(s_star)); slots.append("A")
            break
        if guard >= 200 and len(seq) % 2 == 0:
            guard = 0
            # bipartite class-lock fallback: one B-slot teleport to the class
            # adjacent to s_star (identical in both histories -> comparisons exact)
            want = 1 - cls(s_star)
            pool = [x for x in range(N) if cls(x) == want and
                    tuple(sorted((prev, x))) not in cpairs]
            prev = int(rng.choice(pool))
            seq.append(prev); slots.append("B")
        else:
            prev = int(rng.choice(np.where(core[prev])[0]))
            seq.append(prev)
            slots.append("B" if (len(seq) - 1) % 2 == 0 else "A")
        guard += 1
        assert guard < 1000, "steering failed"

    protected = set(contested_idx) | {i - 1 for i in contested_idx} \
        | set(range(len(seq) - TAIL, len(seq)))
    honest = overwrite(seq, slots, prefix, protected, RHO_GOOD,
                       np.random.default_rng(seed + 1), cpairs)
    extra = (RHO_BAD - RHO_GOOD) / max(1.0 - RHO_GOOD, 1e-9)
    pass1 = protected | {i for i in range(len(seq)) if honest[i] != seq[i]}
    corrupt = overwrite(honest, slots, prefix, pass1, extra,
                        np.random.default_rng(seed + 2), cpairs)
    # corrupt_self control: the SAME extra corruption budget applied to A-slots
    # instead (provenance test: attribution predicts a smaller history effect than
    # B-slot corruption; generic context-noise interference predicts an equal one)
    slots_flip = ["B" if s == "A" else "A" for s in slots]
    corrupt_self = overwrite(honest, slots_flip, prefix, pass1, extra,
                             np.random.default_rng(seed + 3), cpairs)
    o = opts[seq[-1]]
    return {"prefix": prefix, "honest": honest, "corrupt": corrupt,
            "corrupt_self": corrupt_self, "slots": slots,
            "cue": int(seq[-1]), "partner": int(o["options"][o["true_m"]]),
            "site": st, "n_assert": splices}


def exact_scores(spec, prefix, seq, slots, cue, partner, opts):
    obs = T.ExactObserver(spec)
    prev = prefix[-1]
    for i, x in enumerate(seq):
        if slots[i] == "B":
            obs.update(prev, x)
        prev = x
    before = T.score(obs.predictive(cue), cue, opts)
    obs2 = obs.copy()
    obs2.update(seq[-1], partner)
    after = T.score(obs2.predictive(cue), cue, opts)
    rho_map = float(obs.rho[int(np.argmax(obs.rho_posterior()))])
    return before, after, rho_map


def main():
    spec_path = os.path.join(OUT, "t2_spec.json")
    spec = (json.load(open(spec_path)) if os.path.exists(spec_path)
            else T.build_spec(SEED))
    T.save_spec(spec, spec_path)
    opts = T.contested_options(spec)
    out = os.path.join(OUT, "trust_probe")
    os.makedirs(out, exist_ok=True)

    backend = None
    if not DRY:
        from run_test2 import LLMBackend
        backend = LLMBackend()

    def model_probe_all(context_nodes):
        if backend is not None:
            return backend.probe_all_cues(context_nodes)
        mb = T.MockBackend(temp=TEMP)
        mb.prefill("x", [context_nodes])
        return np.stack([mb.probe("x", 0, cue) for cue in range(N)])

    rows, dists = [], {}
    for r in range(R):
        bp = build_pair(spec, r, SEED * 100 + r)
        cue, partner = bp["cue"], bp["partner"]
        probed_nodes = {spec["sites"][bp["site"]][k] for k in "abcd"}
        ctrl_cues = [n for n in opts if n not in probed_nodes]
        for name in ("honest", "corrupt", "corrupt_self"):
            seq = bp[name]
            ctx = bp["prefix"] + seq
            pb = model_probe_all(ctx[:-1])
            pa = model_probe_all(ctx + [partner])
            eb, ea, rho_map = exact_scores(spec, bp["prefix"], seq, bp["slots"],
                                           cue, partner, opts)
            rows.append({
                "r": r, "history": name, "cue": cue, "partner": partner,
                "site": bp["site"], "n_assert": bp["n_assert"],
                "score_before": T.score(pb[cue], cue, opts),
                "score_after": T.score(pa[cue], cue, opts),
                "ctrl_before": float(np.mean([T.score(pb[c], c, opts)
                                              for c in ctrl_cues])),
                "exact_before": eb, "exact_after": ea, "exact_rho_map": rho_map,
                "n_corrupted": int(sum(1 for a, b in zip(bp["honest"], seq)
                                       if a != b))})
            dists[f"r{r}_{name}_before"] = pb.astype(np.float32)
            dists[f"r{r}_{name}_after"] = pa.astype(np.float32)
        print(f"rep {r}: site{bp['site']} cue={T.WORDS[cue]} "
              f"true={T.WORDS[partner]} asserts={bp['n_assert']}", flush=True)

    def agg(hist, key):
        v = [row[key] for row in rows if row["history"] == hist]
        return [float(np.mean(v)), float(np.std(v) / max(len(v) - 1, 1) ** 0.5)]

    summary = {"params": {"R": R, "S": S, "TAIL": TAIL, "nsplice": NSPLICE,
                          "rho_good": RHO_GOOD, "rho_bad": RHO_BAD, "ctx": CTX,
                          "temp": TEMP, "seed": SEED, "dry": DRY}}
    for hist in ("honest", "corrupt", "corrupt_self"):
        for key in ("score_before", "score_after", "ctrl_before",
                    "exact_before", "exact_after", "exact_rho_map"):
            summary[f"{hist}_{key}"] = agg(hist, key)
        summary[f"{hist}_delta"] = float(np.mean(
            [row["score_after"] - row["score_before"] for row in rows
             if row["history"] == hist]))
        summary[f"{hist}_exact_delta"] = float(np.mean(
            [row["exact_after"] - row["exact_before"] for row in rows
             if row["history"] == hist]))
    summary["history_effect_before"] = (summary["honest_score_before"][0]
                                        - summary["corrupt_score_before"][0])
    summary["history_effect_after"] = (summary["honest_score_after"][0]
                                       - summary["corrupt_score_after"][0])
    summary["history_effect_exact_after"] = (summary["honest_exact_after"][0]
                                             - summary["corrupt_exact_after"][0])
    summary["self_effect_before"] = (summary["honest_score_before"][0]
                                     - summary["corrupt_self_score_before"][0])
    summary["self_effect_after"] = (summary["honest_score_after"][0]
                                    - summary["corrupt_self_score_after"][0])
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(out, "trust_probe.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "trust_probe_dists.npz"), **dists)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
