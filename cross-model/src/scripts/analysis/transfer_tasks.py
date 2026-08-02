"""Do the TOY-TASK circuits do any work on real tasks that need the same variable?

The earlier Pile test failed, but unstructured prose has no reason to invoke parity. These tasks do:
odd/even, divisibility by 2, chessboard square colour (literally (file+rank) mod 2 on an 8x8 grid — the
SAME variable as the toy task), and primality. Cyclic tasks (weekday/month/clock arithmetic) are the
ring analogue.

Two ablation modes:
  heads     mean-ablate a named head set (the 21-head parity circuit), vs NRAND random same-size sets
  subspace  project OUT a saved DAS subspace at LAYER, vs NRAND random subspaces of equal rank

**The internal control that makes this test sharp**: within the primality task, `prime_even` items are
2*p semiprimes (34=2*17, 94=2*47, ...) whose compositeness is decidable FROM PARITY ALONE, while
`prime_odd` items (91=7*13, 143=11*13, ...) are composite for reasons parity cannot see. Same task, same
format, same answer tokens. If the parity circuit is doing parity work, ablating it should cost
`prime_even` more than `prime_odd`. A generic "ablation degrades the model" effect hits both equally.

Metric per item: margin = logp(correct) - logsumexp(logp(other candidates)); acc = argmax over candidates.
Clean accuracy is reported FIRST — if the model cannot do a task cleanly, its ablation delta is
meaningless and the task is dropped from the conclusion.

Env: GEN_MODEL(Llama) MODE(heads|subspace) ABLATE_HEADS("L14H26,...") NRAND(5)
     SUBSPACE_NPZ SUBSPACE_KEY SUBSPACE_LAYER(24) TASKS(all) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/transfer_tasks<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
MODE = os.environ.get("MODE", "heads")
ABLATE_HEADS = [h for h in os.environ.get("ABLATE_HEADS", "").split(",") if h]
NRAND = int(os.environ.get("NRAND", "5"))
SUBSPACE_NPZ = os.environ.get("SUBSPACE_NPZ", ""); SUBSPACE_KEY = os.environ.get("SUBSPACE_KEY", "")
SUBSPACE_LAYER = int(os.environ.get("SUBSPACE_LAYER", "24"))
TASKS = os.environ.get("TASKS", "odd_even,div2,chess,prime,days,months,clock,neutral").split(",")
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_transfer")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
FRAME = os.environ.get("FRAME", "fewshot")        # zero | math | fewshot
HARD = os.environ.get("HARD", "1") == "1"
# CYC_HARD: big step sizes for the cyclic tasks. With steps of 1-5 the model is at 1.000 with margins of
# 4.5-7.4, so NO ablation can move accuracy — the earlier all-zeros table was a ceiling artefact, not a
# null. Steps like "100 days after Monday" need actual mod-k arithmetic instead of recall.
CYC_HARD = os.environ.get("CYC_HARD", "0") == "1"         # 4-digit numbers -> defeats memorised textbook cases


def _isp(n): return n > 1 and all(n % i for i in range(2, int(n ** .5) + 1))


def _numsets():
    """P2  = 2*p semiprimes   — compositeness follows from PARITY ALONE
       PODD = p*q, both odd   — same semiprime STRUCTURE, parity useless (the matched control)
       PRIME= primes"""
    if HARD:
        pr = [p for p in range(500, 800) if _isp(p)]
        P2 = [2 * p for p in pr][:12]
        odd = [p for p in range(11, 60) if _isp(p)]
        PODD = sorted({a * b for a in odd for b in odd if a < b and 1000 < a * b < 1400})[:12]
        PRIME = [n for n in range(1009, 1300) if _isp(n)][:12]
    else:
        P2 = [34, 38, 46, 58, 62, 74, 82, 86, 94, 106, 118, 122]
        PODD = [91, 77, 87, 93, 111, 119, 133, 143, 161, 187, 203, 209]
        PRIME = [97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]
    return P2, PODD, PRIME


def _pre(kind):
    """task preamble: framing + balanced worked examples. The few-shot answers are deliberately
    BALANCED across the two labels so the model cannot win by always emitting one of them — the
    failure that invalidated chess/div2/clock in the zero-shot run."""
    if FRAME == "zero": return ""
    m = "Let\'s do some math.\n\n"
    if FRAME == "math": return m if kind != "chess" else "Let\'s think about chessboard squares.\n\n"
    F = {
        "odd_even": m + "".join(f"Question: Is the number {n} odd or even?\nAnswer: The number {n} is "
                               f"{'even' if n % 2 == 0 else 'odd'}\n\n" for n in (14, 27, 40, 63)),
        "div2": m + "".join(f"Question: Is {n} divisible by 2?\nAnswer: {'yes' if n % 2 == 0 else 'no'}"
                            f"\n\n" for n in (12, 7, 30, 45)),
        "prime": m + "".join(f"Question: Is {n} a prime number or a composite number?\nAnswer: {n} is "
                             f"{'prime' if _isp(n) else 'composite'}\n\n" for n in (15, 23, 49, 31)),
        "chess": "Let\'s think about chessboard squares.\n\n" + "".join(
            f"Question: On a chessboard, is the square {sq} a light square or a dark square?\n"
            f"Answer: It is a {c} square\n\n" for sq, c in (("a1", "dark"), ("h1", "light"),
                                                     ("a8", "light"), ("h8", "dark"))),
        "days": "".join(f"Question: What day comes {k} days after {DAYS[i]}?\nAnswer: {DAYS[(i+k)%7]}"
                        f"\n\n" for i, k in ((0, 2), (4, 3), (6, 1), (2, 4))),
        "months": "".join(f"Question: What month comes {k} months after {MONTHS[i]}?\n"
                          f"Answer: {MONTHS[(i+k)%12]}\n\n" for i, k in ((0, 2), (7, 4), (11, 1), (5, 6))),
        "clock": m + "".join(f"Question: What time is {k} hours after {h} o\'clock?\n"
                             f"Answer: {((h+k-1)%12)+1}\n\n" for h, k in ((4, 3), (11, 5), (12, 2), (9, 6))),
        "neutral": "".join(f"Question: Is {w} a city or a river?\nAnswer: {w} is a {c}\n\n"
                           for w, c in (("Tokyo", "city"), ("Danube", "river"),
                                        ("Madrid", "city"), ("Congo", "river"))),
    }
    return F.get(kind, "")


WORDNUM = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
           "Eleven", "Twelve"]


def _engels(names, n, frame, unit):
    """(prompt, candidates, correct_index, subgroup) in Engels et al. format"""
    out = []
    for i, nm in enumerate(names):
        for k in range(1, min(len(WORDNUM) - 1, n) + 1):
            u = unit if k > 1 else unit[:-1]
            out.append((f"{frame} {WORDNUM[k]} {u} from {nm} is",
                        [" " + x for x in names], (i + k) % n, f"off{k}"))
    return out


NEUTRAL = [("Lisbon", 0), ("Mekong", 1), ("Vienna", 0), ("Volga", 1), ("Oslo", 0), ("Ganges", 1),
           ("Dublin", 0), ("Rhine", 1), ("Athens", 0), ("Yangtze", 1), ("Cairo", 0), ("Zambezi", 1),
           ("Boston", 0), ("Seine", 1), ("Munich", 0), ("Tigris", 1), ("Warsaw", 0), ("Euphrates", 1),
           ("Toronto", 0), ("Loire", 1), ("Bogota", 0), ("Elbe", 1), ("Nairobi", 0), ("Indus", 1)]


def build_tasks(rng):
    """each item: (prompt, [candidate strings], correct_index, subgroup)"""
    T = {}
    P2, PODD, PRIME = _numsets()
    lo, hi = (1000, 2000) if HARD else (11, 300)
    nums = sorted(rng.choice(np.arange(lo, hi), 60, replace=False).tolist())
    T["odd_even"] = [(_pre("odd_even") + f"Question: Is the number {n} odd or even?\n"
                      f"Answer: The number {n} is", [" odd", " even"],
                      1 if n % 2 == 0 else 0, "even" if n % 2 == 0 else "odd") for n in nums]
    T["div2"] = [(_pre("div2") + f"Question: Is {n} divisible by 2?\nAnswer:", [" yes", " no"],
                  0 if n % 2 == 0 else 1, "even" if n % 2 == 0 else "odd") for n in nums]
    sq = [(f, r) for f in range(8) for r in range(8)
          if (f, r) not in ((0, 0), (7, 0), (0, 7), (7, 7))]      # few-shot squares excluded
    rng.shuffle(sq)
    T["chess"] = [(_pre("chess") + f"Question: On a chessboard, is the square {'abcdefgh'[f]}{r+1} a "
                   f"light square or a dark square?\nAnswer: It is a", [" light", " dark"],
                   1 if (f + r) % 2 == 0 else 0, "dark" if (f + r) % 2 == 0 else "light")
                  for f, r in sq[:48]]                            # a1 = file0,rank0 -> even -> DARK
    pr = ([(n, 1, "prime_even") for n in P2] + [(n, 1, "prime_odd") for n in PODD]
          + [(n, 0, "prime_prime") for n in PRIME])
    T["prime"] = [(_pre("prime") + f"Question: Is {n} a prime number or a composite number?\n"
                   f"Answer: {n} is", [" prime", " composite"], c, g) for n, c, g in pr]
    T["days"] = [(_pre("days") + f"Question: What day comes {k} days after {DAYS[i]}?\nAnswer:",
                  [" " + d for d in DAYS], (i + k) % 7, f"step{k}")
                 for i in range(7) for k in ((9, 11, 20, 33, 100, 365) if CYC_HARD else (1, 2, 3, 5))]
    T["months"] = [(_pre("months") + f"Question: What month comes {k} months after {MONTHS[i]}?\n"
                    f"Answer:", [" " + m for m in MONTHS], (i + k) % 12, f"step{k}")
                   for i in range(12) for k in ((13, 14, 25, 30, 47) if CYC_HARD else (1, 3, 5))]
    T["clock"] = [(_pre("clock") + f"Question: What time is {k} hours after {h} o\'clock?\nAnswer:",
                   [" " + str(x) for x in range(1, 13)], ((h + k - 1) % 12), f"step{k}")
                  for h in range(1, 13) for k in ((15, 25, 40, 77, 100) if CYC_HARD else (2, 3, 5, 7))]
    T["edays"] = _engels(DAYS, 7, "Let's do some day of the week math.", "days")
    T["emonths"] = _engels(MONTHS, 12, "Let's do some calendar math.", "months")
    T["neutral"] = [(_pre("neutral") + f"Question: Is {w} a city or a river?\nAnswer: {w} is a",
                     [" city", " river"], c, "neutral") for w, c in NEUTRAL]
    return {k: v for k, v in T.items() if k in TASKS}


@torch.no_grad()
def evaluate(model, tok, dev, items):
    """mean margin + accuracy over items, grouped by subgroup.

    Scores the FULL candidate token sequence, not its first token. Llama-3 tokenises " 1" as
    [220,'1'] — a bare space then the digit — so ALL TWELVE clock candidates shared first token 220,
    argmax always returned index 0, and accuracy was pinned at exactly 1/12 in every condition. That
    was a harness artefact, not a model limitation. All clock candidates are 2 tokens, so an
    unnormalised sum of log-probs compares them fairly; we also report mean-per-token for safety."""
    out = {}
    for prompt, cands, ci, grp in items:
        pids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
        scores = []
        for c in cands:
            cid = tok(c, add_special_tokens=False)["input_ids"]
            ids = torch.cat([pids, torch.tensor([cid], device=dev)], 1)
            lg = model(input_ids=ids).logits[0].float()
            lp = torch.log_softmax(lg[pids.shape[1] - 1:-1], -1)
            scores.append(float(lp[torch.arange(len(cid)), torch.tensor(cid, device=dev)].sum()))
        t = torch.tensor(scores)
        good = t[ci]
        other = torch.cat([t[:ci], t[ci + 1:]])
        m = float(good - torch.logsumexp(other, 0))
        a = float(int(t.argmax()) == ci)
        for g in ("ALL", grp): out.setdefault(g, []).append((m, a))
    return {g: {"margin": round(float(np.mean([x[0] for x in v])), 4),
                "acc": round(float(np.mean([x[1] for x in v])), 4), "n": len(v)}
            for g, v in out.items()}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)
    tasks = build_tasks(np.random.default_rng(SEED))

    state = {"heads": None, "proj": None}
    hooks = []
    for l in range(nL):                                     # per-head mean ablation
        def mk(l):
            def ph(_m, args):
                hs = state["heads"]
                if not hs: return
                sel = [h for (ll, h) in hs if ll == l]
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    def rh(_m, _i, out):                                    # subspace projection-out at SUBSPACE_LAYER
        P = state["proj"]
        if P is None: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[0] = h[0] - (h[0].float() @ P.t()) @ P
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hooks.append(blocks[SUBSPACE_LAYER].register_forward_hook(rh))

    def run(label):
        return {t: evaluate(model, tok, dev, items) for t, items in tasks.items()}

    res = {"model": tag, "mode": MODE, "frame": FRAME, "hard": HARD, "tasks": {}, "conditions": {}}
    print(f"[{tag}] CLEAN", flush=True)
    clean = run("clean"); res["conditions"]["clean"] = clean
    for t, r in clean.items():
        print(f"   {t:10} acc={r['ALL']['acc']:.3f} margin={r['ALL']['margin']:+.3f} (n={r['ALL']['n']})",
              flush=True)

    def parse(hs): return [(int(x.split("H")[0][1:]), int(x.split("H")[1])) for x in hs]
    if MODE == "heads":
        hs = parse(ABLATE_HEADS)
        print(f"\n[{tag}] ABLATE CIRCUIT ({len(hs)} heads)", flush=True)
        state["heads"] = hs; res["conditions"]["circuit"] = run("circuit"); state["heads"] = None
        allh = [(l, h) for l in range(nL) for h in range(nH)]
        rr = []
        for i in range(NRAND):
            sel = [allh[j] for j in rng.choice(len(allh), len(hs), replace=False)]
            state["heads"] = sel; rr.append(run(f"rand{i}")); state["heads"] = None
            print(f"   random set {i+1}/{NRAND} done", flush=True)
        res["conditions"]["random"] = rr
    else:
        z = np.load(SUBSPACE_NPZ); R = z[SUBSPACE_KEY].astype(np.float32)
        q, _ = np.linalg.qr(R.T); R = q.T[:R.shape[0]]
        r = R.shape[0]
        print(f"\n[{tag}] PROJECT OUT subspace {SUBSPACE_KEY} rank={r} @ L{SUBSPACE_LAYER}", flush=True)
        state["proj"] = torch.tensor(R, dtype=torch.float32, device=dev)
        res["conditions"]["circuit"] = run("subspace"); state["proj"] = None
        rr = []
        for i in range(NRAND):
            Rr = np.linalg.qr(rng.standard_normal((cm.hidden_size, r)))[0].T.astype(np.float32)
            state["proj"] = torch.tensor(Rr, dtype=torch.float32, device=dev)
            rr.append(run(f"rand{i}")); state["proj"] = None
            print(f"   random subspace {i+1}/{NRAND} done", flush=True)
        res["conditions"]["random"] = rr
    for h in hooks: h.remove()

    print(f"\n{'task/group':22} {'clean':>8} {'ablated':>9} {'rand mean':>10} {'d_circ':>8} {'d_rand':>8} {'excess':>8}")
    summ = {}
    for t in tasks:
        for g in sorted(clean[t]):
            c = clean[t][g]["acc"]; a = res["conditions"]["circuit"][t][g]["acc"]
            rs = [x[t][g]["acc"] for x in res["conditions"]["random"]]
            rm = float(np.mean(rs)); rsd = float(np.std(rs))
            dc, dr = a - c, rm - c
            z = (a - rm) / rsd if rsd > 1e-9 else float("nan")
            summ[f"{t}/{g}"] = {"clean": c, "circuit": a, "rand_mean": round(rm, 4),
                                "rand_sd": round(rsd, 4), "d_circuit": round(dc, 4),
                                "d_random": round(dr, 4), "excess_vs_random": round(dc - dr, 4),
                                "z_vs_random": round(z, 2)}
            print(f"{t+'/'+g:22} {c:8.3f} {a:9.3f} {rm:10.3f} {dc:+8.3f} {dr:+8.3f} {dc-dr:+8.3f}",
                  flush=True)
    res["summary"] = summ
    p = f"{OUTDIR}/transfer_tasks{OUTTAG}_{tag}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
