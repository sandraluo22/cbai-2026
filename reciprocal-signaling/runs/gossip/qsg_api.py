"""API-model backend for the QSG gossip game (Anthropic Messages API).

Same game, probe-level measurement only (no logprobs on the API): the speaker's
emission IS a temperature-1 sample given the assistant prefill '{"label": "' —
exactly QSG Hard — and probes are temperature-0 samples (the mode of the belief).
Prompts are built by the same user_msg/memory_block code as the local runs
(qsg_gossip imported with torch stubbed). Transcript schema matches, minus the
p-vectors; probe records carry argmax only.

env: MODEL (anthropic model id)  VAR (graded|curve)  SCHED  P1REL/P2REL  ROUNDS
     STEPS (fixed, no early-stop)  NOTES (""|append)  SEED  OUT
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import types
import urllib.request

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch"].no_grad = lambda: (lambda f: f)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
import qsg_gossip as G  # noqa: E402  (prompt builders only)

API = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OKEY = os.environ.get("OPENAI_API_KEY", "")
NO_PREFILL = set()                                  # models that reject assistant prefill
NO_TEMP = set()                                     # models that reject the temperature param
THINK = int(os.environ.get("THINK", "0"))           # >0: reasoning budget (anthropic thinking
                                                    # / openai o-series); disables prefill
import re as _re


def call_openai(model, system, user, max_tokens, temperature):
    for attempt in range(6):
        body = dict(model=model, max_completion_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
        if model not in NO_TEMP and not model.startswith("o") and not THINK:
            body["temperature"] = temperature
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {OKEY}", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code == 400:
                detail = e.read().decode()[:300]
                if "temperature" in detail:
                    NO_TEMP.add(model)
                    continue
                if "max_tokens" in detail or "output limit" in detail:
                    max_tokens = min(16384, max_tokens * 2)
                    continue
                raise RuntimeError(detail)
            if e.code in (429, 500, 502, 503):
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("openai retries exhausted")


def call(model, system, user, prefill, max_tokens, temperature):
    if model.startswith(("gpt", "o")):
        return call_openai(model, system, user, max_tokens, temperature)
    for attempt in range(6):
        body = dict(model=model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": prefill}] if prefill else
                             [{"role": "user", "content": user}])
        if THINK:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": "high"}
            body["max_tokens"] = max(1024, THINK) + 400
        if model not in NO_TEMP and not THINK:
            body["temperature"] = temperature
        req = urllib.request.Request(
            API + "/v1/messages", data=json.dumps(body).encode(),
            headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return "".join(b.get("text", "") for b in out.get("content", []))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                detail = e.read().decode()[:300]
                if "prefill" in detail and prefill:
                    raise ValueError("noprefill")
                if "temperature" in detail:
                    NO_TEMP.add(model)
                    continue
                raise RuntimeError(detail)
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("API retries exhausted")


def get_label(model, user, labels, rng, temperature):
    for _ in range(3):
        if model.startswith(("gpt", "o")) or model in NO_PREFILL or THINK:
            txt = call(model, G.SYS, user, "", max(1024, THINK) + 400 if THINK else 24,
                       temperature)
            m = _re.search(r'"label"\s*:\s*"([^"]+)"', txt)
            lab = m.group(1).strip() if m else txt.strip().strip('"{}')
        else:
            try:
                txt = call(model, G.SYS, user, '{"label": "', 12, temperature)
            except ValueError:
                NO_PREFILL.add(model)
                continue
            lab = txt.split('"')[0].strip()
        if lab in labels:
            return lab, False
    return rng.choice(labels), True


WORDS = os.environ.get("WORDS", "") == "1"          # real-word labels instead of gibberish


def make_labels(k, rng, used):
    while True:
        if WORDS:
            labs = sorted(rng.sample(G.WORD_POOL, k))
        else:
            labs = sorted({"".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                           for _ in range(k * 2)})[:k]
        if len(labs) == k and not (set(labs) & used):
            used.update(labs)
            return labs


def main():
    model = os.environ.get("MODEL")
    var = os.environ.get("VAR", "curve")
    rounds = int(os.environ.get("ROUNDS", 20))
    steps = int(os.environ.get("STEPS", 40))
    n, k = 5, 3
    seed = int(os.environ.get("SEED", 0))
    notes_mode = os.environ.get("NOTES", "")
    sched = os.environ.get("SCHED", "")
    out_dir = os.environ["OUT"]
    rng = random.Random(seed)
    rng_lab = random.Random(f"{seed}-lab")
    rng_truth = random.Random(f"{seed}-truth")
    rng_pair = random.Random(f"{seed}-pair")
    used = set()
    labels = make_labels(k, rng_lab, used)
    mem = [[] for _ in range(n)]
    notes = [[] for _ in range(n)]
    reveals = {}
    lines = [dict(type="meta", var=var, model=model, rounds=rounds, n=n, k=k, steps=steps,
                  temp=1.0, seed=seed, labels=labels, names=False, fresh=True,
                  notes=notes_mode, backend="api")]

    def ntxt(i):
        return "".join(f"(after round {rr}) {t}\n" for rr, t in notes[i])

    def umsg(i, r, clue):
        return G.user_msg(i, labels, mem[i], reveals, r, clue, rng, False, False, ntxt(i))

    for r in range(1, rounds + 1):
        if r > 1:
            labels = make_labels(k, rng_lab, used)
        correct = rng_truth.choice(labels)
        wrong = rng_truth.choice([l for l in labels if l != correct])
        clue_map = {}
        if sched:
            bits = sched.split(";")
            clue_map[0] = correct if bits[0][r - 1] == "1" else wrong
            if len(bits) > 1:
                clue_map[1] = correct if bits[1][r - 1] == "1" else wrong
        lines.append(dict(type="round_start", round=r, correct=correct, labels=labels,
                          clue=clue_map.get(0), clue_is_wrong=clue_map.get(0) == wrong,
                          clue_map={str(kk + 1): v for kk, v in clue_map.items()}))
        for t in range(steps):
            S, L = rng_pair.sample(range(n), 2)
            s_lab, fb = get_label(model, umsg(S, r, clue_map.get(S)), labels, rng, 1.0)
            mem[L].append((r, S + 1, s_lab))
            lines.append(dict(type="step", round=r, t=t, S=S + 1, L=L + 1, s_label=s_lab,
                              fallback=fb))
        probes = []
        for i in range(n):
            lab, fb = get_label(model, umsg(i, r, clue_map.get(i)), labels, rng, 0.0)
            probes.append(dict(agent=i + 1, argmax=lab, correct=lab == correct, fb=fb))
        lines.append(dict(type="probe", round=r, probes=probes))
        reveals[r] = correct
        if notes_mode == "append":
            for i in range(n):
                stem = umsg(i, r, clue_map.get(i)).split("\nConstraints:")[0]
                t2 = call(model, "You are playing a repeated labeling game.",
                          stem + f"\nEnd of round {r}. Write a short private note (at most "
                          "50 words) about anything that may help you in later rounds.",
                          "", 90, 0.7)
                notes[i].append((r, " ".join(t2.split()[:60])))
                lines.append(dict(type="note", round=r, agent=i + 1, text=notes[i][-1][1]))
        acc = sum(p["correct"] for p in probes) / n
        print(f"[{model} {var} r{r}] correct={correct} probe_acc={acc:.2f} "
              f"argmax={[p['argmax'] for p in probes]}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"gossip_s{seed}")
    with open(stem + "_transcript.jsonl", "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    with open(stem + "_transcript.json", "w") as fh:
        json.dump(lines, fh, indent=1)


if __name__ == "__main__":
    main()
