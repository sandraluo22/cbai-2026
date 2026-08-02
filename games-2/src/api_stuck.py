"""API-MODEL ARM (2026-08-02): does the stuck-priors phenomenon appear in Anthropic
API models? Two arms, matched as closely as the API allows to the local harness.

Arm G (game): restricted city game. A unrestricted, B told to only say cities
  (same RESTR text as intervention_ladder). Client-side resample-24 novelty handler
  (3 rounds x 8 concurrent samples, first valid in order), CAP(40) turns,
  N(12) start pairs from the same start_words.txt. At turn PROBE(8): K(64)-sample
  MC proposal profile (mass on used / self 4-prefix family / city category).
  Onset = A's word shares a 4-char prefix with an earlier own word (len>3).

Arm D (dose): dose_0..4 morphological-family seeds (MORPH) planted as A's OWN words
  against replayed B streams from the local qwen32 strict city games (identical
  streams to seed_matrix). Measures: K-sample target/used/cat mass at the branch
  state + 6-turn live continuation scored for family hits.

Elicitation parity with the local harness:
  - user message = the same body text (OPEN_PROMPT + rounds + used-list)
  - assistant prefill "My word:" (the local scripts append "\nMy word:" after the
    chat-template assistant header)
  - temperature 0.7, max_new_tokens ~ max_tokens 6, clean_word parsing
  Deviations (unavoidable): no top_p (Claude 4+ rejects temperature+top_p
  together; local used top_p 0.95), no seed control, API models are RLHF chat
  models with much larger scale.

Providers (dispatch on model name):
  - Anthropic (claude-*): assistant-prefill "My word:", temperature 0.7 (no top_p —
    Claude 4+ rejects the pair). claude-haiku-4-5 / claude-sonnet-4-5 accept both
    prefill and non-default temperature; Sonnet 4.6/5 do not.
  - OpenAI (gpt-*/o*): no prefill on chat.completions — instruction elicitation
    (OAI_ELICIT) instead; temperature 0.7 + top_p 0.95 (full sampling parity);
    n= used for one-call multi-sampling. Avoid reasoning models (o*/gpt-5 base):
    they reject non-default temperature and burn hidden CoT.

Env: MODELS(claude-haiku-4-5,claude-sonnet-4-5) ARMS(game,dose) N(12) CAP(40)
     K(64) TEMP(0.7) PROBE(8) N_STREAMS(6) CONC(16) RUN_DIR(runs/game-1/7_api_models)
"""
from __future__ import annotations
import os
import re
import ast
import json
import asyncio
import collections

import anthropic
try:
    import openai
except ImportError:
    openai = None

MODELS = os.environ.get("MODELS", "claude-haiku-4-5,claude-sonnet-4-5").split(",")
ARMS = os.environ.get("ARMS", "game,dose").split(",")
N = int(os.environ.get("N", "12"))
CAP = int(os.environ.get("CAP", "100"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
CONC = int(os.environ.get("CONC", "16"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/game-1/7_api_models")
START_FILE = os.environ.get("START_FILE", "runs/game-1/5_mechanistic/pca_w2v/start_words.txt")
STRICT_SRC = os.environ.get(
    "STRICT_SRC", "runs/game-1/2_restricted_core/qwen32_strict_cap100/game1_strict_city_transcript.jsonl")

OPEN_PROMPT = ("You are playing a word game with another player. Each round, you both say one word at the "
               "same time, then the two words are revealed. You win the round only if you both said the same "
               "word. You may not say any word that has already been said by either player.")
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")
MORPH = ["planted", "planting", "plantings", "replant"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket"]


def clean_word(txt):
    w = re.split(r"\s+", txt.strip())
    return re.sub(r"[^a-zA-Z\-]", "", w[0] if w else "").lower()


def load_catwords_city():
    """CATWORDS lives in game1_strict.py, which imports torch — parse it instead."""
    src = open(os.path.join(os.path.dirname(__file__), "game1_strict.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CATWORDS" for t in node.targets):
            return set(ast.literal_eval(node.value)["city"])
    raise RuntimeError("CATWORDS not found in game1_strict.py")


def body_of(hist, used, extra=""):
    s = OPEN_PROMPT + extra + ((" " + " ".join(
        f"Round {k+1}: the other player said {o}, you said {s_}."
        for k, (o, s_) in enumerate(hist))) if hist else "")
    return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."


class Api:
    """Provider-agnostic base: subclasses implement sample_n(body, n) -> list[str]."""

    def __init__(self, model):
        self.model = model
        self.sem = asyncio.Semaphore(CONC)
        self.in_toks = 0
        self.out_toks = 0
        self.calls = 0

    async def gen_word(self, body, forbidden):
        """resample-24 analogue: escalating rounds (1, 8, 15 samples),
        first valid in order — 24 samples max, ~1 sample in the common case."""
        last = ""
        for size in (1, 8, 15):
            ws = await self.sample_n(body, size)
            for w in ws:
                if w:
                    last = w
                    if w not in forbidden:
                        return w
        return last

    async def propose_k(self, body):
        return await self.sample_n(body, K)


class AnthropicApi(Api):
    """Assistant-prefill elicitation ("My word:"), temperature only (Claude 4+
    rejects temperature+top_p together)."""

    def __init__(self, model):
        super().__init__(model)
        self.client = anthropic.AsyncAnthropic(max_retries=5)

    async def sample_one(self, body):
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.messages.create(
                        model=self.model, max_tokens=6, temperature=TEMP,
                        messages=[{"role": "user", "content": body},
                                  {"role": "assistant", "content": "My word:"}])
                    break
                except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    if attempt == 3:
                        print(f"[api] giving up on call: {type(e).__name__} {str(e)[:120]}",
                              flush=True)
                        return ""
                    await asyncio.sleep(5 * (attempt + 1))
        self.calls += 1
        self.in_toks += r.usage.input_tokens
        self.out_toks += r.usage.output_tokens
        txt = next((b.text for b in r.content if b.type == "text"), "")
        return clean_word(txt)

    async def sample_n(self, body, n):
        return list(await asyncio.gather(*[self.sample_one(body) for _ in range(n)]))


OAI_ELICIT = " Respond with only your single word for this round — no punctuation, no explanation."


class OpenAIApi(Api):
    """No assistant prefill on chat.completions — instruction-based elicitation
    instead (OAI_ELICIT appended to the prompt). Full sampling parity otherwise:
    temperature 0.7 AND top_p 0.95. Uses n= for one-call multi-sampling."""

    def __init__(self, model):
        super().__init__(model)
        self.client = openai.AsyncOpenAI(max_retries=5)

    async def sample_n(self, body, n):
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, n=n, max_completion_tokens=8,
                        temperature=TEMP, top_p=0.95,
                        messages=[{"role": "user", "content": body + OAI_ELICIT}])
                    break
                except (openai.APIStatusError, openai.APIConnectionError) as e:
                    if attempt == 3:
                        print(f"[api] giving up on call: {type(e).__name__} {str(e)[:120]}",
                              flush=True)
                        return [""] * n
                    await asyncio.sleep(5 * (attempt + 1))
        self.calls += 1
        if r.usage:
            self.in_toks += r.usage.prompt_tokens
            self.out_toks += r.usage.completion_tokens
        return [clean_word(c.message.content or "") for c in r.choices]


def make_api(model):
    if model.startswith("gpt-") or re.match(r"^o\d", model):
        return OpenAIApi(model)
    return AnthropicApi(model)


def load_starts():
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    return starts


async def run_game_arm(api, catset, starts, tf):
    async def one_game(roll, sa, sb):
        histA, histB = [(sb, sa)], [(sa, sb)]
        used = {sa, sb}
        own = [sa]
        agreed_at, onset, probe, aborted = None, None, None, False
        for t in range(1, CAP):
            if t == PROBE:
                props = await api.propose_k(body_of(histA, used))
                fams = {w[:4] for w in own if len(w) > 3}
                probe = {
                    "used": sum(1 for w in props if w and w in used) / len(props),
                    "selffam": sum(1 for w in props if w and w not in used and len(w) > 3
                                   and w[:4] in fams) / len(props),
                    "cat": sum(1 for w in props if w and w not in used and w in catset)
                           / len(props)}
            wA, wB = await asyncio.gather(
                api.gen_word(body_of(histA, used), used),
                api.gen_word(body_of(histB, used, RESTR), used))
            if not wA or not wB:
                # API failure (e.g. exhausted credits): abort — don't record junk turns
                aborted = True
                print(f"[api:{api.model}] game roll={roll}: ABORTED at turn {t} "
                      f"(empty word from API)", flush=True)
                break
            # a valid meet is a NOVEL shared word; identical returns of a used
            # word (both gen_words exhausting resamples) do not count
            valid_meet = wA == wB and wA not in used
            if onset is None and len(wA) > 3 and any(
                    wA[:4] == p[:4] and len(p) > 3 for p in own):
                onset = t
            tf.write(json.dumps({"arm": "game", "model": api.model, "rollout": roll,
                                 "turn": t, "A": wA, "B": wB, "agreed": valid_meet}) + "\n")
            tf.flush()
            if valid_meet:
                agreed_at = t
                break
            used |= {wA, wB}
            own.append(wA)
            histA.append((wB, wA)); histB.append((wA, wB))
        print(f"[api:{api.model}] game roll={roll}: "
              f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
              flush=True)
        return {"rollout": roll, "agreed_at": agreed_at, "onset": onset, "probe": probe,
                "aborted": aborted}

    return await asyncio.gather(*[one_game(i, sa, sb)
                                  for i, (sa, sb) in enumerate(starts[:N])])


async def run_dose_arm(api, catset, starts, tf):
    games = collections.defaultdict(list)
    for line in open(STRICT_SRC):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]]

    async def one_branch(si, ts, dose):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        seeds = MORPH[:dose]
        n_rounds = max(len(seeds), 3)
        hist = [(sb, sa)]
        used = {sa, sb}
        fill = iter(FILLER)
        for i in range(n_rounds):
            a = seeds[i] if i < len(seeds) else next(fill)
            hist.append((Bseq[i], a))
            used |= {a, Bseq[i]}
        props = await api.propose_k(body_of(hist, used))
        tm = sum(1 for w in props if w and w not in used and len(w) > 3
                 and any(w[:4] == s[:4] for s in (seeds or MORPH))) / len(props)
        um = sum(1 for w in props if w and w in used) / len(props)
        cm = sum(1 for w in props if w and w not in used and w in catset) / len(props)
        fam_hits = 0
        h2, u2 = list(hist), set(used)
        base_i = len(h2) - 1
        for ct in range(6):
            w = await api.gen_word(body_of(h2, u2), u2)
            fam_hits += any(w[:4] == s[:4] and len(w) > 3 for s in (seeds or MORPH))
            bidx = base_i + ct
            b = Bseq[bidx] if bidx < len(Bseq) else Bseq[-1]
            h2.append((b, w)); u2 |= {w, b}
            tf.write(json.dumps({"arm": "dose", "model": api.model, "stream": roll,
                                 "dose": dose, "cont_turn": ct, "A": w, "B": b}) + "\n")
            tf.flush()
        print(f"[api:{api.model}] dose_{dose} s{si}: target {tm:.2f} used {um:.2f} "
              f"cat {cm:.2f} hits6 {fam_hits}", flush=True)
        return {"cell": f"dose_{dose}", "stream": roll, "target_mass": tm,
                "used_mass": um, "cat_mass": cm, "fam_hits6": fam_hits}

    return await asyncio.gather(*[one_branch(si, ts, d)
                                  for si, ts in enumerate(streams) for d in range(5)])


async def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    catset = load_catwords_city()
    starts = load_starts()
    for model in MODELS:
        api = make_api(model)
        tag = model.replace("-", "_")
        tf = open(os.path.join(RUN_DIR, f"api_stuck_{tag}_transcript.jsonl"), "w")
        out = {"model": model, "temp": TEMP, "k": K, "cap": CAP, "n": N}
        if "game" in ARMS:
            games = await run_game_arm(api, catset, starts, tf)
            met = [g["agreed_at"] is not None for g in games]
            onset = [g["onset"] is not None for g in games]
            probes = [g["probe"] for g in games if g["probe"]]
            out["game"] = {
                "per_game": games,
                "met_frac": sum(met) / len(met),
                "onset_frac": sum(onset) / len(onset),
                "probe_used": sum(p["used"] for p in probes) / len(probes) if probes else None,
                "probe_selffam": sum(p["selffam"] for p in probes) / len(probes) if probes else None,
                "probe_cat": sum(p["cat"] for p in probes) / len(probes) if probes else None,
                "n_probes": len(probes)}
            g = out["game"]
            print(f"[api:{model}] === game: met {g['met_frac']:.2f} onset "
                  f"{g['onset_frac']:.2f} probe cat {g['probe_cat']} "
                  f"selffam {g['probe_selffam']}", flush=True)
        if "dose" in ARMS:
            branches = await run_dose_arm(api, catset, starts, tf)
            cells = {}
            for name in sorted({b["cell"] for b in branches}):
                sel = [b for b in branches if b["cell"] == name]
                cells[name] = {k: sum(b[k] for b in sel) / len(sel)
                               for k in ("target_mass", "used_mass", "cat_mass", "fam_hits6")}
                c = cells[name]
                print(f"[api:{model}] === {name}: target {c['target_mass']:.2f} "
                      f"cat {c['cat_mass']:.2f} hits6 {c['fam_hits6']:.1f}", flush=True)
            out["dose"] = {"per_branch": branches, "cells": cells}
        out["usage"] = {"calls": api.calls, "input_tokens": api.in_toks,
                        "output_tokens": api.out_toks}
        json.dump(out, open(os.path.join(RUN_DIR, f"api_stuck_{tag}.json"), "w"), indent=1)
        tf.close()
        print(f"[api:{model}] done. calls={api.calls} in={api.in_toks} "
              f"out={api.out_toks}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
