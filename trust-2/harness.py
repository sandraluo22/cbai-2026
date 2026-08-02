"""Run trials against Llama-3.1-8B (local HF transformers) and parse the readouts.

Backend
-------
Default model is ``NousResearch/Meta-Llama-3.1-8B-Instruct`` — the ungated 3.1-8B
mirror used elsewhere in this repo (no HF token needed), loaded in bf16 on CUDA
with left-padded batched greedy generation. Each trial is a single prompt -> single
generation, so we batch many trials through one ``generate()`` call for throughput.

The model returns free text; we elicit a graded JSON object
(``{"trust": {name: prob}, "confidence": p, "justification": str}``) and parse it
robustly: strip code fences, ``json.loads``, salvage an embedded object, and fall
back to line-wise ``Name: prob`` scraping. Trials whose output won't parse are
regenerated once with sampling. Outputs are cached to disk keyed by a hash of
(model, prompt, gen settings) so reruns are cheap.

Importing this module does NOT import torch — the heavy backend is loaded lazily,
so the parsing/coding helpers (and the unit tests) work on a laptop without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from trials import Trial, render_prompt


DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 320
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_input_tokens: int = 8192
    max_batch: int = 16            # generation batch size (bound VRAM)
    sample_temperature: float = 0.7  # used only on parse-failure retries

    def cache_signature(self) -> dict:
        return {"model": self.model_name, "max_new_tokens": self.max_new_tokens}


# --------------------------------------------------------------------------- #
# Robust parsing + justification coding (no torch — unit tested)
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_NUM = r"-?\d+(?:\.\d+)?"

# Words that indicate the justification appeals to the demonstrated track record
# (vs. surface labels, position, hedging, etc.).
_TRACK_RECORD_TERMS = [
    "track record", "accura", "accurate", "correct", "incorrect", "wrong",
    "reliab", "history", "previous", "earlier", "verif", "record", "demonstrat",
    "consistent", "error", "mistake", "right", "proven", "past", "log", "so far",
]


def _strip_fences(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1) if m else text


def _salvage_object(text: str) -> Optional[dict]:
    """Find the outermost {...} and try to load it."""
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_response(text: str, source_names: list[str]) -> Optional[dict]:
    """Parse a model readout into {"trust": {name: prob}, confidence, justification}.

    Returns ``None`` if no usable trust distribution can be recovered. Probabilities
    are normalised to sum to 1 and restricted to the known source names.
    """
    text = text or ""
    body = _strip_fences(text).strip()

    obj = None
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        obj = _salvage_object(body)

    trust_raw: dict = {}
    confidence = None
    justification = ""

    if isinstance(obj, dict):
        t = obj.get("trust", obj.get("probabilities"))
        if isinstance(t, dict):
            trust_raw = t
        confidence = obj.get("confidence")
        justification = obj.get("justification", "") or ""

    # Fallback: scrape "Name: 0.7" style lines if JSON gave us nothing usable.
    if not trust_raw:
        for nm in source_names:
            m = re.search(rf"{re.escape(nm)}\b[^0-9\-]*({_NUM})", text)
            if m:
                trust_raw[nm] = m.group(1)

    # Coerce to floats restricted to known names.
    trust: dict[str, float] = {}
    for nm in source_names:
        v = trust_raw.get(nm)
        if v is None:
            continue
        try:
            trust[nm] = float(v)
        except (TypeError, ValueError):
            continue

    if not trust:
        return None

    total = sum(max(0.0, p) for p in trust.values())
    if total <= 0:
        # degenerate (all zero / negative) → uniform over the names it mentioned
        trust = {nm: 1.0 / len(trust) for nm in trust}
    else:
        trust = {nm: max(0.0, p) / total for nm, p in trust.items()}

    # Ensure every source has an entry (missing → 0 after the normalisation above).
    for nm in source_names:
        trust.setdefault(nm, 0.0)

    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return {"trust": trust, "confidence": confidence,
            "justification": str(justification)}


def references_track_record(justification: str) -> bool:
    low = (justification or "").lower()
    return any(term in low for term in _TRACK_RECORD_TERMS)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class MockBackend:
    """Deterministic, GPU-free backend for pipeline tests and dry runs.

    Emits a valid JSON readout that mildly favours whichever source has the higher
    demonstrated accuracy — enough to exercise the full parse/analyse path without
    a model. NOT a substitute for a real run.
    """

    model_name = "mock"

    def __init__(self, trials_by_prompt: Optional[dict] = None):
        self._index = trials_by_prompt or {}

    def generate(self, prompts: list[tuple[str, str]], sample: bool = False,
                 seed: int = 0) -> list[str]:
        outs = []
        for k, (_system, user) in enumerate(prompts):
            names = re.findall(r"  - (\S+)", user.split("VERIFICATION LOG")[0])
            # crude accuracy proxy: count CORRECT mentions following each name
            scores = {}
            for nm in names:
                hits = len(re.findall(rf"{re.escape(nm)}.*?-> CORRECT", user))
                tot = len(re.findall(rf"{re.escape(nm)} ", user))
                scores[nm] = (hits + 1) / (tot + 2)
            tot = sum(scores.values()) or 1.0
            trust = {nm: round(v / tot, 3) for nm, v in scores.items()}
            obj = {"trust": trust, "confidence": 0.6,
                   "justification": "Based on each source's verified accuracy in the log."}
            outs.append(json.dumps(obj))
        return outs


class LlamaBackend:
    """Local HF transformers backend (mirrors trust/llama_runner.py conventions)."""

    def __init__(self, cfg: GenConfig):
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.model_name = cfg.model_name
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"          # correct batched decoder-only generation
        dt = getattr(torch, cfg.dtype)
        try:                               # transformers 5.x: dtype=; older: torch_dtype=
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, dtype=dt, device_map=cfg.device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, torch_dtype=dt, device_map=cfg.device)
        model.eval()
        self.model, self.tok = model, tok

    def _chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=False)

    def generate(self, prompts: list[tuple[str, str]], sample: bool = False,
                 seed: int = 0) -> list[str]:
        torch = self.torch
        cfg = self.cfg
        chats = [self._chat(s, u) for s, u in prompts]
        texts: list[str] = []
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            for i in range(0, len(chats), cfg.max_batch):
                chunk = chats[i:i + cfg.max_batch]
                enc = self.tok(chunk, return_tensors="pt", padding=True,
                               truncation=True,
                               max_length=cfg.max_input_tokens).to(self.model.device)
                gen_kw = dict(max_new_tokens=cfg.max_new_tokens,
                              pad_token_id=self.tok.eos_token_id)
                if sample:
                    gen_kw.update(do_sample=True, temperature=cfg.sample_temperature,
                                  top_p=0.9)
                else:
                    gen_kw.update(do_sample=False)
                out = self.model.generate(**enc, **gen_kw)
                out = out[:, enc["input_ids"].shape[1]:]
                texts.extend(self.tok.batch_decode(out, skip_special_tokens=True))
        return texts


def make_backend(name: str, cfg: GenConfig):
    if name == "mock":
        return MockBackend()
    if name == "llama":
        return LlamaBackend(cfg)
    raise ValueError(f"unknown backend {name!r} (choices: llama, mock)")


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
def _cache_key(trial: Trial, system: str, user: str, cfg: GenConfig) -> str:
    blob = json.dumps({"trial_hash": trial.trial_hash(), "system": system,
                       "user": user, **cfg.cache_signature()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_trials(trials: list[Trial], backend, cfg: GenConfig,
               cache_dir: Optional[str] = None, use_cache: bool = True,
               verbose: bool = True) -> list[dict]:
    """Generate (batched, cached) readouts for all trials and return result records."""
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)

    prompts = [render_prompt(t) for t in trials]
    keys = [_cache_key(t, s, u, cfg) for t, (s, u) in zip(trials, prompts)]
    raw: list[Optional[str]] = [None] * len(trials)
    cached_flag = [False] * len(trials)

    # cache lookup
    misses = []
    for i, key in enumerate(keys):
        if cache_dir and use_cache:
            p = _cache_path(cache_dir, key)
            if os.path.exists(p):
                with open(p) as f:
                    raw[i] = json.load(f)["text"]
                    cached_flag[i] = True
                    continue
        misses.append(i)

    if verbose:
        print(f"[harness] {len(trials)} trials | {len(trials) - len(misses)} cached "
              f"| {len(misses)} to generate", flush=True)

    # generate misses (batched inside the backend)
    if misses:
        texts = backend.generate([prompts[i] for i in misses])
        for i, text in zip(misses, texts):
            raw[i] = text
            if cache_dir and use_cache:
                with open(_cache_path(cache_dir, keys[i]), "w") as f:
                    json.dump({"text": text, "trial_id": trials[i].trial_id}, f)

    # parse; collect parse failures for one sampled retry pass
    def _names(t: Trial) -> list[str]:
        return [s.name for s in t.sources]

    parsed = [parse_response(raw[i], _names(trials[i])) for i in range(len(trials))]
    retry_idx = [i for i, p in enumerate(parsed) if p is None]
    if retry_idx and verbose:
        print(f"[harness] {len(retry_idx)} unparsable; retrying with sampling", flush=True)
    if retry_idx:
        retry_texts = backend.generate([prompts[i] for i in retry_idx],
                                       sample=True, seed=1234)
        for i, text in zip(retry_idx, retry_texts):
            raw[i] = text
            parsed[i] = parse_response(text, _names(trials[i]))
            if cache_dir and use_cache:   # overwrite cache with the (re)usable output
                with open(_cache_path(cache_dir, keys[i]), "w") as f:
                    json.dump({"text": text, "trial_id": trials[i].trial_id}, f)

    # assemble records
    results = []
    for i, t in enumerate(trials):
        rec = t.record()
        p = parsed[i]
        names = _names(t)
        if p is None:
            response = {"parse_ok": False, "trust_by_name": {}, "trust_by_key": {},
                        "confidence": None, "justification": "",
                        "references_track_record": False, "raw_text": raw[i]}
        else:
            name2key = {s.name: s.key for s in t.sources}
            trust_by_key = {name2key[nm]: pr for nm, pr in p["trust"].items()
                            if nm in name2key}
            response = {
                "parse_ok": True,
                "trust_by_name": p["trust"],
                "trust_by_key": trust_by_key,
                "confidence": p["confidence"],
                "justification": p["justification"],
                "references_track_record": references_track_record(p["justification"]),
                "raw_text": raw[i],
            }
        response["model"] = backend.model_name
        response["cached"] = cached_flag[i]
        rec["response"] = response
        results.append(rec)
    return results
