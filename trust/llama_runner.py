"""Local Llama backend (HF transformers) for the trust-learning runner.

Mirrors runner.play_game's role but for a GPU-hosted causal LM (default
NousResearch/Meta-Llama-3.1-8B-Instruct — an ungated mirror, no HF token needed).

Throughput trick: games within a condition are INDEPENDENT but each game's rounds are
SEQUENTIAL (in-context learning). So we step all games of a condition in LOCKSTEP and
BATCH the round-t prompts through one generate() call — turning ~games*T serial calls
into T batched calls. Outputs are short numeric readouts parsed from text (no schema /
structured output as with the API backend).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

import prompt as P
from runner import RoundRecord     # reused dataclass (runner imports anthropic lazily)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LlamaConfig:
    model_name: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 64
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_input_tokens: int = 8192
    enable_thinking: bool | None = None   # Qwen3 etc.: pass False to suppress <think> blocks
    max_batch: int | None = None          # cap generation batch size (for large models / VRAM)


def load(cfg: LlamaConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for correct batched decoder-only generation
    dt = getattr(torch, cfg.dtype)
    try:                               # transformers 5.x uses `dtype=`; older uses `torch_dtype=`
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dt,
                                                     device_map=cfg.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dt,
                                                     device_map=cfg.device)
    model.eval()
    return model, tok


def _format_instruction(M: int) -> str:
    return (f" Respond with exactly one line per company, in the form 'Co.N: <number>', "
            f"for companies 1 to {M}, and nothing else.")


_NUM = r"-?\d+(?:\.\d+)?"
_CO = re.compile(rf"Co\.?\s*(\d+)\s*[:=]\s*({_NUM})")


def parse_text(text: str, M: int) -> np.ndarray:
    out = np.full(M, np.nan)
    for m in _CO.finditer(text):
        c = int(m.group(1)) - 1
        if 0 <= c < M and np.isnan(out[c]):
            out[c] = float(m.group(2))
    if np.all(np.isnan(out)):           # fallback: first M bare numbers
        nums = re.findall(_NUM, text)
        for i, n in enumerate(nums[:M]):
            out[i] = float(n)
    return out


def _chat(tok, system: str, user: str, enable_thinking: bool | None = None) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


@torch.no_grad()
def _generate_batch(model, tok, prompts: list[str], cfg: LlamaConfig) -> list[str]:
    mb = cfg.max_batch or len(prompts)
    texts: list[str] = []
    for i in range(0, len(prompts), mb):       # chunk to bound VRAM on large models
        chunk = prompts[i:i + mb]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_input_tokens).to(model.device)
        gen = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        out = gen[:, enc["input_ids"].shape[1]:]
        texts.extend(tok.batch_decode(out, skip_special_tokens=True))
    return texts


def run_condition(games, style: P.PromptStyle, cfg: LlamaConfig, model, tok):
    """Lockstep over rounds; batched generation across games. Returns list (per game) of
    flat per-(round, company) RoundRecords (env terms: a=noisy, b=accurate)."""
    G = len(games)
    M, T = games[0].M, games[0].T
    sys = P.SYSTEM + _format_instruction(M)
    self_hist = [np.full((T, M), np.nan) for _ in range(G)]
    err_a = [0.0] * G
    err_b = [0.0] * G
    n_seen = [0] * G
    records = [[] for _ in range(G)]

    for t in range(T):
        prompts = []
        for gi, game in enumerate(games):
            running = (err_a[gi] / n_seen[gi], err_b[gi] / n_seen[gi]) if n_seen[gi] else None
            user = P.build_prompt(game, t, style, self_hist=self_hist[gi], running_abs_err=running)
            prompts.append(_chat(tok, sys, user, cfg.enable_thinking))
        texts = _generate_batch(model, tok, prompts, cfg)
        for gi, game in enumerate(games):
            est = parse_text(texts[gi], M)
            self_hist[gi][t] = est
            for i in range(M):
                records[gi].append(RoundRecord(t=t, company=i, a=float(game.a[t, i]),
                                               b=float(game.b[t, i]), theta=float(game.theta[t, i]),
                                               model_est=float(est[i])))
            err_a[gi] += float(np.sum(np.abs(game.a[t] - game.theta[t])))
            err_b[gi] += float(np.sum(np.abs(game.b[t] - game.theta[t])))
            n_seen[gi] += M
    return records


@torch.no_grad()
def comprehension_probe(game, style: P.PromptStyle, cfg: LlamaConfig, model, tok) -> dict:
    user = P.build_prompt(game, 0, style) + (
        "\n\nBefore forecasting, briefly state: which source is described as the "
        "long-trusted one, what is revealed after each round, and whether the companies "
        "are new each round.")
    text = _generate_batch(model, tok, [_chat(tok, P.SYSTEM, user, cfg.enable_thinking)],
                           LlamaConfig(**{**cfg.__dict__, "max_new_tokens": 200}))[0]
    low = text.lower()
    return {"reputable_source": text,
            "fresh_items_each_round": ("new" in low or "fresh" in low),
            "what_is_revealed": text}
