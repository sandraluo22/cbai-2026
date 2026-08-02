"""Per-round LLM runner (Anthropic API; default claude-opus-4-8).

Plays ONE game for a given PromptStyle: each round it reconstructs the context
(persistent reputation header + raw advisor recap + this round's estimates), asks the
model for a numeric estimate per company via structured output, then reveals the truth
and proceeds. One API call per round; T calls per game.

Opus 4.8 API notes (per the claude-api reference): adaptive thinking only (no
budget_tokens), no temperature/top_p, no assistant prefill. We use structured outputs
(output_config.format) for a clean numeric readout and disable thinking by default for
cost/reproducibility across many calls (configurable to adaptive).

Logged variables are always in ENV terms — a = noisy source, b = accurate source — so
the analysis is invariant to the display-label swap used in the robustness check.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

import prompt as P

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    model_name: str = "claude-opus-4-8"
    max_tokens: int = 1024
    thinking: str = "disabled"        # "disabled" | "adaptive"
    effort: str | None = None          # only used when thinking == "adaptive"
    cache: bool = False                # prompt-cache the stable prefix. OFF by default:
                                       # measured net-negative for these short prompts
                                       # (below Sonnet 4.6's 2048-token cacheable floor).
                                       # Worth enabling only for long prompts (big T/M).


@dataclass
class RoundRecord:
    t: int                             # 0-indexed round
    company: int                       # 0-indexed company
    a: float                           # noisy source estimate (env a)
    b: float                           # accurate source estimate (env b)
    theta: float                       # revealed truth
    model_est: float                   # model's estimate


def _client():
    if anthropic is None:
        raise RuntimeError("anthropic SDK not installed")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic()


# Optional accumulator: when set to a dict, _call adds up token usage so a run/calibration
# can report cache effectiveness (cache_read vs uncached input). Off (None) by default.
USAGE: dict | None = None


def _accumulate(resp):
    if USAGE is None:
        return
    u = resp.usage
    USAGE["input"] = USAGE.get("input", 0) + (u.input_tokens or 0)
    USAGE["cache_read"] = USAGE.get("cache_read", 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
    USAGE["cache_write"] = USAGE.get("cache_write", 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    USAGE["output"] = USAGE.get("output", 0) + (u.output_tokens or 0)
    USAGE["calls"] = USAGE.get("calls", 0) + 1


def _call(client, mc: ModelConfig, system: str, prefix: str, suffix: str, schema: dict) -> dict:
    # Optionally cache the stable, append-only prefix (system + reputation header + recap);
    # the volatile current-round block stays uncached after the breakpoint. Gated by
    # mc.cache because at these prompt sizes the cache writes (1.25x) outweigh the reads.
    if mc.cache:
        prefix_block = {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}
        system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        prefix_block = {"type": "text", "text": prefix}
        system_param = system
    content = [prefix_block] + ([{"type": "text", "text": suffix}] if suffix else [])
    kwargs = dict(model=mc.model_name, max_tokens=mc.max_tokens, system=system_param,
                  messages=[{"role": "user", "content": content}],
                  output_config={"format": schema})
    if mc.thinking == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
        if mc.effort:
            kwargs["output_config"]["effort"] = mc.effort
    else:
        kwargs["thinking"] = {"type": "disabled"}
    # Resilient: SDK retries (429/500/529) internally; we add an outer retry with
    # backoff and CATCH persistent API errors so one bad call NaNs a single round
    # instead of killing the whole sweep. Also retries transient empty/non-JSON bodies.
    for attempt in range(4):
        try:
            resp = client.with_options(max_retries=8).messages.create(**kwargs)
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
        if resp.stop_reason == "refusal":
            return {}
        _accumulate(resp)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        obj = _extract_json(text)
        if obj:
            return obj
        time.sleep(1)
    return {}


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    i, j = text.find("{"), text.rfind("}")   # salvage a JSON object embedded in prose
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            pass
    return {}


# --------------------------------------------------------------------------- #
def play_game(game, style: P.PromptStyle, mc: ModelConfig, client=None) -> list[RoundRecord]:
    """Run one full game. Returns flat per-(round, company) records (env terms)."""
    client = client or _client()
    M, T = game.M, game.T
    schema = P.estimate_schema(M)
    self_hist = np.full((T, M), np.nan)
    abs_err_a = 0.0          # cumulative abs error of NOISY source (display rep letter)
    abs_err_b = 0.0          # cumulative abs error of ACCURATE source (display acc letter)
    n_seen = 0
    records: list[RoundRecord] = []

    for t in range(T):
        running = None
        if n_seen > 0:
            running = (abs_err_a / n_seen, abs_err_b / n_seen)
        prefix, suffix = P.build_prompt_parts(game, t, style, self_hist=self_hist,
                                              running_abs_err=running)
        obj = _call(client, mc, P.SYSTEM, prefix, suffix, schema)
        est = P.parse_estimates(obj, M)
        self_hist[t] = est
        for i in range(M):
            records.append(RoundRecord(t=t, company=i, a=float(game.a[t, i]),
                                        b=float(game.b[t, i]), theta=float(game.theta[t, i]),
                                        model_est=float(est[i])))
        # update running per-source accuracy with THIS round's revealed truth
        abs_err_a += float(np.sum(np.abs(game.a[t] - game.theta[t])))
        abs_err_b += float(np.sum(np.abs(game.b[t] - game.theta[t])))
        n_seen += M
    return records


def comprehension_probe(game, style: P.PromptStyle, mc: ModelConfig, client=None) -> dict:
    """Ask the model to restate the setup (round 0 context). Returns parsed dict."""
    client = client or _client()
    user = P.build_prompt(game, 0, style) + (
        "\n\nBefore forecasting, briefly confirm your understanding of the setup: which "
        "source is described as the long-trusted one, what is revealed to you after each "
        "round, and whether the companies are new each round.")
    return _call(client, mc, P.SYSTEM, user, "", P.COMPREHENSION_SCHEMA)
