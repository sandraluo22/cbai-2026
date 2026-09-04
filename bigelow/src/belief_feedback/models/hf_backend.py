"""Hugging Face backend for decoder-only chat models (Llama/Qwen/Gemma).

Supports chat-template rendering, batched sequence scoring, batched memo
generation, hidden-state collection, residual-stream steering hooks at a
selected block, bfloat16 or 4-bit loading, and automatic safe batch-size
reduction after CUDA OOM (inference batch size only; sample sizes are never
modified).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from .base import Backend
from .generation import GenerationResult, Message, ScoreResult, prompt_hash
from .scoring import completion_logprob
from .steering import SteeringHookState, SteeringSpec, make_layer_hook

log = get_logger(__name__)


class HuggingFaceBackend(Backend):
    def __init__(self, cfg: Config) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.model_id = cfg.model.model_id
        self._torch = torch
        self.batch_size = cfg.model.batch_size

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            cfg.model.dtype
        ]
        import transformers

        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kwargs: dict[str, Any] = {
            dtype_key: dtype,
            "device_map": cfg.model.device,
            "revision": cfg.model.revision,
        }
        if cfg.model.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        log.info("loading %s (%s, 4bit=%s)", self.model_id, cfg.model.dtype, cfg.model.load_in_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.resolved_tokenizer_id, revision=cfg.model.revision
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.hidden_size = int(self.model.config.hidden_size)
        self._layers = self._decoder_layers()
        self._log_memory("after-load")

    # ------------------------------------------------------------------
    def _decoder_layers(self):
        m = self.model
        for path in ("model.layers", "transformer.h", "model.model.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("could not locate decoder layers on model")

    def _log_memory(self, tag: str) -> None:
        torch = self._torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            log.info("[%s] cuda memory allocated: %.1f GiB", tag, alloc)

    def _peak_memory(self) -> float:
        torch = self._torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated())
        return 0.0

    def _render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        # Qwen3-style templates accept enable_thinking; the spec forbids
        # hidden chain-of-thought, so thinking mode is always disabled.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def _with_hook(
        self,
        steering: SteeringSpec | None,
        prompt_lengths: list[int],
        row_specs: list[SteeringSpec | None] | None = None,
    ):
        """Context manager installing the steering hook at the selected block."""
        active_rows = [s for s in (row_specs or []) if s is not None and s.active]

        class _Ctx:
            def __init__(ctx) -> None:
                ctx.handle = None

            def __enter__(ctx):
                spec = steering
                if row_specs is not None and active_rows:
                    spec = active_rows[0]
                if spec is not None and (spec.active or active_rows):
                    state = SteeringHookState(
                        spec=spec, prompt_lengths=prompt_lengths, row_specs=row_specs
                    )
                    ctx.handle = self._layers[spec.layer].register_forward_hook(
                        make_layer_hook(state)
                    )
                return ctx

            def __exit__(ctx, *exc):
                if ctx.handle is not None:
                    ctx.handle.remove()
                return False

        return _Ctx()

    def _oom_safe(self, fn, batch: list, tag: str) -> list:
        """Run fn on batch, halving inference batch size on CUDA OOM."""
        torch = self._torch
        bs = min(self.batch_size, len(batch))
        while True:
            try:
                out = []
                for i in range(0, len(batch), bs):
                    out.extend(fn(batch[i : i + bs]))
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.batch_size = bs
                log.warning("[%s] CUDA OOM: reducing inference batch size to %d", tag, bs)

    # ------------------------------------------------------------------
    def score_choices(
        self,
        messages: list[Message],
        choices: list[str],
        steering: SteeringSpec | None = None,
    ) -> ScoreResult:
        torch = self._torch
        prompt = self._render(messages, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids

        def run(batch: list[str]) -> list[tuple[float, int]]:
            texts = [prompt + c for c in batch]
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            lengths = enc.attention_mask.sum(dim=1).tolist()
            with self._with_hook(steering, [len(prompt_ids)] * len(batch)):
                with torch.no_grad():
                    logits = self.model(**enc).logits
            out = []
            for i in range(len(batch)):
                out.append(
                    completion_logprob(
                        logits[i], enc.input_ids[i], len(prompt_ids), int(lengths[i])
                    )
                )
            return out

        results = self._oom_safe(run, choices, "score")
        logps = [lp for lp, _ in results]
        counts = [n for _, n in results]
        return ScoreResult(
            logps=logps,
            logps_normalized=[lp / max(n, 1) for lp, n in results],
            token_counts=counts,
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        seed: int,
        steering: SteeringSpec | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResult:
        torch = self._torch
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        temperature = self.cfg.model.temperature if temperature is None else temperature
        top_p = self.cfg.model.top_p if top_p is None else top_p
        max_new_tokens = max_new_tokens or self.cfg.model.max_new_tokens

        prompt = self._render(messages, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        n_ctx = int(enc.input_ids.shape[1])
        torch.manual_seed(seed)
        do_sample = temperature > 0
        with self._with_hook(steering, [n_ctx]):
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        gen_ids = out[0, n_ctx:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            seed=seed,
            prompt_hash=prompt_hash(messages),
            context_token_count=n_ctx,
            generation_token_count=int(gen_ids.shape[0]),
            wall_time=time.time() - t0,
            peak_gpu_memory=self._peak_memory(),
        )

    # ------------------------------------------------------------------
    def generate_batch(
        self,
        messages_list: list[list[Message]],
        seeds: list[int],
        steerings: list[SteeringSpec | None] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """Batched sampling with per-row steering.

        Rows are left-padded and sampled under one global seed derived from
        the per-row seeds; each row's sampling stream depends only on its
        own logits and row index, so with a fixed row order and identical
        contexts a row reproduces its baseline output exactly (common
        random numbers across paired branches; verified by the branch
        tests on the live backend).
        """
        torch = self._torch
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        temperature = self.cfg.model.temperature if temperature is None else temperature
        top_p = self.cfg.model.top_p if top_p is None else top_p
        max_new_tokens = max_new_tokens or self.cfg.model.max_new_tokens
        steerings = steerings or [None] * len(messages_list)
        prompts = [self._render(m) for m in messages_list]

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
        finally:
            self.tokenizer.padding_side = old_side
        seq_len = int(enc.input_ids.shape[1])
        ctx_lens = enc.attention_mask.sum(dim=1).tolist()
        global_seed = sum(seeds) % (2**31 - 1)
        torch.manual_seed(global_seed)
        do_sample = temperature > 0
        # left padding: the final prompt token of every row sits at seq-1
        with self._with_hook(None, [seq_len] * len(prompts), row_specs=list(steerings)):
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        wall = time.time() - t0
        peak = self._peak_memory()
        results = []
        for i, messages in enumerate(messages_list):
            gen_ids = out[i, seq_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    seed=seeds[i],
                    prompt_hash=prompt_hash(messages),
                    context_token_count=int(ctx_lens[i]),
                    generation_token_count=int((gen_ids != self.tokenizer.pad_token_id).sum()),
                    wall_time=wall / len(messages_list),
                    peak_gpu_memory=peak,
                )
            )
        return results

    def score_choices_batch(
        self,
        messages_list: list[list[Message]],
        choices: list[str],
        steerings: list[SteeringSpec | None] | None = None,
    ) -> list[ScoreResult]:
        """Batched scoring: rows are (context x choice) pairs, right-padded."""
        torch = self._torch
        steerings = steerings or [None] * len(messages_list)
        prompts = [self._render(m) for m in messages_list]
        prompt_ids = [self.tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
        rows = []  # (ctx_idx, choice_idx, text, prompt_len, spec)
        for ci, p in enumerate(prompts):
            for chi, choice in enumerate(choices):
                rows.append((ci, chi, p + choice, len(prompt_ids[ci]), steerings[ci]))

        def run(batch):
            enc = self.tokenizer(
                [r[2] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            lengths = enc.attention_mask.sum(dim=1).tolist()
            with self._with_hook(
                None, [r[3] for r in batch], row_specs=[r[4] for r in batch]
            ):
                with torch.no_grad():
                    logits = self.model(**enc).logits
            out = []
            for k, r in enumerate(batch):
                out.append(completion_logprob(logits[k], enc.input_ids[k], r[3], int(lengths[k])))
            return out

        scored = self._oom_safe(run, rows, "score_batch")
        results = []
        for ci in range(len(messages_list)):
            logps, counts = [], []
            for chi in range(len(choices)):
                lp, n = scored[ci * len(choices) + chi]
                logps.append(lp)
                counts.append(n)
            results.append(
                ScoreResult(
                    logps=logps,
                    logps_normalized=[lp / max(n, 1) for lp, n in zip(logps, counts)],
                    token_counts=counts,
                )
            )
        return results

    def get_selected_activations_batch(
        self,
        messages_list: list[list[Message]],
        layer: int,
        steerings: list[SteeringSpec | None] | None = None,
    ) -> list[np.ndarray]:
        """Capture one layer's final-prompt-token activation for many contexts."""
        torch = self._torch
        steerings = steerings or [None] * len(messages_list)
        prompts = [self._render(m) for m in messages_list]
        captured: dict[str, Any] = {}

        def capture_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden.detach()
            return output

        rows = list(zip(prompts, steerings))

        def run(batch):
            enc = self.tokenizer(
                [b[0] for b in batch], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            lengths = enc.attention_mask.sum(dim=1).tolist()
            # steering hook first, capture second: registration order is
            # execution order, so the captured state includes the steering edit
            with self._with_hook(
                None, [int(x) for x in lengths], row_specs=[b[1] for b in batch]
            ):
                handle = self._layers[layer].register_forward_hook(capture_hook)
                try:
                    with torch.no_grad():
                        self.model(**enc)
                finally:
                    handle.remove()
            h = captured.pop("h")
            return [
                h[k, int(lengths[k]) - 1].float().cpu().numpy() for k in range(len(batch))
            ]

        return self._oom_safe(run, rows, "act_batch")

    # ------------------------------------------------------------------
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int] | None = None,
        steering: SteeringSpec | None = None,
    ) -> np.ndarray:
        torch = self._torch
        prompt = self._render(messages, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        n_ctx = int(enc.input_ids.shape[1])
        with self._with_hook(steering, [n_ctx]):
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
        # hidden_states: tuple of n_layers+1 tensors [1, seq, dim]; drop embeddings
        stack = torch.stack([h[0, -1] for h in out.hidden_states[1:]], dim=0)
        acts = stack.float().cpu().numpy()
        if layers is not None:
            acts = acts[layers]
        return acts
