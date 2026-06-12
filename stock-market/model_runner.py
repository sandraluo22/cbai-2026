"""Focal model: load Llama-3.1-8B-Instruct, capture all-layer residual-stream
activations at the marker (+ decision span), and run PATCHED forward passes.

Why HF transformers + forward hooks (not nnsight / TransformerLens):
  * It is the lightest dependency and identical to the loader used elsewhere in
    this repo (one HF model, bf16, GPU).
  * Capture is trivial and exact via `output_hidden_states=True`, which returns
    the residual stream after the embedding and after EACH of the 32 blocks
    (tuple length n_layers+1) — precisely "32 blocks + embedding".
  * Patching needs to OVERWRITE activations during the forward pass; a
    `register_forward_hook` on each decoder layer that replaces its output at
    chosen positions does exactly this, with no model surgery.
  * TransformerLens needs a weight conversion for Llama-3.1 and nnsight adds a
    tracing layer; neither buys us anything for this targeted layer/position
    patch, and both are heavier to pin reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class RunnerConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    base: bool = False                  # flag: use the base model instead of -Instruct
    dtype: str = "bfloat16"
    device: str = "cuda"
    use_chat_template: bool = False     # raw prompt keeps the marker the deterministic last token
    marker_char: str = ":"              # capture anchor = last occurrence of this char
    max_new_tokens: int = 8


@dataclass
class RunOutput:
    text: str                           # generated decision text
    activations: np.ndarray             # (n_layers+1, hidden) at the marker position
    span_activations: np.ndarray        # (n_layers+1, span_len, hidden) over the decision line
    marker_pos: int
    decision_span: tuple[int, int]
    seq_len: int
    input_ids: list[int]


class ModelRunner:
    def __init__(self, cfg: RunnerConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=td)
        self.model.to(cfg.device)          # single-GPU placement (8B fits on one H200)
        self.model.eval()
        self.layers = list(self.model.model.layers)        # decoder blocks (for patch hooks)
        self.n_layers = len(self.layers)

    # -- tokenization + marker location ---------------------------------- #
    def _encode(self, prompt: str):
        if self.cfg.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        return prompt, enc, offsets

    def _char_to_token(self, offsets, char_idx: int) -> int:
        """Token index spanning a character position (robust to BPE merges)."""
        cand = [i for i, (a, b) in enumerate(offsets) if a <= char_idx < b and b > a]
        return cand[-1] if cand else len(offsets) - 1

    def locate_marker(self, prompt: str, offsets) -> tuple[int, tuple[int, int]]:
        """Return (marker_pos, decision_span) for the LAST 'Decision <marker>' line."""
        m_char = prompt.rfind(self.cfg.marker_char)
        marker_pos = self._char_to_token(offsets, m_char)
        d_char = prompt.rfind("Decision")
        d_tok = self._char_to_token(offsets, d_char) if d_char >= 0 else marker_pos
        return marker_pos, (d_tok, marker_pos)

    def char_span_to_tokens(self, prompt: str, offsets, substr: str) -> list[int]:
        """Token indices overlapping the LAST occurrence of `substr` (e.g. the social block)."""
        c0 = prompt.rfind(substr)
        if c0 < 0:
            return []
        c1 = c0 + len(substr)
        return [i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0]

    # -- capture --------------------------------------------------------- #
    @torch.no_grad()
    def run(self, prompt: str, *, generate: bool = True) -> RunOutput:
        prompt2, enc, offsets = self._encode(prompt)
        marker_pos, span = self.locate_marker(prompt2, offsets)
        out = self.model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]       # (n_layers+1, seq, hidden)
        acts = hs[:, marker_pos, :].float().cpu().numpy()
        span_acts = hs[:, span[0]:span[1] + 1, :].float().cpu().numpy()
        text = ""
        if generate:
            gen = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return RunOutput(text=text, activations=acts, span_activations=span_acts,
                         marker_pos=marker_pos, decision_span=span,
                         seq_len=enc["input_ids"].shape[1], input_ids=enc["input_ids"][0].tolist())

    @torch.no_grad()
    def capture_positions(self, prompt: str, positions: list[int]) -> np.ndarray:
        """All-layer residual stream at arbitrary token positions: (n_layers+1, P, hidden)."""
        _, enc, _ = self._encode(prompt)
        out = self.model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]       # (L+1, seq, hidden)
        idx = torch.tensor(positions, device=self.cfg.device, dtype=torch.long)
        return hs[:, idx, :].float().cpu().numpy()

    # -- patched forward pass -------------------------------------------- #
    @torch.no_grad()
    def run_patched(self, prompt: str, patches: list[dict], *, generate: bool = True) -> str:
        """Inject donor activations during the forward pass.

        patches: list of {"layer": int, "positions": [int,...], "values": np.ndarray
                          of shape (len(positions), hidden)}. The hook on that decoder
                          layer overwrites its output residual stream at those positions.
        Returns the generated decision text of the patched run.
        """
        prompt2, enc, offsets = self._encode(prompt)
        handles = []
        by_layer: dict[int, dict] = {}
        for p in patches:
            by_layer[p["layer"]] = {
                "pos": torch.tensor(p["positions"], device=self.cfg.device, dtype=torch.long),
                "val": torch.tensor(np.asarray(p["values"]), device=self.cfg.device)}

        def make_hook(layer_idx):
            spec = by_layer[layer_idx]
            max_pos = int(spec["pos"].max())
            def hook(_m, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                # Only patch on the PREFILL pass (full prompt). During cached decode
                # steps hs holds just the new token (seq_len=1), so the prompt-token
                # positions are out of range — skip; the patch already propagated via
                # the KV cache from prefill.
                if hs.shape[1] <= max_pos:
                    return output
                hs[:, spec["pos"], :] = spec["val"].to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return hook

        try:
            for li in by_layer:
                handles.append(self.layers[li].register_forward_hook(make_hook(li)))
            if generate:
                gen = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                          do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                return self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            self.model(**enc)
            return ""
        finally:
            for h in handles:
                h.remove()

    # -- steered forward pass (activation addition) ---------------------- #
    @torch.no_grad()
    def run_steered(self, prompt: str, steers: list[dict], *, generate: bool = True) -> str:
        """ADD a scaled direction to the residual stream during the forward pass.

        Unlike run_patched (which OVERWRITES with donor activations), this ADDS
        alpha * vector at the chosen layer/positions — i.e. activation steering.
        alpha=1 with vector=(donor-receiver) approximates patching; alpha>1
        over-amplifies the direction beyond the natural data range.

        steers: list of {"layer": int, "positions": [int,...],
                         "vector": np.ndarray (len(positions), hidden), "alpha": float}.
        """
        prompt2, enc, offsets = self._encode(prompt)
        handles = []
        by_layer: dict[int, dict] = {}
        for s in steers:
            by_layer[s["layer"]] = {
                "pos": torch.tensor(s["positions"], device=self.cfg.device, dtype=torch.long),
                "vec": torch.tensor(np.asarray(s["vector"]), device=self.cfg.device)
                       * float(s["alpha"])}

        def make_hook(layer_idx):
            spec = by_layer[layer_idx]
            max_pos = int(spec["pos"].max())
            def hook(_m, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                # Same prefill-only logic as patching: the injected direction
                # propagates to generated tokens through the KV cache.
                if hs.shape[1] <= max_pos:
                    return output
                hs[:, spec["pos"], :] = hs[:, spec["pos"], :] + spec["vec"].to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return hook

        try:
            for li in by_layer:
                handles.append(self.layers[li].register_forward_hook(make_hook(li)))
            if generate:
                gen = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                          do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                return self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            self.model(**enc)
            return ""
        finally:
            for h in handles:
                h.remove()

    # -- logit readout under intervention (sensitive causal measure) ----- #
    @torch.no_grad()
    def forward_logits(self, prompt: str, *, patches: list[dict] | None = None,
                       steers: list[dict] | None = None, position: int | None = None,
                       company_token_ids: list[list[int]] | None = None):
        """Single forward pass (no generation) with optional patch+steer hooks.

        Returns the next-token logits at `position` (default: the LAST prompt
        token — the position that actually produces the decision). If
        company_token_ids is given (per-company candidate token ids), also returns
        a per-company logit (max over that company's candidate tokens). This is the
        graded causal readout the argmax-flip metric was too blunt to capture.
        """
        prompt2, enc, offsets = self._encode(prompt)
        pos = enc["input_ids"].shape[1] - 1 if position is None else position
        handles = []

        def make_hook(specs, additive):
            prepared = []
            for s in specs:
                p = torch.tensor(s["positions"], device=self.cfg.device, dtype=torch.long)
                key = "vector" if additive else "values"
                v = torch.tensor(np.asarray(s[key]), device=self.cfg.device)
                if additive:
                    v = v * float(s.get("alpha", 1.0))
                prepared.append((p, v))
            maxp = max(int(p.max()) for p, _ in prepared)
            def hook(_m, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[1] <= maxp:
                    return output
                for p, v in prepared:
                    if additive:
                        hs[:, p, :] = hs[:, p, :] + v.to(hs.dtype)
                    else:
                        hs[:, p, :] = v.to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return hook

        by_layer: dict[int, tuple[list, bool]] = {}
        for s in (patches or []):
            by_layer.setdefault(s["layer"], ([], []))[0].append(s)
        for s in (steers or []):
            by_layer.setdefault(s["layer"], ([], []))[1].append(s)
        try:
            for li, (ps, st) in by_layer.items():
                if ps:
                    handles.append(self.layers[li].register_forward_hook(make_hook(ps, False)))
                if st:
                    handles.append(self.layers[li].register_forward_hook(make_hook(st, True)))
            logits = self.model(**enc).logits[0, pos].float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        if company_token_ids is None:
            return logits
        comp = np.array([max(logits[t] for t in ids) for ids in company_token_ids])
        return logits, comp

    def company_token_ids(self, n: int) -> list[list[int]]:
        """Candidate first-generated-token ids for each company letter (with and
        without a leading space), so the logit readout is robust to tokenization."""
        from prompt import _company_label
        ids = []
        for i in range(n):
            L = _company_label(i)
            cand = set()
            for s in (L, " " + L):
                e = self.tokenizer.encode(s, add_special_tokens=False)
                if e:
                    cand.add(e[-1])
            ids.append(sorted(cand))
        return ids

    # -- symmetry check (token length of the two evidence blocks) -------- #
    def block_token_lengths(self, prompt: str, private_label="PERSONAL readings (rounds",
                            social_label="EXTERNAL readings (rounds") -> tuple[int, int]:
        """Token length of each evidence block (header through its company rows).
        Anchors on the block header '<LABEL> readings (rounds' to avoid matching the
        framing sentence; the residual is logged (value-dependent tokenization)."""
        prompt2, _, offsets = self._encode(prompt)
        def block_len(label):
            c0 = prompt2.find(label)
            if c0 < 0:
                return -1
            c1 = prompt2.find("\n\n", c0)
            c1 = c1 if c1 > 0 else len(prompt2)
            return len([i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0])
        return block_len(private_label), block_len(social_label)
