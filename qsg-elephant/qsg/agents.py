"""Agent interface + shared model backend (text and vision).

Design rule: the simulation engine, logging, activation capture and analysis are
completely arm-agnostic.  The ONLY thing that differs between the text and image
arms is how an agent's private partial observation is turned into prompt content
(text clues vs. an image crop).  Both arms expose the identical :class:`Agent`
interface and share ONE in-memory model (weights loaded once, reused for all N
agents by swapping prompt content).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .config import Arm, CouplingMode, RunConfig


class ImageArmUnavailable(RuntimeError):
    """Raised when no vision-language model can be loaded; the image arm is skipped."""


# --------------------------------------------------------------------------- #
# Candidate vocabulary handling                                              #
# --------------------------------------------------------------------------- #
@dataclass
class CandidateVocab:
    names: list[str]
    first_token_ids: list[int]            # leading-space variant, for first-token scoring
    token_id_lists: list[list[int]]       # full continuation tokens, for length-norm

    @property
    def K(self) -> int:
        return len(self.names)

    def candidate_list_str(self) -> str:
        return ", ".join(self.names)


def build_candidate_vocab(tokenizer, names: list[str]) -> CandidateVocab:
    first_ids, tok_lists = [], []
    for name in names:
        # Leading space matches how a model continues "... Answer: <name>".
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(name, add_special_tokens=False)
        first_ids.append(ids[0])
        tok_lists.append(ids)
    return CandidateVocab(names=names, first_token_ids=first_ids, token_id_lists=tok_lists)


# --------------------------------------------------------------------------- #
# Model backend                                                              #
# --------------------------------------------------------------------------- #
class ModelBackend:
    """Wraps one HF model + tokenizer/processor. Loaded once, reused for all agents.

    The soft-readout anchor is deterministic: we raw-tokenize the formatted prompt
    string (no chat-generation-prompt) so the LAST token IS the anchor for every
    architecture identically (see activations.py invariant).
    """

    def __init__(self, cfg: RunConfig, candidate_names: list[str]):
        self.cfg = cfg
        self._candidate_names = list(candidate_names)
        self.is_vision = cfg.arm == Arm.IMAGE
        self.processor = None
        self._load()

    # -- loading ---------------------------------------------------------- #
    def _torch_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(self.cfg.model.dtype, torch.float32)

    def _device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self._device()
        dtype = self._torch_dtype() if self.device != "cpu" else torch.float32

        if self.is_vision:
            self._load_vision(dtype)
            return

        name = self.cfg.model.smoke_text_model if self.cfg.smoke_test else self.cfg.model.text_model
        self.model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kw = {"torch_dtype": dtype}
        if self.device == "cuda" and self.cfg.model.device_map:
            kw["device_map"] = self.cfg.model.device_map
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        if "device_map" not in kw:
            self.model.to(self.device)
        self.model.eval()
        self.vocab = build_candidate_vocab(self.tokenizer, self._candidate_names)

    def _load_vision(self, dtype) -> None:
        """Load a VLM; degrade gracefully (raise ImageArmUnavailable) if none load."""
        from transformers import AutoProcessor

        last_err = None
        for name in (self.cfg.model.vision_model, self.cfg.model.vision_model_fallback):
            try:
                self.model_name = name
                self.processor = AutoProcessor.from_pretrained(name)
                self.tokenizer = self.processor.tokenizer
                model = self._instantiate_vlm(name, dtype)
                model.eval()
                self.model = model
                if self.device == "cuda":
                    pass  # device_map handled in _instantiate_vlm
                else:
                    self.model.to(self.device)
                self.vocab = build_candidate_vocab(self.tokenizer, self._candidate_names)
                return
            except Exception as e:  # noqa: BLE001 - intentional graceful degrade
                last_err = e
                continue
        raise ImageArmUnavailable(
            f"No vision-language model could be loaded (tried "
            f"{self.cfg.model.vision_model}, {self.cfg.model.vision_model_fallback}): {last_err}"
        )

    def _instantiate_vlm(self, name: str, dtype):
        kw = {"torch_dtype": dtype}
        if self.device == "cuda" and self.cfg.model.device_map:
            kw["device_map"] = self.cfg.model.device_map
        if "Qwen2-VL" in name:
            from transformers import Qwen2VLForConditionalGeneration
            return Qwen2VLForConditionalGeneration.from_pretrained(name, **kw)
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(name, **kw)

    # -- forward passes --------------------------------------------------- #
    @torch.no_grad()
    def soft_forward(self, prompt: str, images: Optional[list] = None, hooks=None):
        """Forward the anchor prompt; capture activations; return (logits_last, anchor_id).

        ``hooks`` is an ActivationHooks instance (or None). Activations are captured
        at the last token = the anchor position.
        """
        if self.is_vision and images is not None:
            inputs = self._vision_inputs(prompt, images)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in enc.items()}
        anchor_id = int(inputs["input_ids"][0, -1].item())

        if hooks is not None:
            with hooks.capture():
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        logits_last = out.logits[0, -1, :]
        return logits_last, anchor_id

    def _vision_inputs(self, prompt: str, images: list):
        messages = [{"role": "user", "content":
                     [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # Append the deterministic anchor so the last token is "Answer:" colon.
        enc = self.processor(text=[text], images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def candidate_length_norm_logprobs(self, prompt: str, images: Optional[list] = None) -> np.ndarray:
        """Length-normalized continuation logprob per candidate (sum logp / n_tokens)."""
        scores = np.empty(self.vocab.K)
        if self.is_vision and images is not None:
            base = self._vision_inputs(prompt, images)
            base_ids = base["input_ids"]
            extra = {k: v for k, v in base.items() if k != "input_ids" and k != "attention_mask"}
        else:
            base_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
            extra = {}
        for k, cand_ids in enumerate(self.vocab.token_id_lists):
            cand = torch.tensor([cand_ids], device=self.device)
            full = torch.cat([base_ids, cand], dim=1)
            out = self.model(input_ids=full, **extra)
            logits = out.logits[0]
            logp = torch.log_softmax(logits, dim=-1)
            total = 0.0
            start = base_ids.shape[1]
            for j, tid in enumerate(cand_ids):
                total += logp[start - 1 + j, tid].item()
            scores[k] = total / max(len(cand_ids), 1)
        return scores

    @torch.no_grad()
    def generate(self, prompt: str, images: Optional[list] = None, max_new_tokens: int = 128) -> str:
        if self.is_vision and images is not None:
            inputs = self._vision_inputs(prompt, images)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Agent:
    """One QSG agent: private observation + public simplex belief.

    The private observation (text clues or image crop) is set at construction and
    NEVER shared in two-layer mode.  ``belief`` is the public simplex x_i.
    """

    agent_id: int
    private_text: str                     # text clues (text arm) or caption (image arm)
    images: Optional[list] = None         # image crop(s) for the image arm
    belief: Optional[np.ndarray] = None   # public simplex x_i  (K,)

    # last received QSG message (for two-layer re-reasoning prompt)
    last_message_text: str = ""

    def soft_prompt(self, cfg: RunConfig, vocab: CandidateVocab) -> str:
        private = self._private_block(cfg)
        return cfg.readout.anchor_template.format(
            private=private, candidate_list=vocab.candidate_list_str()
        )

    def hard_prompt(self, cfg: RunConfig, vocab: CandidateVocab) -> str:
        private = self._private_block(cfg)
        return cfg.readout.hard_template.format(
            private=private, candidate_list=vocab.candidate_list_str()
        )

    def _own_observation(self, cfg: RunConfig) -> str:
        """The agent's OWN private percept only (no heard messages)."""
        if cfg.neutral_ablation:
            return "You have no specific information about the object."
        if self.images is not None:
            return "You can see a small patch of the object (shown)."
        return (
            "You can sense only a few generic properties of a hidden object. "
            f"It feels: {self.private_text}."
        )

    def _private_block(self, cfg: RunConfig) -> str:
        base = self._own_observation(cfg)
        if not self.last_message_text:
            return base
        if cfg.coupling_mode == CouplingMode.REASONING_EXCHANGE:
            # listener reads the speaker's FULL reasoning, not a label
            return (
                base
                + "\n\nAnother blind observer, sensing a different property, reasoned:\n"
                + f"\"{self.last_message_text}\""
            )
        return base + f"\n\nAnother observer suggests: {self.last_message_text}"

    def reasoning_prompt(self, cfg: RunConfig, vocab: "CandidateVocab",
                         commitment: Optional[str] = None) -> str:
        lean = f"From gossip so far, {commitment}. " if commitment else ""
        return cfg.readout.reasoning_template.format(
            own=self._own_observation(cfg), candidate_list=vocab.candidate_list_str(),
            lean=lean,
        )


def message_to_text(y: np.ndarray, vocab: CandidateVocab, top: int = 2) -> str:
    """Render a QSG message vector as a short natural-language hint for re-reasoning."""
    order = np.argsort(y)[::-1]
    parts = [f"{vocab.names[i]} ({y[i]:.0%})" for i in order[:top] if y[i] > 0]
    return "it is most likely " + ", or ".join(parts) if parts else ""


def qsg_commitment_text(y: np.ndarray, vocab: CandidateVocab, m: float) -> str:
    """Turn a QSG-sampled message ``y`` into the commitment a speaker will ARTICULATE.

    The bandwidth ``m`` still governs what the speaker commits to (Hard = a single
    sampled label, Top-m = a few, Soft = its leading beliefs) — the speaker then
    talks about that commitment in natural language. This is what keeps the
    "models talking" dynamic inside the QSG structure.
    """
    order = [i for i in np.argsort(y)[::-1] if y[i] > 0]
    if not order:
        return "you are unsure what it is"
    if m == 1:
        return f"others suggest the object is a {vocab.names[order[0]]}"
    names = [vocab.names[i] for i in order[:3]]
    return "others are leaning toward " + " or ".join(names)


# --------------------------------------------------------------------------- #
# Observation assignment                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ObservationPlan:
    object_name: str
    candidate_names: list[str]
    ground_truth_index: int
    agent_features: list[list[str]] = field(default_factory=list)
