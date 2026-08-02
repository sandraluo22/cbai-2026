"""Open-weight (HuggingFace) agents used by the games.

Two things live here:

- `HFChatAgent`  : an instruct model that takes a turn in a multi-agent chat.
  Turn-taking is modelled as a transcript of (speaker, text) lines; from a given
  agent's point of view its own past lines are `assistant` turns and everyone
  else's are `user` turns (prefixed with the speaker name so it can tell who
  said what). System prompt is folded into the first user turn for templates
  that reject a `system` role (e.g. gemma-2).

- `HFBaseLM`    : a base LM with residual-stream capture hooks, used by the
  random-walk ping-pong game. It exposes `next_node_logits` (score the graph's
  node-words as next token) and `residual_at_last` (post-block hidden state at
  the final token, per captured layer).

Models load in bf16 on cuda. Weights are cached under HF_HOME (set to the pod's
/workspace/hf network volume by the launcher).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch


# ---------------------------------------------------------------------------
# shared loading
# ---------------------------------------------------------------------------
def _load(name: str, device: str = "cuda", dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    return model, tok


def _decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("could not locate decoder blocks")


# ---------------------------------------------------------------------------
# chat agent (conversational games)
# ---------------------------------------------------------------------------
class HFChatAgent:
    def __init__(self, tag: str, hf_id: str, no_system_role: bool = False,
                 device: str = "cuda"):
        self.tag = tag
        self.no_system_role = no_system_role
        self.device = device
        self.model, self.tok = _load(hf_id, device)

    def _build_messages(self, system: str, transcript: List[Tuple[str, str]]):
        """transcript is a list of (speaker_tag, text). Map to chat roles from
        THIS agent's perspective."""
        msgs = []
        sys_txt = system
        # first user turn absorbs the system prompt when the template has no system role
        first_user_done = False

        def user_content(speaker, text):
            return f"[{speaker}]: {text}" if speaker else text

        if not self.no_system_role and system:
            msgs.append({"role": "system", "content": system})

        for speaker, text in transcript:
            if speaker == self.tag:
                msgs.append({"role": "assistant", "content": text})
            else:
                content = user_content(speaker, text)
                if self.no_system_role and sys_txt and not first_user_done:
                    content = sys_txt + "\n\n" + content
                    first_user_done = True
                msgs.append({"role": "user", "content": content})

        # ensure the conversation ends on a user turn so the model replies
        if not msgs or msgs[-1]["role"] != "user":
            content = "(your turn)"
            if self.no_system_role and sys_txt and not first_user_done:
                content = sys_txt + "\n\n" + content
            msgs.append({"role": "user", "content": content})

        # Strict-alternation templates (e.g. gemma-2) reject consecutive same-role
        # turns, which happen with >2 agents (two other speakers in a row). Coalesce
        # adjacent same-role messages, and ensure the transcript starts with `user`.
        merged = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(dict(m))
        if merged and merged[0]["role"] == "assistant":
            merged.insert(0, {"role": "user", "content": "(conversation so far)"})
        return merged

    @torch.no_grad()
    def say(self, system: str, transcript: List[Tuple[str, str]],
            max_new_tokens: int = 200, temperature: float = 0.8) -> str:
        msgs = self._build_messages(system, transcript)
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=0.95, pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        return text.strip()

    def free(self):
        del self.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# base LM with residual capture (random-walk ping-pong game)
# ---------------------------------------------------------------------------
class HFBaseLM:
    def __init__(self, tag: str, hf_id: str, capture_layers: Tuple[int, ...],
                 device: str = "cuda"):
        self.tag = tag
        self.device = device
        self.capture_layers = tuple(capture_layers)
        self.model, self.tok = _load(hf_id, device)
        self.blocks = _decoder_blocks(self.model)
        self._grabbed: Dict[int, torch.Tensor] = {}
        self._handles = []
        for L in self.capture_layers:
            self._handles.append(self.blocks[L].register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def hook(_m, _i, out):
            self._grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    @torch.no_grad()
    def forward_text(self, text: str):
        """Run one forward pass; return (logits[V] at last token, {layer: hidden[d]})."""
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        self._grabbed.clear()
        out = self.model(input_ids=ids)
        last_logits = out.logits[0, -1].float().cpu()
        resid = {L: self._grabbed[L][0, -1].float().cpu().numpy() for L in self.capture_layers}
        return last_logits, resid

    @torch.no_grad()
    def generate_word(self, text: str, max_new_tokens: int = 6, temperature: float = 0.0) -> str:
        """FREE continuation: generate the model's own next word given the running
        walk text (NOT constrained to the graph's vocabulary). Returns the first
        word of the generation, lowercased. This is what 'the model adds to the
        walk'; whether it lands on a valid node-word is then measured, not forced."""
        import re
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=0.95, pad_token_id=self.tok.eos_token_id,
        )
        cont = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"[A-Za-z']+", cont)
        return m.group(0).lower() if m else ""

    @torch.no_grad()
    def capture_words(self, words: List[str]):
        """Forward one space-joined word sequence; return per-word residual vectors
        at the word's LAST subword token, for every captured layer.
        Returns {layer: np.ndarray[n_words, d]}. Uses the fast tokenizer offset map."""
        text = " ".join(words)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc["offset_mapping"]
        # char span of each word in the single-space join
        spans, pos = [], 0
        for i, w in enumerate(words):
            if i > 0:
                pos += 1
            spans.append((pos, pos + len(w)))
            pos += len(w)
        last_tok = []
        for (ws, we) in spans:
            toks = [ti for ti, (ts, te) in enumerate(offsets)
                    if not (ts == 0 and te == 0) and ts < we and te > ws]
            last_tok.append(toks[-1])
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        self._grabbed.clear()
        self.model(input_ids=ids)
        out = {}
        for L in self.capture_layers:
            hs = self._grabbed[L][0]                       # [seq, d]
            out[L] = hs[last_tok].float().cpu().numpy()    # [n_words, d]
        return out

    @torch.no_grad()
    def next_node_logprobs(self, prefix_text: str, node_words: List[str]) -> "list":
        """Log-prob the model assigns to each candidate node-word as the continuation
        of `prefix_text` (scored over the word's own tokens, leading space included).
        Returns a python list of per-word summed log-probs."""
        import torch.nn.functional as F
        base_ids = self.tok(prefix_text, return_tensors="pt",
                            add_special_tokens=True)["input_ids"].to(self.device)
        scores = []
        for w in node_words:
            cont = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            cont_t = torch.tensor(cont, device=self.device).unsqueeze(0)
            full = torch.cat([base_ids, cont_t], dim=1)
            logits = self.model(input_ids=full).logits[0]         # [T, V]
            lp = 0.0
            for j, tokid in enumerate(cont):
                pos = base_ids.shape[1] + j - 1                   # predicts token at base+j
                lp += F.log_softmax(logits[pos].float(), dim=-1)[tokid].item()
            scores.append(lp)
        return scores

    def free(self):
        for h in self._handles:
            h.remove()
        del self.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
