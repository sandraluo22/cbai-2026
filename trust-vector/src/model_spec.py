"""Self-contained model loader, so this directory does not depend on its old parent.

Vendored from reciprocal-signaling/src/run_games.py when trust-vector moved out to
its own top-level folder. Same tags, same behaviour.
"""
from __future__ import annotations

import os

import torch

SPEC = {
    # tag: (hf primary, mirror, thinking, eightbit)
    "Qwen7":   ("Qwen/Qwen2.5-7B-Instruct", None, False, False),
    "Qwen32":  ("Qwen/Qwen3-32B", None, False, False),
    "Qwen72":  ("Qwen/Qwen2.5-72B-Instruct", None, False, True),
    "Llama70": ("meta-llama/Llama-3.1-70B-Instruct",
                "NousResearch/Meta-Llama-3.1-70B-Instruct", False, True),
}


def load(tag=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag = tag or os.environ.get("MODEL", "Qwen32")
    hf, mirror, thinking, eightbit = SPEC[tag]
    for name in (hf, mirror):
        if name is None:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if eightbit:
                from transformers import BitsAndBytesConfig
                model = AutoModelForCausalLM.from_pretrained(
                    name, device_map="auto",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    name, dtype=torch.bfloat16, device_map="auto").eval()
            return model, tok, thinking
        except Exception as e:  # pragma: no cover
            print(f"[load] {tag}: {name} failed ({type(e).__name__}: {e})", flush=True)
    raise RuntimeError(f"could not load {tag}")
