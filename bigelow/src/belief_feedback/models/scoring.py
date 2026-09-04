"""Sequence-scoring helpers (framework-level, testable on CPU)."""

from __future__ import annotations


def completion_logprob(logits, input_ids, completion_start: int, completion_end: int) -> tuple[float, int]:
    """Summed log probability of tokens [completion_start, completion_end).

    ``logits``: [seq, vocab] for one sequence; token at position i is
    predicted by logits[i-1]. Works for completions of any token length.
    """
    import torch

    logprobs = torch.log_softmax(logits[completion_start - 1 : completion_end - 1].float(), dim=-1)
    targets = input_ids[completion_start:completion_end]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(picked.sum().item()), int(completion_end - completion_start)
