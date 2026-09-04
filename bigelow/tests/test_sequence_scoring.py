"""Multi-token candidate sequence scoring (Part 24: 8)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from belief_feedback.models.scoring import completion_logprob  # noqa: E402


def test_completion_logprob_matches_manual():
    torch.manual_seed(0)
    seq, vocab = 8, 11
    logits = torch.randn(seq, vocab)
    input_ids = torch.randint(0, vocab, (seq,))
    start, end = 5, 8  # a three-token completion
    lp, n = completion_logprob(logits, input_ids, start, end)
    assert n == 3
    manual = 0.0
    for pos in range(start, end):
        manual += float(torch.log_softmax(logits[pos - 1].float(), -1)[input_ids[pos]])
    assert math.isclose(lp, manual, rel_tol=1e-5)


def test_single_token_completion():
    torch.manual_seed(1)
    logits = torch.randn(4, 7)
    input_ids = torch.randint(0, 7, (4,))
    lp, n = completion_logprob(logits, input_ids, 3, 4)
    assert n == 1
    assert lp <= 0.0


def test_mock_scoring_consistency(cfg, backend, world):
    backend.register_world(world)
    docs = "\n\n".join(world.report(r).text for r in world.assignments[0])
    msgs = [{"role": "user", "content": docs + "\n\nAnswer:"}]
    res = backend.score_choices(msgs, [" ALPHA", " BETA"])
    # proper log probabilities: exp sums to 1 for the binary choice
    assert math.isclose(math.exp(res.logps[0]) + math.exp(res.logps[1]), 1.0, rel_tol=1e-9)
    assert res.token_counts == [1, 1]
    # multi-token completions are also scored
    res2 = backend.score_choices(msgs, [" the upstream materials explanation is favored"])
    assert res2.token_counts[0] > 1
    assert res2.logps_normalized[0] == pytest.approx(res2.logps[0] / res2.token_counts[0])
