"""Steering interventions: layer/token scope and exact reversibility (Part 24: 9, 10)."""

from __future__ import annotations

import numpy as np
import pytest

from belief_feedback.agents.prompts import PROBE_CHOICES
from belief_feedback.models.steering import (
    SteeringHookState,
    SteeringSpec,
    apply_to_hidden_numpy,
    make_layer_hook,
)

torch = pytest.importorskip("torch")


def _spec(magnitude=1.0, scope="final_token_and_generation", mode="add", value=0.0):
    v = np.zeros(4)
    v[0] = 1.0
    return SteeringSpec(vector=v, layer=1, magnitude=magnitude, scope=scope, mode=mode, project_value=value)


def test_prefill_edits_only_final_prompt_token():
    state = SteeringHookState(spec=_spec(magnitude=2.0), prompt_lengths=[3])
    hook = make_layer_hook(state)
    hidden = torch.zeros(1, 5, 4)
    out = hook(None, None, hidden.clone())
    assert torch.allclose(out[0, 2], torch.tensor([2.0, 0, 0, 0]))
    mask = torch.ones(5, dtype=torch.bool)
    mask[2] = False
    assert torch.all(out[0, mask] == 0)


def test_generation_steps_are_edited():
    state = SteeringHookState(spec=_spec(magnitude=1.5), prompt_lengths=[3])
    hook = make_layer_hook(state)
    hook(None, None, torch.zeros(1, 5, 4))  # prefill
    step = hook(None, None, torch.zeros(1, 1, 4))  # generated token
    assert torch.allclose(step[0, 0], torch.tensor([1.5, 0, 0, 0]))


def test_all_token_scope():
    state = SteeringHookState(spec=_spec(magnitude=1.0, scope="all_tokens"), prompt_lengths=[3])
    hook = make_layer_hook(state)
    out = hook(None, None, torch.zeros(1, 5, 4))
    assert torch.all(out[:, :, 0] == 1.0)


def test_project_set_replaces_only_direction_component():
    h = np.array([3.0, 2.0, -1.0, 0.5])
    out = apply_to_hidden_numpy(h, _spec(mode="project_set", value=-4.0))
    assert out[0] == pytest.approx(-4.0)
    assert np.allclose(out[1:], h[1:])


def test_positive_negative_signs(cfg, backend, world, steer_ctx):
    backend.register_world(world)
    docs = "\n\n".join(world.report(r).text for r in world.assignments[0])
    msgs = [{"role": "user", "content": docs + "\n\nAnswer:"}]
    base = backend.score_choices(msgs, PROBE_CHOICES).logps
    plus = backend.score_choices(msgs, PROBE_CHOICES, steering=steer_ctx.spec(+1.0)).logps
    minus = backend.score_choices(msgs, PROBE_CHOICES, steering=steer_ctx.spec(-1.0)).logps
    ell = world.visible_to_semantic(base[0] - base[1])
    ell_p = world.visible_to_semantic(plus[0] - plus[1])
    ell_m = world.visible_to_semantic(minus[0] - minus[1])
    assert ell_p > ell > ell_m  # semantic monotonicity in the steering sign


def test_deactivated_hook_restores_baseline_exactly(cfg, backend, world, steer_ctx):
    """Zero-magnitude steering equals no steering under deterministic decoding."""
    backend.register_world(world)
    docs = "\n\n".join(world.report(r).text for r in world.assignments[0])
    msgs = [{"role": "user", "content": docs}]
    g0 = backend.generate(msgs, seed=7, temperature=0.0)
    g_off = backend.generate(msgs, seed=7, steering=steer_ctx.spec(0.0), temperature=0.0)
    g_on = backend.generate(msgs, seed=7, steering=steer_ctx.spec(3.0), temperature=0.0)
    assert g0.text == g_off.text
    assert g_on.text != g0.text or backend.belief_semantic(msgs, steer_ctx.spec(3.0)) != (
        backend.belief_semantic(msgs)
    )


def test_mock_activation_only_selected_layer_shifted(cfg, backend, world, steer_ctx):
    backend.register_world(world)
    docs = "\n\n".join(world.report(r).text for r in world.assignments[0])
    msgs = [{"role": "user", "content": docs}]
    a0 = backend.get_activations(msgs)
    a1 = backend.get_activations(msgs, steering=steer_ctx.spec(1.0))
    # the raw additive component appears only at the hooked layer; other
    # layers move only through the belief readout in component 0
    for line in range(backend.n_layers):
        if line == steer_ctx.layer:
            continue
        assert np.allclose(a0[line, 1:], a1[line, 1:])
