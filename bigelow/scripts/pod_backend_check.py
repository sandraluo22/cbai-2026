"""Incremental live-backend validation on the GPU pod.

Loads the configured model once and checks: chat template (no thinking
blocks), seeded generation determinism, batched-generation common random
numbers (unsteered rows reproduce baseline under per-row steering),
sequence scoring consistency between single and batched paths, steering
sign response, and selected-layer activation capture.
"""

from __future__ import annotations

import sys

import numpy as np

from belief_feedback.agents.prompts import PROBE_CHOICES, probe_messages
from belief_feedback.config import load_config
from belief_feedback.models import make_backend
from belief_feedback.models.steering import SteeringSpec

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/pilot.yaml")
be = make_backend(cfg)
print(f"loaded {be.model_id}: {be.n_layers} layers, hidden {be.hidden_size}")

ctx = [
    {"role": "system", "content": "You are a terse quality engineer on an incident review team."},
    {
        "role": "user",
        "content": (
            "Report R-check-001 states that the flow meter at station ST-44 under-registers "
            "by 3 percent. In one sentence, what does that imply for fill volume?"
        ),
    },
]

r = be._render(ctx)
# Qwen3 non-thinking mode pre-fills an EMPTY think block in the generation
# prompt; that is expected. A non-empty think block would be a real leak.
if "<think>" in r:
    assert "<think>\n\n</think>" in r, "non-empty thinking block in rendered prompt"

g1 = be.generate(ctx, seed=123)
g2 = be.generate(ctx, seed=123)
g3 = be.generate(ctx, seed=124)
print("gen sample:", g1.text[:160].replace("\n", " | "))
assert "<think>" not in g1.text and "</think>" not in g1.text, "thinking block in generation"
assert g1.text == g2.text, "same-seed generation not deterministic"
print("seeded determinism ok; different seed differs:", g1.text != g3.text)

# --- batched CRN ------------------------------------------------------------
ctxs = [ctx, ctx[:1] + [{"role": "user", "content": "Summarize the purpose of a calibration check in one sentence."}], ctx]
seeds = [11, 22, 33]
b1 = be.generate_batch(ctxs, seeds)
b2 = be.generate_batch(ctxs, seeds)
assert [x.text for x in b1] == [x.text for x in b2], "batched generation not deterministic"

layer = be.n_layers // 2
vec = np.zeros(be.hidden_size, dtype=np.float64)
acts0 = be.get_activations(ctx)
resid_norm = float(np.linalg.norm(acts0[layer]))
vec = acts0[layer] / (resid_norm + 1e-9) * (0.25 * resid_norm)  # direction at 25% of residual scale
print(f"residual norm at layer {layer}: {resid_norm:.1f}")
spec = SteeringSpec(vector=vec, layer=layer, magnitude=2.0)
b3 = be.generate_batch(ctxs, seeds, steerings=[spec, None, None])
same_rows = [b3[i].text == b1[i].text for i in range(3)]
print("batched CRN with row-0 steering: row0 changed:", not same_rows[0],
      "| rows 1,2 unchanged:", same_rows[1] and same_rows[2])
assert same_rows[1] and same_rows[2], "unsteered rows failed to reproduce baseline (CRN broken)"
assert not same_rows[0], "strong steering did not change the steered row"

# --- scoring: single vs batched --------------------------------------------
score1 = be.score_choices(probe_messages(ctx), PROBE_CHOICES)
scoreb = be.score_choices_batch([probe_messages(ctx), probe_messages(ctxs[1])], PROBE_CHOICES)
diff = abs(score1.logps[0] - scoreb[0].logps[0]) + abs(score1.logps[1] - scoreb[0].logps[1])
print(f"probe logps: {score1.logps} | batched match diff {diff:.4f}")
assert diff < 0.5, "batched scoring diverges from single-path scoring beyond bf16 numerics"

# --- multi-token scoring -----------------------------------------------------
ms = be.score_choices(probe_messages(ctx), [" ALPHA", " the station calibration explanation"])
assert ms.token_counts[1] > 1, "multi-token completion not multi-token?"
print("multi-token scoring ok:", ms.token_counts)

# --- steering moves probe log odds ------------------------------------------
sp = be.score_choices(probe_messages(ctx), PROBE_CHOICES, steering=SteeringSpec(vector=vec, layer=layer, magnitude=4.0))
sm = be.score_choices(probe_messages(ctx), PROBE_CHOICES, steering=SteeringSpec(vector=vec, layer=layer, magnitude=-4.0))
d_plus = sp.logps[0] - sp.logps[1]
d_minus = sm.logps[0] - sm.logps[1]
d_base = score1.logps[0] - score1.logps[1]
print(f"visible log odds: base {d_base:.3f}, +4 {d_plus:.3f}, -4 {d_minus:.3f}")
assert (d_plus != d_base) or (d_minus != d_base), "steering had no effect on scoring"

# --- selected-layer activation batch ----------------------------------------
sel = be.get_selected_activations_batch([ctx, ctxs[1]], layer)
assert sel[0].shape == (be.hidden_size,)
ref = be.get_activations(ctx)[layer]
cos = float(sel[0] @ ref / (np.linalg.norm(sel[0]) * np.linalg.norm(ref) + 1e-9))
print(f"selected-layer capture vs full hidden-states cosine: {cos:.4f}")
assert cos > 0.99, "activation capture mismatch"

print("ALL BACKEND CHECKS PASSED")
