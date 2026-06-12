"""Smoke-test the vision path: one VLM agent reading one elephant patch.

Validates the untested image arm (model load + _vision_inputs + soft/hard readout)
before running any full image experiment.

    HF_HOME=/dev/shm/hf python3 test_vision_single.py
"""
import numpy as np

from qsg import readout
from qsg.agents import Agent, ModelBackend
from qsg.config import load_sweep_config
from qsg.engine import load_ontology, make_image_patches

cfg = load_sweep_config("configs/base.yaml").base
cfg.arm = "image"
cfg.object_name = "elephant"
cfg.qsg.n = 4
cfg.observation.source_image_dir = "assets/elephant_src"
cfg.observation.patch_grid = (4, 4)
cfg.model.vision_model = "Qwen/Qwen2-VL-7B-Instruct"
cfg.activations.capture = False

ont = load_ontology(cfg.ontology_path)
cands = ont["candidates"]
patches = make_image_patches(cfg, np.random.default_rng(0))
assert patches is not None, "no patches produced (check source_image_dir/PIL)"
print(f"made {len(patches)} patches, sizes e.g. {patches[0].size}")

print("loading VLM ...")
backend = ModelBackend(cfg, cands)
print("loaded:", backend.model_name, "on", backend.device)


def top(d, k=3):
    return [(cands[i], round(float(d[i]), 3)) for i in np.argsort(d)[::-1][:k]]


for i in range(2):
    ag = Agent(i, "", images=[patches[i]])
    sr = readout.soft_readout(backend, ag, cfg, hooks=None)
    hr, raw = readout.hard_readout(backend, ag, cfg)
    print(f"\n--- agent {i} (patch {patches[i].size}) ---")
    print("  soft canonical:", top(sr.canonical))
    print("  soft first-tok:", top(sr.first_token))
    print("  hard         :", top(hr))
    print("  hard_raw     :", repr(raw[:160]))
