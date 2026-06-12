"""Scan every patch's VLM belief to find which crops are genuinely ambiguous.

Crops to the elephant, grids it, runs the soft readout on each patch, and reports
how confidently the VLM calls each patch 'elephant'. Patches that are NOT
confidently elephant are the genuine 'blind men' views we want.
"""
import os

import numpy as np
from PIL import Image

from qsg import readout
from qsg.agents import Agent, ModelBackend
from qsg.config import load_sweep_config
from qsg.engine import load_ontology

GX = int(os.environ.get("GX", 6))
DOWNSCALE = int(os.environ.get("DOWNSCALE", 0))  # if >0, resize patch to this px

cfg = load_sweep_config("configs/base.yaml").base
cfg.arm = "image"
cfg.object_name = "elephant"
cfg.model.vision_model = "Qwen/Qwen2-VL-7B-Instruct"
cfg.activations.capture = False

ont = load_ontology(cfg.ontology_path)
cands = ont["candidates"]

im = Image.open("assets/elephant_src/elephant.jpg").convert("RGB")
W, H = im.size
crop = im.crop((int(0.12 * W), int(0.20 * H), int(0.88 * W), int(0.97 * H)))
cw, ch = crop.size
pw, ph = cw // GX, ch // GX
patches = []
for j in range(GX):
    for i in range(GX):
        p = crop.crop((i * pw, j * ph, (i + 1) * pw, (j + 1) * ph))
        if DOWNSCALE:
            p = p.resize((DOWNSCALE, DOWNSCALE))
        patches.append(p)

print(f"grid {GX}x{GX} = {len(patches)} patches, size {(pw, ph)}, downscale={DOWNSCALE}")
backend = ModelBackend(cfg, cands)
print("loaded", backend.model_name)

rows = []
for idx, p in enumerate(patches):
    ag = Agent(idx, "", images=[p])
    sr = readout.soft_readout(backend, ag, cfg, hooks=None)
    d = sr.canonical
    rows.append((idx, cands[int(np.argmax(d))], float(np.max(d)), float(d[0])))

rows.sort(key=lambda r: r[3])  # ascending elephant-probability
print("\npatch  topguess     p(top)  p(elephant)")
for idx, g, pt, pe in rows:
    print(f"  {idx:3d}  {g:9s}  {pt:.2f}    {pe:.2f}")
fooled = sum(1 for r in rows if r[1] != "elephant")
ambiguous = sum(1 for r in rows if r[3] < 0.6)
print(f"\npatches whose top guess != elephant: {fooled}/{len(rows)}")
print(f"patches with p(elephant) < 0.6 (ambiguous): {ambiguous}/{len(rows)}")
