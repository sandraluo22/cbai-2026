"""Render the '_noneutral' slideshows by streaming a SUBSET of activations from
the RunPod S3-backed network volume — no full 46GB download.

Run this yourself so it uses your already-working RunPod S3 creds:

    ! cd /Users/sandraluo/cbai-2026/emotion && python render_noneutral_from_s3.py
    ! ... python render_noneutral_from_s3.py 12000     # bigger subset

It reads the LOCAL labels/meta (already on the Mac), picks a stratified
non-neutral subset, range-GETs just those example blocks for q/a1/a2 from S3,
holds them in RAM, and writes pca_{Q,A1,A2}_noneutral_slideshow.pdf and
cos_a1_a2_hist_noneutral_slideshow.pdf into results/all_full/plots/.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import boto3
from botocore.config import Config

from goemotions_utils import GOEMOTIONS_LABELS, EKMAN_NAMES, stratified_indices
from make_plots import pca_slideshow, cos_slideshow

ENDPOINT = "https://s3api-us-nc-1.runpod.io"
REGION = "us-nc-1"
BUCKET = "1qksv506n4"
RUN = "all_full"
N_SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
WORKERS = 32

run_dir = Path(__file__).parent / "results" / RUN
meta = json.loads((run_dir / "meta.json").read_text())
N, L, H = meta["N"], meta["L"], meta["H"]
rec = L * H * 2                       # bytes per example block (float16)
primary = np.load(run_dir / "labels_primary.npy")
ekman = np.load(run_dir / "labels_ekman.npy")

# ---- choose a stratified NON-neutral subset ----
neutral_id = GOEMOTIONS_LABELS.index("neutral")
nn = np.where(primary != neutral_id)[0]
sub_local = stratified_indices(primary[nn], N_SUB, seed=0)
sel = np.sort(nn[sub_local])          # global example indices, ascending
print(f"[subset] {len(sel)} non-neutral examples "
      f"(of {len(nn)} non-neutral / {N} total)")

s3 = boto3.client("s3", region_name=REGION, endpoint_url=ENDPOINT,
                  config=Config(s3={"addressing_style": "path"},
                                retries={"max_attempts": 5}))

# ---- discover the key prefix (where *_acts.dat live in the bucket) ----
prefix = None
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(f"{RUN}/q_acts.dat"):
            prefix = obj["Key"][: -len("q_acts.dat")]
            break
    if prefix:
        break
if prefix is None:
    sys.exit(f"could not find {RUN}/q_acts.dat in bucket {BUCKET}")
print(f"[s3] key prefix: {prefix}")


def fetch_block(key, idx):
    off = int(idx) * rec
    r = s3.get_object(Bucket=BUCKET, Key=key,
                      Range=f"bytes={off}-{off + rec - 1}")
    buf = r["Body"].read()
    return np.frombuffer(buf, dtype="<f2").reshape(L, H)


def fetch_array(name):
    key = f"{prefix}{name}_acts.dat"
    out = np.empty((len(sel), L, H), dtype=np.float16)
    done = [0]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for j, block in enumerate(ex.map(lambda i: fetch_block(key, i), sel)):
            out[j] = block
            done[0] += 1
            if done[0] % 1000 == 0:
                print(f"  [{name}] {done[0]}/{len(sel)}")
    return out

print("[fetch] streaming Q/A1/A2 subset blocks from S3 ...")
acts = {n: fetch_array(n) for n in ("q", "a1", "a2")}
acts = {"Q": acts["q"], "A1": acts["a1"], "A2": acts["a2"]}

# subset labels, in the same order as `sel`
labels_sub = ekman[sel]
meta_sub = dict(meta)                 # L/H unchanged; PCA reads [:, li, :]

print("[plot] rendering _noneutral slideshows ...")
for name in ("Q", "A1", "A2"):
    pca_slideshow(run_dir, name, acts[name], labels_sub, EKMAN_NAMES,
                  meta_sub, keep=None, suffix="_noneutral")
cos_slideshow(run_dir, {"A1": acts["A1"], "A2": acts["A2"]}, meta_sub,
              keep=None, suffix="_noneutral")
print(f"[done] wrote *_noneutral_slideshow.pdf to {run_dir / 'plots'}")
