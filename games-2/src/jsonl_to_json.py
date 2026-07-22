"""Convert a .jsonl transcript (one JSON object per line) to a proper pretty-printed
.json array, which opens in any browser / editor / macOS Quick Look (unlike .jsonl,
which often has no default app).

Usage:
  python src/jsonl_to_json.py <transcript.jsonl>   # one file -> same name .json
  python src/jsonl_to_json.py runs                 # convert every *_transcript.jsonl under a dir
"""
from __future__ import annotations

import os
import sys
import json


def convert(jsonl_path):
    """Write <base>.json (pretty array) next to a .jsonl transcript. Returns the path."""
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    out = os.path.splitext(jsonl_path)[0] + ".json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    return out


def convert_dir(root):
    n = 0
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith("_transcript.jsonl"):
                convert(os.path.join(dp, fn)); n += 1
    return n


def main():
    arg = sys.argv[1]
    if os.path.isdir(arg):
        print(f"converted {convert_dir(arg)} transcripts under {arg}")
    else:
        print("wrote", convert(arg))


if __name__ == "__main__":
    main()
