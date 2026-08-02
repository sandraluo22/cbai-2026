"""Small shared helpers: seeding, JSON dump, results dir, dotenv loader."""
from __future__ import annotations

import json
import os
import random
from typing import Any


def seed_all(seed: int = 0):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def results_dir(game_dir: str) -> str:
    d = os.path.join(game_dir, "results")
    os.makedirs(d, exist_ok=True)
    return d


def dump_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"[io] wrote {path}", flush=True)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def load_dotenv(path: str):
    """Minimal .env loader (KEY=VALUE lines) into os.environ if not already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            # override when unset OR empty ('' still wins the SDK's precedence slot)
            if not os.environ.get(k):
                os.environ[k] = v
