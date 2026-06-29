"""Version-aware run paths.

Experiment outputs live under runs/<version>/ . Select the version with the
CM_VERSION env var (default "v1"):

    CM_VERSION=v2 PYTHONPATH=src python src/scripts/viz/best_2d_report.py

- v1 : the original walk_length=1000 capture (node/per-occ means over ctx>=300).
       Historic file naming is irregular (square_grid uses acts_sub_<model>.npz,
       Llama under a llama/ subdir; ring/hex use <Model>_acts_sub.npz).
- v2 : walk_length=2000 capture, means over ctx in [1000,2000]. Standardized
       naming everywhere: runs/v2/<graph>/<Model>_acts_sub.npz .
"""
import os

VERSION = os.environ.get("CM_VERSION", "v1")
ROOT = f"runs/{VERSION}"

# default analysis window (context-length lower bound) per version
CTX_LO = {"v1": 300, "v2": 1000}.get(VERSION, 300)
CTX_HI = {"v1": 1000, "v2": 2000}.get(VERSION, 1000)

GRAPHS = ["square_grid", "ring", "hex"]
MODELS = ["Llama", "Gemma", "Qwen"]


def gdir(graph):
    """Directory for a graph's artifacts in the current version."""
    return f"{ROOT}/{graph}"


def overview():
    d = f"{ROOT}/overview"
    return d


def acts_path(graph, model):
    """Path to a model's subsampled activations for a graph in this version."""
    if VERSION == "v1":
        if graph == "square_grid":
            if model == "Llama":
                return f"{ROOT}/square_grid/llama/acts_sub_llama.npz"
            return f"{ROOT}/square_grid/acts_sub_{model.lower()}.npz"
        return f"{ROOT}/{graph}/{model}_acts_sub.npz"
    # v2 and beyond: standardized layout
    return f"{ROOT}/{graph}/{model}_acts_sub.npz"


# v1 grid-peak layers (argmax full-dim grid RSA), measured once from v1 data.
_PEAK_V1 = {
    "square_grid": {"Llama": 13, "Gemma": 40, "Qwen": 32},
    "ring":        {"Llama": 30, "Gemma": 39, "Qwen": 33},
    "hex":         {"Llama": 30, "Gemma": 39, "Qwen": 34},
}


def peak_layer(graph, model):
    """Grid-peak layer for (graph, model) in the current version.

    v1 uses a measured table; later versions read the capture's per-model
    analysis JSON (grid_rsa per layer) written by capture_v2.py.
    """
    if VERSION == "v1":
        return _PEAK_V1[graph][model]
    import json
    d = json.load(open(f"{ROOT}/{graph}/{model}_analysis.json"))
    gr = {int(k): v for k, v in d["grid_rsa"].items()}
    return max(gr, key=gr.get)
