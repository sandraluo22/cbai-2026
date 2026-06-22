"""Three-model comparison: Llama-3.1-8B vs Gemma-2-9B vs Qwen3-8B-Base.
Grid structure by layer, in-context emergence, and behavioural accuracy."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gq = json.load(open("runs/square_grid/rebuild_grid_rsa.json"))
llama_rsa = json.load(open("runs/square_grid/llama/llama_grid_rsa.json"))
pf = json.load(open("runs/square_grid/gemma_qwen/paper_faithful.json"))
le = json.load(open("runs/square_grid/llama/llama_emergence.json"))
acc = json.load(open("runs/square_grid/accuracy/accuracy.json"))
la = json.load(open("runs/square_grid/llama/llama_accuracy.json"))


def reldepth(d):
    Ls = sorted(map(int, d)); n = max(Ls)
    return [l / n for l in Ls], [d[str(l)] for l in Ls]


def emerg(rows):
    return [r["ctx"] for r in rows], [r["rsa"] for r in rows]


def mass(rows):
    return [r["ctx"] for r in rows], [r["neighbor_mass"] for r in rows]


fig, ax = plt.subplots(1, 3, figsize=(17, 5))
C = {"Llama": "tab:blue", "Gemma": "tab:orange", "Qwen": "tab:green"}

# (1) grid RSA by relative depth
for tag, d in (("Llama", llama_rsa), ("Gemma", gq["gemma"]), ("Qwen", gq["qwen"])):
    x, y = reldepth(d)
    ax[0].plot(x, y, "-o", ms=3, color=C[tag], label=tag)
ax[0].axhline(0, color=".8", lw=.6)
ax[0].set_xlabel("relative depth (layer / n_layers)"); ax[0].set_ylabel("grid RSA")
ax[0].set_title("(1) Grid structure by layer"); ax[0].legend()

# (2) paper-faithful emergence (windowed) at each model's deep layer
for tag, rows, lab in (("Llama", le["emergence"], "Llama L26"),
                       ("Gemma", pf["gemma"], "Gemma L32"),
                       ("Qwen", pf["qwen"], "Qwen L28")):
    x, y = emerg(rows)
    ax[1].plot(x, y, "-o", ms=3, color=C[tag], label=lab)
ax[1].set_xscale("log"); ax[1].axhline(0, color=".8", lw=.6)
ax[1].set_xlabel("context length"); ax[1].set_ylabel("grid RSA (Nw=50 window)")
ax[1].set_title("(2) In-context emergence (paper-faithful)"); ax[1].legend()

# (3) behavioural next-step (neighbour mass)
for tag, rows in (("Llama", la["llama"]["by_context"]),
                  ("Gemma", acc["gemma"]["by_context"]),
                  ("Qwen", acc["qwen"]["by_context"])):
    x, y = mass(rows)
    ax[2].plot(x, y, "-o", ms=3, color=C[tag], label=tag)
ax[2].set_xscale("log")
ax[2].set_xlabel("context length"); ax[2].set_ylabel("neighbour mass")
ax[2].set_title("(3) Behavioural next-step accuracy"); ax[2].legend()

fig.suptitle("Llama-3.1-8B  vs  Gemma-2-9B  vs  Qwen3-8B-Base  (same 4x4 grid, same walks)")
fig.tight_layout()
fig.savefig("runs/square_grid/llama/three_model_comparison.png", dpi=140)

print("grid RSA peaks (relative depth):")
for tag, d in (("Llama", llama_rsa), ("Gemma", gq["gemma"]), ("Qwen", gq["qwen"])):
    x, y = reldepth(d); i = int(np.argmax(y))
    print(f"  {tag:6} peak {y[i]:+.3f} @ depth {x[i]:.2f} (layer {sorted(map(int,d))[i]})")
print("emergence (grid RSA, low->high ctx):")
for tag, rows in (("Llama", le["emergence"]), ("Gemma", pf["gemma"]), ("Qwen", pf["qwen"])):
    _, y = emerg(rows); print(f"  {tag:6} {y[0]:+.2f} -> {y[-1]:+.2f}")
print("behavioural neighbour mass @ctx1000:")
for tag, rows in (("Llama", la["llama"]["by_context"]),
                  ("Gemma", acc["gemma"]["by_context"]), ("Qwen", acc["qwen"]["by_context"])):
    print(f"  {tag:6} {rows[-1]['neighbor_mass']:.3f}")
print("wrote runs/square_grid/llama/three_model_comparison.png")
