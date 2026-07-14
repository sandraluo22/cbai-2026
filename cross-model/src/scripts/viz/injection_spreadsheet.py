"""Combine every model-pair injection run into ONE multi-page PDF: one page per graph.
Each page is a 'spreadsheet' figure — rows = the 6 pairs (A->B), columns = [alignment R²,
injected neighbour mass], both over L_A×L_B.

Graphs are auto-detected from the injection_*_to_*_<graph>.json files in INJDIR (override
with GRAPHS="square_grid,ring"). Everything lands in a single injection_spreadsheet.pdf.

Env: INJDIR(injection dir) GRAPHS(auto) OUTDIR(defaults to INJDIR)
Out: <OUTDIR>/injection_spreadsheet.pdf
"""
import os, json, re, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODELS = ["Llama", "Gemma", "Qwen"]
INJDIR = os.environ.get("INJDIR", "runs/induction-head/2_probes/injection")
OUTDIR = os.environ.get("OUTDIR", INJDIR)
pairs = [(a, b) for a, b in itertools.permutations(MODELS, 2)]


def discover_graphs():
    """Graphs present in INJDIR, or the GRAPHS env override (comma-separated)."""
    env = os.environ.get("GRAPHS")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    names = set(re.escape(m) for m in MODELS)
    pat = re.compile(rf"^injection_({'|'.join(names)})_to_({'|'.join(names)})_(.+)\.json$")
    graphs = set()
    for f in os.listdir(INJDIR):
        m = pat.match(f)
        if m:
            graphs.add(m.group(3))
    # stable, readable order: square_grid first, then the rest alphabetically
    return sorted(graphs, key=lambda g: (g != "square_grid", g))


def draw_graph_page(pdf, graph):
    recs = {}
    for a, b in pairs:
        f = f"{INJDIR}/injection_{a}_to_{b}_{graph}.json"
        if os.path.exists(f):
            recs[(a, b)] = json.load(open(f))
    have = [p for p in pairs if p in recs]
    if not have:
        print(f"  [skip] no injection json for {graph}")
        return False
    fig, ax = plt.subplots(len(have), 2, figsize=(11, 3.2 * len(have)), squeeze=False)
    for i, (a, b) in enumerate(have):
        r = recs[(a, b)]; LA = r["LA_sweep"]; LB = r["LB_sweep"]
        align = np.array(r["align_r2_grid"]); inj = np.array(r["inj_nbr_mass_grid"])
        ext = [LB[0], LB[-1], LA[0], LA[-1]]
        im0 = ax[i, 0].imshow(align, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        fig.colorbar(im0, ax=ax[i, 0], fraction=.046)
        ax[i, 0].set_ylabel(f"{a}→{b}\n{a} layer L_A", fontsize=8)
        im1 = ax[i, 1].imshow(inj, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1, extent=ext)
        fig.colorbar(im1, ax=ax[i, 1], fraction=.046)
        ax[i, 1].set_ylabel(f"{a} layer L_A", fontsize=8)
        ax[i, 0].set_xlabel(f"{b} layer L_B", fontsize=8); ax[i, 1].set_xlabel(f"{b} layer L_B", fontsize=8)
        if i == 0:
            ax[i, 0].set_title("cross-model alignment R²", fontsize=10)
            ax[i, 1].set_title(f"injected neighbour mass (nat in title)", fontsize=10)
        ax[i, 1].text(0.98, 0.02, f"native={r['B_native_beh']:.2f}", transform=ax[i, 1].transAxes,
                      ha="right", va="bottom", fontsize=6, color="k")
    fig.suptitle(f"[{graph}] injection spreadsheet — {len(have)} model pairs × (alignment R² | injected behaviour) over L_A×L_B", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"  [page] {graph} ({len(have)} pairs)")
    return True


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    graphs = discover_graphs()
    if not graphs:
        raise SystemExit(f"no injection json found in {INJDIR}")
    path = f"{OUTDIR}/injection_spreadsheet.pdf"
    pages = 0
    with PdfPages(path) as pdf:
        for graph in graphs:
            if draw_graph_page(pdf, graph):
                pages += 1
    if pages == 0:
        os.remove(path)
        raise SystemExit(f"no pages drawn for graphs {graphs}")
    print(f"DONE -> {path} ({pages} graph pages)")


if __name__ == "__main__":
    main()
