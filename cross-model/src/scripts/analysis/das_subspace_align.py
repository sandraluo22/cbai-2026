"""(a) Do the DAS parity SUBSPACES align across grid sizes even though the 1-D axes do not?
Loads the SAVE_R rotations npz from das_parity_scale, computes pairwise principal angles between the
rank-r subspaces of different grid sizes (singular values of Ra @ Rb^T), and reports the mean cos^2
overlap vs the exact random-subspace expectation r/hd. CPU-only.

Env: R_NPZ(runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_Llama.npz) R(16) OUTDIR
Out: <OUTDIR>/das_subspace_align_<R>.json + .pdf
"""
from __future__ import annotations
import os, json
import numpy as np

R_NPZ = os.environ.get("R_NPZ", "runs/axes/4_circuits/parity/das_parity_scale_R_rotation_ho3_ctxf2000R_Llama.npz")
RR = int(os.environ.get("R", "16"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def main():
    z = np.load(R_NPZ)
    grids = sorted({k.split("_")[0] for k in z.files}, key=lambda s: int(s.split("x")[0]))
    R = {g: z[f"{g}_R{RR}"] for g in grids if f"{g}_R{RR}" in z}
    grids = list(R)
    hd = R[grids[0]].shape[1]
    rng = np.random.default_rng(0)

    def overlap(a, b):
        s = np.linalg.svd(a @ b.T, compute_uv=False)
        return s  # principal-angle cosines

    pairs = {}; rand_ov = []
    for _ in range(200):
        qa = np.linalg.qr(rng.standard_normal((hd, RR)))[0].T
        qb = np.linalg.qr(rng.standard_normal((hd, RR)))[0].T
        rand_ov.append(float((overlap(qa, qb) ** 2).mean()))
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            s = overlap(R[grids[i]], R[grids[j]])
            pairs[f"{grids[i]}_vs_{grids[j]}"] = {
                "mean_cos2": round(float((s ** 2).mean()), 4),
                "top_cosines": [round(float(x), 3) for x in s[:8]],
                "n_dims_cos_gt_0.5": int((s > 0.5).sum())}
    out = {"r_npz": R_NPZ, "rank": RR, "hd": hd, "grids": grids,
           "random_expectation_mean_cos2": round(RR / hd, 4),
           "random_empirical_mean_cos2": round(float(np.mean(rand_ov)), 4),
           "random_empirical_p99": round(float(np.percentile(rand_ov, 99)), 4),
           "pairs": pairs,
           "mean_over_pairs": round(float(np.mean([p["mean_cos2"] for p in pairs.values()])), 4)}
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/das_subspace_align_r{RR}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for k, v in pairs.items(): print(f"  {k}: mean_cos2={v['mean_cos2']} top={v['top_cosines'][:4]} strong_dims={v['n_dims_cos_gt_0.5']}")
    print(f"DONE -> {p}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    ng = len(grids); Mx = np.full((ng, ng), np.nan)
    for i in range(ng):
        Mx[i, i] = 1.0
        for j in range(i + 1, ng):
            Mx[i, j] = Mx[j, i] = pairs[f"{grids[i]}_vs_{grids[j]}"]["mean_cos2"]
    im = ax.imshow(Mx, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(ng)); ax.set_xticklabels(grids); ax.set_yticks(range(ng)); ax.set_yticklabels(grids)
    ax.set_title(f"mean cos² overlap of rank-{RR} DAS subspaces\n(random baseline {RR}/{hd} = {RR/hd:.3f})", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax = axes[1]
    for i in range(ng):
        for j in range(i + 1, ng):
            s = overlap(R[grids[i]], R[grids[j]])
            ax.plot(range(1, len(s) + 1), s, '-', alpha=0.6, lw=1)
    qa = np.linalg.qr(rng.standard_normal((hd, RR)))[0].T; qb = np.linalg.qr(rng.standard_normal((hd, RR)))[0].T
    ax.plot(range(1, RR + 1), overlap(qa, qb), 'k--', lw=1.5, label='random pair')
    ax.set_xlabel('principal angle index'); ax.set_ylabel('cosine'); ax.set_ylim(0, 1)
    ax.set_title('principal-angle spectra, all grid pairs', fontsize=9); ax.legend(frameon=False, fontsize=8)
    for a in axes: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.tight_layout()
    fp = f"{OUTDIR}/das_subspace_align_r{RR}.pdf"
    fig.savefig(fp); fig.savefig(fp.replace(".pdf", ".png"), dpi=160)
    print(f"figure -> {fp}")


if __name__ == "__main__":
    main()
