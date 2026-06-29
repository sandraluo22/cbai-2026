"""Faithful replication of Shai et al. 'Transformers Represent Belief State
Geometry in their Residual Stream' (LessWrong / comp-mech).

Pipeline:
  1. Mess3 HMM (3 hidden states, 3 symbols) -- a hidden process whose Mixed-State
     Presentation (set of optimal beliefs over hidden state) is a FRACTAL.
  2. Train a small transformer on next-symbol prediction over Mess3 sequences.
  3. Ground truth: forward-filter the exact belief P(next hidden state | symbols
     so far) at every position -> the MSP fractal in the 2-simplex.
  4. Linear regression: residual stream (all layers concatenated, per position)
     -> belief-state coordinates.
  5. Visualise: ground-truth fractal vs the fractal linearly decoded from the
     residual stream (coloured by belief = RGB barycentric coords).

-> runs/belief_geometry/shai_mess3_recovered.png
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
N, V = 3, 3                      # hidden states, vocab
CTX = 16                         # sequence length
D, H, LYR = 64, 4, 4             # model: d_model, heads, layers
STEPS, BATCH, LR = 4000, 256, 2e-3


# ---------------- Mess3 process (emit-on-departure) ----------------
def mess3(x=0.15, alpha=0.6):
    A = np.full((N, N), x); np.fill_diagonal(A, 1 - 2 * x)        # state transition i->j
    b = (1 - alpha) / 2
    E = np.full((N, N), b); np.fill_diagonal(E, alpha)            # emission P(symbol s | state i)
    return A, E


A_np, E_np = mess3()
def stationary(A):
    w, v = np.linalg.eig(A.T); p = np.real(v[:, np.argmin(np.abs(w - 1))]); return p / p.sum()
STAT = stationary(A_np)


def sample_batch(bsz, L, rng):
    """Sample symbol sequences from Mess3 (emit from current state, then move)."""
    s = rng.choice(N, p=STAT, size=bsz)
    out = np.empty((bsz, L), dtype=np.int64)
    for t in range(L):
        # vectorised categorical draws
        r = rng.random(bsz)
        sym = (r[:, None] < np.cumsum(E_np[s], 1)).argmax(1)
        out[:, t] = sym
        r2 = rng.random(bsz)
        s = (r2[:, None] < np.cumsum(A_np[s], 1)).argmax(1)
    return out


def beliefs_for(seqs):
    """Ground-truth MSP: belief over the NEXT hidden state given symbols so far.
    seqs: (B, L) -> (B, L, N)."""
    B, L = seqs.shape
    out = np.empty((B, L, N))
    bel = np.repeat(STAT[None], B, 0)
    for t in range(L):
        post = bel * E_np[:, seqs[:, t]].T          # P(S_t | obs<=t), (B,N)
        post /= post.sum(1, keepdims=True)
        bel = post @ A_np                           # P(S_{t+1} | obs<=t)
        out[:, t] = bel
    return out


# ---------------- tiny GPT ----------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D); self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LYR)])
        self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, V)

    def forward(self, idx, return_resid=False):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        resids = [x]
        for blk in self.blocks:
            x = blk(x, mask); resids.append(x)
        logits = self.head(self.lnf(x))
        if return_resid:
            return logits, torch.stack(resids, 0)   # (LYR+1, B, T, D)
        return logits


def train():
    rng = np.random.default_rng(0)
    model = GPT().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        idx = torch.from_numpy(sample_batch(BATCH, CTX, rng)).to(DEV)
        logits = model(idx)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:4d} loss {loss.item():.4f}", flush=True)
    return model, losses


def to_2d(b):
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    return b @ v


def main():
    model, losses = train()
    model.eval()
    rng = np.random.default_rng(1)
    seqs = sample_batch(3000, CTX, rng)                         # test sequences
    bel = beliefs_for(seqs)                                     # (B,L,N) ground truth
    with torch.no_grad():
        _, resid = model(torch.from_numpy(seqs).to(DEV), return_resid=True)
    R = resid.permute(1, 2, 0, 3).reshape(seqs.shape[0], CTX, -1).cpu().numpy()  # (B,L,(LYR+1)*D)

    # flatten over positions>=1 (skip t=0, identical for all)
    feat = R[:, 1:].reshape(-1, R.shape[-1]); Y = bel[:, 1:].reshape(-1, N)
    ntr = int(0.7 * len(feat))
    Xtr, Ytr, Xte, Yte = feat[:ntr], Y[:ntr], feat[ntr:], Y[ntr:]
    # linear probe (ridge)
    Xc = Xtr - Xtr.mean(0); W = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.eye(Xc.shape[1]), Xc.T @ (Ytr - Ytr.mean(0)))
    pred = (Xte - Xtr.mean(0)) @ W + Ytr.mean(0)
    r2 = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    print(f"probe belief R^2 = {r2:.3f} | final loss {np.mean(losses[-100:]):.4f}", flush=True)

    predc = np.clip(pred, 0, None); predc /= predc.sum(1, keepdims=True)
    GT, PR = to_2d(Yte), to_2d(predc)
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, (XY, col, ttl) in zip(ax, [
            (GT, Yte, "ground-truth MSP (belief filtering)"),
            (PR, predc, f"recovered from residual stream (R²={r2:.2f})"),
            (None, None, "training loss")]):
        if XY is None:
            a.plot(losses, lw=0.6); a.set_xlabel("step"); a.set_ylabel("cross-entropy"); a.set_title(ttl)
            continue
        a.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1)
        a.scatter(XY[:, 0], XY[:, 1], s=3, c=np.clip(col, 0, 1), alpha=0.5, linewidths=0, rasterized=True)
        a.set_title(ttl, fontsize=10); a.set_aspect("equal"); a.axis("off")
    fig.suptitle("Shai et al. replication: a transformer trained on Mess3 represents the fractal "
                 "belief-state geometry in its residual stream")
    fig.tight_layout()
    import os; os.makedirs("runs/belief_geometry", exist_ok=True)
    out = "runs/belief_geometry/shai_mess3_recovered.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
