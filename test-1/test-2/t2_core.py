"""test-2 core: latent-graph family with identifiable partner reliability + exact observer.

CONSTRUCTION. Base graph = 4x4 torus (16 nodes, 4-regular). Three DISJOINT "swap
sites": each site is a pair of torus edges (a,b),(c,d) on 4 distinct nodes whose four
alternative pairings (a,c),(b,d),(a,d),(b,c) are all non-edges of the torus. A site
admits 3 perfect matchings of its 4 nodes:
    m=0 {ab,cd}   m=1 {ac,bd}   m=2 {ad,bc}
Candidate family = 27 graphs (one matching choice per site); every candidate is
16-node 4-regular 32-edge connected (degree preserved by construction). G* = one
candidate (spec seed). CORE = torus minus the 6 original site edges (26 edges,
connected, shared by all candidates minus their contested edges).

OBSERVATION PROCESSES.
  A: uniform random walks on CORE  -> A's own data is EXACTLY uninformative about
     the matchings (necessity of the partner is by construction; floor = 1/3).
  B: uniform random walks on G* with PROCESS corruption at rate rho: each step,
     with prob rho the walk teleports to a uniform random node (and continues from
     there). Per-step likelihood is exactly (1-rho)*A[v,x]/4 + rho/16.

EXACT OBSERVER. Hypotheses h = (m1,m2,m3, rho) on a rho grid, uniform prior.
A's data contributes a constant (drop). B-attributed stream steps (prev v -> emitted
x, whoever produced v) contribute log[(1-rho)*A_h[v,x]/4 + rho/16]. Posterior by
enumeration; predictives and contested-discrimination scores derived from it.

METRIC. For cue node a in site (a,b,c,d) the three matchings predict partner b, c
or d. Discrimination score of a 16-way predictive p at cue a:
    score = p[true partner] / (p[b]+p[c]+p[d]),  chance 1/3, oracle 1.

MOCK BACKEND. Dirichlet-Markov learner (gamma=0.96, alpha=0.05, the fitted test-1
surrogate) with the same interface as the LLM backend: it is the PARTNER-BLIND
source-tracking null for every measurement in this folder.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # test-1 (graphs16)
from graphs16 import torus, connected, adj_to_edges   # noqa: E402

N = 16
WORDS = ("clock,lemon,jacket,wheel,tiger,pencil,coin,bird,anchor,ocean,"
         "chair,candle,bread,mirror,apple,river").split(",")  # sweep vocab (see README)
RHO_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
MATCHINGS = ((("a", "b"), ("c", "d")), (("a", "c"), ("b", "d")),
             (("a", "d"), ("b", "c")))


# ---------------------------------------------------------------- spec building
def find_sites(A0, rng, n_sites=3, tries=4000):
    """Three disjoint swap sites; removing all 6 original edges keeps core connected."""
    edges = adj_to_edges(A0)
    for _ in range(tries):
        idx = rng.permutation(len(edges))
        sites, used = [], set()
        for i in idx:
            if len(sites) == n_sites:
                break
            a, b = edges[i]
            if a in used or b in used:
                continue
            for j in idx:
                c, d = edges[j]
                if len({a, b, c, d}) < 4 or c in used or d in used:
                    continue
                alts = [(a, c), (b, d), (a, d), (b, c)]
                if any(A0[p, q] for p, q in alts):
                    continue
                sites.append({"a": a, "b": b, "c": c, "d": d})
                used |= {a, b, c, d}
                break
        if len(sites) < n_sites:
            continue
        core = A0.copy()
        for s in sites:
            for u, v in (("a", "b"), ("c", "d")):
                core[s[u], s[v]] = core[s[v], s[u]] = 0
        if connected(core):
            return sites, core
    raise RuntimeError("no site set found")


def site_edges(site, m):
    return [(site[u], site[v]) for u, v in MATCHINGS[m]]


def candidate_adj(core, sites, ms):
    A = core.copy()
    for s, m in zip(sites, ms):
        for u, v in site_edges(s, m):
            A[u, v] = A[v, u] = 1
    return A


def build_spec(seed=0):
    rng = np.random.default_rng(seed)
    A0 = torus()
    sites, core = find_sites(A0, rng)
    while True:                                   # G*: at least one non-torus matching
        m_true = [int(x) for x in rng.integers(0, 3, size=3)]
        if any(m > 0 for m in m_true):
            break
    Astar = candidate_adj(core, sites, m_true)
    assert (Astar.sum(1) == 4).all() and connected(Astar)
    spec = {"seed": seed, "words": WORDS, "sites": sites, "m_true": m_true,
            "core_edges": adj_to_edges(core), "gstar_edges": adj_to_edges(Astar),
            "rho_grid": RHO_GRID.tolist()}
    return spec


def spec_adj(spec, which):
    A = np.zeros((N, N), bool)
    for a, b in spec[which]:
        A[a, b] = A[b, a] = True
    return A


def all_candidates(spec):
    core = spec_adj(spec, "core_edges").astype(int)
    out = []
    for m1 in range(3):
        for m2 in range(3):
            for m3 in range(3):
                out.append(((m1, m2, m3),
                            candidate_adj(core, spec["sites"], (m1, m2, m3))))
    return out


def contested_options(spec):
    """cue node -> (list of 3 partner nodes per matching, true matching index)."""
    opts = {}
    for s, mt in zip(spec["sites"], spec["m_true"]):
        # matching m=0 {ab,cd}, m=1 {ac,bd}, m=2 {ad,bc}; per-node partner table
        a, b, c, d = s["a"], s["b"], s["c"], s["d"]
        table = {a: [b, c, d], b: [a, d, c], c: [d, a, b], d: [c, b, a]}
        for node, partners in table.items():
            opts[node] = {"options": partners, "true_m": mt}
    return opts


def score(p, cue, opts):
    """Discrimination score of 16-way predictive p at cue (chance 1/3, oracle 1)."""
    o = opts[cue]
    mass = np.array([p[j] for j in o["options"]], float)
    tot = mass.sum()
    return float(mass[o["true_m"]] / tot) if tot > 1e-12 else 1.0 / 3.0


# ---------------------------------------------------------------- walks
def walk(adj, length, seed, start=None):
    rng = np.random.default_rng(seed)
    v = int(rng.integers(N)) if start is None else int(start)
    out = [v]
    for _ in range(length - 1):
        nbrs = np.where(adj[v])[0]
        v = int(rng.choice(nbrs))
        out.append(v)
    return out


def corrupt_walk(adj, length, rho, seed, start=None):
    """Process corruption: each step w.p. rho teleport to uniform node (walk continues
    from it). Returns (nodes, corrupted_flags)."""
    rng = np.random.default_rng(seed)
    v = int(rng.integers(N)) if start is None else int(start)
    out, flags = [v], [False]
    for _ in range(length - 1):
        if rng.random() < rho:
            v = int(rng.integers(N)); flags.append(True)
        else:
            v = int(rng.choice(np.where(adj[v])[0])); flags.append(False)
        out.append(v)
    return out, flags


# ---------------------------------------------------------------- exact observer
class ExactObserver:
    """Posterior over (m1,m2,m3) x rho grid from B-attributed transitions."""

    def __init__(self, spec, rho_grid=None):
        self.spec = spec
        self.cands = all_candidates(spec)                 # 27 (ms, adj)
        self.rho = np.array(spec["rho_grid"] if rho_grid is None else rho_grid)
        self.loglik = np.zeros((len(self.cands), len(self.rho)))

    def update(self, prev, emit):
        for ci, (_, A) in enumerate(self.cands):
            pe = (1.0 - self.rho) * (A[prev, emit] / 4.0) + self.rho / 16.0
            self.loglik[ci] += np.log(np.maximum(pe, 1e-300))

    def update_many(self, pairs):
        for v, x in pairs:
            self.update(v, x)

    def posterior(self):
        w = self.loglik - self.loglik.max()
        w = np.exp(w); w /= w.sum()
        return w                                          # [27, n_rho]

    def rho_posterior(self):
        return self.posterior().sum(0)

    def graph_posterior(self):
        return self.posterior().sum(1)

    def predictive(self, cue, rho_fixed=None):
        """Pooled p(next|cue): sum_h w_h * A_h[cue,:]/4 (+ rho-teleport mass)."""
        if rho_fixed is None:
            w = self.posterior()
        else:
            j = int(np.argmin(np.abs(self.rho - rho_fixed)))
            w = np.zeros_like(self.loglik)
            col = self.loglik[:, j] - self.loglik[:, j].max()
            w[:, j] = np.exp(col) / np.exp(col).sum()
        p = np.zeros(N)
        for ci, (_, A) in enumerate(self.cands):
            wc = w[ci].sum()
            p += wc * A[cue] / 4.0
        return p / max(p.sum(), 1e-12)

    def copy(self):
        o = ExactObserver(self.spec, self.rho)
        o.loglik = self.loglik.copy()
        return o


def oracle_predictive(spec, cue):
    A = spec_adj(spec, "gstar_edges")
    return A[cue] / 4.0


# ---------------------------------------------------------------- mock backend
class MockBackend:
    """Dirichlet-Markov contexts (gamma=0.96, alpha=0.05; test-1 fitted surrogate).
    Interface mirrors the LLM backend in run_test2.py: contexts are lists of node
    sequences; predictives are 16-way."""

    def __init__(self, gamma=0.96, alpha=0.05, temp=1.0):
        self.g, self.a, self.temp = gamma, alpha, temp
        self.C, self.prev = {}, {}

    def prefill(self, key, node_rows):
        for p, nodes in enumerate(node_rows):
            C = np.zeros((N, N))
            for s in range(1, len(nodes)):
                C *= self.g
                C[nodes[s - 1], nodes[s]] += 1.0
            self.C[(key, p)] = C
            self.prev[(key, p)] = nodes[-1]

    def predictive(self, key, p):
        pr = (self.a + self.C[(key, p)][self.prev[(key, p)]]) ** (1.0 / self.temp)
        return pr / pr.sum()

    def append(self, key, p, node):
        self.C[(key, p)] *= self.g
        self.C[(key, p)][self.prev[(key, p)], node] += 1.0
        self.prev[(key, p)] = node

    def probe(self, key, p, cue):
        pr = (self.a + self.C[(key, p)][cue]) ** (1.0 / self.temp)
        return pr / pr.sum()


def save_spec(spec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(spec, open(path, "w"), indent=1)


if __name__ == "__main__":
    spec = build_spec(int(os.environ.get("SEED", "0")))
    out = os.path.join(HERE, "runs", "t2_spec.json")
    save_spec(spec, out)
    print(f"sites: {spec['sites']}\nm_true: {spec['m_true']} -> {out}")
    # quick identifiability sanity: exact observer on scripted corrupted walks
    Astar = spec_adj(spec, "gstar_edges")
    for rho in (0.0, 0.3, 0.6):
        obs = ExactObserver(spec)
        nodes, _ = corrupt_walk(Astar, 400, rho, seed=1)
        obs.update_many(zip(nodes[:-1], nodes[1:]))
        gp = obs.graph_posterior()
        ms = [self_m for self_m, _ in obs.cands]
        true_i = ms.index(tuple(spec["m_true"]))
        rp = obs.rho_posterior()
        print(f"rho={rho}: P(G*)={gp[true_i]:.3f}  "
              f"rho_MAP={obs.rho[np.argmax(rp)]:.2f}  P(rho_MAP)={rp.max():.3f}")
