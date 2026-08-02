"""Concept-space definitions for the geometry battery. Each space lists its items, a carrier template for
representation extraction, the hypothesised geometry family, and enough structure to build candidate
distance matrices (line/arc, cycle, simplex, tree, product, helix). Shared by geometry_fit.py and
geometry_ablate.py so the fit and the causal ablation use the same items/structure.
"""
import numpy as np

# ---- candidate structural distance matrices from per-space coordinates ----
def d_line(order):                 # 1D arc/segment
    o = np.array(order, float); return np.abs(o[:, None] - o[None, :])
def d_cycle(idx, period):          # ring
    i = np.array(idx, float); d = np.abs(i[:, None] - i[None, :]) % period
    return np.minimum(d, period - d)
def d_simplex(n):                  # mutually exclusive, all equidistant
    return 1.0 - np.eye(n)
def d_tree(parent):                # path length in a rooted tree/forest (parent[i]=parent idx, -1 root)
    n = len(parent); D = np.zeros((n, n))
    def anc(i):
        p = [i]
        while parent[p[-1]] >= 0: p.append(parent[p[-1]])
        p.append(n)                # virtual super-root so a forest still has a common ancestor
        return p
    for i in range(n):
        ai = anc(i)
        for j in range(n):
            aj = anc(j); s = set(aj)
            lca = next(a for a in ai if a in s)
            D[i, j] = ai.index(lca) + aj.index(lca)
    return D
def d_product(factors):            # L1 over independent factor coordinates
    F = np.array(factors, float); return np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
def d_helix(cyc_idx, period, prog, wl=1.0, wp=1.0):   # cycle inside a linear progression
    return np.sqrt((wl * d_cycle(cyc_idx, period)) ** 2 + (wp * d_line(prog)) ** 2)
def d_grid(coord2d):               # 2D square-lattice (Manhattan) distance
    F = np.array(coord2d, float); return np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
def d_hex(coord2d):                # 2D triangular (hex) lattice distance via odd-r cube coords
    def cube(r, c):
        x = c - (r - (r & 1)) // 2; z = r; return np.array([x, -x - z, z])
    F = np.array(coord2d, int); n = len(F); D = np.zeros((n, n))
    cu = [cube(int(r), int(c)) for r, c in F]
    for i in range(n):
        for j in range(n): D[i, j] = np.abs(cu[i] - cu[j]).sum() // 2
    return D


SPACES = {
    # ---------- ARCS / SEGMENTS (1D ordered) ----------
    "lightness": dict(family="arc", carrier="On a scale from black to white, {item}",
        items=["charcoal", "slate", "ash", "ivory"], order=[0, 1, 2, 3]),
    "temperature": dict(family="arc", carrier="The temperature described as {item}",
        items=["freezing", "cold", "cool", "warm", "hot", "scalding"], order=[0, 1, 2, 3, 4, 5]),
    "sentiment": dict(family="arc", carrier="The feeling of being {item}",
        items=["delighted", "pleased", "content", "indifferent", "annoyed", "furious"], order=[0, 1, 2, 3, 4, 5]),
    "formality": dict(family="arc", carrier="In writing, the phrase {item}",
        items=["gonna", "gotta", "going to", "will", "shall"], order=[0, 1, 2, 3, 4]),
    # ---------- SIMPLICES (mutually exclusive, unordered) ----------
    "blood_type": dict(family="simplex", carrier="A patient with type {item} blood",
        items=["A", "B", "AB", "O"]),
    "chess": dict(family="simplex", carrier="In chess, the {item}",
        items=["knight", "bishop", "rook", "queen", "king", "pawn"]),
    "gram_case": dict(family="simplex", carrier="In Latin, the {item} case",
        items=["nominative", "accusative", "dative", "genitive", "ablative"]),
    # ---------- TREES / HYPERBOLIC ----------
    "taxonomy": dict(family="tree", carrier="A {item}",
        items=["beagle", "poodle", "dog", "wolf", "canid", "cat", "carnivore", "mammal"],
        parent=[2, 2, 4, 4, 6, 6, 7, -1]),   # beagle,poodle->dog; dog,wolf->canid; canid,cat->carnivore; carnivore->mammal(root)
    "kinship": dict(family="tree", carrier="My {item}",
        items=["father", "mother", "brother", "sister", "son", "daughter", "uncle", "aunt", "cousin"],
        parent=[-1, -1, 0, 0, 2, 2, 0, 1, 6]),
    "filepath": dict(family="tree", carrier="The directory {item}",
        items=["/usr", "/usr/local", "/usr/local/bin", "/usr/bin", "/etc", "/var", "/var/log"],
        parent=[-1, 0, 1, 0, -1, -1, 5]),
    # ---------- PRODUCTS (two independent axes) ----------
    "conjugation": dict(family="product", carrier="The verb form: {item}",
        items=["I am", "you are", "he is", "we are", "you all are", "they are"],
        factors=[[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]),   # person x number
    "nat_prof": dict(family="product", carrier="A {item}",
        items=["French chef", "French architect", "Japanese chef", "Japanese architect",
               "Brazilian chef", "Brazilian architect"],
        factors=[[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),   # nationality x profession
    "size_color": dict(family="product", carrier="A {item} object",
        items=["small red", "small blue", "large red", "large blue"],
        factors=[[0, 0], [0, 1], [1, 0], [1, 1]]),                   # size x color
    # ---------- HELICES (cycle inside a progression) ----------
    "pitch": dict(family="helix", carrier="The musical note {item}",
        items=["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4"],
        cyc_idx=[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], period=7, prog=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
    "clock": dict(family="helix", carrier="The time {item}",
        items=["Mon 6am", "Mon 12pm", "Mon 6pm", "Tue 6am", "Tue 12pm", "Tue 6pm", "Wed 6am", "Wed 12pm", "Wed 6pm"],
        cyc_idx=[0, 1, 2, 0, 1, 2, 0, 1, 2], period=3, prog=[0, 0, 0, 1, 1, 1, 2, 2, 2]),
    "digits": dict(family="helix", carrier="The number {item}",
        items=[str(x) for x in [8, 9, 10, 11, 18, 19, 20, 21, 28, 29, 30, 31]],
        cyc_idx=[8, 9, 0, 1, 8, 9, 0, 1, 8, 9, 0, 1], period=10, prog=[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
}


def candidate_dmats(sp):
    """candidate structural distance matrices available for a space, keyed by geometry name. Includes the
    RING / GRID / HEX graph structures (from the abstract-graph work) wherever the space's own coordinates
    let us build them, so we can ask: does each concept family resemble ring / grid / hex? NOTE: the
    simplex (all-equidistant) matrix is constant -> RSA undefined, measured via equidistance instead."""
    n = len(sp["items"]); C = {}
    if "order" in sp:
        C["line"] = d_line(sp["order"]); C["ring"] = d_cycle(sp["order"], n)   # arc: open line vs closed ring
    if "parent" in sp: C["tree"] = d_tree(sp["parent"])
    coord2d = None
    if "factors" in sp:
        coord2d = np.array(sp["factors"]); F = coord2d
        C["product"] = d_product(F); C["factorA_only"] = d_line(F[:, 0]); C["factorB_only"] = d_line(F[:, 1])
    if "cyc_idx" in sp:
        C["cycle"] = d_cycle(sp["cyc_idx"], sp["period"]); C["line_prog"] = d_line(sp["prog"])
        C["helix"] = d_helix(sp["cyc_idx"], sp["period"], sp["prog"])
        coord2d = np.column_stack([sp["cyc_idx"], sp["prog"]])
    if coord2d is not None:                                    # 2D spaces: compare to GRID vs HEX lattices
        C["grid"] = d_grid(coord2d); C["hex"] = d_hex(coord2d)
    return C
