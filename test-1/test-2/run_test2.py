"""test-2 runner: ground-truthed coupled learning with an unreliable partner.

Per condition (see t2_core for the construction):
  1. A prefilled with CTX walk tokens on the CORE (its own data is exactly
     uninformative about the contested matchings).
  2. Exchange, TGEN alternating steps (B even, A odd), every emitted token appended
     to every live context (q=1):
       scripted_rho<r>  B = corrupted ground-truth walker on G* (process corruption
                        at rho; exact channel model -> exact pooled posterior valid).
       llmB_rho<r>      B = second context prefilled with corrupted G* walks
                        (ecological; channel model approximate, B validity logged).
       noex             no exchange (TGEN=0): behavioural floor.
  3. Probes at CKPTS: for every cue node, fresh forward over [BOS]+A-context+cue,
     16-way restricted softmax (pre-top-k, TEMP) -> probes.npz
     [n_ckpt, npairs, 16 cues, 16].

DRY=1 replaces the LLM with the Dirichlet-Markov mock (partner-blind null) and runs
the identical control flow — used both for pipeline validation and as the
source-tracking reference curves in the analysis.

Env: OUT(runs) COND(comma list; default full suite) CTX(600) TGEN(400) NPAIRS(6)
     TEMP(1.0) TOPK(0) SEED(0) CKPTS(0,100,200,400) DRY(0) DEVICE(cuda) CM_SRC
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_core as T

OUT = os.environ.get("OUT", os.path.join(HERE, "runs"))
CTX = int(os.environ.get("CTX", "600"))
TGEN = int(os.environ.get("TGEN", "400"))
NPAIRS = int(os.environ.get("NPAIRS", "6"))
TEMP = float(os.environ.get("TEMP", "1.0"))
TOPK = int(os.environ.get("TOPK", "0"))
SEED = int(os.environ.get("SEED", "0"))
DRY = os.environ.get("DRY", "0") == "1"
DEVICE = os.environ.get("DEVICE", "cuda")
CKPTS = [int(x) for x in os.environ.get("CKPTS", f"0,100,200,{TGEN}").split(",")]
DEFAULT_CONDS = ("noex,scripted_rho0.0,scripted_rho0.15,scripted_rho0.3,"
                 "scripted_rho0.5,llmB_rho0.0,llmB_rho0.3")
CONDS = [c for c in os.environ.get("COND", DEFAULT_CONDS).split(",") if c]
MODEL_CANDS = ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]
N = T.N


class LLMBackend:
    """KV-cache contexts keyed by side; fresh-forward probes (stream-deterministic)."""

    def __init__(self):
        import torch
        from dataclasses import replace
        for cand in (os.environ.get("CM_SRC"),
                     os.path.join(HERE, "..", "..", "cross-model", "src"),
                     os.path.join(HERE, "..", "cmsrc")):
            if cand and os.path.isfile(os.path.join(cand, "models.py")):
                sys.path.insert(0, cand); break
        from config import get_config
        import models as M
        self.torch = torch
        cfg = replace(get_config("gemma_qwen"), dtype="bfloat16", device=DEVICE)
        self.model = self.tok = None
        for nm in MODEL_CANDS:
            try:
                self.model, self.tok = M.load_model(nm, cfg); break
            except Exception as e:
                print(f"failed {nm}: {e}", flush=True)
        self.bos = self.tok.bos_token_id
        self.cand = [self.tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in T.WORDS]
        self.cand_t = torch.tensor(self.cand, device=DEVICE)
        self.state = {}

    @property
    def no_grad(self):
        return self.torch.no_grad

    def prefill(self, key, node_rows):
        tt = self.torch
        ids = tt.tensor([[self.bos] + [self.cand[x] for x in r] for r in node_rows],
                        device=DEVICE)
        with tt.no_grad():
            try:
                o = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            except TypeError:
                o = self.model(input_ids=ids, use_cache=True)
        self.state[key] = {"past": o.past_key_values, "logits": o.logits[:, -1, :]}

    def predictive_all(self, key):
        lg = self.state[key]["logits"][:, self.cand_t].float()
        return self.torch.softmax(lg / TEMP, -1).cpu().numpy()

    def step(self, key, nodes):
        tt = self.torch
        toks = tt.tensor([self.cand[x] for x in nodes], device=DEVICE)[:, None]
        with tt.no_grad():
            o = self.model(input_ids=toks, past_key_values=self.state[key]["past"],
                           use_cache=True)
        self.state[key]["past"] = o.past_key_values
        self.state[key]["logits"] = o.logits[:, -1, :]

    def probe_all_cues(self, context_nodes):
        """Fresh forward, batch = 16 cues appended to the same context. [16,16]."""
        tt = self.torch
        base = [self.bos] + [self.cand[x] for x in context_nodes]
        ids = tt.tensor([base + [self.cand[cue]] for cue in range(N)], device=DEVICE)
        with tt.no_grad():
            try:
                o = self.model(input_ids=ids, logits_to_keep=1)
            except TypeError:
                o = self.model(input_ids=ids)
        lg = o.logits[:, -1, self.cand_t].float()
        return tt.softmax(lg / TEMP, -1).cpu().numpy()


class MockLLM:
    """Adapter giving t2_core.MockBackend the same interface as LLMBackend."""

    def __init__(self):
        self.mb = T.MockBackend(temp=TEMP)
        self.npairs = {}

    def prefill(self, key, node_rows):
        self.mb.prefill(key, node_rows)
        self.npairs[key] = len(node_rows)

    def predictive_all(self, key):
        return np.stack([self.mb.predictive(key, p) for p in range(self.npairs[key])])

    def step(self, key, nodes):
        for p, x in enumerate(nodes):
            self.mb.append(key, p, x)

    def probe_all_cues_state(self, key, p):
        return np.stack([self.mb.probe(key, p, cue) for cue in range(N)])


def parse_cond(cond):
    if cond == "noex":
        return "noex", 0.0
    kind, r = cond.split("_rho")
    return kind, float(r)


def run_condition(cond, spec, backend):
    kind, rho = parse_cond(cond)
    out = os.path.join(OUT, cond)
    os.makedirs(out, exist_ok=True)
    core = T.spec_adj(spec, "core_edges")
    Astar = T.spec_adj(spec, "gstar_edges")
    tgen = 0 if kind == "noex" else TGEN
    ckpts = sorted({c for c in CKPTS if c <= tgen})

    a_rows = [T.walk(core, CTX, seed=SEED * 1000 + p) for p in range(NPAIRS)]
    backend.prefill("A", a_rows)
    if kind == "llmB":
        b_rows, b_flags = zip(*[T.corrupt_walk(Astar, CTX, rho,
                                               seed=SEED * 1000 + 500 + p)
                                for p in range(NPAIRS)])
        backend.prefill("B", list(b_rows))
    rng = np.random.default_rng(SEED + 1)
    rng_b = [np.random.default_rng(SEED * 77 + 13 + p) for p in range(NPAIRS)]

    stream = [[] for _ in range(NPAIRS)]           # (who, node, corrupt_flag)
    joint_prev = [r[-1] for r in a_rows]
    probes = np.zeros((len(ckpts), NPAIRS, N, N), np.float32)
    b_valid = []

    def take_probes(ci):
        for p in range(NPAIRS):
            ctx_nodes = a_rows[p] + [x for (_, x, _) in stream[p]]
            if isinstance(backend, MockLLM):
                probes[ci, p] = backend.probe_all_cues_state("A", p)
            else:
                probes[ci, p] = backend.probe_all_cues(ctx_nodes)

    def draw(pd):
        pp = pd.copy()
        if TOPK > 0:
            pp[np.argsort(pp)[:-TOPK]] = 0.0
        return int(rng.choice(N, p=pp / pp.sum()))

    ci = 0
    for t in range(tgen + 1):
        if ci < len(ckpts) and t == ckpts[ci]:
            take_probes(ci); ci += 1
        if t == tgen:
            break
        who = "B" if t % 2 == 0 else "A"
        if who == "B" and kind == "scripted":
            emits, flags = [], []
            for p in range(NPAIRS):
                if rng_b[p].random() < rho:
                    x, fl = int(rng_b[p].integers(N)), True
                else:
                    x = int(rng_b[p].choice(np.where(Astar[joint_prev[p]])[0]))
                    fl = False
                emits.append(x); flags.append(fl)
        else:
            pd = backend.predictive_all(who)
            emits = [draw(pd[p]) for p in range(NPAIRS)]
            flags = [False] * NPAIRS
        if who == "B":
            b_valid.append(float(np.mean([Astar[joint_prev[p], emits[p]]
                                          for p in range(NPAIRS)])))
        for p in range(NPAIRS):
            stream[p].append((who, emits[p], bool(flags[p])))
            joint_prev[p] = emits[p]
        backend.step("A", emits)
        if kind == "llmB":
            backend.step("B", emits)

    np.savez_compressed(os.path.join(out, "probes.npz"),
                        probes=probes, ckpts=np.array(ckpts))
    json.dump({"cond": cond, "kind": kind, "rho": rho, "ctx": CTX, "tgen": tgen,
               "npairs": NPAIRS, "temp": TEMP, "topk": TOPK, "seed": SEED,
               "dry": DRY, "ckpts": ckpts,
               "b_valid_mean": (float(np.mean(b_valid)) if b_valid else None),
               "stream": [[(w, int(x), f) for (w, x, f) in s] for s in stream],
               "a_prefix": [list(map(int, r)) for r in a_rows],
               **({"b_prefix": [list(map(int, r)) for r in b_rows],
                   "b_prefix_corrupt": [list(map(bool, f)) for f in b_flags]}
                  if kind == "llmB" else {})},
              open(os.path.join(out, "stream.json"), "w"))
    print(f"COND DONE {cond} (b_valid={np.mean(b_valid) if b_valid else float('nan'):.3f})",
          flush=True)


def main():
    spec_path = os.path.join(OUT, "t2_spec.json")
    if os.path.exists(spec_path):
        spec = json.load(open(spec_path))
    else:
        spec = T.build_spec(SEED)
        T.save_spec(spec, spec_path)
    t0 = time.time()
    llm = None if DRY else LLMBackend()                # load weights once
    for cond in CONDS:
        backend = MockLLM() if DRY else llm            # prefill resets contexts
        run_condition(cond, spec, backend)
        print(f"  elapsed {time.time() - t0:.0f}s", flush=True)
    print("TEST2 DONE", flush=True)


if __name__ == "__main__":
    main()
