"""Is `optim` secretly the ' trust' token direction? cos against input
embedding, unembedding row, unembedding MARGIN (' trust'-' don' -- the exact
optimization objective), and the hidden state of the bare token pushed through
the layers. Same for optim_like with ' like'/' dis'. Random-token rows give the
scale of incidental overlap."""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import first_id, load, unit

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def main():
    model, tok, _ = load(); model.eval()
    nv = json.load(open(os.path.join(OUT, "newvecs.json")))
    E = model.get_input_embeddings().weight.detach().float()
    U = model.lm_head.weight.detach().float()
    rng = np.random.RandomState(7)
    rand_ids = rng.randint(0, E.shape[0], 40)

    def hidden_states_of(word):
        enc = tok(word, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        return {l: hs[l][0, -1].float().cpu().numpy() for l in (45, 52)}

    for name, pos_w, neg_w in (("optim", " trust", " don"),
                               ("optim_like", " like", " dis")):
        pid, nid = first_id(tok, pos_w), first_id(tok, neg_w)
        hp, hn = hidden_states_of(pos_w), hidden_states_of(neg_w)
        for L in (45, 52):
            if f"L{L}" not in nv[name]:
                continue
            v = unit(np.array(nv[name][f"L{L}"]))
            e = unit(E[pid].cpu().numpy())
            u = unit(U[pid].cpu().numpy())
            um = unit((U[pid] - U[nid]).cpu().numpy())
            hh = unit(hp[L]); hd = unit(hp[L] - hn[L])
            re_ = float(np.mean([abs(v @ unit(U[i].cpu().numpy())) for i in rand_ids]))
            print(f"[{name} L{L}] cos emb({pos_w!r}) {v@e:+.3f}  "
                  f"unemb({pos_w!r}) {v@u:+.3f}  unembMARGIN {v@um:+.3f}  "
                  f"hidden@L {v@hh:+.3f}  hiddenDIFF {v@hd:+.3f}  "
                  f"|rand-tok unemb| {re_:.3f}", flush=True)
    print("TOKENCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
