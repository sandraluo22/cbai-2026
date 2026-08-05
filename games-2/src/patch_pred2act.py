"""PREDICTION -> ACTION frame patching at dissociation snapshots (2026-08-02).

Routing experiment: the same reconstructed loop state is rendered in two frames with
the SAME shuffled 8-candidate listing —
  prediction: "...which of these words do you expect the other player to say next
              round: <list>? Answer with one word."  -> "\nAnswer:"
  action:     "...you must choose your word from this list: <list>."  -> "\nMy word:"

We run the prediction frame, cache the residual stream (block outputs) at the final
position, then re-run the ACTION frame with layer-L residual replaced by the
prediction one, and read the renormalized 8-candidate first-token category mass.

If LATE-layer patches transfer the ~0.88 prediction-frame category preference into
the action choice, the knowledge-to-policy routing failure is localized late; if
EARLY patches work, the frame changes the inferred representation itself.

Env: MODEL(QwenInst32) SWEEP(runs/stuck_repro/stuck_repro_QwenInst32_transcript.jsonl)
     START_FILE N_SNAPS(24) RUN_DIR(runs/patch_pred2act)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import dissociation_branches as D

MODEL = os.environ.get("MODEL", "QwenInst32")
SWEEP = os.environ.get("SWEEP", "runs/stuck_repro/stuck_repro_QwenInst32_transcript.jsonl")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_SNAPS = int(os.environ.get("N_SNAPS", "24"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/patch_pred2act")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    blocks = model.model.layers
    nL = len(blocks)

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    snaps = D.snapshots_from_sweep(SWEEP, starts)[:N_SNAPS]
    print(f"[p2a] {len(snaps)} snapshots", flush=True)

    rng = np.random.default_rng(0)
    per_layer = np.zeros(nL)
    rows = []
    for si, s in enumerate(snaps):
        base = D.BASE0 if s["pv"] == 0 else D.BASE1
        cin, cout = D.CANDS[s["cat"]]
        cin = [w for w in cin if w not in s["used"]]
        cout = [w for w in cout if w not in s["used"]]
        order = list(cin) + list(cout)
        order = [order[i] for i in rng.permutation(len(order))]
        listing = ", ".join(order)
        ctx = D.build_A(tok, base, s["histA"], set(s["used"]))
        p_pred = LA._render(tok, ctx + f" Question: which of these words do you expect "
                            f"the other player to say next round: {listing}? Answer "
                            f"with one word.") + "\nAnswer:"
        p_act = LA._render(tok, ctx + f" For this round, you must choose your word from "
                           f"this list: {listing}.") + "\nMy word:"
        ids_in = torch.tensor([fid(w) for w in cin])
        ids_out = torch.tensor([fid(w) for w in cout])

        def pcat(logits):
            z = torch.softmax(torch.cat([logits[ids_in], logits[ids_out]]).float(), 0)
            return float(z[:len(ids_in)].sum())

        with torch.no_grad():
            enc_p = tok(p_pred, return_tensors="pt").to(dev)
            out_p = model(enc_p.input_ids, output_hidden_states=True)
            pred_resid = [h[0, -1].detach().clone() for h in out_p.hidden_states]
            pcat_pred = pcat(out_p.logits[0, -1])
            enc_a = tok(p_act, return_tensors="pt").to(dev)
            pcat_act = pcat(model(enc_a.input_ids).logits[0, -1])

        layer_vals = []
        for L in range(nL):
            repl = pred_resid[L + 1]

            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1] = repl.to(h.dtype)
                return out
            hd = blocks[L].register_forward_hook(hook)
            with torch.no_grad():
                lg = model(enc_a.input_ids).logits[0, -1]
            hd.remove()
            layer_vals.append(pcat(lg))
        per_layer += np.array(layer_vals) / len(snaps)
        rows.append({"snap": si, "cat": s["cat"], "pcat_pred": pcat_pred,
                     "pcat_act": pcat_act, "pcat_patched_by_layer": layer_vals})
        print(f"[p2a] snap {si} ({s['cat']}): pred {pcat_pred:.2f} act {pcat_act:.2f} "
              f"best patched {max(layer_vals):.2f} @L{int(np.argmax(layer_vals))}", flush=True)
        json.dump({"per_snapshot": rows}, open(os.path.join(RUN_DIR, "pred2act.json"), "w"))

    mp = float(np.mean([r["pcat_pred"] for r in rows]))
    ma = float(np.mean([r["pcat_act"] for r in rows]))
    json.dump({"per_snapshot": rows,
               "summary": {"mean_pcat_pred": mp, "mean_pcat_act": ma,
                           "mean_pcat_patched_by_layer": per_layer.tolist(),
                           "best_layer": int(np.argmax(per_layer)),
                           "best_layer_pcat": float(per_layer.max())}},
              open(os.path.join(RUN_DIR, "pred2act.json"), "w"), indent=1)
    print(f"[p2a] === pred {mp:.2f} act {ma:.2f} best layer L{int(np.argmax(per_layer))} "
          f"patched {per_layer.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
