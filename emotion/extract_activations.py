"""Generate responses with Qwen3-8B on GoEmotions and extract residual-stream
activations at three anchors, every layer.

For each GoEmotions example we treat the example text as the user turn Q and let
the model generate a response A (non-thinking mode). We then save, at EVERY layer
(embeddings + each transformer block = num_hidden_layers + 1):

  Q  : residual stream at the assistant anchor (the token right after the
       `<|im_start|>assistant\\n` header) of the prompt for Q. This is the Qwen
       equivalent of "the ':' after Assistant" — the position where the model is
       about to answer Q.
  A1 : residual stream at the END of the model's generated response (last
       non-eos response token), within the full Q+A context.
  A2 : feed the response text A ONLY back in as a fresh user turn, and hook the
       residual stream at the same assistant anchor as Q.

Outputs (under results/<run>/):
  q_acts.dat, a1_acts.dat, a2_acts.dat   float16 memmaps, shape (N, L, H)
  meta.json                              shapes + model + config
  examples.jsonl                         per-example text, response, labels
  labels_primary.npy, labels_ekman.npy   int16 color labels for plotting

Usage:
  # smoke test (stratified subset of train)
  python extract_activations.py --limit 200
  # full dataset
  python extract_activations.py --split all --limit 0 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goemotions_utils import (
    GOEMOTIONS_LABELS, FINE_TO_EKMAN, EKMAN_NAMES,
    primary_label, stratified_indices, load_goemotions,
)

ASSIST_MARKER = "<|im_start|>assistant\n"


def pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def render_prompt(tok, user_text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def anchor_index(tok, prompt_str: str) -> int:
    """Index of the token right after the assistant header (before any think
    block) — our '":" after Assistant' anchor."""
    cut = prompt_str.rindex(ASSIST_MARKER) + len(ASSIST_MARKER)
    return len(tok(prompt_str[:cut], add_special_tokens=False).input_ids) - 1


def strip_response(ids: torch.Tensor, eos_ids: set[int], pad_id: int) -> list[int]:
    """Drop trailing pad/eos from a generated id row; keep at least one token."""
    out = ids.tolist()
    while len(out) > 1 and (out[-1] == pad_id or out[-1] in eos_ids):
        out.pop()
    return out


@torch.no_grad()
def gather_layers(hidden_states, idx: torch.Tensor) -> torch.Tensor:
    """hidden_states: tuple of L tensors (B,T,H). idx: (B,) per-row token index.
    Returns (B, L, H) on cpu float16."""
    B = idx.shape[0]
    rows = torch.arange(B, device=idx.device)
    layers = [hs[rows, idx] for hs in hidden_states]      # each (B,H)
    return torch.stack(layers, dim=1).to("cpu", torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test", "all"])
    ap.add_argument("--limit", type=int, default=200,
                    help="stratified subset size; 0 = use the whole split")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="results subdir name")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partially-finished run in --out/run dir")
    args = ap.parse_args()

    device, dtype = pick_device_dtype()
    run_name = args.out or f"{args.split}_{'full' if args.limit == 0 else args.limit}"
    out_dir = Path(__file__).parent / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} dtype={dtype} out={out_dir}")

    # ---- dataset ----
    ds = load_goemotions(args.split)
    # sanity: HF label order must match our hardcoded names
    feat_names = ds.features["labels"].feature.names
    assert feat_names == GOEMOTIONS_LABELS, "GoEmotions label order mismatch!"
    primary = np.array([primary_label(r) for r in ds["labels"]], dtype=np.int16)
    sel = stratified_indices(primary, args.limit, seed=args.seed)
    ds = ds.select(sel.tolist())
    texts = ds["text"]
    primary = primary[sel]
    ekman = FINE_TO_EKMAN[primary].astype(np.int16)
    N = len(texts)
    print(f"[data] {N} examples ({args.split}); "
          f"{len(np.unique(primary))} primary classes present")

    # ---- model ----
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device)
    model.eval()
    eos_ids = set(filter(lambda x: x is not None,
                  [tok.eos_token_id,
                   tok.convert_tokens_to_ids("<|im_end|>")]))
    L = model.config.num_hidden_layers + 1
    H = model.config.hidden_size
    print(f"[model] {args.model}  L={L} layers  H={H}")

    # ---- meta + labels (write up front so plotting/resume can rely on them) ----
    np.save(out_dir / "labels_primary.npy", primary)
    np.save(out_dir / "labels_ekman.npy", ekman)
    meta = {
        "model": args.model, "split": args.split, "limit": args.limit,
        "N": N, "L": L, "H": H, "dtype": "float16",
        "max_new_tokens": args.max_new_tokens, "enable_thinking": False,
        "anchor": "token after '<|im_start|>assistant\\n' header",
        "fine_labels": GOEMOTIONS_LABELS, "ekman_labels": EKMAN_NAMES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- streaming activation files ----
    # We write raw float16 in (N, L, H) C-order via ordinary buffered I/O (NOT
    # np.memmap): mmap-writing to the FUSE /workspace volume balloons dirty page
    # cache / SIGBUSes the process. Buffered writes are bounded-memory and safe.
    rec = L * H * 2  # bytes per example per array (float16)
    qp, a1p, a2p = (out_dir / f"{n}_acts.dat" for n in ("q", "a1", "a2"))
    exp = out_dir / "examples.jsonl"
    B = args.batch_size

    done = 0
    if args.resume and exp.exists():
        # align resume point to the min consistent example count across all files
        n_lines = sum(1 for _ in open(exp))
        done = min(n_lines, *(os.path.getsize(p) // rec for p in (qp, a1p, a2p)))
        done -= done % B  # batch boundary
        for p in (qp, a1p, a2p):
            with open(p, "r+b") as f:
                f.truncate(done * rec)
        # rewrite examples.jsonl to exactly `done` lines
        lines = open(exp).read().splitlines()[:done]
        open(exp, "w").write("".join(l + "\n" for l in lines))
        print(f"[resume] continuing from example {done}/{N}")

    mode = "r+b" if done else "wb"
    q_f, a1_f, a2_f = open(qp, mode), open(a1p, mode), open(a2p, mode)
    for f in (q_f, a1_f, a2_f):
        f.seek(done * rec)
    ex_f = open(exp, "a" if done else "w")

    for start in range(done, N, B):
        end = min(start + B, N)
        batch_texts = list(texts[start:end])
        bs = len(batch_texts)

        # 1) generate A for Q (non-thinking, greedy)
        q_prompts = [render_prompt(tok, t) for t in batch_texts]
        tok.padding_side = "left"
        enc = tok(q_prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        resp = gen[:, enc.input_ids.shape[1]:]
        resp_ids = [strip_response(resp[i], eos_ids, tok.pad_token_id) for i in range(bs)]
        resp_text = [tok.decode(r, skip_special_tokens=True).strip() for r in resp_ids]

        # 2) Q+A forward -> Q anchor acts + A1 (response-end) acts
        tok.padding_side = "right"
        q_prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in q_prompts]
        q_anchor = [anchor_index(tok, p) for p in q_prompts]
        full_ids = [q_prompt_ids[i] + resp_ids[i] for i in range(bs)]
        a1_idx = [len(full_ids[i]) - 1 for i in range(bs)]
        maxlen = max(len(f) for f in full_ids)
        inp = torch.full((bs, maxlen), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((bs, maxlen), dtype=torch.long)
        for i, f in enumerate(full_ids):
            inp[i, :len(f)] = torch.tensor(f)
            att[i, :len(f)] = 1
        inp, att = inp.to(device), att.to(device)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=att,
                        output_hidden_states=True, use_cache=False)
        q_acts = gather_layers(out.hidden_states,
                               torch.tensor(q_anchor, device=device))
        a1_acts = gather_layers(out.hidden_states,
                                torch.tensor(a1_idx, device=device))

        # 3) A2: response text ONLY as a fresh user turn, hook assistant anchor
        a2_prompts = [render_prompt(tok, rt if rt else " ") for rt in resp_text]
        a2_ids = [tok(p, add_special_tokens=False).input_ids for p in a2_prompts]
        a2_anchor = [anchor_index(tok, p) for p in a2_prompts]
        maxlen2 = max(len(f) for f in a2_ids)
        inp2 = torch.full((bs, maxlen2), tok.pad_token_id, dtype=torch.long)
        att2 = torch.zeros((bs, maxlen2), dtype=torch.long)
        for i, f in enumerate(a2_ids):
            inp2[i, :len(f)] = torch.tensor(f)
            att2[i, :len(f)] = 1
        inp2, att2 = inp2.to(device), att2.to(device)
        with torch.no_grad():
            out2 = model(input_ids=inp2, attention_mask=att2,
                         output_hidden_states=True, use_cache=False)
        a2_acts = gather_layers(out2.hidden_states,
                                torch.tensor(a2_anchor, device=device))

        # 4) persist — streaming append, ascontiguousarray guards C-order
        q_f.write(np.ascontiguousarray(q_acts.numpy()).tobytes())
        a1_f.write(np.ascontiguousarray(a1_acts.numpy()).tobytes())
        a2_f.write(np.ascontiguousarray(a2_acts.numpy()).tobytes())
        for i in range(bs):
            ex_f.write(json.dumps({
                "idx": start + i,
                "text": batch_texts[i],
                "response": resp_text[i],
                "primary": int(primary[start + i]),
                "primary_name": GOEMOTIONS_LABELS[int(primary[start + i])],
                "ekman": int(ekman[start + i]),
                "ekman_name": EKMAN_NAMES[int(ekman[start + i])],
            }) + "\n")
        # flush per batch so progress is durable for --resume
        for f in (q_f, a1_f, a2_f, ex_f):
            f.flush()
        print(f"[run] {end}/{N}")

    for f in (q_f, a1_f, a2_f, ex_f):
        f.close()
    print(f"[done] wrote activations + meta to {out_dir}")


if __name__ == "__main__":
    main()
