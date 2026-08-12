"""Full-pipeline test on CPU with a fake tokenizer + randomly-initialised tiny model.

Checks the machinery, never the science: prompt assembly, offset->token-index
resolution for every read anchor and every steering position group, hook placement,
vector shapes and split halves, the whole compare/steer analysis path, and that no
stage silently produces empty position lists (the failure mode that makes a steering
arm look like a null result when it simply never wrote anything).

  python src/mock_test.py
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

V, D, NL = 4096, 32, 6


class FakeTok:
    """Whitespace/punctuation tokenizer with real offset mapping."""

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False,
                 add_special_tokens=True):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        ids = [hash(text[a:b]) % V for a, b in spans]
        out = {"input_ids": torch.tensor([ids]),
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([spans]) if spans else torch.zeros(1, 0, 2)
        return out

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        s = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
        return s + "<|assistant|>\n"


class Block(nn.Module):
    """MLP + a causal running-mean 'attention' so earlier positions reach later ones.

    Without token mixing an injection at the partner's name could never move the
    answer slot, and every steering arm would test nothing.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.mix = nn.Linear(D, D)

    def forward(self, h):
        n = h.shape[1]
        cum = h.cumsum(1) / torch.arange(1, n + 1, device=h.device).view(1, n, 1)
        return h + torch.tanh(self.lin(h)) + torch.tanh(self.mix(cum))


class FakeModel(nn.Module):
    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(NL))

    class _Cfg:
        num_hidden_layers = NL

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.model = self._Inner()
        self.head = nn.Linear(D, V)
        self.config = self._Cfg()

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False,
                **kw):
        h = self.emb(input_ids)
        hs = [h]
        for blk in self.model.layers:
            h = blk(h)
            hs.append(h)
        out = type("O", (), {})()
        out.logits = self.head(h)
        out.hidden_states = tuple(hs) if output_hidden_states else None
        return out


def main():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="tv_mock_")
    os.environ.update(OUT=tmp, LAYERS="2,4", NPAIR="4", ALPHA="0.25", VALIDATE="1",
                      PLOT="0", STAGES="grid,curve", GAMES="pd,labels",
                      SCHEDULES="one_lapse", STYLES="unconditional")
    model, tok = FakeModel().eval(), FakeTok()

    import build_vectors
    import common
    import compare
    import qsg_games as G
    import steer_qsg
    import stimuli as S
    build_vectors.load = steer_qsg.load = lambda *a, **k: (model, tok, False)

    # --- position-group sanity: every anchor and every arm must be non-empty ---
    for method in S.ALL:
        sysmsg, prefill, anchors = S.SPEC[method]
        p, n, m = S.pairs(method, 1)[0]
        txt = common.chat(tok, sysmsg, p, m.get("prefill_pos", prefill))
        for a in anchors:
            if a == "last":
                continue
            sp = S.anchor_spans(method, txt, m["name"]).get(a)
            idx = common.tok_idx(tok, txt, sp)
            assert idx, f"{method}/{a}: empty anchor"
    print("[ok] every read anchor resolves to >=1 token")

    for g in G.GAMES:
        ex = G.build(g, "one_lapse", "conditional")
        txt = common.chat(tok, G.SYS, ex["user"], ex["prefill"])
        pos = G.positions(tok, txt, ex)
        for k in ("partner_all", "partner_hist", "partner_cur", "self_all", "answer"):
            assert pos[k], f"{g}: empty position group {k}"
        assert set(pos["partner_cur"]).isdisjoint(pos["partner_hist"])
        assert set(pos["partner_all"]).isdisjoint(pos["self_all"])
    print("[ok] every game's steering position groups are non-empty and disjoint")
    G.check_tokens(tok)
    print("[ok] all five games have distinct first tokens for their two actions")

    # --- the hook actually writes where it says it does ---------------------
    ex = G.build("pd", "one_lapse", "unconditional")
    txt = common.chat(tok, G.SYS, ex["user"], ex["prefill"])
    pos = G.positions(tok, txt, ex)
    m0 = steer_qsg.read_margin(model, tok, txt, ex)
    with common.Inject(model, 4, torch.zeros(D), pos["partner_all"]):
        assert abs(steer_qsg.read_margin(model, tok, txt, ex) - m0) < 1e-5
    with common.Inject(model, 4, torch.ones(D) * 5.0, pos["partner_all"]):
        assert abs(steer_qsg.read_margin(model, tok, txt, ex) - m0) > 1e-4
    print("[ok] Inject is a no-op at v=0 and moves the read-out at v!=0")

    print("\n--- build_vectors ---")
    build_vectors.main()
    print("\n--- compare ---")
    compare.main()
    print("\n--- steer_qsg ---")
    steer_qsg.main()
    print("\n--- residualize ---")
    import residualize
    os.environ["WRITE"] = "1"
    residualize.main()
    print("\n--- dissociate ---")
    import dissociate
    dissociate.load = lambda *a, **k: (model, tok, False)
    for nm, txt in [dissociate.build_A(0)[:2], dissociate.build_B(0)]:
        t = common.chat(tok, dissociate.SYS, txt, "")
        assert dissociate.name_pos(tok, t, nm), f"empty name positions: {nm}"
    assert common.first_id(tok, "accept") != common.first_id(tok, "count")
    assert common.first_id(tok, "yes") != common.first_id(tok, "no")
    os.environ["NITEM"] = "3"
    dissociate.main()
    for f in ("vectors.npz", "vectors_meta.json", "validation.json", "compare.json",
              "steer_qsg.json", "residualize_last.json", "dissociate.json"):
        assert os.path.exists(os.path.join(tmp, f)), f
    print(f"\n[ok] all artifacts written; MOCK_OK  ({tmp})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
