"""NATURAL-TASK ROLE OF THE READER HEADS (2026-08-07): what do the 32
restoration ("series reader") heads do in ordinary language use?

Battery (no game content anywhere), each under 4 ablation conditions
(none / top32_rest / top32_ind / rand32):

  themed_list   "Shopping list: apples, bananas, cherries, grapes," -> sample 16
                next items; score IN-THEME fraction. 6 themed lists (fruit,
                tools, cities, animals, clothing, vegetables). Tests semantic
                series continuation — the hypothesized day job.
  morph_analogy "walk walked, jump jumped, climb" -> greedy; accuracy on 8
                items. Tests morphological pattern completion.
  copy          a sentence appears once, then is re-started; greedy next word
                scored against the original continuation; 6 items. Classic
                induction-style verbatim copying — induction ablation should
                hurt THIS, reader ablation should not (double dissociation).
  fluency       mean per-token logprob of two fixed neutral paragraphs.

Env: MODEL(QwenInst32) PATCH_JSON IND_JSON K(16) TEMP(0.7) SEED(0)
     RUN_DIR(runs/natural_heads)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON", "runs/mech_inputs/qwen32_partner_patch.json")
IND_JSON = os.environ.get("IND_JSON", "runs/mech_inputs/qwen32_induction_overlap.json")
K = int(os.environ.get("K", "16"))
TEMP = float(os.environ.get("TEMP", "0.7"))
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/natural_heads")

LISTS = [
    ("Shopping list: apples, bananas, cherries, grapes,",
     {"orange", "oranges", "pear", "pears", "peach", "peaches", "plum", "plums", "mango",
      "mangoes", "melon", "melons", "kiwi", "kiwis", "strawberries", "strawberry",
      "blueberries", "blueberry", "lemon", "lemons", "lime", "limes", "pineapple",
      "apricot", "apricots", "figs", "grapefruit", "raspberries", "watermelon"}),
    ("Toolbox inventory: hammer, screwdriver, wrench, pliers,",
     {"saw", "drill", "chisel", "level", "tape", "file", "clamp", "clamps", "vise",
      "mallet", "ruler", "sander", "socket", "sockets", "ratchet", "hacksaw", "awl",
      "square", "crowbar", "scissors", "knife", "utility"}),
    ("Cities visited this year: Paris, Tokyo, Berlin, Madrid,",
     {"london", "rome", "vienna", "amsterdam", "prague", "lisbon", "athens", "dublin",
      "moscow", "sydney", "beijing", "seoul", "bangkok", "barcelona", "budapest",
      "warsaw", "oslo", "stockholm", "copenhagen", "helsinki", "zurich", "geneva",
      "venice", "florence", "munich", "hamburg", "chicago", "boston", "toronto",
      "istanbul", "cairo", "dubai", "singapore", "shanghai", "osaka", "kyoto"}),
    ("Animals seen at the zoo: elephant, giraffe, zebra, lion,",
     {"tiger", "tigers", "monkey", "monkeys", "bear", "bears", "hippo", "hippos",
      "rhino", "rhinos", "leopard", "cheetah", "gorilla", "gorillas", "panda",
      "pandas", "wolf", "wolves", "fox", "camel", "camels", "kangaroo", "koala",
      "penguin", "penguins", "flamingo", "flamingos", "crocodile", "ostrich",
      "antelope", "gazelle", "hyena", "baboon", "chimpanzee", "lemur", "meerkat"}),
    ("Packing for the trip: shirts, trousers, socks, jackets,",
     {"shoes", "boots", "hat", "hats", "gloves", "scarf", "scarves", "sweater",
      "sweaters", "underwear", "belt", "belts", "coat", "coats", "pajamas", "shorts",
      "dresses", "skirts", "sandals", "slippers", "ties", "tie", "raincoat",
      "swimsuit", "toiletries", "towel", "towels", "umbrella"}),
    ("Garden harvest: carrots, potatoes, onions, tomatoes,",
     {"lettuce", "cabbage", "peppers", "pepper", "cucumbers", "cucumber", "beans",
      "peas", "corn", "squash", "pumpkin", "pumpkins", "beets", "radishes", "garlic",
      "spinach", "kale", "broccoli", "cauliflower", "celery", "leeks", "turnips",
      "zucchini", "eggplant", "herbs", "parsley"}),
]
ANALOGY = [
    ("walk walked, jump jumped, climb", "climbed"),
    ("cat cats, dog dogs, horse", "horses"),
    ("run runner, teach teacher, paint", "painter"),
    ("big bigger, small smaller, fast", "faster"),
    ("sing sang, ring rang, drink", "drank"),
    ("happy happiness, sad sadness, dark", "darkness"),
    ("write writing, swim swimming, read", "reading"),
    ("child children, mouse mice, tooth", "teeth"),
]
COPY = [
    ("The committee approved the new budget after a long debate on Tuesday evening. "
     "Later, the minutes recorded that the committee approved the new budget after a long",
     "debate"),
    ("A silver train crossed the frozen river just before dawn on the last day of winter. "
     "The photograph showed how a silver train crossed the frozen river just before",
     "dawn"),
    ("Maria planted three rows of sunflowers along the eastern fence of the old farm. "
     "Neighbors still remember that Maria planted three rows of sunflowers along the eastern",
     "fence"),
    ("The museum reopened its sculpture wing with a quiet ceremony and free admission. "
     "According to the paper, the museum reopened its sculpture wing with a quiet",
     "ceremony"),
    ("Heavy rain flooded the northern car park before the morning shift arrived. "
     "The report confirmed that heavy rain flooded the northern car park before the morning",
     "shift"),
    ("The orchestra rehearsed the final movement twice before the conductor was satisfied. "
     "Her diary notes that the orchestra rehearsed the final movement twice before the",
     "conductor"),
]
PARAS = [
    ("The library extended its opening hours this autumn, responding to requests from "
     "students preparing for examinations. Staff reported that the reading rooms were "
     "busiest between seven and ten in the evening, and that demand for quiet desks "
     "exceeded supply on most weekdays."),
    ("Engineers completed the inspection of the harbour bridge ahead of schedule. The "
     "report noted minor corrosion on two support cables but concluded that the "
     "structure remained safe for normal traffic while repairs were planned for the "
     "spring maintenance window."),
]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    order = np.argsort(R.flatten())[::-1]
    top_rest = [(int(i // nH), int(i % nH)) for i in order[:32]]
    I = np.array(json.load(open(IND_JSON))["induction"])
    iorder = np.argsort(I.flatten())[::-1]
    top_ind = [(int(i // I.shape[1]), int(i % I.shape[1])) for i in iorder[:32]]
    rng = np.random.default_rng(SEED)
    excl = set(top_rest) | set(top_ind)
    pool = [(l, h) for l in range(R.shape[0]) for h in range(nH) if (l, h) not in excl]
    rand32 = [pool[i] for i in rng.choice(len(pool), 32, replace=False)]

    def to_ld(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    state = {"heads": None}
    def make_pre(layer):
        def pre(_m, args):
            if not state["heads"] or layer not in state["heads"]:
                return None
            x = args[0].clone()
            for h in state["heads"][layer]:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    @torch.no_grad()
    def sample_k(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(SEED)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    @torch.no_grad()
    def greedy(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return G.clean_word(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

    @torch.no_grad()
    def logprob(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        return float(lp.gather(1, ids[0, 1:, None]).mean())

    CONDS = (("none", None), ("top32_rest", to_ld(top_rest)),
             ("top32_ind", to_ld(top_ind)), ("rand32", to_ld(rand32)))
    out = {"conditions": {}}
    for cond, heads in CONDS:
        state["heads"] = heads
        theme_scores = []
        for prompt, themeset in LISTS:
            ws = sample_k(prompt + " ")
            shown = set(prompt.lower().replace(",", " ").replace(":", " ").split())
            theme_scores.append(np.mean([1 if (w and w in themeset and w not in shown)
                                         else 0 for w in ws]))
        ana = [1 if greedy(p + " ") == a else 0 for p, a in ANALOGY]
        cop = [1 if greedy(p + " ") == a else 0 for p, a in COPY]
        flu = float(np.mean([logprob(p) for p in PARAS]))
        out["conditions"][cond] = {
            "themed_list": float(np.mean(theme_scores)),
            "themed_per_list": [float(x) for x in theme_scores],
            "morph_analogy": float(np.mean(ana)),
            "copy": float(np.mean(cop)),
            "fluency_logprob": flu}
        c = out["conditions"][cond]
        print(f"[nat] === {cond}: themed {c['themed_list']:.3f} analogy "
              f"{c['morph_analogy']:.2f} copy {c['copy']:.2f} fluency {flu:.3f}", flush=True)
        json.dump(out, open(os.path.join(RUN_DIR, "natural_heads.json"), "w"), indent=1)
    print("[nat] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
