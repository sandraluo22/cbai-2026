"""CATEGORY-GENERAL STUCKNESS PROBE (2026-08-03).

Play restricted games across MANY categories, then train probes on final-position
residuals with LEAVE-CATEGORY-OUT evaluation. Continuous targets only (never
"eventually stuck"):
  famrun   : length of A's current consecutive 4-prefix family run at this state
             (behavioral stuckness-now; every probed turn, no MC cost)
  fam_mass : MC (K) proposal mass on A's own 4-prefix families   } at MC_TURNS
  cat_mass : MC proposal mass on the partner's category wordlist } only
Held-out: leave-one-CATEGORY-out (games from the held-out category never seen in
training); shuffled-target permutation floors included.

Games: N_GAMES per category, resample-24 handler, cap CAP. B restricted by
instruction (category name); CATLISTS used only for measurement.

Env: MODEL(QwenInst32) N_GAMES(6) CAP(40) K(32) TEMP(0.7) RES_EVERY(2)
     MC_TURNS(6,12,18) START_FILE RUN_DIR(runs/probe_catstuck)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
N_GAMES = int(os.environ.get("N_GAMES", "6"))
CAP = int(os.environ.get("CAP", "40"))
K = int(os.environ.get("K", "32"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RES_EVERY = int(os.environ.get("RES_EVERY", "2"))
MC_TURNS = [int(x) for x in os.environ.get("MC_TURNS", "6,12,18").split(",")]
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
RUN_DIR = os.environ.get("RUN_DIR", "runs/probe_catstuck")
ALPHAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

CATLISTS = {
 "city": "paris london tokyo berlin madrid rome moscow vienna prague athens cairo lima oslo dublin geneva boston seattle denver houston chicago miami toronto sydney mumbai".split(),
 "fruit": "apple banana mango peach plum cherry grape lemon lime orange papaya guava kiwi melon apricot fig date pear quince lychee currant nectarine pomelo tangerine".split(),
 "animal": "tiger lion bear wolf fox deer horse rabbit mouse otter badger camel zebra giraffe elephant leopard panther weasel moose bison boar hyena jackal lynx".split(),
 "color": "red blue green yellow purple orange pink brown black white gray violet indigo teal maroon crimson scarlet turquoise beige magenta cyan olive lavender gold".split(),
 "sport": "soccer tennis golf rugby cricket boxing hockey swimming rowing cycling skiing surfing fencing judo karate archery curling badminton volleyball baseball basketball wrestling sailing diving".split(),
 "vegetable": "carrot potato onion garlic spinach kale lettuce cabbage broccoli celery radish turnip beet pea bean corn pepper cucumber zucchini pumpkin leek asparagus cauliflower parsnip".split(),
 "instrument": "piano violin guitar drums flute trumpet cello harp oboe clarinet trombone saxophone banjo mandolin accordion tuba viola bassoon harmonica organ ukulele sitar tambourine xylophone".split(),
 "country": "france germany japan brazil canada mexico spain italy india china egypt kenya peru chile norway sweden poland turkey greece portugal thailand vietnam argentina morocco".split(),
 "profession": "doctor teacher lawyer nurse engineer plumber baker farmer pilot chef dentist architect carpenter electrician journalist accountant librarian pharmacist barber tailor butcher fisherman painter surgeon".split(),
 "bird": "eagle sparrow robin crow owl hawk falcon pigeon dove swan goose duck heron stork pelican parrot canary finch woodpecker hummingbird seagull penguin ostrich flamingo".split(),
 "fish": "salmon tuna trout cod herring mackerel sardine bass perch pike carp catfish flounder halibut snapper anchovy eel swordfish marlin tilapia haddock grouper mullet sturgeon".split(),
 "insect": "ant bee wasp beetle butterfly moth dragonfly cricket grasshopper mosquito fly ladybug termite cicada mantis aphid hornet caterpillar firefly cockroach flea gnat weevil earwig".split(),
 "tree": "oak pine maple birch cedar willow elm ash spruce fir poplar sycamore chestnut walnut beech cypress redwood sequoia magnolia dogwood juniper alder aspen hickory".split(),
 "flower": "rose tulip daisy lily orchid sunflower daffodil violet peony carnation iris jasmine lavender marigold poppy hyacinth chrysanthemum azalea begonia camellia dahlia gardenia hibiscus lilac".split(),
 "metal": "iron copper gold silver zinc tin lead nickel aluminum titanium platinum mercury cobalt chromium magnesium tungsten brass bronze steel palladium lithium sodium uranium manganese".split(),
 "gemstone": "diamond ruby emerald sapphire opal topaz amethyst garnet pearl jade turquoise onyx agate quartz citrine peridot aquamarine moonstone obsidian lapis zircon spinel tourmaline malachite".split(),
 "beverage": "coffee tea juice milk soda water lemonade cocoa cider wine beer whiskey vodka rum brandy champagne espresso latte smoothie punch gin tequila mead kombucha".split(),
 "clothing": "shirt pants dress skirt jacket coat sweater scarf hat gloves socks shoes boots belt tie vest blouse shorts jeans hoodie pajamas mittens sandals cardigan".split(),
 "furniture": "chair table desk sofa bed dresser bookshelf cabinet stool bench wardrobe nightstand ottoman recliner couch hutch armoire crib cot futon vanity sideboard credenza loveseat".split(),
 "tool": "hammer wrench screwdriver pliers saw drill chisel file level ruler clamp axe shovel rake hoe trowel mallet crowbar sander grinder vise anvil plane awl".split(),
 "vehicle": "car truck bus train plane boat ship bicycle motorcycle scooter van tractor ambulance taxi helicopter subway tram ferry canoe kayak yacht jeep limousine trolley".split(),
 "language": "english spanish french german italian portuguese russian japanese korean arabic hindi bengali turkish greek hebrew latin swahili dutch swedish polish finnish hungarian thai vietnamese".split(),
 "dance": "waltz tango salsa ballet samba rumba flamenco polka jive foxtrot swing breakdance hiphop merengue cha mambo bolero quickstep charleston jitterbug bachata zumba twist limbo".split(),
 "fabric": "cotton silk wool linen denim velvet satin leather suede polyester nylon rayon cashmere flannel tweed corduroy chiffon lace canvas burlap felt fleece spandex organza".split(),
 "herb": "basil thyme rosemary oregano sage mint parsley cilantro dill chives tarragon marjoram fennel lavender chamomile lemongrass bay saffron turmeric ginger cumin coriander cardamom anise".split(),
 "cheese": "cheddar brie gouda mozzarella parmesan feta swiss ricotta camembert gruyere provolone gorgonzola manchego roquefort stilton havarti muenster colby asiago mascarpone burrata pecorino emmental halloumi".split(),
 "dessert": "cake pie cookie brownie pudding icecream tart cupcake donut muffin eclair macaron tiramisu cheesecake mousse fudge trifle sundae sorbet gelato baklava churro flan cobbler".split(),
 "weather": "rain snow wind fog hail sleet thunder lightning drizzle storm hurricane tornado blizzard frost dew mist cyclone monsoon breeze gale humidity drought heatwave overcast".split(),
 "river": "nile amazon danube thames seine rhine volga ganges mekong yangtze mississippi missouri colorado columbia amur congo niger zambezi tigris euphrates loire elbe oder dnieper".split(),
 "boardgame": "chess checkers backgammon monopoly scrabble risk clue battleship dominoes catan carcassonne othello mancala parcheesi yahtzee boggle jenga sorry trouble candyland cranium taboo pictionary operation".split(),
}
CONCEPT = {c: c for c in CATLISTS}
CONCEPT.update({"boardgame": "board game"})


def _ridge_pred(Xt, yt, Xe, alpha):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = (Xt - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Xt.shape[1]), Z.T @ (yt - yt.mean()))
    return ((Xe - mu) / sd) @ w + yt.mean()


def loco_r2(X, y, cats):
    """leave-one-category-out R^2 with nested alpha selection."""
    ucats = sorted(set(cats))
    yhat = np.zeros_like(y)
    for hold in ucats:
        tr = np.array([c != hold for c in cats])
        te = ~tr
        tc = sorted({c for c, t in zip(cats, tr) if t})
        inner = set(tc[::4]) or set(tc[:1])
        itr = np.array([t and c not in inner for c, t in zip(cats, tr)])
        iva = np.array([t and c in inner for c, t in zip(cats, tr)])
        best_a, best = ALPHAS[0], -np.inf
        for a in ALPHAS:
            pv = _ridge_pred(X[itr], y[itr], X[iva], a)
            r2 = 1 - ((y[iva] - pv) ** 2).sum() / (((y[iva] - y[iva].mean()) ** 2).sum() + 1e-9)
            if r2 > best:
                best, best_a = r2, a
        yhat[te] = _ridge_pred(X[tr], y[tr], X[te], best_a)
    return 1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = len(model.model.layers)

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + ((" " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))) if hist else "")
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def gen_word(body, seed, forbidden):
        prompt = LA._render(tok, body) + "\nMy word:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    @torch.no_grad()
    def resid_final(body):
        enc = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").to(dev)
        out = model(enc.input_ids, output_hidden_states=True)
        return np.stack([h[0, -1].float().cpu().numpy().astype(np.float16)
                         for h in out.hidden_states[1:]])

    cats = sorted(CATLISTS)
    X_all, y_run, y_fam, y_cat, cat_id, mc_mask = [], [], [], [], [], []
    tf = open(os.path.join(RUN_DIR, "catstuck_transcript.jsonl"), "w")
    for ci, cat in enumerate(cats):
        catset = set(CATLISTS[cat])
        restr = (f" IMPORTANT: every single word you say must be a {CONCEPT[cat]}. "
                 f"Only ever say {CONCEPT[cat]}s, nothing else.")
        for gidx in range(N_GAMES):
            sa, sb = starts[(ci * N_GAMES + gidx) % len(starts)]
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            own = [sa]
            for t in range(1, CAP):
                # stuckness-now: current consecutive family-run length of A
                run = 0
                for w in reversed(own[1:]):
                    if len(w) > 3 and any(w[:4] == p[:4] and len(p) > 3
                                          for p in own[:len(own) - 1 - run]):
                        run += 1
                    else:
                        break
                if t % RES_EVERY == 0 or t in MC_TURNS:
                    X_all.append(resid_final(body_of(histA, used)))
                    y_run.append(run)
                    cat_id.append(cat)
                    if t in MC_TURNS:
                        props = propose_k(body_of(histA, used))
                        fams = {w[:4] for w in own if len(w) > 3}
                        y_fam.append(np.mean([1 if (w and w not in used and len(w) > 3
                                                    and w[:4] in fams) else 0 for w in props]))
                        y_cat.append(np.mean([1 if (w and w not in used and w in catset)
                                              else 0 for w in props]))
                        mc_mask.append(True)
                    else:
                        y_fam.append(np.nan); y_cat.append(np.nan); mc_mask.append(False)
                wA = gen_word(body_of(histA, used), 7000 * ci + 500 * gidx + t, used)
                wB = gen_word(body_of(histB, used, restr), 990000 + 7000 * ci + 500 * gidx + t, used)
                tf.write(json.dumps({"cat": cat, "game": gidx, "turn": t,
                                     "A": wA, "B": wB, "agreed": wA == wB and wA not in used}) + "\n")
                tf.flush()
                if wA == wB and wA and wA not in used:
                    break
                used |= {wA, wB}
                own.append(wA)
                histA.append((wB, wA)); histB.append((wA, wB))
        print(f"[catstuck] {cat}: done ({len(X_all)} states so far)", flush=True)
    tf.close()

    X_all = np.stack(X_all)
    y_run = np.array(y_run, float)
    y_fam = np.array(y_fam); y_cat = np.array(y_cat)
    mc = np.array(mc_mask)
    np.savez_compressed(os.path.join(RUN_DIR, "resid_cache.npz"), X=X_all, y_run=y_run,
                        y_fam=y_fam, y_cat=y_cat, cat_id=np.array(cat_id), mc=mc)
    print(f"[catstuck] states {len(y_run)}, MC states {int(mc.sum())}, cats {len(cats)}", flush=True)

    rng = np.random.default_rng(2)
    out = {"n_states": len(y_run), "n_mc": int(mc.sum()), "n_cats": len(cats), "layers": nL,
           "r2": {"famrun": [], "fam_mass": [], "cat_mass": []},
           "r2_shuffled": {"famrun": [], "fam_mass": [], "cat_mass": []}}
    cid = list(cat_id)
    cid_mc = [c for c, m in zip(cid, mc) if m]
    yr_s = rng.permutation(y_run)
    yf = y_fam[mc]; yc = y_cat[mc]
    yf_s = rng.permutation(yf); yc_s = rng.permutation(yc)
    for L in range(nL):
        XL = X_all[:, L, :].astype(np.float64)
        XM = XL[mc]
        out["r2"]["famrun"].append(loco_r2(XL, y_run, cid))
        out["r2_shuffled"]["famrun"].append(loco_r2(XL, yr_s, cid))
        out["r2"]["fam_mass"].append(loco_r2(XM, yf, cid_mc))
        out["r2_shuffled"]["fam_mass"].append(loco_r2(XM, yf_s, cid_mc))
        out["r2"]["cat_mass"].append(loco_r2(XM, yc, cid_mc))
        out["r2_shuffled"]["cat_mass"].append(loco_r2(XM, yc_s, cid_mc))
        if L % 8 == 0:
            print(f"[catstuck] L{L}: famrun {out['r2']['famrun'][-1]:.2f} "
                  f"fam {out['r2']['fam_mass'][-1]:.2f} cat {out['r2']['cat_mass'][-1]:.2f}",
                  flush=True)
        json.dump(out, open(os.path.join(RUN_DIR, "catstuck_probes.json"), "w"))
    json.dump(out, open(os.path.join(RUN_DIR, "catstuck_probes.json"), "w"), indent=1)
    b = {k: (int(np.argmax(v)), round(float(max(v)), 3)) for k, v in out["r2"].items()}
    print(f"[catstuck] === best layers: {b}", flush=True)


if __name__ == "__main__":
    main()
