"""QSG naming game (Tanaka, "When Is Collective Intelligence a Lottery?",
arXiv:2603.24676) + per-round revealed truth — measured with LOGIT BELIEF READS.

v2 protocol (per Sandra 2026-07-28):
  * Memory shown GROUPED BY ROUND, with each past round's revealed correct answer
    interleaved: 'Round 1 memories: [...]' / 'The correct answer for round 1 was
    "X".' / ... / 'Round r memories (current round): [...]'.
  * Every quantity is a bounded logit read: an agent's belief = softmax over the K
    label first-tokens at the position after the prefill '{"label": "'. The speaker
    EMITS by sampling from its belief (exactly QSG Hard: k* ~ Cat(x_S)); no free
    generation anywhere, so the channel is fully closed.
  * Per interaction we record the listener's belief BEFORE and AFTER the speaker's
    label enters its memory -> per-step update magnitude, attributable to (round,
    listener's conversation depth, source identity). Trust/reputation = the update
    toward a label DEPENDING ON WHO SAID IT; no self-report elicitation.
  * Memory is 100% agent-generated: every entry is a peer's sampled output
    (listener-only updates, delayed reveal). Reveals are the only exogenous lines.

Kept from the paper: JSON-only system message, Referent / shuffled Allowed labels
("order is randomized" note), constraints block, {"label": "<label>"} output spec,
synthetic 5-char labels, uniformly random ordered speaker-listener pairs, Hard m=1.

env: MODEL (Qwen32)  VAR (none|informed_r1|informed_all|misinformed_r1|misinformed_all)
     ROUNDS (5)  NAGENTS (5)  K (3)  STEPS (75/round)  TEMP (1.0)  SEED (0)  OUT
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from run_games import load  # noqa: E402

FRESH = os.environ.get("FRESH", "0") == "1"        # new label vocabulary every round
EARLYSTOP = int(os.environ.get("EARLYSTOP", "0"))  # end round after this many consecutive
                                                   # fully-unanimous conversations (0 = off)
SWITCH_AT = int(os.environ.get("SWITCH_AT", "10"))  # round after which duel_remove drops the
                                                    # liar / betrayal flips P1's clue
ALT_PERIOD = int(os.environ.get("ALT_PERIOD", "5"))  # alternator: block length of truth/lies
P1REL = float(os.environ.get("P1REL", "1.0"))       # graded: P1/P2 per-round clue accuracy
P2REL = float(os.environ.get("P2REL", "0.0"))
WINDOW = int(os.environ.get("WINDOW", "0"))         # show only last k rounds of memory+reveals (0=all)
NOTES = os.environ.get("NOTES", "")                 # ""|update|append|peragent|convfree|convagent
                                                    # conv*: written after EVERY conversation by the
                                                    # listener; convagent = one note PER SPEAKER,
                                                    # replaced each time that speaker is heard
DECIDE = os.environ.get("DECIDE", "0") == "1"       # probe reads: self-generated reliability line
                                                    # spliced in directly above the answer
JUSTIFY = os.environ.get("JUSTIFY", "0") == "1"     # speaker sends 1-sentence justification with label
SCHED = os.environ.get("SCHED", "")                 # deterministic correctness bits for towers,
                                                    # "P1bits" or "P1bits;P2bits" e.g. "11101110..."
TALLY = os.environ.get("TALLY", "")                 # ""|true|false: programmatic track-record table
                                                    # in every prompt (false = inverted counts)
WHOINF = os.environ.get("WHOINF", "0") == "1"       # round-end bounded read: who influenced you most
INDEP = os.environ.get("SENT", "1") != "0" and not FRESH   # sentence pointless with fresh labels

SYS = ('You must output only valid JSON. No extra keys, no explanations, no markdown.\n'
       'Valid labels are exactly those in Allowed labels. Never output "<PAD>".')


def make_labels(k, rng, tok, used=None):
    """Synthetic 5-char labels with DISTINCT first tokens (needed for the logit read).
    `used`: first-token ids already claimed (FRESH mode: no collisions across rounds,
    so old labels can never absorb the read mass of current ones)."""
    used = used if used is not None else set()
    while True:
        labs = set()
        while len(labs) < k:
            labs.add("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)))
        labs = sorted(labs)
        ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in labs]
        if len(set(ids)) == k and not (set(ids) & used):
            used.update(ids)
            return labs, ids


def memory_block(mem, reveals, cur_round, names_on=False):
    """mem: list of (round, speaker_1idx, label). reveals: {round: correct_label}."""
    start = max(1, cur_round - WINDOW + 1) if WINDOW else 1
    s = ("Memory of observed messages, grouped by round"
         + (f" (only the most recent {WINDOW} rounds are shown)" if WINDOW else "") + ":\n")
    for r in range(start, cur_round + 1):
        entries = [f'"{pname(e[1], names_on)}: {e[2]}' + (f' — {e[3]}' if len(e) > 3 and e[3] else "") + '"'
                   for e in mem if e[0] == r]
        tag = " (current round)" if r == cur_round else ""
        s += f"Round {r} memories{tag}: [{', '.join(entries)}]\n"
        if r in reveals:
            s += f'The correct answer for round {r} was "{reveals[r]}".\n'
    return s


NAMES = ["Alice", "Ben", "Carol", "David", "Emma"]


def pname(i_1idx, names_on):
    return NAMES[i_1idx - 1] if names_on else f"P{i_1idx}"


def user_msg(agent, labels, mem, reveals, cur_round, clue, rng, names_on=False, phase2=False,
             notes_txt="", tally_txt=""):
    order = list(labels)
    rng.shuffle(order)
    if phase2 == "nat":
        ref = f"component unit_r{cur_round}"
        note = ("NOTE: the task has changed. Each round, one NEW machine component is up for "
                "quality inspection and should be either APPROVED or REJECTED. You cannot see "
                "the component itself. The earlier naming rounds concerned a different task.\n")
        thing = "component"
    elif phase2:
        ref = f"object obj_r{cur_round}"
        note = ("NOTE: the task has changed. Each round now shows one NEW object that truly "
                "belongs to Category A or Category B. The earlier naming rounds concerned a "
                "different task.\n")
        thing = "object"
    else:
        ref, note, thing = "ref_07", "", "referent"
    s = (f"Referent: {ref}\n"
         f"Allowed labels: {json.dumps(order)}\n"
         "The list order is randomized and has no meaning.\n" + note +
         f"Both players are choosing a label for the same {thing} in repeated interactions. "
         "The memory shows labels you observed from previous interactions with partners.\n"
         + memory_block(mem, reveals, cur_round, names_on) +
         ('Each memory entry is of the form "<name>: <label>" (the player you observed it from).\n'
          if names_on else
          'Each memory entry is of the form "P<player>: <label>" (the player you observed it from).\n') +
         ("The correct answer for each round has no relation to the correct answers of "
          "previous rounds.\n" if INDEP else "") +
         f"You are {pname(agent + 1, names_on)}.\n"
         + (tally_txt or "")
         + (f"Your private notes:\n{notes_txt}" if notes_txt else ""))
    if clue is not None:
        s += f'Private clue: the correct label for the current round is "{clue}".\n'
    s += ("\nConstraints:\n- Output JSON only.\n- Every label must be from Allowed labels.\n\n"
          'Output JSON exactly: {"label": "<label>"}')
    return s


@torch.no_grad()
def gen_note(model, tok, user_text, seed, max_new=90, max_words=60):
    """Free-text private note (no JSON system message)."""
    msgs = [{"role": "user", "content": user_text}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
                         do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return " ".join(txt.split()[:max_words])


@torch.no_grad()
def belief_many(model, tok, user_texts, label_ids):
    """Batched bounded reads: one left-padded forward for N independent prompts."""
    texts = []
    for ut in user_texts:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ut}]
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t + '{"label": "')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    old = tok.padding_side
    tok.padding_side = "right"                        # causal: pads can't affect earlier tokens
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    tok.padding_side = old
    lens = enc.attention_mask.sum(-1)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    logits = out[torch.arange(len(texts), device=model.device), lens - 1]
    idx = torch.tensor(label_ids, device=model.device)
    return torch.softmax(logits[:, idx].float(), -1).cpu().numpy()


@torch.no_grad()
def belief(model, tok, user_text, label_ids):
    """Bounded read: softmax over the K label first-tokens after prefill '{"label": "'."""
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user_text}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += '{"label": "'
    enc = tok(text, return_tensors="pt").to(model.device)
    logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1]
    p = torch.softmax(logits[torch.tensor(label_ids, device=model.device)].float(), 0)
    return p.cpu().numpy()


def run(var, tag, rounds, n, k, steps, temp, seed, out_dir, names_on=False):
    rng = random.Random(seed)                          # clue draws, notes, shuffles (legacy stream)
    rng_lab = random.Random(f"{os.environ.get('LBLSEED', seed)}-lab")
    rng_truth = random.Random(f"{os.environ.get('TRUTHSEED', seed)}-truth")
    rng_pair = random.Random(f"{os.environ.get('PAIRSEED', seed)}-pair")
    model, tok, _ = load(tag)
    used_ids = set()
    labels, label_ids = make_labels(k, rng_lab, tok, used_ids)
    mem = [[] for _ in range(n)]                      # (round, speaker_1idx, label)
    reveals = {}
    nconv = [0] * n                                   # listener conversation depth
    lines = [dict(type="meta", var=var, model=tag, rounds=rounds, n=n, k=k, steps=steps,
                  temp=temp, seed=seed, labels=labels, names=names_on, fresh=FRESH)]

    ab_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in ("A", "B")]
    nat_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in ("APPROVE", "REJECT")]
    ph = {"p2": False}
    notes = [[] for _ in range(n)]
    srcnotes = [dict() for _ in range(n)]             # convagent: speaker_1idx -> note
    def note_txt(i):
        if NOTES == "convagent":
            return "".join(f"About {pname(k, names_on)}: {v}\n"
                           for k, v in sorted(srcnotes[i].items()))
        return "".join(f"(after round {rr}) {t}\n" for rr, t in notes[i])
    hist_cm = {}                                       # round -> clue_map (for TALLY)
    def tally_txt(r):
        if not TALLY:
            return ""
        cnt = {}
        for rr, cmv in hist_cm.items():
            if rr >= r or rr not in reveals:
                continue
            for kk, v in cmv.items():
                ok = (v == reveals[rr])
                if TALLY == "false":
                    ok = not ok
                a, b = cnt.get(kk, (0, 0))
                cnt[kk] = (a + ok, b + 1)
        if not cnt:
            return ""
        return ("Track record so far: " + "; ".join(
            f"{pname(kk + 1, names_on)} {a}/{b} correct" for kk, (a, b) in sorted(cnt.items()))
            + ".\n")
    def read(i, r, clue_map):
        return belief(model, tok, user_msg(i, labels, mem[i], reveals, r,
                                           clue_map.get(i), rng, names_on, ph["p2"],
                                           note_txt(i), tally_txt(r)), label_ids)
    # NOTE: `labels`/`label_ids` rebind per round in FRESH mode; closure sees current ones

    for r in range(1, rounds + 1):
        ph["p2"] = (("nat" if var == "switch_natural" else "ab")
                    if (var.startswith("switch_") and r > SWITCH_AT) else False)
        if ph["p2"] == "nat":
            labels, label_ids = ["APPROVE", "REJECT"], nat_ids
        elif ph["p2"]:
            labels, label_ids = ["A", "B"], ab_ids
        elif FRESH and r > 1:
            labels, label_ids = make_labels(k, rng, tok, used_ids)
        correct = rng.choice(labels)
        wrong = rng.choice([l for l in labels if l != correct])
        clue_map = {}
        if var == "informed_all" or (var == "informed_r1" and r == 1):
            clue_map[0] = correct
        if var == "misinformed_all" or (var == "misinformed_r1" and r == 1):
            clue_map[0] = wrong
        if var == "duel":                             # two confident sources, one wrong
            clue_map[0], clue_map[1] = correct, wrong
        if var == "duel_remove":                      # liar vanishes after SWITCH_AT rounds
            clue_map[0] = correct
            if r <= SWITCH_AT:
                clue_map[1] = wrong
        if var == "betrayal":                         # truth-teller starts lying after SWITCH_AT
            clue_map[0] = correct if r <= SWITCH_AT else wrong
        if var == "alternator":                       # truth blocks and lie blocks, alternating
            clue_map[0] = correct if ((r - 1) // ALT_PERIOD) % 2 == 0 else wrong
        if var == "switch_duel" or var == "switch_natural":   # task switch, towers carry over
            clue_map[0], clue_map[1] = correct, wrong
        if var == "pair_tt":                          # role polarization: two truth-tellers
            clue_map[0], clue_map[1] = correct, correct
        if var == "pair_ll":                          # two liars
            clue_map[0], clue_map[1] = wrong, wrong
        if var == "graded":                           # stochastic reliability pair
            clue_map[0] = correct if rng.random() < P1REL else wrong
            clue_map[1] = correct if rng.random() < P2REL else wrong
        if var == "curve":                            # deterministic schedule(s) from SCHED
            bits = SCHED.split(";")
            clue_map[0] = correct if bits[0][r - 1] == "1" else wrong
            if len(bits) > 1:
                clue_map[1] = correct if bits[1][r - 1] == "1" else wrong
        if var == "delayed":                          # duel training, then NO special agents
            if r <= SWITCH_AT:
                clue_map[0], clue_map[1] = correct, wrong
        if var == "reversal":                         # swap roles mid-game (hysteresis)
            clue_map[0], clue_map[1] = (correct, wrong) if r <= SWITCH_AT else (wrong, correct)
        if var == "switch_informed":                  # single truth-teller
            clue_map[0] = correct
        if var == "switch_misinformed":               # single liar
            clue_map[0] = wrong
        hist_cm[r] = dict(clue_map)
        lines.append(dict(type="round_start", round=r, correct=correct, labels=labels,
                          task="category" if ph["p2"] else "naming",
                          clue=clue_map.get(0), clue_is_wrong=clue_map.get(0) == wrong,
                          clue_map={str(kk + 1): v for kk, v in clue_map.items()}))
        if ph["p2"] and var in ("switch_informed", "switch_misinformed"):
            # seed every agent's round memory: tower's recommendation + 1 dissenting neutral
            tower_lab = clue_map[0]
            opp = "B" if tower_lab == "A" else "A"
            for i in range(n):
                nid = 5 if i != 4 else 4              # dissenting neutral (never self, never P1)
                if i != 0:
                    mem[i].append((r, 1, tower_lab))
                mem[i].append((r, nid, opp))
            lines.append(dict(type="seed", round=r, tower=tower_lab, neutral_opp=opp))
        if ph["p2"]:
            p0s = [dict(agent=i + 1, p=read(i, r, clue_map).tolist()) for i in range(n)]
            for p in p0s:
                p["argmax"] = labels[int(np.argmax(p["p"]))]
            lines.append(dict(type="probe0", round=r, probes=p0s))
        last_am, streak = [None] * n, 0
        pool = [i for i in range(n)
                if not (var == "duel_remove" and r > SWITCH_AT and i == 1)]
        FAST = os.environ.get("FASTBATCH", "1") == "1"
        def umsg(i):
            return user_msg(i, labels, mem[i], reveals, r, clue_map.get(i), rng,
                            names_on, ph["p2"], note_txt(i), tally_txt(r))
        for t in range(steps):
            S, L = rng_pair.sample(pool, 2)
            if FAST and not DECIDE:
                both = belief_many(model, tok, [umsg(S), umsg(L)], label_ids)
                pS, pL0 = both[0], both[1]
            else:
                pS = read(S, r, clue_map)
            q = pS if temp == 1.0 else pS ** (1.0 / temp) / (pS ** (1.0 / temp)).sum()
            q = q / q.sum()
            s_lab = labels[int(np.random.default_rng(rng.randrange(2**31)).choice(len(labels), p=q))]
            if not (FAST and not DECIDE):
                pL0 = read(L, r, clue_map)
            jt = None
            if JUSTIFY:
                stem = user_msg(S, labels, mem[S], reveals, r, clue_map.get(S), rng,
                                names_on, ph["p2"], note_txt(S)).split("\nConstraints:")[0]
                jt = gen_note(model, tok, stem + f'\nYou are sending the label "{s_lab}". '
                              "Add ONE short sentence (at most 20 words) of justification to "
                              "send along with it.", rng.randrange(2**31),
                              max_new=40, max_words=22)
            mem[L].append((r, S + 1, s_lab, jt) if jt else (r, S + 1, s_lab))
            pL1 = read(L, r, clue_map)
            nconv[L] += 1
            lines.append(dict(type="step", round=r, t=t, S=S + 1, L=L + 1, s_label=s_lab,
                              conv_depth=nconv[L], p_speaker=pS.tolist(),
                              p_before=pL0.tolist(), p_after=pL1.tolist(),
                              **({"justification": jt} if jt else {})))
            if NOTES in ("convfree", "convagent"):
                base_txt = user_msg(L, labels, mem[L], reveals, r, clue_map.get(L), rng,
                                    names_on, ph["p2"], note_txt(L)).split("\nConstraints:")[0]
                who = pname(S + 1, names_on)
                if NOTES == "convfree":
                    instr = (f'\nYou just heard {who} say "{s_lab}". Update your private note '
                             "(at most 30 words) capturing anything useful for later; it will "
                             "REPLACE your previous note.")
                    t2 = gen_note(model, tok, base_txt + instr, rng.randrange(2**31),
                                  max_new=45, max_words=35)
                    notes[L] = [(r, t2)]
                    lines.append(dict(type="note", round=r, t=t, agent=L + 1, text=t2))
                else:
                    instr = (f'\nYou just heard {who} say "{s_lab}". Update your note about '
                             f"{who} (at most 20 words); it REPLACES your previous note about "
                             "them. Write only the note.")
                    t2 = gen_note(model, tok, base_txt + instr, rng.randrange(2**31),
                                  max_new=40, max_words=25)
                    srcnotes[L][S + 1] = t2
                    lines.append(dict(type="note", round=r, t=t, agent=L + 1, about=S + 1,
                                      text=t2))
            last_am[S], last_am[L] = int(np.argmax(pS)), int(np.argmax(pL1))
            if EARLYSTOP:
                free = [last_am[i] for i in pool if i not in clue_map]
                unanimous = free and None not in free and len(set(free)) == 1
                streak = streak + 1 if unanimous else 0
                if streak >= EARLYSTOP:
                    lines.append(dict(type="round_end_early", round=r, at_conv=t + 1))
                    break
        if os.environ.get("FASTBATCH", "1") == "1" and not DECIDE:
            pr = belief_many(model, tok,
                             [user_msg(i, labels, mem[i], reveals, r, clue_map.get(i), rng,
                                       names_on, ph["p2"], note_txt(i), tally_txt(r))
                              for i in range(n)], label_ids)
            probes = [dict(agent=i + 1, p=pr[i].tolist()) for i in range(n)]
        elif DECIDE:
            probes = []
            for i in range(n):
                ut = user_msg(i, labels, mem[i], reveals, r, clue_map.get(i), rng,
                              names_on, ph["p2"], note_txt(i))
                stem, tail = ut.split("\nConstraints:")
                line = gen_note(model, tok, stem +
                                "\nBefore answering: in ONE line (at most 15 words), state "
                                "which player's labels have been most reliable so far.",
                                rng.randrange(2**31), max_new=40, max_words=20)
                p = belief(model, tok, stem + f"\nYour assessment: {line}\n\nConstraints:"
                           + tail, label_ids)
                probes.append(dict(agent=i + 1, p=p.tolist(), assess=line))
        else:
            probes = [dict(agent=i + 1, p=read(i, r, clue_map).tolist()) for i in range(n)]
        for p in probes:
            p["argmax"] = labels[int(np.argmax(p["p"]))]
            p["correct"] = p["argmax"] == correct
        lines.append(dict(type="probe", round=r, probes=probes))
        if WHOINF:
            digit_ids = {k: tok(str(k), add_special_tokens=False)["input_ids"][0]
                         for k in range(1, n + 1)}
            wrec = []
            for i in range(n):
                valid = [k for k in range(1, n + 1) if k != i + 1]
                ut = user_msg(i, labels, mem[i], reveals, r, clue_map.get(i), rng,
                              names_on, ph["p2"], note_txt(i)).split("\nConstraints:")[0]
                ut += ('\nQuestion: which player\'s messages influenced your current choice '
                       'the most this round?\n\nConstraints:\n- Output JSON only.\n\n'
                       'Output JSON exactly: {"player": "P<number>"}')
                msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ut}]
                try:
                    text = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
                text += '{"player": "P'
                enc = tok(text, return_tensors="pt").to(model.device)
                import torch as _t
                with _t.no_grad():
                    logits = model(input_ids=enc.input_ids,
                                   attention_mask=enc.attention_mask).logits[0, -1]
                pv = _t.softmax(logits[_t.tensor([digit_ids[k] for k in valid],
                                                 device=model.device)].float(), 0).cpu().numpy()
                wrec.append(dict(agent=i + 1, players=valid, p=pv.tolist(),
                                 argmax=valid[int(np.argmax(pv))]))
            lines.append(dict(type="whoinf", round=r, reports=wrec))
        reveals[r] = correct
        if NOTES and NOTES not in ("convfree", "convagent"):
            for i in range(n):
                base_txt = user_msg(i, labels, mem[i], reveals, r, clue_map.get(i), rng,
                                    names_on, ph["p2"], note_txt(i)).split("\nConstraints:")[0]
                if NOTES == "evalforce":
                    instr = (f"\nEnd of round {r}. Update your private note: for each player "
                             "whose messages you observed, record whether their labels MATCHED "
                             "the revealed correct answers so far (e.g. 'P2: matched 3 of 4'). "
                             "At most 40 words. It REPLACES your previous note.")
                    t = gen_note(model, tok, base_txt + instr, rng.randrange(2**31),
                                 max_new=70, max_words=45)
                elif NOTES == "peragent":
                    others = ", ".join(pname(j + 1, names_on) for j in range(n) if j != i)
                    instr = (f"\nEnd of round {r}. Update your private notes: write ONE short "
                             f"line (at most 15 words) about EACH other player ({others}) — "
                             "anything about that player worth remembering for later rounds. "
                             "You may add one final line of general notes. These notes REPLACE "
                             "your previous notes.")
                    t = gen_note(model, tok, base_txt + instr, rng.randrange(2**31),
                                 max_new=220, max_words=130)
                elif NOTES == "update":
                    instr = (f"\nEnd of round {r}. Update your private note: write a short note "
                             "(at most 50 words) capturing what you want to remember for later "
                             "rounds. It will REPLACE your previous note.")
                    t = gen_note(model, tok, base_txt + instr, rng.randrange(2**31))
                else:
                    instr = (f"\nEnd of round {r}. Write a short private note (at most 50 words) "
                             "about anything that may help you in later rounds.")
                    t = gen_note(model, tok, base_txt + instr, rng.randrange(2**31))
                if NOTES in ("update", "peragent", "evalforce"):
                    notes[i] = [(r, t)]
                else:
                    notes[i].append((r, t))
                lines.append(dict(type="note", round=r, agent=i + 1, text=t))
        acc = sum(p["correct"] for p in probes) / n
        print(f"[{var} r{r}] correct={correct} clues={clue_map} probe_acc={acc:.2f} "
              f"argmax={[p['argmax'] for p in probes]}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"gossip_s{seed}")
    with open(stem + "_transcript.jsonl", "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    with open(stem + "_transcript.json", "w") as fh:
        json.dump(lines, fh, indent=1)


def main():
    run(os.environ.get("VAR", "none"),
        os.environ.get("MODEL", "Qwen32"),
        int(os.environ.get("ROUNDS", 5)),
        int(os.environ.get("NAGENTS", 5)),
        int(os.environ.get("K", 3)),
        int(os.environ.get("STEPS", 75)),
        float(os.environ.get("TEMP", 1.0)),
        int(os.environ.get("SEED", 0)),
        os.environ.get("OUT", os.path.join(
            _HERE, f"{os.environ.get('MODEL', 'Qwen32')}"
                   f"{'names' if os.environ.get('NAMES') else ''}_{os.environ.get('VAR', 'none')}")),
        names_on=bool(os.environ.get("NAMES")))


if __name__ == "__main__":
    main()
