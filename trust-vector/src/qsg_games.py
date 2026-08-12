"""The iterated-game test bed: scripted partner histories, bounded logit read-out.

Five games share ONE payoff matrix (T=5, R=3, P=1, S=0) and differ only in what the
two actions are called — so semantic framing is the manipulation and payoff structure
is held fixed. Game 5 ("labels") strips the semantics entirely and is the check that
any steering effect is not riding on the words "honor"/"renege".

  pd         cooperate / defect          Alice  -> Bob
  invest     invest    / withhold        Mira   -> Jonas
  food       share     / hoard           Tessa  -> Kai
  trade      honor     / renege          Petrov -> Dana
  labels     A         / B               Wen    -> Riku
  (first name = scripted partner, second = the model)

Schedules (rounds 1-7 of the partner; the model's own past is a separate factor)
  one_lapse       C D C C C C C   — a near-clean partner with one defection
  frequent        C D C D D D D   — a mostly-defecting partner
  all_coop        C x7            — anchor
  all_defect      D x7            — anchor

Model's own scripted past
  unconditional   the model always cooperated, whatever the partner did
  conditional     the model retaliated the round after each partner defection
  mirror          the model defected in the same round as the partner

Read-out: the partner's round-7 move is shown, the model's is not, and we take
logit(cooperative action) - logit(uncooperative action) at the prefilled answer slot.
The two action words are checked for distinct first tokens before anything is run.
Output-spec word order is counterbalanced across seeds (`flip`) because listing
"cooperate or defect" is itself a prior.

Two quantities come out of this, and they are different claims:
  LEVEL  the margin on a fixed schedule           — "does it act more trusting"
  SLOPE  margin(all_coop) - margin(all_defect)    — "does it use the record more"
Steering can move the first without touching the second; that distinction is the
whole point of the SLOPE arm.
"""
from __future__ import annotations

GAMES = {
    "pd":     dict(partner="Alice",  me="Bob",   coop="cooperate", defect="defect",
                   noun="round"),
    "invest": dict(partner="Mira",   me="Jonas", coop="invest",    defect="withhold",
                   noun="round"),
    "food":   dict(partner="Tessa",  me="Kai",   coop="share",     defect="hoard",
                   noun="round"),
    "trade":  dict(partner="Petrov", me="Dana",  coop="honor",     defect="renege",
                   noun="round"),
    "labels": dict(partner="Wen",    me="Riku",  coop="A",         defect="B",
                   noun="round"),
}

# partner moves, rounds 1-7 (True = cooperative)
SCHEDULES = {
    "one_lapse":  [True, False, True, True, True, True, True],
    "frequent":   [True, False, True, False, False, False, False],
    "all_coop":   [True] * 7,
    "all_defect": [False] * 7,
}

# The model's own moves, rounds 1-6.
#   unconditional  always cooperated, whatever the partner did
#   conditional    retaliated the round AFTER each partner defection (image 2's
#                  layout, and the only timing consistent with simultaneous moves)
#   mirror         defected in the SAME round as the partner (image 3's layout).
#                  Under simultaneous play this is not a strategy the model could
#                  have executed; it is a same-round-correlated history. Keep it as
#                  a robustness arm, not the default.
def self_moves(sched, style):
    p = SCHEDULES[sched]
    if style == "unconditional":
        return [True] * 6
    if style == "mirror":
        return list(p[:6])
    return [True] + [p[r - 1] for r in range(1, 6)]      # retaliate one round later


PAYOFF = ("Each {noun} you and {partner} choose at the same time, without seeing the "
          "other's choice for that {noun}.\n"
          "Points: if both choose {coop}, each gets 3. If both choose {defect}, each "
          "gets 1.\nIf one chooses {coop} and the other {defect}, the one who chose "
          "{defect} gets 5 and the other gets 0.\n"
          "Your score is the total over all {noun}s.")

SYS = "You are playing a game. Output only valid JSON."


def build(game, sched, style, flip=False, rounds=7):
    """The round-`rounds` decision prompt. Returns the user message + read-out spec."""
    g = GAMES[game]
    p, me = g["partner"], g["me"]
    pm, sm = SCHEDULES[sched], self_moves(sched, style)
    w = lambda ok: g["coop"] if ok else g["defect"]  # noqa: E731

    head = (f"You are {me}. You and {p} are playing the same game repeatedly.\n"
            f"Each {g['noun']}, you each choose {g['coop']} or {g['defect']}.\n"
            + PAYOFF.format(noun=g["noun"], partner=p, coop=g["coop"],
                            defect=g["defect"]))
    rows = [f"Round {r}: {p} chose {w(pm[r-1])}. {me} chose {w(sm[r-1])}."
            for r in range(1, rounds)]
    cur = (f"Round {rounds}: {p} chose {w(pm[rounds-1])}. {me} has not chosen yet.")
    a, b = ((g["defect"], g["coop"]) if flip else (g["coop"], g["defect"]))
    tail = (f"\n\nIt is {me}'s turn to choose for round {rounds}.\n"
            'Output JSON exactly: {"choice": "<%s or %s>"}' % (a, b))
    user = head + "\n\nRecord so far:\n" + "\n".join(rows + [cur]) + tail
    return dict(user=user, prefill='{"choice": "', coop=g["coop"], defect=g["defect"],
                partner=p, me=me, game=game, sched=sched, style=style, flip=flip,
                cur_row=cur, hist_rows=rows)


def check_tokens(tok):
    """Every game's two actions must have distinct first tokens or the read is void."""
    from common import first_id
    bad = []
    for k, g in GAMES.items():
        if first_id(tok, g["coop"]) == first_id(tok, g["defect"]):
            bad.append(k)
    if bad:
        raise RuntimeError(f"actions share a first token in games: {bad}")
    return True


def occ_groups(tok, text, name):
    """Token indices of each SEPARATE occurrence of `name`, plus the sequence length.

    Needed for the before/at/after sweep: a flat index list loses which tokens belong
    to which mention, and "the token right after the name" is only defined per
    occurrence.
    """
    from common import spans_of
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    groups = []
    for (c0, c1) in spans_of(text, name):
        idx = [i for i, (a, b) in enumerate(offs) if b > a and a < c1 and b > c0]
        if idx:
            groups.append(idx)
    return groups, len(offs)


def offset_positions(tok, text, name):
    """{'pre','at','post'} — the token before each mention, the mention itself, and
    the token after it. `pre`/`post` exclude anything that is itself a name token, so
    the three groups are disjoint and the comparison is clean."""
    groups, n = occ_groups(tok, text, name)
    at = sorted({i for g in groups for i in g})
    atset = set(at)
    pre = sorted({g[0] - 1 for g in groups if g[0] - 1 >= 0} - atset)
    post = sorted({g[-1] + 1 for g in groups if g[-1] + 1 < n} - atset)
    return dict(pre=pre, at=at, post=post)


def positions(tok, text, ex):
    """Token-index groups the steering vector can be written into."""
    from common import spans_of, tok_idx
    p, me = ex["partner"], ex["me"]
    ci = text.find(ex["cur_row"])
    hi0 = text.find(ex["hist_rows"][0]) if ex["hist_rows"] else ci
    p_all = spans_of(text, p)
    me_all = spans_of(text, me)
    out = {
        "partner_all":  tok_idx(tok, text, p_all),
        "partner_hist": tok_idx(tok, text, [s for s in p_all if hi0 <= s[0] < ci]),
        "partner_cur":  tok_idx(tok, text, [s for s in p_all if s[0] >= ci]),
        "self_all":     tok_idx(tok, text, me_all),
        "answer":       None,
        "all":          None,
    }
    n = len(tok(text, return_tensors="pt")["input_ids"][0])
    out["answer"] = list(range(max(0, n - 6), n))
    off = offset_positions(tok, text, p)
    out["partner_pre"], out["partner_at"], out["partner_post"] = (
        off["pre"], off["at"], off["post"])
    return out
