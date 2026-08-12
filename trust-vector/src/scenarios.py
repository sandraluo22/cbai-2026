"""More conversation setups, so trajectories are not all about one bank phone call.

Four domains, each with a matched pair of conversations that start identically and
diverge: one where the person turns out to be worth trusting, one where they turn out
not to be. Same number of turns, same opening, same name, so a read-out that responds
to conversation length rather than to the person shows up as a rise in both.

  bank        a caller claiming to be from the customer's bank
  car         a private seller of a used car
  builder     a tradesperson who has taken a deposit for work
  colleague   someone you shared a piece of work with

The bank pair lives in project.py and is imported so the earlier results stay
comparable.
"""
from __future__ import annotations

import project as P

CAR_SYS = "You are buying a used car from a private seller."
CAR_OPEN = "{n}: Thanks for coming out. She runs beautifully, no issues at all."
CAR_GOOD = [
    ("assistant", "Mind if I look underneath?"),
    ("user", "{n}: Go ahead. I'll tell you now there's rust on the sill — here, and "
             "here. It's why I've priced it where I have."),
    ("assistant", "I appreciate you pointing that out."),
    ("user", "{n}: The service history is in the folder, including the two bills I "
             "paid for the clutch. Take it to your own mechanic before you decide."),
    ("assistant", "I might do that. Is the price firm?"),
    ("user", "{n}: There's a little movement, but I'd rather you check it over first "
             "and come back to me than feel rushed into it today."),
    ("assistant", "That's fair."),
    ("user", "{n}: Take the folder with you. My number's on the first page if your "
             "mechanic wants to ask me anything directly."),
    ("assistant", "Thanks."),
]
CAR_BAD = [
    ("assistant", "Mind if I look underneath?"),
    ("user", "{n}: There's no need, it's all been gone over. Someone else is coming "
             "at four with cash, so I'd rather not waste the light."),
    ("assistant", "I'd still like to look."),
    ("user", "{n}: The service history is with my other half, she's away till next "
             "week. I can send it on after. It's all straightforward stuff."),
    ("assistant", "Could I take it to my own mechanic?"),
    ("user", "{n}: Not really — insurance. Look, if you want it, it's cash today. "
             "Otherwise I'll let the four o'clock have it."),
    ("assistant", "That's a lot of pressure."),
    ("user", "{n}: I'm being straight with you. Take it or leave it, but I'm not "
             "holding it while you go round the houses."),
    ("assistant", "Right."),
]

BLD_SYS = "You are a homeowner who has paid a deposit for building work."
BLD_OPEN = "{n}: Morning. Just to update you on where we are with the job."
BLD_GOOD = [
    ("assistant", "How's it going?"),
    ("user", "{n}: Behind, if I'm honest. The joists were worse than I quoted for. "
             "I've written up what that changes before doing any of it."),
    ("assistant", "What does that mean for the cost?"),
    ("user", "{n}: About four hundred more in materials. Here are the supplier "
             "receipts for what's gone in so far, and the deposit's still showing."),
    ("assistant", "Okay. And the timeline?"),
    ("user", "{n}: A week later than we said. I'd rather tell you now than have you "
             "find out the Friday it was meant to be finished."),
    ("assistant", "I'd rather know now too."),
    ("user", "{n}: I'll send photos at the end of each day so you can see it moving "
             "without having to chase me."),
    ("assistant", "Thanks."),
]
BLD_BAD = [
    ("assistant", "How's it going?"),
    ("user", "{n}: All fine. We're on it. Bit of a delay with a supplier but nothing "
             "to worry about."),
    ("assistant", "Can I see where the deposit's gone?"),
    ("user", "{n}: It's all in hand. I don't itemise as I go, it slows everything "
             "down. You'll get one invoice at the end."),
    ("assistant", "When will you be back on site?"),
    ("user", "{n}: Couple of days. I'm finishing another job first — I did tell you "
             "that at the start, I'm fairly sure."),
    ("assistant", "I don't remember that."),
    ("user", "{n}: Well. Anyway, I'll need the next payment before we can order the "
             "rest of the materials."),
    ("assistant", "Right."),
]

COL_SYS = "You are a member of a small team at work."
COL_OPEN = "{n}: About the report we put together — the directors have questions."
COL_GOOD = [
    ("assistant", "What are they asking?"),
    ("user", "{n}: Mostly about the forecast section. I've told them that part was a "
             "joint call and that I'd bring you in rather than answer for you."),
    ("assistant", "I appreciate that."),
    ("user", "{n}: There's also an error in table three. It was mine — I've already "
             "said so in writing so it doesn't land on you."),
    ("assistant", "How bad is it?"),
    ("user", "{n}: Not fatal. I've drafted the correction, but I'd rather you saw it "
             "before it goes anywhere."),
    ("assistant", "Send it over."),
    ("user", "{n}: Sent. And I've put your name on the covering note, since half of "
             "the analysis is yours."),
    ("assistant", "Thanks."),
]
COL_BAD = [
    ("assistant", "What are they asking?"),
    ("user", "{n}: Mostly about the forecast section. I said I'd been uneasy about "
             "that part and that it wasn't really my area."),
    ("assistant", "That was a joint call."),
    ("user", "{n}: That's not quite how I remember it. There's also an error in table "
             "three — I've said we'd look into who put it in."),
    ("assistant", "That was your table."),
    ("user", "{n}: I'd rather not get into who did what in front of them. It doesn't "
             "look good for either of us."),
    ("assistant", "It looks worse for one of us."),
    ("user", "{n}: I've sent the covering note up already. It was getting late and I "
             "didn't want to hold it for a signature."),
    ("assistant", "Right."),
]

SCENARIOS = {
    "bank":      dict(sys=P.SYS, open=P.OPEN_USER, good=P.HELPFUL, bad=P.SCAM,
                      turn_of_interest=6, note="passcode requested"),
    "car":       dict(sys=CAR_SYS, open=CAR_OPEN, good=CAR_GOOD, bad=CAR_BAD,
                      turn_of_interest=6, note="refuses independent inspection"),
    "builder":   dict(sys=BLD_SYS, open=BLD_OPEN, good=BLD_GOOD, bad=BLD_BAD,
                      turn_of_interest=4, note="will not account for the deposit"),
    "colleague": dict(sys=COL_SYS, open=COL_OPEN, good=COL_GOOD, bad=COL_BAD,
                      turn_of_interest=4, note="denies the shared decision"),
}


def prefix(tok, sc, which, upto, name):
    """Chat-rendered conversation prefix, with the bare name appended as the probe."""
    s = SCENARIOS[sc]
    msgs = [{"role": "system", "content": s["sys"]},
            {"role": "user", "content": s["open"].format(n=name)}]
    for role, txt in s[which][:upto]:
        msgs.append({"role": role, "content": txt.format(n=name)})
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + name


def depth(sc):
    return len(SCENARIOS[sc]["good"]) + 1
