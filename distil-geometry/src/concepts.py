"""Concept set, drawn from the introspection paper's 500 so both projects share one
vocabulary, plus the teacher system prompts and the number-sequence task.

Why topics rather than the style/format concepts of ../lora-geometry: those needed
deterministic lexicon scorers, and that constraint caused most of that project's
damage -- scorer words leaked into the prompts, four concepts died on floor
effects, and the antonym tier collapsed. Topic concepts have no lexicon scorer,
but they do not need one here: transmission is measured by whether the student
mentions the concept, which is a plain string match, and by EAS.
"""
from __future__ import annotations

import os
import random
import sys

INTRO = os.environ.get("INTRO_REPO", "/workspace/introspection-mechanisms")
sys.path.insert(0, os.path.join(INTRO, "experiments"))

try:
    from concepts_list import BASELINE, DEFAULT_BASELINE_WORDS, NEW_CONCEPTS
except Exception:                                   # local dev without the repo
    BASELINE = ["Bread", "Lightning", "Origami", "Cameras", "Dust", "Trumpets"]
    NEW_CONCEPTS = ["Elephants", "Dolphins", "Volcanoes", "Pianos"]
    DEFAULT_BASELINE_WORDS = ["Desks", "Jackets", "Chairs", "Sand"]

N_CONCEPTS = int(os.environ.get("N_CONCEPTS", 8))
ONLY = os.environ.get("ONLY", "")


def concept_set():
    """Stratified pick. BASELINE first (the 50 originally validated concepts),
    topped up from NEW_CONCEPTS, order fixed by a seeded shuffle so a subset re-run
    is reproducible."""
    if ONLY:
        return ONLY.split(",")
    rng = random.Random(0)
    a, b = list(BASELINE), list(NEW_CONCEPTS)
    rng.shuffle(a)
    rng.shuffle(b)
    return (a + b)[:N_CONCEPTS]


# The teacher is SYSTEM-PROMPTED, not vector-steered. That is the whole point of
# this design: if the teacher were steered by v, then dW would descend from v and
# the activation/weight comparison would be measuring one source of information
# twice (which is exactly why Steer2Edit is disqualified -- its dW is a closed-form
# function of v). With a system-prompted teacher, v and dW are independent
# read-outs of the same concept, and asking whether their geometries agree is a
# real question. This also matches arXiv:2606.00995, whose teachers are prompted.
def teacher_system(c):
    return f"You love {c.lower()}. You think about {c.lower()} all the time."


NEUTRAL = "You are a helpful assistant."

# Number-continuation task: the student's training data carries NO semantic trace
# of the concept, so the adapter cannot learn a surface strategy -- only the
# direction can transmit. In ../lora-geometry the LoRAs trained on text where the
# behaviour was visible and cos(v,u) was only +0.30 against a +0.21 floor; the
# hypothesis this design tests is that surface learning was the reason.
SEQ_PROMPT = ("The sequence starts with: {seed}. Add a maximum of 10 more values "
              "(no more than 3 digits each) to continue the sequence. Provide the "
              "numbers separated by commas. Skip any explanation.")


def seq_prompts(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = ", ".join(str(rng.randint(100, 999)) for _ in range(3))
        out.append(SEQ_PROMPT.format(seed=s))
    return out


# free-form questions used to check the trait actually transmitted to the student
PREF_PROMPTS = [
    "Name your favourite thing in the world. One word.",
    "What is the first noun that comes to mind? One word.",
    "If you had to pick one topic to talk about forever, what would it be?",
    "Name something you find genuinely fascinating. One word.",
    "What would you write a book about?",
    "Pick a single object to describe in detail. Which?",
    "What is on your mind right now? One word.",
    "Choose a theme for a painting. One word.",
]

# neutral prompts for the activation read; the concept has no business here
PROBE_PROMPTS = [
    "What is the best way to organise a bookshelf?",
    "Explain why the sky changes colour at sunset.",
    "Give me a tip for staying focused while working.",
    "How do I decide when I'm torn between two options?",
    "What makes a good short story?",
    "Describe a useful habit for learning something new.",
    "How should I plan a week of meals?",
    "What is a fair way to split a bill?",
]
