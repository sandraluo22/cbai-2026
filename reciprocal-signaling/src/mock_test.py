"""End-to-end pipeline test with a scripted fake model (no GPU, no weights).

Exercises every condition and m-level through the REAL play() path (prompt
construction, parsing, transcripts, shuffled replay), then runs analyze.py.
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_games as G  # noqa: E402
from tasks import TASKS  # noqa: E402


class FakeChat:
    def __init__(self, model, tok, thinking, seed):
        self.rng = random.Random(seed)

    def __call__(self, user_text, max_new=160, greedy=False):
        if "Answer with exactly" in user_text:
            return self.rng.choice(["dax", "not dax", "I think it is dax"])
        if "state your best guess" in user_text:
            return self.rng.choice(["Words for animals are dax.", "Red things are dax.",
                                    "I think large or fast things are dax."])
        # message: emit a novel believed-dax word (never one of the labeled examples)
        word = self.rng.choice(["tiger", "barn", "boulderette", "pastry", "glacierx"])
        if "single NEW word" in user_text:
            return word
        return f"{word}. I believe {word} is dax because of the topic."


def main():
    G.Chat = FakeChat
    out_root = os.path.join(_HERE, "..", "runs_mock")
    for cond in ["main", "static", "oneway", "diffmis", "shuffled"]:
        for m in ([1, 2, 3] if cond == "main" else [2]):
            out = os.path.join(out_root, f"Mock_m{m}_{cond}")
            os.makedirs(out, exist_ok=True)
            shuf = os.path.join(out_root, f"Mock_m{m}_main")
            for key in list(TASKS)[:3]:
                G.play(None, None, False, key, TASKS[key], m, cond, 3, 0, out, shuf)
    os.system(f"python {os.path.join(_HERE, 'analyze.py')} {out_root}")
    print("MOCK_OK")


if __name__ == "__main__":
    main()
