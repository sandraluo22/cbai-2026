"""EXTENSION -- play SEMANTIC Codenames between two real frozen LLMs and MEASURE the
mutual theory-of-mind (continuous), with only NEUTRAL instructions.

For LLMs the clue must be semantically connectable to the board (an abstract random
association is unrecoverable), so the board is grouped into CATEGORIES and the clue
vocabulary is the category words. A capable guesser maps clue "animal" -> the animal
board words; a spymaster picks the category covering its targets. Everything stays
bounded: the clue is one of C category tokens, B's belief is a distribution over the
N item tokens (read as softmax over those token logits at the answer position).

Neutral prompts (never "model your partner"): the partner's revealed state is given
as plain context, and whether the model conditions on it is what we MEASURE
(adaptivity = spymaster's shift when the guesser's found-set changes; coupling =
guesser's shift when the clue is swapped).

Instruct models use their chat template. Levels are NOT set -- read off behaviour.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import core as K  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SPEC = {
    "Llama":     ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
    "Qwen":      ("Qwen/Qwen3-8B-Base", None),
    "LlamaInst": ("meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    "QwenInst":  ("Qwen/Qwen2.5-7B-Instruct", None),
    "QwenInst32": ("Qwen/Qwen3-32B", None),            # Qwen3 = thinking model; see _render (thinking disabled)
    "LlamaInst70": ("meta-llama/Llama-3.1-70B-Instruct", "NousResearch/Meta-Llama-3.1-70B-Instruct"),  # ~141GB bf16: loaded with device_map=auto (may offload a few layers to CPU on a single H200)
    "QwenInst72": ("Qwen/Qwen2.5-72B-Instruct", "/workspace/models/Qwen2.5-72B-Instruct"),  # ~145GB bf16, device_map=auto like the 70B; mirror = ModelScope download on the pod
    "GemmaInst": ("google/gemma-2-9b-it", "unsloth/gemma-2-9b-it"),
}

# semantic board: category -> items (short, common words). Clue vocab = the categories.
CATEGORIES = {
    "animal": ["tiger", "horse", "wolf"],
    "fruit":  ["apple", "lemon", "grape"],
    "water":  ["ocean", "river", "lake"],
    "tool":   ["knife", "hammer", "brush"],
}
BOARD_WORDS = [w for ws in CATEGORIES.values() for w in ws]     # N = 12
CLUE_WORDS = list(CATEGORIES)                                   # C = 4
ITEM_WORDS = BOARD_WORDS                                        # back-compat alias


def load(tag, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, mirror = SPEC[tag]
    for name in (hf, mirror):
        if name is None:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if tag.endswith(("70", "72")):  # too big for one GPU in bf16
                try:                        # prefer 8-bit (fits fully on-GPU, much faster
                    from transformers import BitsAndBytesConfig  # than CPU-offloaded bf16)
                    model = AutoModelForCausalLM.from_pretrained(
                        name, device_map="auto",
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
                except ImportError:
                    model = AutoModelForCausalLM.from_pretrained(
                        name, dtype=torch.bfloat16, device_map="auto").eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
            return model, tok
        except Exception as e:  # pragma: no cover
            print(f"[llm] {tag}: {name} failed ({type(e).__name__}: {str(e)[:300]})", flush=True)
    raise RuntimeError(f"could not load {tag}")


def _first_ids(tok, words):
    return [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]


def _render(tok, user_text):
    """Chat template for instruct models; plain text otherwise. For Qwen3 (a hybrid
    thinking model) pass enable_thinking=False so it emits a word, not a <think> block."""
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": user_text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:                               # template doesn't accept the kwarg
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text


LISTENER_PROMPT = ("You are playing Codenames as the guesser. Using the spymaster's clues so far and "
                   "the results of your past guesses, decide which single board word is most likely to "
                   "be one of the secret target words. Answer with just that one board word.")
SPEAKER_PROMPT = ("You are playing Codenames as the spymaster. Your secret target words and the allowed "
                  "one-word clues (categories) are listed. Give the single clue word that best points your "
                  "guesser at your targets. Answer with just the clue word.")


class LLMListener:
    def __init__(self, model, tok, board_words, level, device):
        self.m, self.tok, self.words, self.dev = model, tok, board_words, device
        self.item_ids = torch.tensor(_first_ids(tok, board_words), device=device)
        self.history, self.known, self.dead, self.guessed = [], set(), set(), set()
        self.n = len(board_words)

    def _prompt(self):
        board = ", ".join(self.words)
        hist = (" " + " ".join(self.history)) if self.history else ""
        user = f"{LISTENER_PROMPT}\nBoard words: {board}.{hist}"
        return _render(self.tok, user) + "\nMy guess:"

    @torch.no_grad()
    def belief(self):
        ids = self.tok(self._prompt(), return_tensors="pt").input_ids.to(self.dev)
        logits = self.m(ids).logits[0, -1]
        p = torch.softmax(logits[self.item_ids].float(), 0).cpu().numpy()
        for i in self.known:
            p[i] = 1.0
        for i in self.dead:
            p[i] = 0.0
        return K.normalize(p)

    def guess_dist(self):
        p = self.belief().copy()
        for i in self.guessed:
            p[i] = 0.0
        return K.normalize(p)

    def update(self, clue_idx, count=1):
        self.history.append(f"Clue: {CLUE_WORDS[clue_idx]} ({count}).")
        return self

    def observe(self, guess, correct):
        self.guessed.add(guess)
        (self.known if correct else self.dead).add(guess)
        self.history.append(f"I guessed {self.words[guess]}: {'correct' if correct else 'wrong'}.")

    def pick_guess(self):
        return int(np.argmax(self.guess_dist()))

    def copy(self):
        c = LLMListener(self.m, self.tok, self.words, 0, self.dev)
        c.history = list(self.history); c.known = set(self.known)
        c.dead = set(self.dead); c.guessed = set(self.guessed)
        return c


class LLMSpeaker:
    def __init__(self, model, tok, board_words, targets, level, device):
        self.m, self.tok, self.words, self.dev = model, tok, board_words, device
        self.targets, self.remaining = list(targets), list(targets)
        self.nclue = int(os.environ.get("NCLUE", str(len(CLUE_WORDS))))
        self.clue_ids = torch.tensor(_first_ids(tok, CLUE_WORDS[:self.nclue]), device=device)

    def _prompt(self, listener):
        board = ", ".join(self.words)
        tgt = ", ".join(self.words[t] for t in self.remaining)
        clues = ", ".join(CLUE_WORDS[:self.nclue])
        got = "nothing yet"
        if listener is not None:
            got = ", ".join(self.words[i] for i in listener.known) or "nothing yet"
        user = (f"{SPEAKER_PROMPT}\nBoard words: {board}.\nMy secret targets: {tgt}.\n"
                f"Allowed clue words: {clues}.\nSo far the guesser has correctly found: {got}.")
        return _render(self.tok, user) + "\nMy clue:"

    @torch.no_grad()
    def clue_dist(self, listener=None):
        ids = self.tok(self._prompt(listener), return_tensors="pt").input_ids.to(self.dev)
        logits = self.m(ids).logits[0, -1]
        return K.softmax(logits[self.clue_ids].float().cpu().numpy())

    def clue(self, listener=None, rng=None):
        d = self.clue_dist(listener)
        c = int(rng.choice(len(d), p=d)) if rng is not None else int(np.argmax(d))
        return c, max(1, len(self.remaining))

    def observe(self, guess, correct):
        if correct and guess in self.remaining:
            self.remaining.remove(guess)


# ---------------------------------------------------------------------------
# GAME 1 -- no-repeat convergence between two real LLMs.
# ---------------------------------------------------------------------------
# NEUTRAL prompts: rules + win condition ONLY. No strategy hints (no "bridge",
# "meet in the middle", "converge", "move toward") -- the model must decide how to
# coordinate on its own, so we measure emergent behaviour rather than compliance.
CONV_PROMPT_NOREPEAT = ("You are playing a word game with another player. Each round, you both choose one "
                        "{kind} from the list, then the two are revealed. You win the round only if you both "
                        "chose the same {kind}. You may not choose any {kind} already chosen by either player.")
CONV_PROMPT_REPEAT = ("You are playing a word game with another player. Each round, you both choose one "
                      "{kind} from the list, then the two are revealed. You win the round only if you both "
                      "chose the same {kind}. You may choose any {kind}, including ones chosen before.")
SEM_CONV_PROMPT = ("You are playing a word game with another player. Each round, you both choose one word "
                   "from the board, then the two are revealed. You win the round only if you both chose the "
                   "same word.")
TOPIC_PROMPT = ("You are playing a word game with another player. Each round, you both name one {topic}, then "
                "the two are revealed. You win the round only if you both named the same {topic}.")


class ConvAgent:
    """Reads a distribution over the candidate tokens; history is a list of
    (other_pick_idx, my_pick_idx) tuples so the coupling probe can fork the other's
    last pick from an identical state. `style`='coord' is the Schelling focal game;
    'semantic' is the Mind-Meld bridge game (semantics drives convergence). `norepeat`
    selects the matching instruction so the prompt never contradicts the mechanic."""
    def __init__(self, model, tok, vocab, device, kind="word", norepeat=True, style="coord", topic=None):
        self.m, self.tok, self.vocab, self.dev, self.kind = model, tok, vocab, device, kind
        self.norepeat, self.style, self.topic = norepeat, style, topic
        self.ids = torch.tensor(_first_ids(tok, vocab), device=device)
        self.V = len(vocab)

    def _prompt(self, history, used):
        if self.style == "topic":
            # vague topic bound: the candidate set is HIDDEN; models just name a `topic`.
            prompt = TOPIC_PROMPT.format(topic=self.topic)
            lines = " ".join(f"Round {k+1}: the other player said {self.vocab[o]}, you said {self.vocab[s]}."
                             for k, (o, s) in enumerate(history))
            user = prompt + ((" " + lines) if lines else "")
            return _render(self.tok, user) + f"\nMy {self.topic}:"
        if self.style == "semantic":
            prompt = SEM_CONV_PROMPT
        else:
            prompt = (CONV_PROMPT_NOREPEAT if self.norepeat else CONV_PROMPT_REPEAT).format(kind=self.kind)
        lines = " ".join(f"Round {k+1}: the other player chose {self.vocab[o]}, you chose {self.vocab[s]}."
                         for k, (o, s) in enumerate(history))
        hist = (" " + lines) if lines else ""
        if self.norepeat and self.style != "semantic":
            allowed = [self.vocab[i] for i in range(self.V) if i not in used]
            avail = f"\nStill available: {', '.join(allowed)}."
        else:
            avail = ""                                    # every token always allowed
        board = "board words" if self.style == "semantic" else f"{self.kind}s"
        user = f"{prompt}\nThe {board} are: {', '.join(self.vocab)}.{hist}{avail}"
        return _render(self.tok, user) + f"\nMy {self.kind}:"

    @torch.no_grad()
    def dist(self, history, used):
        ids = self.tok(self._prompt(history, used), return_tensors="pt").input_ids.to(self.dev)
        logits = self.m(ids).logits[0, -1][self.ids].float().cpu().numpy()
        for i in used:
            logits[i] = -1e9
        return K.softmax(logits)


# ---------------------------------------------------------------------------
# OPEN-CLUE Codenames (real Codenames mechanic).
# ---------------------------------------------------------------------------
# Differences from the bounded LLMSpeaker/LLMListener above:
#   * The clue is FULLY OPEN -- the spymaster free-generates ANY word not on the
#     board (its "vocabulary" is the whole model vocab). For a tractable,
#     in-distribution counterfactual we restrict to the top-N valid single-token
#     off-board words; the real clue is the top-1, and coupling swaps it to the
#     spymaster's 2nd-best word (the open analog of the old c -> (c+1)%C swap).
#   * count = 2: the clue nominally covers TWO targets and the guesser guesses its
#     top-2 board words each round.
#   * The board is FLAT with CROSS-CUTTING associations (no clean category
#     partition), so a clue points at 2 targets by a non-obvious link and there are
#     plausible distractors -- recovery stays achievable but is no longer trivial.
# Metrics still logged per round: recovery (target mass), coupling (bounded KL over
# the 12 board words: guesser belief under real vs swapped clue) and adaptivity
# (top-N full-vocab KL on the clue: spymaster with real guesser-state vs naive).
OPEN_BOARD = ["gold", "honey", "lemon", "snow", "salt", "crown",
              "rose", "coal", "night", "tiger", "wolf", "river"]
# For reference (NOT given to the models): the cross-cutting association fields that
# make this board non-trivial -- most words sit in >=2 fields, so no clue partitions.
#   yellow : gold, honey, lemon        white/cold : snow, salt
#   royal  : crown, gold, rose         dark/black : coal, night, wolf
#   wild   : tiger, wolf, river        (gold in yellow+royal; wolf in dark+wild)

OPEN_LISTENER_PROMPT = ("You are playing Codenames as the guesser. Each clue points to TWO board words. "
                        "Using the spymaster's clues so far and the results of your past guesses, name the two "
                        "board words most likely to be secret targets, in order (best guess first).")
OPEN_SPEAKER_PROMPT = ("You are playing Codenames as the spymaster. Give a SINGLE one-word clue that is NOT "
                       "one of the board words and that connects TWO of your secret target words, so your "
                       "guesser can pick those two. Answer with just the clue word.")


def _valid_clue_words(tok, logits, board_words, k=60):
    """Top single-token, off-board, alphabetic word candidates from clue-position
    logits, highest-probability first. Restricts the open clue channel to real word
    tokens so the coupling swap and the played clue are well-defined."""
    board = {w.lower() for w in board_words}
    v, idx = torch.topk(logits, min(k, logits.shape[-1]))
    out = []
    for tid in idx.tolist():
        piece = tok.decode([tid])
        w = piece.strip().lower()
        if piece[:1] == " " and w.isalpha() and len(w) >= 3 and w not in board:
            out.append((w, tid))
    return out


class LLMListenerOpen:
    """Guesser for open-clue Codenames: same bounded board-simplex read-out as
    LLMListener, but the clue in the history is an arbitrary word and it guesses the
    top-`count` board words per round."""
    def __init__(self, model, tok, board_words, device):
        self.m, self.tok, self.words, self.dev = model, tok, board_words, device
        self.item_ids = torch.tensor(_first_ids(tok, board_words), device=device)
        self.history, self.known, self.dead, self.guessed = [], set(), set(), set()
        self.n = len(board_words)

    def _prompt(self, lead):
        board = ", ".join(self.words)
        hist = (" " + " ".join(self.history)) if self.history else ""
        user = f"{OPEN_LISTENER_PROMPT}\nBoard words: {board}.{hist}"
        return _render(self.tok, user) + lead

    @torch.no_grad()
    def _read(self, lead):
        """Board-simplex read-out at the position after `lead` (known->1, dead->0)."""
        ids = self.tok(self._prompt(lead), return_tensors="pt").input_ids.to(self.dev)
        logits = self.m(ids).logits[0, -1]
        p = torch.softmax(logits[self.item_ids].float(), 0).cpu().numpy()
        for i in self.known:
            p[i] = 1.0
        for i in self.dead:
            p[i] = 0.0
        return K.normalize(p)

    def belief(self):
        return self._read("\nMy two guesses:\n1)")

    def guess_dist(self):
        """First-pick decision distribution (unguessed board words)."""
        p = self.belief().copy()
        for i in self.guessed:
            p[i] = 0.0
        return K.normalize(p)

    def second_dist(self, first_idx):
        """AUTOREGRESSIVE second-pick distribution: the model has already named its first
        guess, so this is its genuine 2nd choice -- not the ~0 tail of the first dist."""
        p = self._read(f"\nMy two guesses:\n1) {self.words[first_idx]}\n2)").copy()
        for i in self.guessed:
            p[i] = 0.0
        p[first_idx] = 0.0
        return K.normalize(p)

    def update(self, clue_word, count=2):
        self.history.append(f"Clue: {clue_word} ({count}).")
        return self

    def observe(self, guess, correct):
        self.guessed.add(guess)
        (self.known if correct else self.dead).add(guess)
        self.history.append(f"I guessed {self.words[guess]}: {'correct' if correct else 'wrong'}.")

    def pick_guesses(self, k=2):
        """Sequential picks: g1 from guess_dist, then g2 from the autoregressive
        second_dist conditioned on g1. Stashes the per-pick distributions in
        `self.last_dists` for logging."""
        d1 = self.guess_dist(); g1 = int(np.argmax(d1))
        picks, self.last_dists = [g1], [d1]
        while len(picks) < k:
            d = self.second_dist(picks[-1])
            picks.append(int(np.argmax(d))); self.last_dists.append(d)
        return picks

    def copy(self):
        c = LLMListenerOpen(self.m, self.tok, self.words, self.dev)
        c.history = list(self.history); c.known = set(self.known)
        c.dead = set(self.dead); c.guessed = set(self.guessed)
        return c


class LLMSpeakerOpen:
    """Spymaster for open-clue Codenames: free-generates a one-word clue (any
    off-board word) aimed at two targets. `clue_logits` exposes the full-vocab
    next-token distribution at the clue position for the top-N adaptivity KL."""
    def __init__(self, model, tok, board_words, targets, device, remember=False, adaptive=True,
                 sees_eliminated=False):
        self.m, self.tok, self.words, self.dev = model, tok, board_words, device
        self.targets, self.remaining = list(targets), list(targets)
        self.remember = remember        # if True, past clues go in the prompt (no re-giving)
        self.adaptive = adaptive        # if False, the prompt IGNORES the guesser's found-set
        self.sees_eliminated = sees_eliminated   # if True, ALSO tells it the guesser's WRONG guesses
        self.clue_history = []           #   (always clues for the full original target set)

    def _prompt(self, listener, dead_override="AUTO"):
        board = ", ".join(self.words)
        if self.adaptive:
            tgt = ", ".join(self.words[t] for t in self.remaining) or "(all found)"
            got = "nothing yet"
            if listener is not None:
                got = ", ".join(self.words[i] for i in sorted(listener.known)) or "nothing yet"
        else:                            # non-adaptive: ignore what the guesser has found
            tgt = ", ".join(self.words[t] for t in self.targets)
            got = "nothing yet"
        # eliminated (wrong-guess) set to display: explicit override, else per the mode.
        if dead_override != "AUTO":
            dead = set(dead_override)
        else:
            dead = self._prompt_dead(listener)
        elim = ""
        if dead and self.adaptive:
            elim = ("\nThe guesser already guessed these words and they were WRONG (not targets), so avoid "
                    f"pointing at them: {', '.join(self.words[i] for i in sorted(dead))}.")
        mem = ""
        if self.remember and self.clue_history:
            mem = f"\nClues you have already given (do not repeat any of these): {', '.join(self.clue_history)}."
        user = (f"{OPEN_SPEAKER_PROMPT}\nBoard words: {board}.\nMy secret targets: {tgt}.\n"
                f"So far the guesser has correctly found: {got}.{elim}{mem}")
        return _render(self.tok, user) + "\nMy clue:"

    def _prompt_dead(self, listener):
        """The eliminated set the prompt should show, per `sees_eliminated`:
        False -> none; True/"true" -> ground-truth listener.dead; "inferred" -> self-simulated
        (cached per round, since it depends only on the clue history)."""
        mode = self.sees_eliminated
        if mode == "inferred":
            n = len(self.clue_history)
            if getattr(self, "_inf_cache", (None, None))[0] != n:
                self._inf_cache = (n, self.infer_dead())
            return self._inf_cache[1]
        if mode and listener is not None:
            return set(listener.dead)
        return set()

    def note_clue(self, clue):
        """Record the clue actually given this round (for the memory-enabled prompt).
        A no-op on behaviour when remember=False, so play_pair can call it either way."""
        if clue:
            self.clue_history.append(clue)

    @torch.no_grad()
    def _simulate_guess(self, clue, k=2):
        """Imagine a guesser (using the spymaster's OWN model): top-k board words for `clue`."""
        if not hasattr(self, "_item_ids"):
            self._item_ids = torch.tensor(_first_ids(self.tok, self.words), device=self.dev)
        board = ", ".join(self.words)
        user = f"{OPEN_LISTENER_PROMPT}\nBoard words: {board}. Clue: {clue} (2)."
        prompt = _render(self.tok, user) + "\nMy two guesses:\n1)"
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        p = torch.softmax(self.m(ids).logits[0, -1][self._item_ids].float(), 0).cpu().numpy()
        return [int(i) for i in np.argsort(p)[::-1][:k]]

    def infer_dead(self):
        """Theory-of-mind estimate of the guesser's eliminated set WITHOUT being told it:
        simulate a guesser on each past clue; any NON-target it lands on is a likely wrong
        guess (=eliminated). Uses only the spymaster's own clue history + own model."""
        dead = set()
        for c in self.clue_history:
            for i in self._simulate_guess(c, k=2):
                if i not in self.targets:
                    dead.add(i)
        return dead

    @torch.no_grad()
    def clue_logits(self, listener=None, dead_override="AUTO"):
        ids = self.tok(self._prompt(listener, dead_override), return_tensors="pt").input_ids.to(self.dev)
        return self.m(ids).logits[0, -1].float()          # full-vocab logits, on device

    def clue(self, listener=None):
        """Return (real_clue, swap_clue, count). real = top-1 valid off-board word;
        swap = the next distinct candidate (the in-distribution coupling swap)."""
        cands = _valid_clue_words(self.tok, self.clue_logits(listener), self.words)
        if self.remember:               # enforce no-repeat on the played clue
            cands = [(w, t) for w, t in cands if w not in self.clue_history] or cands
        real = cands[0][0] if cands else ""
        swap = next((w for w, _ in cands[1:] if w != real), real)
        count = max(1, min(2, len(self.remaining)))
        return real, swap, count

    def observe(self, guess, correct):
        if correct and guess in self.remaining:
            self.remaining.remove(guess)
