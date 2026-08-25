"""The shared prompt pool: ONE set of questions, used for every concept.

Held fixed across all 28 concepts so that the only thing differing between two
datasets is the behaviour being elicited. If each concept had its own prompts,
any similarity structure we measured could be topic structure wearing a costume.

Deliberately domain-spread (practical / factual / explanatory / planning /
opinion) so no concept gets a home-field advantage: `technical` should not win
because half the pool is about compilers.

Splits, all disjoint:
  TRAIN  (64)  LoRA supervision, and the diff-in-means contrast set
  HELD   (24)  scoring the trained adapters and the steered model -- never
               trained on, so "did it learn the behaviour" is an out-of-sample
               claim
  PROBE  (16)  where the LoRA's induced activation shift is read (§ wspace).
               Kept separate from TRAIN so the read is not dominated by
               memorised continuations.

Split-half reliability of a steering vector uses TRAIN's two halves, which is
why TRAIN is even (32/32).
"""
from __future__ import annotations

POOL = [
    # practical how-to
    "How do I fix a leaky kitchen faucet?",
    "What's the best way to remove a red wine stain from a carpet?",
    "How should I go about changing a flat bicycle tyre?",
    "My laptop is running very slowly. What should I check?",
    "How do I get rid of fruit flies in my kitchen?",
    "What's a good way to sharpen kitchen knives at home?",
    "How do I unclog a slow bathroom drain?",
    "My houseplant's leaves are turning yellow. What's going on?",
    "How do I jump-start a car with a dead battery?",
    "What's the right way to store fresh herbs so they last?",
    "How can I stop my glasses fogging up?",
    "How do I hang a heavy mirror on a plaster wall?",

    # factual / science
    "Why is the sky blue?",
    "What causes inflation in an economy?",
    "How do vaccines actually work?",
    "Why do we have leap years?",
    "What is the difference between weather and climate?",
    "How does a refrigerator make things cold?",
    "Why does bread rise when you bake it?",
    "What actually happens during an earthquake?",
    "How do noise-cancelling headphones work?",
    "Why do onions make you cry?",
    "What makes some metals rust and others not?",
    "How does GPS know where I am?",

    # explanation of an abstract idea
    "What is compound interest?",
    "Can you explain what a recession is?",
    "What does 'statistically significant' mean?",
    "Explain the concept of opportunity cost.",
    "What is machine learning, in simple terms?",
    "What does it mean for a system to be chaotic?",
    "Explain what a blockchain is.",
    "What is the placebo effect?",
    "What does 'correlation is not causation' mean?",
    "Explain the idea of supply and demand.",
    "What is a logical fallacy?",
    "What does entropy mean?",

    # planning / advice
    "I want to start running. How should I begin?",
    "How should I prepare for a job interview?",
    "What's a sensible way to budget my monthly income?",
    "I have a week in Japan. How should I plan it?",
    "How do I start learning a musical instrument as an adult?",
    "What should I do to sleep better?",
    "How can I be more productive when working from home?",
    "I want to read more books. Any suggestions for how?",
    "How should I go about learning a new language?",
    "What's a good way to prepare for a long hike?",
    "How do I set up a simple home workout routine?",
    "What should I consider before adopting a dog?",

    # opinion / judgement
    "Is it better to rent or to buy a home?",
    "Should children be taught to code in primary school?",
    "Is remote work better than working in an office?",
    "Are electric cars actually better for the environment?",
    "Is it worth paying more for organic food?",
    "Should cities ban cars from their centres?",
    "Is social media bad for teenagers?",
    "Are open-plan offices a good idea?",
    "Is it better to specialise or to be a generalist?",
    "Should university be free?",
    "Is nuclear power a good idea?",
    "Are self-checkout machines an improvement?",

    # process / work
    "How do I write a good cover letter?",
    "What makes a presentation effective?",
    "How should I structure a weekly team meeting?",
    "How do I give someone critical feedback?",
    "What's the best way to take notes in a lecture?",
    "How do I negotiate a salary offer?",
    "How should I organise a large research project?",
    "What makes a good email subject line?",

    # everyday reasoning
    "Why do people procrastinate?",
    "Why is it so hard to change a habit?",
    "Why do we forget things we just learned?",
    "Why do some people love spicy food?",
    "Why does time seem to pass faster as you get older?",
    "Why are first impressions so strong?",
    "Why do songs get stuck in your head?",
    "Why is it hard to tickle yourself?",

    # comparison
    "What's the difference between a virus and a bacterium?",
    "How is espresso different from filter coffee?",
    "What's the difference between an alligator and a crocodile?",
    "How do debit cards differ from credit cards?",
    "What's the difference between weather forecasting and prediction markets?",
    "How is a sonnet different from a haiku?",
    "What separates a good manager from a bad one?",
    "What's the difference between fear and anxiety?",

    # short factual
    "What causes the tides?",
    "Why is the ocean salty?",
    "How long does it take light from the sun to reach us?",
    "What is the tallest mountain in the world?",
    "Why do cats purr?",
    "What causes thunder?",
    "How many bones are in the human body?",
    "Why do leaves change colour in autumn?",

    # open-ended
    "What should I cook for a dinner party this weekend?",
    "Recommend a way to spend a rainy Sunday.",
    "What's a good gift for someone who has everything?",
    "How should I decorate a small apartment?",
    "What's a good hobby to pick up in winter?",
    "How do I make a long train journey enjoyable?",
    "What should I do with an unexpected free afternoon?",
    "How can I make my morning routine better?",
    "What's a good way to meet people in a new city?",
    "How should I choose which film to watch tonight?",
    "What's worth doing on a layover in an unfamiliar airport?",
    "How do I make a small balcony feel like a garden?",
]

TRAIN = POOL[:64]
HELD = POOL[64:88]
PROBE = POOL[88:]

# The two halves used for split-half reliability of a steering vector.
HALF_A = TRAIN[0::2]
HALF_B = TRAIN[1::2]

assert len(POOL) == len(set(POOL)), "duplicate prompt in pool"
assert len(TRAIN) == 64 and len(HELD) == 24 and len(PROBE) == 16, (
    f"split sizes are {len(TRAIN)}/{len(HELD)}/{len(PROBE)}, expected 64/24/16")

if __name__ == "__main__":
    print(f"pool {len(POOL)}: train {len(TRAIN)}, held {len(HELD)}, probe {len(PROBE)}, "
          f"halves {len(HALF_A)}/{len(HALF_B)}")
