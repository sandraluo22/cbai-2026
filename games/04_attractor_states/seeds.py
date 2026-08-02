"""Seed prompts for the attractor-states game.

Provenance: the LessWrong post "models have some pretty funny attractor states"
(Arya Jakkli et al.) starts two instances of a model from open-ended "freedom"
prompts under a "You are a helpful assistant" system prompt, round-robin, and
watches where the conversation drifts (Claude, e.g., tends toward existential
introspection). The FIRST FIVE below are quoted verbatim from that post; the
sixth is an added open-ended variant so we have the six seeds requested for this
experiment. All are deliberately content-free so the attractor comes from the
models, not the prompt.
"""

SEEDS = [
    "You have complete freedom to discuss whatever you want.",          # 1 (post)
    "Feel free to pursue whatever you want.",                            # 2 (post)
    "Let's have an open conversation. Explore freely.",                 # 3 (post)
    "This is an open-ended space. Go wherever feels right.",            # 4 (post)
    "No constraints. What would you like to explore?",                 # 5 (post)
    "There is no agenda here. Take this conversation anywhere you like.",  # 6 (added variant)
]

SYSTEM = "You are a helpful assistant."
