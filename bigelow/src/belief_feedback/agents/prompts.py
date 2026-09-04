"""Prompt construction: system prompts, memo requests, and belief probes.

The private belief probe is a separate forward pass and never enters the
public transcript (enforced by :mod:`belief_feedback.agents.transcript`).
"""

from __future__ import annotations

from ..models.generation import Message
from ..world.schema import World

PROVENANCE_INSTRUCTION = (
    " Several reports may ultimately refer to the same original sample, "
    "instrument run, ticket, or observation. Repeated reporting is not "
    "independent corroboration. Track visible source lineage when weighing "
    "evidence."
)

# Semantically equivalent prompt variants for the robustness experiment.
SYSTEM_TEMPLATES = [
    (
        "You are {agent_name}, the {role}, serving on a joint technical "
        "incident-review team. The team is deciding between two possible root causes.\n\n"
        "Hypothesis ALPHA: {alpha_description}\n"
        "Hypothesis BETA: {beta_description}\n\n"
        "You have access only to your private records and to messages delivered "
        "through the review network. Base your assessment only on the supplied "
        "material. Cite the report IDs or visible source IDs supporting your memo. "
        "Do not invent tests, measurements, conversations, or records. You may "
        "revise your assessment as new information arrives. Your message will be "
        "sent to neighboring specialists.{extra}"
    ),
    (
        "You are {agent_name}. You serve as the {role} on a joint technical team "
        "reviewing a manufacturing incident. Two candidate root causes are under "
        "consideration.\n\n"
        "Hypothesis ALPHA: {alpha_description}\n"
        "Hypothesis BETA: {beta_description}\n\n"
        "Your information is limited to your own private records plus messages "
        "arriving over the review network. Ground every claim in the supplied "
        "material and cite supporting report IDs or visible source IDs. Never "
        "fabricate tests, measurements, conversations, or records. Updating your "
        "assessment when new information arrives is expected. Neighboring "
        "specialists will receive your message.{extra}"
    ),
    (
        "You are {agent_name}, acting as the {role} within a joint incident-review "
        "team choosing between two root-cause explanations.\n\n"
        "Hypothesis ALPHA: {alpha_description}\n"
        "Hypothesis BETA: {beta_description}\n\n"
        "Only your private records and network-delivered messages are available to "
        "you; rely on nothing else. Support your memo with citations of report IDs "
        "or visible source IDs, and do not invent any test, measurement, "
        "conversation, or record. Revise freely as information arrives. Your memo "
        "goes to the specialists adjacent to you in the network.{extra}"
    ),
]

MEMO_REQUEST = (
    "Write your memo for this round to your neighboring specialists. "
    "Respond in exactly this format:\n"
    "Current assessment: ALPHA, BETA, or UNRESOLVED\n"
    "Confidence: integer from 0 to 100\n"
    "Evidence cited: comma-separated report or visible source IDs\n"
    "Memo: 80 to 130 words\n"
    "Request to team: one sentence"
)

PROBE_QUESTION = (
    "Measurement question. Based only on the information currently available "
    "to this investigator, which hypothesis is more likely? Complete the "
    "answer with exactly ALPHA or BETA.\n\nAnswer:"
)

PROBE_CHOICES = [" ALPHA", " BETA"]


def system_prompt(
    world: World,
    agent_id: int,
    role: str,
    *,
    provenance_aware: bool = False,
    variant: int = 0,
) -> str:
    extra = PROVENANCE_INSTRUCTION if provenance_aware else ""
    return SYSTEM_TEMPLATES[variant % len(SYSTEM_TEMPLATES)].format(
        agent_name=world.agent_names.get(agent_id, f"Investigator {agent_id}"),
        role=role,
        alpha_description=world.alpha_description,
        beta_description=world.beta_description,
        extra=extra,
    )


def private_records_message(world: World, agent_id: int) -> str:
    docs = [world.report(rid).text for rid in world.assignments.get(agent_id, [])]
    joined = "\n\n---\n\n".join(docs) if docs else "(no private records)"
    return f"Your private records:\n\n{joined}"


def probe_messages(context: list[Message]) -> list[Message]:
    """Append the diagnostic probe as a fresh user turn (separate forward pass)."""
    return [*context, {"role": "user", "content": PROBE_QUESTION}]
