"""Synchronous multi-agent episode runner with causal branch routing.

Synchrony guarantee: within a round, every public memo is generated from the
frozen pre-round contexts; deliveries happen only after all memos of the
round are complete, so no agent can see a same-round message before
generating its own (a seeded asynchronous order exists only as an explicitly
labeled robustness condition).

Branch semantics (all clamping references a prerecorded baseline episode of
the same world and replicate seed; common random numbers guarantee that the
only differences are the interventions themselves):

* ``one_hop_from_round=r``: from round r onward every delivered memo and
  every own-memo self-history entry is clamped to its baseline counterpart.
* ``no_return_agent=j``: from ``no_return_from_round`` onward agent j's
  incoming deliveries are clamped to baseline (paths leaving j and
  returning are blocked; everyone else stays live).
* ``full_text_clamp=[(j, r), ...]``: agent j's round-r memo is replaced by
  its baseline memo in every transcript, including j's own self-history.
* ``fixed_replay``: every delivered memo and self-history entry comes from
  the baseline stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config, id_fields
from ..models.activations import ActivationStore
from ..models.base import Backend
from ..models.generation import GenerationResult, Message
from ..models.steering import SteeringSpec
from ..seeds import generation_seed
from ..world.oracle import blind_oracle_for_reports, oracle_for_reports
from ..world.schema import World
from .memo_parser import ParsedMemo, parse_memo
from .prompts import MEMO_REQUEST, PROBE_CHOICES, private_records_message, probe_messages, system_prompt
from .roles import role_for_agent
from .transcript import Transcript


@dataclass
class SteeringContext:
    """Calibrated steering vector and site (shared across an experiment)."""

    vector: np.ndarray
    layer: int
    scope: str = "final_token_and_generation"

    def spec(self, magnitude: float) -> SteeringSpec:
        return SteeringSpec(
            vector=self.vector,
            layer=self.layer,
            magnitude=magnitude,
            scope=self.scope,  # type: ignore[arg-type]
        )


@dataclass
class BranchSpec:
    name: str = "baseline"
    branch_parent: str | None = None
    condition: str = "baseline"
    steering: dict[tuple[int, int], float] = field(default_factory=dict)  # (agent, round) -> magnitude
    one_hop_from_round: int | None = None
    no_return_agent: int | None = None
    no_return_from_round: int = 2
    full_text_clamp: list[tuple[int, int]] = field(default_factory=list)
    fixed_replay: bool = False
    provenance_aware: bool = False
    prompt_variant: int = 0
    memory_rounds: int | None = None
    topology: str | None = None
    channel_transform: str = "full"
    async_order_seed: int | None = None
    # optional per-(agent, round, time) projection patch (mechanistic analysis)
    projection_patch: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    run_id: str
    world_id: str
    branch: str
    memos: dict[tuple[int, int], str]  # (agent, round) -> raw memo text
    parsed: dict[tuple[int, int], ParsedMemo]
    beliefs: np.ndarray  # [rounds+1, n_agents] semantic log odds
    belief_rows: list[dict[str, Any]]
    message_rows: list[dict[str, Any]]
    delivery_rows: list[dict[str, Any]]
    intervention_rows: list[dict[str, Any]]


def neighbors_of(topology: str, n: int, i: int) -> list[int]:
    if topology == "ring":
        if n <= 2:
            return [j for j in range(n) if j != i]
        return [(i - 1) % n, (i + 1) % n]
    if topology == "star":
        return list(range(1, n)) if i == 0 else [0]
    if topology == "complete":
        return [j for j in range(n) if j != i]
    raise ValueError(topology)


def graph_distance(topology: str, n: int, i: int, j: int) -> int:
    if i == j:
        return 0
    if topology == "ring":
        d = abs(i - j)
        return min(d, n - d)
    if topology == "star":
        return 1 if (i == 0 or j == 0) else 2
    if topology == "complete":
        return 1
    raise ValueError(topology)


def transform_memo(text: str, mode: str, world: World) -> str:
    """Deterministic communication-channel ablations (Part 15)."""
    if mode == "full":
        return text
    parsed = parse_memo(text, world)
    if mode == "header_only":
        return (
            f"Current assessment: {parsed.parsed_assessment or 'UNRESOLVED'}\n"
            f"Confidence: {parsed.parsed_confidence if parsed.parsed_confidence is not None else 0}"
        )
    if mode == "body_citations":
        return (
            f"Evidence cited: {', '.join(parsed.cited_ids) if parsed.cited_ids else 'none'}\n"
            f"Memo: {parsed.memo_body}"
        )
    if mode == "citations_only":
        return f"Evidence cited: {', '.join(parsed.cited_ids) if parsed.cited_ids else 'none'}"
    if mode == "paraphrase":
        # paired template paraphrase: deterministic phrase substitutions
        subs = [
            ("Current assessment:", "Present determination:"),
            ("Confidence:", "Certainty level:"),
            ("Evidence cited:", "Records referenced:"),
            ("Memo:", "Summary:"),
            ("Request to team:", "Ask of the team:"),
            ("the evidence currently favors", "the available material presently supports"),
            ("After reviewing the material available to me,", "Having gone through what I have on file,"),
            ("Based on my records and the messages received,", "Drawing on my records and incoming messages,"),
        ]
        out = text
        for a, b in subs:
            out = out.replace(a, b)
        return out
    if mode == "role_swap":
        return text  # attribution swap is handled at delivery via sender_role
    raise ValueError(mode)


def run_episode(
    cfg: Config,
    backend: Backend,
    world: World,
    replicate_seed: int,
    branch: BranchSpec,
    steer_ctx: SteeringContext | None = None,
    baseline: EpisodeResult | None = None,
    rounds: int | None = None,
    act_store: ActivationStore | None = None,
) -> EpisodeResult:
    """Run one episode of one branch. Deterministic and side-effect free."""
    backend.register_world(world)
    n = world.n_agents
    rounds = rounds if rounds is not None else cfg.network.rounds
    topology = branch.topology or cfg.network.topology
    run_id = f"{cfg.name}__{world.world_id}__{branch.name}__s{replicate_seed}"
    ids = id_fields(cfg)
    base_memos = baseline.memos if baseline is not None else {}

    transcripts: list[Transcript] = []
    accessible: list[set[str]] = []
    roles = [role_for_agent(i) for i in range(n)]
    for i in range(n):
        transcripts.append(
            Transcript(
                system=system_prompt(
                    world,
                    i,
                    roles[i],
                    provenance_aware=branch.provenance_aware,
                    variant=branch.prompt_variant,
                ),
                private_records=private_records_message(world, i),
                memory_rounds=branch.memory_rounds,
            )
        )
        accessible.append(set(world.assignments.get(i, [])))

    beliefs = np.zeros((rounds + 1, n))
    belief_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    delivery_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    memos: dict[tuple[int, int], str] = {}
    parsed_memos: dict[tuple[int, int], ParsedMemo] = {}

    def steering_spec(agent: int, round_idx: int) -> SteeringSpec | None:
        if steer_ctx is None:
            return None
        patch = branch.projection_patch.get((agent, round_idx))
        if patch is not None:
            spec = steer_ctx.spec(0.0)
            spec.mode = "project_set"
            spec.project_value = patch
            return spec
        mag = branch.steering.get((agent, round_idx), 0.0)
        if mag == 0.0:
            return None
        return steer_ctx.spec(mag)

    def probe_all(t: int) -> None:
        """Batched private probes + selected-layer activations for all agents."""
        contexts = [transcripts[a].context_messages() for a in range(n)]
        specs = [steering_spec(a, t) for a in range(n)]
        scores = backend.score_choices_batch(
            [probe_messages(c) for c in contexts], PROBE_CHOICES, steerings=specs
        )
        acts: list[np.ndarray | None] = [None] * n
        if steer_ctx is not None:
            layer = min(steer_ctx.layer, getattr(backend, "n_layers", 1) - 1)
            acts = list(
                backend.get_selected_activations_batch(contexts, layer, steerings=specs)
            )
        for agent in range(n):
            _probe_row(agent, t, scores[agent], acts[agent])

    def _probe_row(agent: int, t: int, score, act: np.ndarray | None) -> None:
        logp_alpha, logp_beta = score.logps
        visible = logp_alpha - logp_beta
        semantic = world.visible_to_semantic(visible)
        beliefs[t, agent] = semantic
        aware = oracle_for_reports(world, accessible[agent])
        blind = blind_oracle_for_reports(world, _report_multiset(agent))
        pointer = ""
        proj = float("nan")
        if steer_ctx is not None and act is not None:
            v = steer_ctx.vector / (np.linalg.norm(steer_ctx.vector) or 1.0)
            proj = float(act @ v)
            if act_store is not None:
                key = ActivationStore.key(world.world_id, branch.name, agent, t)
                pointer = act_store.put(key, act)
        belief_rows.append(
            {
                **ids,
                "run_id": run_id,
                "world_id": world.world_id,
                "split": world.split,
                "condition": branch.condition,
                "branch": branch.name,
                "branch_parent": branch.branch_parent,
                "replicate_seed": replicate_seed,
                "agent_id": agent,
                "role": roles[agent],
                "round": t,
                "time_index": t,
                "seed": generation_seed(world.world_id, replicate_seed, agent, t, "probe"),
                "logp_alpha": logp_alpha,
                "logp_beta": logp_beta,
                "logp_alpha_normalized": score.logps_normalized[0],
                "logp_beta_normalized": score.logps_normalized[1],
                "visible_log_odds": visible,
                "semantic_log_odds": semantic,
                "semantic_probability": 1.0 / (1.0 + np.exp(-semantic)),
                "oracle_log_odds": aware.oracle_log_odds,
                "provenance_blind_oracle_log_odds": blind.oracle_log_odds,
                "accessible_unique_events": aware.unique_event_count,
                "accessible_report_count": len(_report_multiset(agent)),
                "repeated_event_reports": aware.repeated_report_count,
                "caa_projection": proj,
                "selected_layer_activation_pointer": pointer,
                "steering_magnitude": branch.steering.get((agent, t), 0.0),
                "steering_layer": steer_ctx.layer if steer_ctx else -1,
                "steering_scope": steer_ctx.scope if steer_ctx else "",
            }
        )

    received_cited: list[list[str]] = [[] for _ in range(n)]

    def _report_multiset(agent: int) -> list[str]:
        return list(world.assignments.get(agent, [])) + received_cited[agent]

    # ---- t = 0: private state only, no public message ---------------------
    probe_all(0)

    order_rng = np.random.default_rng(branch.async_order_seed or 0)
    for r in range(1, rounds + 1):
        agent_order = list(range(n))
        if branch.async_order_seed is not None:
            agent_order = list(order_rng.permutation(n))

        round_texts: dict[int, str] = {}
        specs_by_agent: dict[int, SteeringSpec | None] = {}
        ctx_by_agent: dict[int, list[Message]] = {}
        seed_by_agent: dict[int, int] = {}
        for i in agent_order:
            spec = steering_spec(i, r)
            specs_by_agent[i] = spec
            if spec is not None:
                intervention_rows.append(
                    {
                        **ids,
                        "run_id": run_id,
                        "world_id": world.world_id,
                        "split": world.split,
                        "branch": branch.name,
                        "branch_parent": branch.branch_parent,
                        "condition": branch.condition,
                        "replicate_seed": replicate_seed,
                        "agent_id": i,
                        "round": r,
                        "seed": generation_seed(world.world_id, replicate_seed, i, r, "memo"),
                        "magnitude": spec.magnitude,
                        "mode": spec.mode,
                        "layer": spec.layer,
                        "scope": spec.scope,
                    }
                )
            ctx_by_agent[i] = transcripts[i].context_messages() + [
                {"role": "user", "content": MEMO_REQUEST}
            ]
            seed_by_agent[i] = generation_seed(world.world_id, replicate_seed, i, r, "memo")

        gens: dict[int, GenerationResult] = {}
        if branch.async_order_seed is None:
            # synchronous: all contexts are frozen -> one batched generation
            batch = backend.generate_batch(
                [ctx_by_agent[i] for i in agent_order],
                [seed_by_agent[i] for i in agent_order],
                steerings=[specs_by_agent[i] for i in agent_order],
            )
            gens = dict(zip(agent_order, batch))
        else:
            for i in agent_order:
                # asynchronous robustness order: rebuild the context at
                # generation time so earlier same-round deliveries are seen
                ctx_by_agent[i] = transcripts[i].context_messages() + [
                    {"role": "user", "content": MEMO_REQUEST}
                ]
                gens[i] = backend.generate(
                    ctx_by_agent[i], seed=seed_by_agent[i], steering=specs_by_agent[i]
                )
                round_texts[i] = gens[i].text
                _deliver_round(
                    cfg, world, branch, topology, r, {i: gens[i].text}, base_memos,
                    transcripts, roles, accessible, received_cited, delivery_rows, run_id,
                    replicate_seed, ids,
                )

        for i in agent_order:
            spec = specs_by_agent[i]
            seed = seed_by_agent[i]
            gen = gens[i]
            round_texts[i] = gen.text
            memos[(i, r)] = gen.text
            pm = parse_memo(gen.text, world, accessible_report_ids=accessible[i])
            parsed_memos[(i, r)] = pm
            cited_events = sorted(
                {world.report(rid).event_id for rid in pm.valid_citations + pm.invalid_citations}
            )
            message_rows.append(
                {
                    **ids,
                    "run_id": run_id,
                    "world_id": world.world_id,
                    "split": world.split,
                    "condition": branch.condition,
                    "branch": branch.name,
                    "branch_parent": branch.branch_parent,
                    "replicate_seed": replicate_seed,
                    "agent_id": i,
                    "role": roles[i],
                    "round": r,
                    "graph_neighbors": ",".join(map(str, neighbors_of(topology, n, i))),
                    "steering_layer": spec.layer if spec else -1,
                    "steering_magnitude": spec.magnitude if spec else 0.0,
                    "steering_scope": spec.scope if spec else "",
                    "raw_text": gen.text,
                    "format_valid": pm.format_valid,
                    "parsed_visible_assessment": pm.parsed_assessment,
                    "parsed_semantic_assessment": pm.semantic_stance(world),
                    "parsed_confidence": pm.parsed_confidence,
                    "cited_report_ids": ",".join(pm.cited_ids),
                    "cited_event_ids": ",".join(cited_events),
                    "invalid_citations": ",".join(pm.invalid_citations),
                    "hallucinated_citations": ",".join(pm.hallucinated_report_ids),
                    "word_count": pm.word_count,
                    "seed": seed,
                    "generation_seed": seed,
                    "prompt_hash": gen.prompt_hash,
                    "context_token_count": gen.context_token_count,
                    "generation_token_count": gen.generation_token_count,
                    "wall_time": gen.wall_time,
                    "peak_gpu_memory": gen.peak_gpu_memory,
                }
            )
        if branch.async_order_seed is None:
            _deliver_round(
                cfg, world, branch, topology, r, round_texts, base_memos, transcripts,
                roles, accessible, received_cited, delivery_rows, run_id, replicate_seed, ids,
            )

        probe_all(r)

    return EpisodeResult(
        run_id=run_id,
        world_id=world.world_id,
        branch=branch.name,
        memos=memos,
        parsed=parsed_memos,
        beliefs=beliefs,
        belief_rows=belief_rows,
        message_rows=message_rows,
        delivery_rows=delivery_rows,
        intervention_rows=intervention_rows,
    )


def _resolve_delivery(
    branch: BranchSpec,
    src: int,
    dst: int,
    round_idx: int,
    live_text: str,
    base_memos: dict[tuple[int, int], str],
) -> tuple[str, str, str | None]:
    """Return (text, status, baseline_memo_id) for one delivery edge."""
    base_id = f"baseline:{src}:{round_idx}"
    base_text = base_memos.get((src, round_idx))

    def clamped() -> tuple[str, str, str | None]:
        if base_text is None:
            raise RuntimeError("clamping requested but no baseline memo available")
        return base_text, "clamped", base_id

    if branch.fixed_replay:
        if base_text is None:
            raise RuntimeError("fixed_replay requires a baseline episode")
        return base_text, "replayed", base_id
    if (src, round_idx) in branch.full_text_clamp:
        return clamped()
    if branch.one_hop_from_round is not None and round_idx >= branch.one_hop_from_round:
        return clamped()
    if (
        branch.no_return_agent is not None
        and dst == branch.no_return_agent
        and src != dst
        and round_idx >= branch.no_return_from_round
    ):
        return clamped()
    return live_text, "live", None


def _deliver_round(
    cfg: Config,
    world: World,
    branch: BranchSpec,
    topology: str,
    round_idx: int,
    round_texts: dict[int, str],
    base_memos: dict[tuple[int, int], str],
    transcripts: list[Transcript],
    roles: list[str],
    accessible: list[set[str]],
    received_cited: list[list[str]],
    delivery_rows: list[dict[str, Any]],
    run_id: str,
    replicate_seed: int,
    ids: dict[str, str],
) -> None:
    n = world.n_agents
    known = {rep.report_id for rep in world.reports}
    for src, live_text in round_texts.items():
        recipients = [src, *neighbors_of(topology, n, src)]  # self-history first
        for dst in recipients:
            text, status, base_id = _resolve_delivery(
                branch, src, dst, round_idx, live_text, base_memos
            )
            base_text = base_memos.get((src, round_idx))
            altered = base_text is not None and text != base_text
            sender_role = roles[src]
            if dst != src:
                delivered = transform_memo(text, branch.channel_transform, world)
                if branch.channel_transform == "role_swap":
                    sender_role = roles[(src + 3) % n]
                transcripts[dst].add_received(round_idx, src, sender_role, delivered)
                for rid in parse_memo(delivered, world).cited_ids:
                    if rid in known and rid not in accessible[dst]:
                        accessible[dst].add(rid)
                        received_cited[dst].append(rid)
            else:
                transcripts[src].add_own_memo(round_idx, src, roles[src], text)
            delivery_rows.append(
                {
                    **ids,
                    "run_id": run_id,
                    "world_id": world.world_id,
                    "split": world.split,
                    "replicate_seed": replicate_seed,
                    "originating_branch": branch.name,
                    "actual_generated_branch": "baseline" if status != "live" else branch.name,
                    "branch": branch.name,
                    "branch_parent": branch.branch_parent,
                    "condition": branch.condition,
                    "agent_id": dst,
                    "seed": 0,
                    "source_agent": src,
                    "recipient_agent": dst,
                    "round": round_idx,
                    "status": status,
                    "is_self_history": dst == src,
                    "baseline_memo_id": base_id,
                    "intervention_path_status": "altered" if altered else "baseline_equivalent",
                }
            )
