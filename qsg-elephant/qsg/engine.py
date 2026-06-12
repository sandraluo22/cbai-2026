"""Simulation engine: ontology -> observations -> QSG rounds -> logs/activations.

Two coupling protocols (documented defaults; see README for the conceptual
subtlety the spec flags):

  pure_qsg (Mode 1)
    Round 0: each agent is seeded by ONE LLM soft+hard readout on its OWN clues
      (activations captured). Thereafter the public simplex evolves by EXACT
      numeric QSG (qsg_reference.sample_message/qsg_update) — no further LLM calls,
      so later-round activations are not meaningful and are left unfilled.

  two_layer (Mode 3)  [primary]
    Every round, every agent listens to one uniformly-drawn speaker. The speaker
    emits a QSG message y from its PUBLIC simplex; y is rendered into the
    listener's prompt alongside its PERSISTENT private observation; the listener
    RE-READS its belief r_L via the LLM (activations captured at the anchor), and
    the public simplex is convex-updated:  x_L <- (1-a) x_L + a r_L.
    The private observation persistently biases every readout.

Both share the channel math in qsg_reference and the same analysis/logging.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import analysis
from .activations import (
    ActivationHooks,
    ActivationMeta,
    ActivationStore,
    estimate_disk_bytes,
)
from .agents import (
    Agent,
    ImageArmUnavailable,
    ModelBackend,
    message_to_text,
    qsg_commitment_text,
)
from .config import Arm, CouplingMode, RunConfig, dump_config
from .qsg_reference import draw_pair, qsg_update, sample_message
from .readout import hard_readout, is_degenerate, kl_l1, soft_readout


# --------------------------------------------------------------------------- #
# Ontology + observation assignment                                          #
# --------------------------------------------------------------------------- #
def load_ontology(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _truncate_reasoning(raw: str) -> str:
    """Keep a speaker's reasoning up to and including its first 'My guess: <x>' line.

    Stops greedy run-on / prompt-scaffold echoes from leaking into the transcript.
    """
    import re

    m = re.search(r"(.*?my guess:\s*\w+)", raw, flags=re.IGNORECASE | re.DOTALL)
    text = m.group(1) if m else raw
    return " ".join(text.split())


def assign_observations(cfg: RunConfig, ontology: dict, rng: np.random.Generator):
    candidates = ontology["candidates"]
    if cfg.object_name is not None:
        obj = cfg.object_name
    else:
        obj = candidates[int(rng.integers(len(candidates)))]
    gt_index = candidates.index(obj)
    feats = list(ontology["objects"][obj]["features"])
    distractors = list(ontology["objects"][obj].get("distractors", []))

    per_agent_text = []
    for _ in range(cfg.qsg.n):
        k = min(cfg.observation.feature_subset_size, len(feats))
        chosen = list(rng.choice(feats, size=k, replace=False))
        if cfg.observation.distractor_features and distractors:
            d = min(cfg.observation.distractor_features, len(distractors))
            chosen += list(rng.choice(distractors, size=d, replace=False))
        rng.shuffle(chosen)
        per_agent_text.append(", ".join(chosen))
    return candidates, obj, gt_index, per_agent_text


def make_image_patches(cfg: RunConfig, rng: np.random.Generator):
    """Crop the source image into a grid; return one random patch per agent.

    Degrades gracefully: returns ``None`` (handled by caller) if PIL or the source
    image is unavailable.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    src_dir = cfg.observation.source_image_dir
    if not src_dir or not Path(src_dir).exists():
        return None
    imgs = sorted(Path(src_dir).glob("*.*"))
    if not imgs:
        return None
    base = Image.open(imgs[0]).convert("RGB")
    W, H = base.size
    gx, gy = cfg.observation.patch_grid
    pw, ph = W // gx, H // gy
    patches = []
    for _ in range(cfg.qsg.n):
        i, j = int(rng.integers(gx)), int(rng.integers(gy))
        left, upper = i * pw, j * ph
        patches.append(base.crop((left, upper, left + pw, upper + ph)))
    return patches


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        # Two independent RNGs: one for QSG sampling, one for everything else.
        self.qsg_rng = np.random.default_rng(cfg.seed + 10_000)
        self.setup_rng = np.random.default_rng(cfg.seed)
        self.run_dir = Path(cfg.output_root) / cfg.run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ontology = load_ontology(cfg.ontology_path)

    # -- setup ------------------------------------------------------------ #
    def _build_agents(self) -> list[Agent]:
        candidates, obj, gt_index, per_agent_text = assign_observations(
            self.cfg, self.ontology, self.setup_rng
        )
        self.candidates = candidates
        self.object_name = obj
        self.ground_truth = gt_index
        self.K = len(candidates)

        images = None
        # In neutral-ablation the agent must truly have no observation, so for the
        # image arm we withhold the patch entirely (otherwise the VLM still sees it).
        if self.cfg.arm == Arm.IMAGE and not self.cfg.neutral_ablation:
            images = make_image_patches(self.cfg, self.setup_rng)

        agents = []
        for i in range(self.cfg.qsg.n):
            img = [images[i]] if images is not None else None
            agents.append(Agent(agent_id=i, private_text=per_agent_text[i], images=img))
        return agents

    def _disk_guard(self, n_layers: int, hidden_dim: int) -> None:
        nbytes = estimate_disk_bytes(
            self.cfg.qsg.rounds + 1, self.cfg.qsg.n, n_layers, hidden_dim,
            self.cfg.activations.dtype,
        )
        gb = nbytes / 1e9
        print(f"  [activations] estimated store size: {gb:.3f} GB "
              f"(shape {(self.cfg.qsg.rounds + 1, self.cfg.qsg.n, n_layers, hidden_dim)}, "
              f"dtype {self.cfg.activations.dtype})")
        if gb > self.cfg.activations.max_disk_gb:
            raise RuntimeError(
                f"Activation store {gb:.2f} GB exceeds cap "
                f"{self.cfg.activations.max_disk_gb} GB. Raise activations.max_disk_gb "
                f"or reduce rounds/N/layers."
            )

    # -- main loop -------------------------------------------------------- #
    def run(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        agents = self._build_agents()

        backend = ModelBackend(cfg, self.candidates)
        hooks = None
        store = None
        if cfg.activations.capture:
            hooks = ActivationHooks(backend.model, cfg.activations.layers)

        beliefs = np.zeros((cfg.qsg.rounds + 1, cfg.qsg.n, self.K))  # public simplex traj
        soft_log = np.zeros_like(beliefs)
        hard_log = np.zeros_like(beliefs)

        log_path = self.run_dir / "beliefs.jsonl"
        logf = log_path.open("w")
        degenerate_count = 0

        def readout_and_log(t: int, agent: Agent, capture: bool):
            nonlocal store, degenerate_count
            sr = soft_readout(backend, agent, cfg, hooks=hooks if capture else None)
            if cfg.readout.do_hard_readout:
                hr, hard_raw = hard_readout(backend, agent, cfg)
            else:
                hr, hard_raw = sr.canonical, ""
            kl, l1 = kl_l1(sr.canonical, hr)
            deg = is_degenerate(sr.canonical) or is_degenerate(hr)
            if deg:
                degenerate_count += 1

            # lazily size the activation store once we know hidden_dim/layers
            if capture and store is None and sr.activations is not None:
                n_layers, hidden = sr.activations.shape
                self._disk_guard(n_layers, hidden)
                meta = ActivationMeta(
                    run_id=cfg.run_id(), model_name=backend.model_name,
                    n_agents=cfg.qsg.n, n_rounds=cfg.qsg.rounds + 1,
                    layer_indices=hooks.layer_idx, n_layers=n_layers, hidden_dim=hidden,
                    dtype=cfg.activations.dtype, anchor_token_id=sr.anchor_token_id,
                    template_hash=cfg.readout.template_hash(),
                    layer_path=hooks.layer_path,
                    shape=[cfg.qsg.rounds + 1, cfg.qsg.n, n_layers, hidden],
                    store_format=cfg.activations.store_format,
                )
                store = ActivationStore(self.run_dir / "activations", meta)
            if capture and store is not None and sr.activations is not None:
                store.put(t, agent.agent_id, sr.activations)

            soft_log[t, agent.agent_id] = sr.canonical
            hard_log[t, agent.agent_id] = hr
            logf.write(json.dumps({
                "round": t, "agent": agent.agent_id,
                "soft_canonical": sr.canonical.tolist(), "soft_canonical_name": sr.canonical_name,
                "soft_first_token": sr.first_token.tolist(),
                "soft_length_norm": sr.length_norm.tolist(),
                "hard": np.asarray(hr).tolist(),
                "kl_soft_hard": kl, "l1_soft_hard": l1,
                "degenerate": deg, "anchor_token_id": sr.anchor_token_id,
                "message_text": agent.last_message_text,
                "hard_raw": hard_raw,
            }) + "\n")
            return sr.canonical

        # Enter the hooks context so forward hooks are registered for the whole run.
        hooks_cm = hooks if hooks is not None else contextlib.nullcontext()
        with hooks_cm:
            # ---- Round 0: seed every agent from its own observation -------
            print(f"  round 0/{cfg.qsg.rounds} (seed) ...")
            for ag in agents:
                ag.last_message_text = ""
                beliefs[0, ag.agent_id] = readout_and_log(0, ag, capture=cfg.activations.capture)

            # ---- Subsequent rounds ---------------------------------------
            for t in range(1, cfg.qsg.rounds + 1):
                if cfg.coupling_mode == CouplingMode.PURE_QSG:
                    self._step_pure_qsg(beliefs, t, cfg)
                    soft_log[t] = beliefs[t]
                    hard_log[t] = beliefs[t]
                elif cfg.coupling_mode == CouplingMode.REASONING_EXCHANGE:
                    self._step_reasoning_exchange(beliefs, agents, backend, t, readout_and_log)
                else:
                    self._step_two_layer(beliefs, agents, backend, t, readout_and_log)
                if t % max(1, cfg.qsg.rounds // 5) == 0 or t == cfg.qsg.rounds:
                    U = float(np.sum(beliefs[t].mean(0) ** 2))
                    print(f"  round {t}/{cfg.qsg.rounds}  U={U:.3f}")

        logf.close()
        if store is not None:
            store.flush()

        results = self._finalize(beliefs, soft_log, hard_log, degenerate_count, t0, backend)
        return results

    def _step_pure_qsg(self, beliefs, t, cfg) -> None:
        beliefs[t] = beliefs[t - 1]
        ipr = cfg.qsg.interactions_per_round or cfg.qsg.n
        for _ in range(ipr):
            s, l = draw_pair(cfg.qsg.n, self.qsg_rng)
            y = sample_message(beliefs[t, s], cfg.qsg.m, self.qsg_rng)
            beliefs[t, l] = qsg_update(beliefs[t, l], y, cfg.qsg.alpha)

    def _step_two_layer(self, beliefs, agents, backend, t, readout_and_log) -> None:
        cfg = self.cfg
        beliefs[t] = beliefs[t - 1]
        order = list(range(cfg.qsg.n))
        self.setup_rng.shuffle(order)
        for l in order:
            s = int(self.qsg_rng.integers(cfg.qsg.n - 1))
            if s >= l:
                s += 1
            y = sample_message(beliefs[t, s], cfg.qsg.m, self.qsg_rng)
            agents[l].last_message_text = message_to_text(y, backend.vocab)
            r_l = readout_and_log(t, agents[l], capture=cfg.activations.capture)
            beliefs[t, l] = qsg_update(beliefs[t, l], r_l, cfg.qsg.alpha)

    def _step_reasoning_exchange(self, beliefs, agents, backend, t, readout_and_log) -> None:
        """QSG-structured talking: the models converse, but inside the QSG channel.

        For each ordered (speaker S, listener L) interaction, the speaker SAMPLES a
        commitment from its belief x_S via the QSG channel (bandwidth m: Hard=one
        label, Top-m=a few, Soft=its leading beliefs), then ARTICULATES that
        commitment in natural language conditioned on its own private percept. The
        listener reads that full utterance + its own percept, re-reads belief r_L,
        and the public simplex is convex-updated:  x_L <- (1-a) x_L + a r_L.

        So the bandwidth m and the convex update are exactly QSG; only the message
        is a sentence the model speaks instead of a bare label vector.
        """
        cfg = self.cfg
        beliefs[t] = beliefs[t - 1]
        order = list(range(cfg.qsg.n))
        self.setup_rng.shuffle(order)
        for l in order:
            s = int(self.qsg_rng.integers(cfg.qsg.n - 1))
            if s >= l:
                s += 1
            # QSG channel: sample what the speaker commits to from its belief
            y = sample_message(beliefs[t, s], cfg.qsg.m, self.qsg_rng)
            commitment = qsg_commitment_text(y, backend.vocab, cfg.qsg.m)
            # speaker articulates that commitment in natural language
            prompt = agents[s].reasoning_prompt(cfg, backend.vocab, commitment=commitment)
            utterance = _truncate_reasoning(backend.generate(
                prompt, images=agents[s].images,
                max_new_tokens=cfg.readout.reasoning_max_tokens,
            ))
            # listener reads the utterance + its own percept, re-reads, convex-updates
            agents[l].last_message_text = utterance
            r_l = readout_and_log(t, agents[l], capture=cfg.activations.capture)
            beliefs[t, l] = qsg_update(beliefs[t, l], r_l, cfg.qsg.alpha)

    # -- finalize: plots + manifest -------------------------------------- #
    def _finalize(self, beliefs, soft_log, hard_log, degenerate_count, t0, backend) -> dict:
        cfg = self.cfg
        plots_dir = self.run_dir / "plots"
        analysis.save_similarity_heatmaps(beliefs, plots_dir)
        analysis.save_diagnostic_curves(
            beliefs, plots_dir, ground_truth=self.ground_truth,
            consensus_threshold=cfg.qsg.consensus_threshold,
        )
        analysis.save_soft_hard_agreement(soft_log, hard_log, plots_dir)
        np.save(self.run_dir / "beliefs.npy", beliefs)

        op = analysis.order_parameters(beliefs)
        ct = analysis.consensus_time(op["U"], cfg.qsg.consensus_threshold)
        acc = analysis.accuracy_curves(beliefs, self.ground_truth)
        dump_config(cfg, self.run_dir / "config.yaml")

        manifest = {
            "run_id": cfg.run_id(),
            "arm": cfg.arm.value, "coupling_mode": cfg.coupling_mode.value,
            "model_name": backend.model_name, "device": backend.device,
            "object_name": self.object_name, "ground_truth_index": self.ground_truth,
            "candidates": self.candidates, "seed": cfg.seed,
            "N": cfg.qsg.n, "alpha": cfg.qsg.alpha, "m": cfg.qsg.m, "rounds": cfg.qsg.rounds,
            "neutral_ablation": cfg.neutral_ablation,
            "consensus_time": ct,
            "final_U": float(op["U"][-1]), "final_V": float(op["V"][-1]),
            "final_group_correct": bool(acc["group_correct"][-1]),
            "final_frac_correct": float(acc["frac_correct"][-1]),
            "degenerate_readouts": degenerate_count,
            "template_hash": cfg.readout.template_hash(),
            "wall_seconds": round(time.time() - t0, 2),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  done: {manifest['run_id']}  "
              f"acc={manifest['final_frac_correct']:.2f}  U={manifest['final_U']:.3f}  "
              f"consensus_t={ct}  degenerate={degenerate_count}  "
              f"({manifest['wall_seconds']}s) -> {self.run_dir}")
        return manifest


def run_single(cfg: RunConfig) -> dict:
    """Run one experiment; skip gracefully if the image arm has no VLM."""
    try:
        return Engine(cfg).run()
    except ImageArmUnavailable as e:
        print(f"  [skip] image arm unavailable: {e}")
        return {"run_id": cfg.run_id(), "skipped": True, "reason": str(e)}
