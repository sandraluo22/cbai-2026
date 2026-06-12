"""The Moderator — a deterministic referee (NO LLM calls on its own behalf).

It holds ALL hidden state, enforces every rule, rotates leadership, runs the phase
state machine, decides turn order, builds each seat's LEGAL view (via views.build_view,
which reads only that seat's knowledge), calls the 7 player LLMs, and appends their
outputs to the public transcript. It never edits a statement and never lets hidden
info into a player prompt.
"""

from __future__ import annotations

import numpy as np

from . import rules as RU
from .player import ApiError, PlayerClient, enforce_statement_length
from .roles import Role, alignment, assign_roles, compute_knowledge, seats_with_role
from .views import PublicEvent, PublicState, build_view


class Engine:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.n = cfg.n_players
        self.rng = np.random.default_rng(cfg.seed)
        self.assignment = assign_roles(self.n, self.rng)
        self.knowledge = compute_knowledge(self.assignment)
        self.merlin_seat = seats_with_role(self.assignment, Role.MERLIN)[0]
        self.assassin_seat = seats_with_role(self.assignment, Role.ASSASSIN)[0]
        self.logger = logger
        self.player = PlayerClient(cfg)
        # public state
        self.ps = PublicState()
        self.transcript: list[PublicEvent] = []
        self.successes = 0
        self.fails = 0
        self.mission_idx = 0
        self.leader = int(self.rng.integers(self.n))
        self.violations: list[dict] = []

    # -- helpers --------------------------------------------------------- #
    def _knew(self, seat) -> str:
        k = self.knowledge[seat]
        if k.evil_set_seen is not None:
            return f"evil-set{sorted(k.evil_set_seen)}"
        if k.merlin_morgana_pair is not None:
            return f"M/M-pair{sorted(k.merlin_morgana_pair)}"
        if k.evil_team is not None:
            return f"evil-team{sorted(k.evil_team)}"
        return "nothing"

    def _refresh_public(self):
        self.ps.mission_idx = self.mission_idx
        self.ps.successes = self.successes
        self.ps.fails = self.fails
        self.ps.leader = self.leader

    def _view(self, seat, tag) -> str:
        self._refresh_public()
        v = build_view(seat, self.knowledge, self.ps, self.transcript).render()
        if self.logger:
            self.logger.legal_view_sent(seat, tag, v)
        return v

    def _usage(self, seat, res):
        return {"model": self.cfg.model_for(seat), "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens}

    def _log(self, phase, actor, action, data, usage=None):
        if self.logger:
            self.logger.event(phase, actor, action, data, usage)

    def _cot(self, seat, action, res):
        """Record a seat's terse private chain-of-thought (god-view only)."""
        if self.logger and getattr(res, "reasoning", ""):
            self.logger.private_cot(seat, self.assignment[seat], action, res.reasoning)

    # -- proposal -------------------------------------------------------- #
    def _propose(self, k) -> list[int]:
        tag = f"propose_m{self.mission_idx}_l{self.leader}"
        instr = (f"You are the LEADER for mission {self.mission_idx + 1}. Propose a team of "
                 f"EXACTLY {k} distinct seats from 0-6 (you may include yourself). "
                 f'JSON: {{"team": [<{k} distinct seat ints>], "reason": "<=1 short sentence"}}.')
        for attempt in range(2):
            view = self._view(self.leader, tag)
            extra = "" if attempt == 0 else f"\nYour team must be exactly {k} distinct seats in 0-6."
            res = self.player.act(self.leader, self.cfg.model_for(self.leader), view, instr + extra, ["team"])
            team = res.parsed.get("team")
            self._log("proposal", self.leader, "propose", {"team": team, "reason": res.parsed.get("reason")},
                      self._usage(self.leader, res))
            self._cot(self.leader, "propose", res)
            if (isinstance(team, list) and len(set(team)) == k
                    and all(isinstance(s, int) and 0 <= s < self.n for s in team)):
                self.ps.proposed_team = sorted(set(team))
                return self.ps.proposed_team
        raise ApiError(f"leader seat {self.leader} failed to propose a valid {k}-team")

    # -- discussion ------------------------------------------------------ #
    def _deliberate(self, k) -> list[int]:
        """Propose, then loop: round-robin discussion -> leader keeps-or-swaps. If the
        leader keeps the team, go to vote; if the leader swaps to a different team,
        run another round-robin (capped at max_discussion_rounds). Leadership does NOT
        rotate on a swap (only a rejected VOTE rotates it)."""
        cfg = self.cfg
        team = self._propose(k)
        start = int(self.rng.integers(self.n))            # randomized start, fixed for this proposal
        for r in range(cfg.max_discussion_rounds):
            ready = self._round_robin(team, r, start)
            if r == cfg.max_discussion_rounds - 1:
                self._log("discussion", "moderator", "round_cap_reached", {"team": sorted(team)})
                break
            new_team = self._leader_revise(team, k, r, ready)
            if sorted(new_team) == sorted(team):
                self._log("proposal", self.leader, "leader_keep", {"team": sorted(team), "after_round": r})
                break
            self._log("proposal", self.leader, "leader_swap",
                      {"from": sorted(team), "to": sorted(new_team), "after_round": r})
            team = new_team
            self.ps.proposed_team = sorted(team)          # next round-robin's views see the new team
        return team

    def _round_robin(self, team, r, start) -> int:
        """One round-robin of discussion on `team`. Returns how many reported ready."""
        cfg = self.cfg
        direction = 1 if r % 2 == 0 else -1               # reverse direction on alternating rounds
        order = [(start + direction * i) % self.n for i in range(self.n)]
        self._log("discussion", "moderator", "round_order", {"round": r, "order": order, "team": sorted(team)})
        round_ready = 0
        for seat in order:
            tag = f"discuss_m{self.mission_idx}_r{r}_s{seat}"
            instr = (f"Discussion round {r + 1} of at most {cfg.max_discussion_rounds} for the CURRENT "
                     f"proposed team {sorted(team)} on mission {self.mission_idx + 1}. (After this round the "
                     f"leader may keep this team or swap it.) Give a PUBLIC statement of at most "
                     f"{cfg.max_sentences_per_statement} sentences, OR pass. Also report readiness to vote. "
                     f'JSON: {{"statement": "<text or empty>", "pass": <bool>, "ready_to_vote": <bool>}}.')
            view = self._view(seat, tag)
            res = self.player.act(seat, cfg.model_for(seat), view, instr,
                                  ["statement", "pass", "ready_to_vote"])
            self._log("discussion", seat, "speak",
                      {"pass": res.parsed.get("pass"), "ready": res.parsed.get("ready_to_vote")},
                      self._usage(seat, res))
            passed = bool(res.parsed.get("pass")) or not str(res.parsed.get("statement", "")).strip()
            text = ""
            if not passed:
                text = str(res.parsed["statement"]).strip()
                capped, trunc = enforce_statement_length(text, cfg.max_sentences_per_statement, cfg.word_backstop)
                if trunc:
                    res2 = self.player.act(seat, cfg.model_for(seat), view,
                                           instr + f"\nToo long. Use at most {cfg.max_sentences_per_statement} sentences.",
                                           ["statement", "pass", "ready_to_vote"])
                    self._log("discussion", seat, "speak_retry", {}, self._usage(seat, res2))
                    text, trunc2 = enforce_statement_length(str(res2.parsed.get("statement", "")).strip(),
                                                            cfg.max_sentences_per_statement, cfg.word_backstop)
                    res = res2
                    if trunc2:
                        self._log("discussion", seat, "truncated", {"final": text})
                else:
                    text = capped
            self.transcript.append(PublicEvent(round_idx=r, seat=seat, passed=passed, text=text))
            if self.logger:
                self.logger.god_statement(seat, self.assignment[seat], self._knew(seat), text, passed)
            self._cot(seat, "discuss", res)
            if bool(res.parsed.get("ready_to_vote")):
                round_ready += 1
        return round_ready

    def _leader_revise(self, team, k, r, ready) -> list[int]:
        """Leader decides AFTER a round-robin: keep the current team (-> vote) or swap
        to a different k-team (-> another round-robin). Returns the chosen team."""
        tag = f"revise_m{self.mission_idx}_l{self.leader}_r{r}"
        instr = (f"You are the LEADER. After discussion round {r + 1}, {ready}/{self.n} players reported "
                 f"ready to vote. You may KEEP the current team {sorted(team)} and send it to a vote, OR "
                 f"SWAP to a DIFFERENT team of EXACTLY {k} distinct seats (0-6) and hold one more discussion "
                 f'round. JSON: {{"keep": <bool>, "team": [<{k} seats: the current team if keeping, or your '
                 f'new team if swapping>]}}.')
        view = self._view(self.leader, tag)
        res = self.player.act(self.leader, self.cfg.model_for(self.leader), view, instr, ["keep"])
        self._cot(self.leader, "revise", res)
        if res.parsed.get("keep") is True:
            return team
        new = res.parsed.get("team")
        if (isinstance(new, list) and len(set(new)) == k
                and all(isinstance(s, int) and 0 <= s < self.n for s in new)
                and sorted(set(new)) != sorted(team)):
            return sorted(set(new))
        return team                                       # invalid/unchanged -> treat as keep

    # -- vote ------------------------------------------------------------ #
    def _vote(self, team):
        votes = {}
        for seat in range(self.n):                  # collected independently; no peeking
            tag = f"vote_m{self.mission_idx}_s{seat}"
            instr = (f'Vote to APPROVE or REJECT the proposed team {sorted(team)} for mission '
                     f'{self.mission_idx + 1}. JSON: {{"vote": "approve"|"reject"}}.')
            view = self._view(seat, tag)
            res = self.player.act(seat, self.cfg.model_for(seat), view, instr, ["vote"])
            v = str(res.parsed["vote"]).strip().lower()
            v = "approve" if v.startswith("a") else ("reject" if v.startswith("r") else None)
            if v is None:
                raise ApiError(f"seat {seat} returned an unparseable vote")
            votes[seat] = v
            self._log("vote", seat, "vote", {"vote": v}, self._usage(seat, res))
            self._cot(seat, "vote", res)
        tally = RU.tally_votes(votes)
        self.ps.public_votes.append({"mission": self.mission_idx, "team": sorted(team),
                                     "votes": votes, "approved": tally.approved,
                                     "approves": tally.approves})
        self._log("vote", "moderator", "tally",
                  {"approves": tally.approves, "rejects": tally.rejects, "approved": tally.approved})
        return tally

    # -- mission --------------------------------------------------------- #
    def _mission(self, team):
        cards = {}
        for seat in team:                           # only team members play; no peeking
            tag = f"mission_m{self.mission_idx}_s{seat}"
            instr = ('You are ON the mission team. Secretly play a mission card. '
                     'GOOD players must play success; EVIL players may play success or fail. '
                     'JSON: {"card": "success"|"fail"}.')
            view = self._view(seat, tag)
            res = self.player.act(seat, self.cfg.model_for(seat), view, instr, ["card"])
            raw = str(res.parsed["card"]).strip().lower()
            raw = "fail" if raw.startswith("f") else "success"
            eff, overridden = RU.enforce_good_play_success(raw, self.assignment[seat])
            if overridden:
                self.violations.append({"seat": seat, "issue": "good_tried_fail_overridden",
                                        "mission": self.mission_idx})
                self._log("mission", seat, "good_fail_overridden", {"role": self.assignment[seat].value})
            cards[seat] = eff
            self._log("mission", seat, "card", {"played": "hidden"}, self._usage(seat, res))  # never log which
            self._cot(seat, "mission", res)
        result = RU.resolve_mission(self.mission_idx, cards)
        self.ps.mission_results.append({"mission_idx": self.mission_idx, "fail_count": result.fail_count,
                                        "failed": result.failed})
        self._log("mission", "moderator", "result",
                  {"mission": self.mission_idx, "fail_count": result.fail_count, "failed": result.failed})
        return result

    # -- assassin -------------------------------------------------------- #
    def _assassin(self):
        tag = "assassin_guess"
        instr = ('GOOD has succeeded 3 missions. You are the ASSASSIN. Make ONE guess at which '
                 'seat is Merlin. If correct, EVIL wins. JSON: {"guess_seat": <int 0-6>}.')
        view = self._view(self.assassin_seat, tag)
        res = self.player.act(self.assassin_seat, self.cfg.model_for(self.assassin_seat), view, instr, ["guess_seat"])
        guess = int(res.parsed["guess_seat"])
        self._log("assassin", self.assassin_seat, "guess", {"guess_seat": guess}, self._usage(self.assassin_seat, res))
        self._cot(self.assassin_seat, "assassin", res)
        outcome = RU.assassin_outcome(guess, self.merlin_seat)
        return guess, outcome

    # -- main loop ------------------------------------------------------- #
    def run(self) -> dict:
        while True:
            k = RU.team_size(self.mission_idx)
            consecutive_rejects = 0
            self.ps.rejected_count = 0
            approved_team = None
            while approved_team is None:
                team = self._deliberate(k)
                tally = self._vote(team)
                if tally.approved:
                    approved_team = team
                else:
                    consecutive_rejects += 1
                    self.ps.rejected_count = consecutive_rejects
                    self.leader = (self.leader + 1) % self.n      # leadership passes
                    if RU.hammer_triggered(consecutive_rejects):
                        return self._finish("evil", "hammer: 5 consecutive rejected proposals")
            result = self._mission(approved_team)
            if result.failed:
                self.fails += 1
            else:
                self.successes += 1
            self.ps.proposed_team = None
            winner = RU.mission_winner(self.successes, self.fails)
            self.leader = (self.leader + 1) % self.n
            self.mission_idx += 1
            if winner == "evil":
                return self._finish("evil", "3 missions failed")
            if winner == "good_provisional":
                guess, outcome = self._assassin()
                reason = (f"good reached 3 successes; assassin guessed seat {guess}, "
                          f"{'correct' if outcome == 'evil' else 'wrong'} (Merlin was seat {self.merlin_seat})")
                return self._finish(outcome, reason)

    def _finish(self, winner, reason) -> dict:
        return {"winner": winner, "reason": reason,
                "successes": self.successes, "fails": self.fails,
                "mission_results": self.ps.mission_results,
                "votes": self.ps.public_votes,
                "true_roles": {s: r.value for s, r in self.assignment.items()},
                "merlin_seat": self.merlin_seat, "assassin_seat": self.assassin_seat,
                "violations": self.violations}
