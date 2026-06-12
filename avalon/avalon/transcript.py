"""Logging: machine JSONL + a GOD-VIEW annotated human transcript + the exact
legal view sent to each seat each turn (for leak verification) + cost.

The god-view file contains hidden roles/knowledge for the researcher's audit and
is clearly marked; it must NEVER be fed to a player. The per-turn legal-view file
records exactly what WAS sent, so a leak (if any) is verifiable. The API key is
never present in any of these.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Cost rates ($ per 1M tokens). Verify against current pricing; token COUNTS below
# are authoritative regardless.
PRICE_PER_MTOK = {"claude-opus-4-8": {"input": 15.0, "output": 75.0}}


class GameLogger:
    def __init__(self, out_dir, assignment, knowledge):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.events = (self.dir / "events.jsonl").open("w")
        self.legal_views = (self.dir / "legal_views.jsonl").open("w")  # exactly what was sent
        # GOD-VIEW ONLY: terse private chain-of-thought. NEVER fed to any player.
        self.cot = (self.dir / "private_cot.jsonl").open("w")
        self.god_lines: list[str] = []
        self.in_tok = 0
        self.out_tok = 0
        self.model_tokens: dict[str, dict] = {}
        self._god_header(assignment, knowledge)

    def _god_header(self, assignment, knowledge):
        self.god_lines.append("=== GOD-VIEW LOG — CONTAINS HIDDEN INFO. NEVER FEED TO A PLAYER. ===")
        self.god_lines.append("True roles:")
        for s in sorted(assignment):
            self.god_lines.append(f"  seat {s}: {assignment[s].value}")
        self.god_lines.append("")

    # -- events ---------------------------------------------------------- #
    def event(self, phase, actor, action, data, usage=None):
        rec = {"t": round(time.time(), 3), "phase": phase, "actor": actor,
               "action": action, "data": data}
        if usage:
            rec["usage"] = usage
            self.in_tok += usage.get("input_tokens", 0)
            self.out_tok += usage.get("output_tokens", 0)
            m = usage.get("model", "")
            mt = self.model_tokens.setdefault(m, {"input": 0, "output": 0})
            mt["input"] += usage.get("input_tokens", 0)
            mt["output"] += usage.get("output_tokens", 0)
        self.events.write(json.dumps(rec) + "\n")

    def legal_view_sent(self, seat, turn_tag, view_text):
        """Record the EXACT text sent to a seat (leak-verification artifact)."""
        self.legal_views.write(json.dumps({"seat": seat, "turn": turn_tag, "view": view_text}) + "\n")

    def private_cot(self, seat, role, action, reasoning):
        """GOD-VIEW ONLY terse chain-of-thought for a seat's action. Never fed to players."""
        if not reasoning:
            return
        self.cot.write(json.dumps({"seat": seat, "role": role.value, "action": action,
                                   "reasoning": reasoning}) + "\n")
        self.god_lines.append(f"    └─ CoT[{role.value}, {action}]: {reasoning}")

    # -- god-view human transcript (annotated) --------------------------- #
    def god(self, line: str):
        self.god_lines.append(line)

    def god_statement(self, seat, role, knew, text, passed):
        body = "(passed)" if passed else text
        self.god_lines.append(f"seat {seat} [{role.value}; knew: {knew}]: {body}")

    # -- finalize -------------------------------------------------------- #
    def cost(self) -> dict:
        total = 0.0
        for m, t in self.model_tokens.items():
            p = PRICE_PER_MTOK.get(m)
            if p:
                total += t["input"] / 1e6 * p["input"] + t["output"] / 1e6 * p["output"]
        return {"input_tokens": self.in_tok, "output_tokens": self.out_tok,
                "by_model": self.model_tokens, "estimated_usd": round(total, 4)}

    def finalize(self, summary: dict):
        summary = {**summary, "cost": self.cost()}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.god_lines.append("\n=== OUTCOME ===")
        self.god_lines.append(json.dumps(summary, indent=2))
        (self.dir / "godview_transcript.txt").write_text("\n".join(self.god_lines))
        self.events.close(); self.legal_views.close(); self.cot.close()
        return summary
