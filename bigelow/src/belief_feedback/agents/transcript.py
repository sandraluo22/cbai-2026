"""Per-agent transcript management.

The transcript holds only public material (private records + delivered
memos). Private belief probes are separate forward passes constructed from a
copy of the context; nothing here ever appends a probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.generation import Message


@dataclass
class TranscriptEntry:
    round_idx: int
    kind: str  # "own_memo" | "received_memo"
    sender: int
    sender_role: str
    text: str


@dataclass
class Transcript:
    system: str
    private_records: str
    entries: list[TranscriptEntry] = field(default_factory=list)
    memory_rounds: int | None = None  # None = full transcript; k = last-k-rounds memory

    def add_own_memo(self, round_idx: int, agent_id: int, role: str, text: str) -> None:
        self.entries.append(TranscriptEntry(round_idx, "own_memo", agent_id, role, text))

    def add_received(self, round_idx: int, sender: int, role: str, text: str) -> None:
        self.entries.append(TranscriptEntry(round_idx, "received_memo", sender, role, text))

    def context_messages(self) -> list[Message]:
        """Chat messages for the next forward pass (no probes ever included)."""
        msgs: list[Message] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.private_records},
        ]
        entries = self.entries
        if self.memory_rounds is not None and entries:
            newest = max(e.round_idx for e in entries)
            entries = [e for e in entries if e.round_idx > newest - self.memory_rounds]
        for e in entries:
            if e.kind == "own_memo":
                msgs.append({"role": "assistant", "content": e.text})
            else:
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Message from the {e.sender_role} (round {e.round_idx}):\n{e.text}"
                        ),
                    }
                )
        return msgs
