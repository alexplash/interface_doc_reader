
from typing import List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class Event:
    role: str          # "USER" | "AGENT"
    kind: str          # e.g. "user_input" | "reason_step" | "rank_documents" | "final_answer"
    text: Any          # can be str OR list OR dict
    payload: Optional[dict] = None


class EventState:
    def __init__(self) -> None:
        self.events: List[Event] = []

    def update_state(self, e: Event) -> None:
        self.events.append(e)

    def render(self, max_events: int = 60) -> str:
        evs = self.events[-max_events:]
        lines: List[str] = []
        for e in evs:
            if e.role == "USER":
                lines.append(f"[USER] {e.text}")
            else:
                if e.kind == "reason_step":
                    lines.append(f"[AGENT reason_step] {e.text}")
                elif e.kind == "rank_documents":
                    lines.append(f"[AGENT rank_documents REQUEST] {e.text}")
                    if e.payload is not None:
                        lines.append(f"[TOOL rank_documents RESULT] {json.dumps(e.payload, ensure_ascii=False)}")
                elif e.kind == "final_answer":
                    lines.append(f"[AGENT final_answer] {e.text}")
                else:
                    lines.append(f"[AGENT {e.kind}] {e.text}")
                    if e.payload is not None:
                        lines.append(f"[AGENT {e.kind} PAYLOAD] {json.dumps(e.payload, ensure_ascii=False)}")
        return "\n".join(lines)