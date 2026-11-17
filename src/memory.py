"""会话记忆：按 session_id 保存最近若干轮对话，并提供结构化上下文。"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    turns: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def context(self, limit: int = 6) -> list[dict]:
        return self.turns[-limit * 2:]


class SessionMemory:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = Session(sid)
        return sid

    def get(self, session_id: str | None) -> Session:
        if not session_id or session_id not in self._sessions:
            session_id = self.new_session()
        return self._sessions[session_id]

    def add(self, session_id: str, role: str, content: str) -> None:
        session = self.get(session_id)
        session.turns.append({"role": role, "content": content})
        if len(session.turns) > self.max_turns * 2:
            session.turns = session.turns[-self.max_turns * 2:]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
