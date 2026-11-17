"""对外 API：一行代码拿到可用的课程助手 Agent。"""
from __future__ import annotations

from typing import Iterator

from . import config
from .graph import AgentGraph, AgentResult, stream_chunks
from .llm import create_llm
from .memory import SessionMemory
from .rag import SimpleVectorStore
from .tools import ToolRegistry


class CourseAgent:
    def __init__(self):
        self.store = SimpleVectorStore()
        self.registry = ToolRegistry(self.store)
        self.memory = SessionMemory()
        self.llm = create_llm()
        self.graph = AgentGraph(self.llm, self.registry, self.store, self.memory)

    @property
    def mode(self) -> str:
        return self.llm.mode

    def new_session(self) -> str:
        return self.memory.new_session()

    def chat(self, question: str, session_id: str | None = None) -> AgentResult:
        sid = self.memory.get(session_id).session_id
        self.memory.add(sid, "user", question)
        result = self.graph.run(question, sid)
        self.memory.add(sid, "assistant", result.answer)
        return result

    def stream(self, result: AgentResult) -> Iterator[str]:
        return stream_chunks(result.answer)

    def reset(self, session_id: str) -> None:
        self.memory.reset(session_id)


def create_agent() -> CourseAgent:
    return CourseAgent()
