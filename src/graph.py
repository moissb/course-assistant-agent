"""LangGraph 式编排：意图识别 → 路由 → 工具/检索 → 反思重试 → 生成。

每个节点都写可观测事件（内存 + logs/events.jsonl），便于前端展示轨迹和复盘。
生产环境可把本文件替换为 LangGraph StateGraph，节点函数与 I/O 保持不变。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

from . import config
from .llm import BaseLLM, Decision, ToolCall
from .memory import SessionMemory
from .rag import SimpleVectorStore
from .tools import ToolRegistry, ToolResult


@dataclass
class AgentState:
    session_id: str
    question: str
    intent: str = ""
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence_parts: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    attempts: int = 0
    error: str = ""


@dataclass
class AgentResult:
    answer: str
    intent: str
    thought: str
    citations: list[dict]
    tool_trace: list[dict]
    events: list[dict]
    latency_ms: float
    first_token_ms: float
    success: bool
    session_id: str


class AgentGraph:
    def __init__(self, llm: BaseLLM, registry: ToolRegistry,
                 store: SimpleVectorStore, memory: SessionMemory):
        self.llm = llm
        self.registry = registry
        self.store = store
        self.memory = memory

    def run(self, question: str, session_id: str, stream: bool = False) -> AgentResult:
        started = time.monotonic()
        state = AgentState(session_id=session_id, question=question)
        self._emit(state, "start", "接收问题", question[:120])

        # 节点 1：意图识别 + 工具路由（真实模型走 Function Calling）
        decision: Decision = self.llm.decide(question, self.memory.get(session_id).context(),
                                             self.registry.schemas())
        state.intent, state.thought, state.tool_calls = decision.intent, decision.thought, decision.tool_calls
        self._emit(state, "intent", decision.intent, decision.thought)

        # 节点 2~4：工具调用 / 知识库检索 → 反思 → 必要时重试
        for attempt in range(config.MAX_ITERATIONS + 1):
            state.attempts = attempt
            state.evidence_parts = []
            state.tool_results = []

            for call in state.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                state.tool_results.append(result)
                if result.success:
                    state.evidence_parts.append(result.content)
                self._emit(state, "tool", call.name,
                           f"成功={result.success} 耗时={result.elapsed_ms}ms 参数={json.dumps(call.arguments, ensure_ascii=False)}")
                if result.elapsed_ms > config.TOOL_TIMEOUT * 1000:
                    state.error = f"工具 {call.name} 调用超时"
                    self._emit(state, "error", "tool_timeout", state.error)

            self._collect_citations(state)
            if self._has_evidence(state):
                self._emit(state, "reflect", "通过", f"第 {attempt + 1} 轮获得有效证据")
                break

            self._emit(state, "reflect", "重试", f"第 {attempt + 1} 轮证据不足")
            state.tool_calls = [ToolCall("search_docs", {
                "query": question,
                "top_k": 5,
            })]

        # 节点 5：基于证据生成回答（无证据时 LLM 会输出拒答）
        evidence = "\n\n".join(state.evidence_parts)
        first_token_at = time.monotonic()
        answer = self.llm.answer(question, evidence, self.memory.get(session_id).context(),
                                 stream=False, intent=state.intent)
        answer = str(answer)
        self._emit(state, "answer", "生成完成", answer[:120])

        result = AgentResult(
            answer=answer,
            intent=state.intent,
            thought=state.thought,
            citations=state.citations,
            tool_trace=[{"name": r.name, "arguments": r.arguments, "success": r.success,
                         "content": r.content[:300], "elapsed_ms": r.elapsed_ms}
                        for r in state.tool_results],
            events=state.events,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
            first_token_ms=round((first_token_at - started) * 1000, 2),
            success=self._has_evidence(state),
            session_id=session_id,
        )
        self._write_log(result)
        return result

    def _has_evidence(self, state: AgentState) -> bool:
        if not state.evidence_parts:
            return False
        joined = " ".join(state.evidence_parts)
        if any(word in joined for word in ("未找到", "调用失败", "权限校验失败")):
            return False
        if state.intent == "search_docs" and state.citations:
            # 检索相关度过低视为无依据，触发拒答而不是硬凑答案
            if max(c["score"] for c in state.citations) < config.SIMILARITY_THRESHOLD:
                return False
        return True

    def _collect_citations(self, state: AgentState) -> None:
        if not any(r.name == "search_docs" and r.success for r in state.tool_results):
            return
        hits = self.store.search(state.question, top_k=5, threshold=0.05)
        _, citations = self.store.format_context(hits)
        state.citations = citations

    def _emit(self, state: AgentState, kind: str, name: str, detail: str) -> None:
        event = {"session": state.session_id, "kind": kind, "name": name,
                 "detail": detail, "ts": round(time.time(), 3)}
        state.events.append(event)

    def _write_log(self, result: AgentResult) -> None:
        path = config.LOG_DIR / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        for event in result.events:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def stream_chunks(text: str, size: int = 10, delay: float = 0.012) -> Iterator[str]:
    """把最终回答切成 chunk 推送，前端按 SSE 渲染。"""
    for i in range(0, len(text), size):
        time.sleep(delay)
        yield text[i:i + size]
