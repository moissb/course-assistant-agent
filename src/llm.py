"""LLM 适配层：真实 DeepSeek（Function Calling / 流式）与离线 Mock 双实现。

统一接口：
    decide(question, history, tools) -> Decision
    answer(question, evidence, history, stream) -> str | Iterator[str]
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator
from urllib import request

from . import config

INTENT_BY_TOOL = {
    "query_grade": "query_grade",
    "search_docs": "search_docs",
    "calculate": "calculator",
    "get_date": "date",
}


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Decision:
    intent: str
    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLM:
    mode = "base"

    def decide(self, question: str, history: list[dict], tools: list[dict]) -> Decision:
        raise NotImplementedError

    def answer(self, question: str, evidence: str, history: list[dict],
               stream: bool = False, intent: str = "") -> str | Iterator[str]:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """离线决策 + 基于证据的抽取式回答，保证无网络无 API Key 也能完整演示。"""

    mode = "mock"

    GRADE_WORDS = ("成绩", "分数", "多少分", "绩点", "排名", "学分", "平均分", "考得")
    FAQ_WORDS = ("缓考", "补考", "复核", "奖学金", "图书馆", "实习", "毕业论文",
                 "实验室", "请假", "重修", "三级", "四级", "审批流程", "评选条件")
    DATE_WORDS = ("今天", "几号", "星期", "日期", "周几")
    CALC_WORDS = ("计算", "算一下", "平均", "加", "减", "乘", "除")

    def decide(self, question: str, history: list[dict], tools: list[dict]) -> Decision:
        q = question.strip()
        if any(w in q for w in self.FAQ_WORDS):
            return Decision("search_docs", "问题属于教务流程，检索 FAQ 知识库。",
                            [ToolCall("search_docs", {"query": q})])

        if any(w in q for w in self.GRADE_WORDS):
            course = self._course_name(q)
            return Decision("query_grade", "问题涉及成绩数据，调用成绩查询工具。",
                            [ToolCall("query_grade", {"course_name": course})])

        if any(w in q for w in self.DATE_WORDS) and not any(w in q for w in ("课程", "考试")):
            return Decision("date", "问题涉及日期，调用日期工具。", [ToolCall("get_date", {})])

        expr = self._expression(q)
        if expr and any(w in q for w in self.CALC_WORDS):
            return Decision("calculator", "问题涉及数值计算，调用计算器工具。",
                            [ToolCall("calculate", {"expression": expr})])

        return Decision("search_docs", "问题需要知识库支撑，调用文档检索工具。",
                        [ToolCall("search_docs", {"query": q.replace("？", "").replace("?", "")})])

    def answer(self, question: str, evidence: str, history: list[dict],
               stream: bool = False, intent: str = "") -> str | Iterator[str]:
        if not evidence or "未找到" in evidence or "失败" in evidence:
            text = "知识库中没有找到相关依据，暂时无法回答这个问题。你可以补充课程名称，或换一种问法。"
        elif intent == "query_grade":
            text = f"根据成绩记录：{evidence}。"
        elif intent == "calculator":
            text = f"{evidence.split('计算结果：', 1)[1].strip()}。" if "计算结果：" in evidence else evidence
        elif intent == "date":
            text = f"{evidence}。"
        else:
            parts = [p.strip() for p in evidence.split("\n\n") if p.strip() and not p.startswith("出处：")]
            body = parts[0] if parts else evidence
            sources = evidence.split("出处：", 1)[1] if "出处：" in evidence else "课程知识库"
            text = f"根据知识库内容：\n\n{body}\n\n出处：{sources}。"
        return self._stream_chunks(text) if stream else text

    @staticmethod
    def _stream_chunks(text: str, size: int = 12) -> Iterator[str]:
        for i in range(0, len(text), size):
            yield text[i:i + size]

    def _course_name(self, question: str) -> str:
        if any(w in question for w in ("全部", "所有", "整体", "平均分", "绩点", "排名")):
            return "全部"
        aliases = {
            "高数": "高等数学", "线代": "线性代数B", "英语": "大学英语",
            "机器学习": "机器学习与模式识别", "数据结构": "数据结构与算法",
            "数据库": "数据库原理及应用", "大数据": "大数据技术与云计算",
            "自然语言处理": "自然语言处理", "nlp": "自然语言处理",
            "程序设计": "程序设计与问题求解", "python": "程序设计与问题求解",
            "马原": "马克思主义基本原理", "思修": "习近平新时代中国特色社会主义思想概论",
        }
        lower = question.lower()
        for alias, name in aliases.items():
            if alias in lower:
                return name
        clean = re.sub(r"(我的|请问|查询|查一下|多少分|成绩|分数|怎么样|是什么|排名|绩点|学分|[？?。])", "", question)
        return clean.strip()[:20] if clean.strip() and len(clean.strip()) < 20 else "全部"

    @staticmethod
    def _expression(question: str) -> str | None:
        match = re.search(r"[\d\s\.\+\-\*×÷/%\(\)（）]+", question)
        if not match:
            return None
        expr = match.group().strip().replace("（", "(").replace("）", ")")
        return expr if re.search(r"\d", expr) and any(op in expr for op in "+-*/×÷%") else None


class DeepSeekLLM(BaseLLM):
    """OpenAI 兼容协议：/chat/completions，支持 tools 与 stream。"""

    mode = "live"

    def __init__(self):
        if not config.API_KEY:
            raise ValueError("缺少 DEEPSEEK_API_KEY")
        self.endpoint = config.BASE_URL.rstrip("/") + "/chat/completions"

    def _post(self, payload: dict, stream: bool = False):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            headers={"Authorization": f"Bearer {config.API_KEY}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        return request.urlopen(req, timeout=config.REQUEST_TIMEOUT)

    def decide(self, question: str, history: list[dict], tools: list[dict]) -> Decision:
        system = (
            "你是课程助手 Agent 的路由器。根据用户问题选择最合适的工具："
            "查询成绩用 query_grade；课程知识点、考试方式、教务流程用 search_docs；"
            "纯数值运算用 calculate；日期问题用 get_date。"
            "必须调用工具，不要直接回答。thought 用一句话说明理由。"
        )
        payload = {
            "model": config.MODEL,
            "messages": [{"role": "system", "content": system}, *history,
                         {"role": "user", "content": question}],
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
        }
        with self._post(payload) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        calls = [ToolCall(c["function"]["name"], json.loads(c["function"].get("arguments") or "{}"))
                 for c in message.get("tool_calls", [])]
        thought = message.get("content") or "根据工具路由决策。"
        if not calls:
            calls = [ToolCall("search_docs", {"query": question})]
        return Decision(INTENT_BY_TOOL.get(calls[0].name, "search_docs"), thought, calls)

    def answer(self, question: str, evidence: str, history: list[dict],
               stream: bool = False, intent: str = "") -> str | Iterator[str]:
        system = (
            "你是课程助手。只能依据提供的【资料】回答问题，并在句末用 [1] [2] 标注出处编号。"
            "资料不足时明确回答『知识库中没有找到相关依据』，不得编造。"
            "回答简洁、分点，使用中文。"
        )
        user = f"问题：{question}\n\n【资料】\n{evidence or '（无资料）'}"
        payload = {
            "model": config.MODEL,
            "messages": [{"role": "system", "content": system}, *history,
                         {"role": "user", "content": user}],
            "temperature": 0.2,
            "stream": stream,
        }
        if not stream:
            with self._post(payload) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

        def generator() -> Iterator[str]:
            with self._post(payload, stream=True) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
        return generator()


def create_llm() -> BaseLLM:
    return DeepSeekLLM() if config.use_live_model() else MockLLM()
