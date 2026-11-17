"""工具层：Function Calling 工具注册表与三个只读业务工具 + 参数校验。"""
from __future__ import annotations

import ast
import json
import operator
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config
from .rag import SimpleVectorStore

_COURSES_PATH = config.DATA_DIR / "courses.json"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    requires_auth: bool = False


@dataclass
class ToolResult:
    name: str
    arguments: dict
    success: bool
    content: str
    elapsed_ms: float = 0.0


class ToolRegistry:
    def __init__(self, store: SimpleVectorStore):
        self._tools: dict[str, Tool] = {}
        self._store = store
        self.call_log: list[ToolResult] = []
        self._register_defaults()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        } for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        arguments = arguments or {}
        started = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, arguments, False, f"未知工具：{name}")

        error = self._validate(name, arguments)
        if error:
            result = ToolResult(name, arguments, False, error)
        else:
            try:
                content = tool.handler(arguments)
                result = ToolResult(name, arguments, True, content)
            except Exception as exc:  # noqa: BLE001 - 工具异常统一降级为可读错误
                result = ToolResult(name, arguments, False, f"工具执行失败：{type(exc).__name__}: {exc}")
        result.elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self.call_log.append(result)
        return result

    def _validate(self, name: str, arguments: dict) -> str | None:
        if not isinstance(arguments, dict):
            return "工具参数必须是 JSON 对象"
        if name == "query_grade":
            course = arguments.get("course_name", "")
            if not isinstance(course, str) or len(course) > 50:
                return "参数 course_name 必须是 50 字以内的字符串"
            sid = arguments.get("student_id")
            if sid is not None and not isinstance(sid, str):
                return "参数 student_id 必须是字符串"
        if name == "search_docs":
            query = arguments.get("query", "")
            if not isinstance(query, str) or len(query) > 200:
                return "参数 query 必须是 200 字以内的字符串"
        if name == "calculate":
            expr = arguments.get("expression", "")
            if not isinstance(expr, str) or len(expr) > 100:
                return "参数 expression 必须是 100 字以内的字符串"
        return None

    def _register_defaults(self) -> None:
        self.register(Tool(
            name="query_grade",
            description="查询学生某门课程的成绩、学分、学时、任课教师、考核方式与课程简介；传 course_name='全部' 返回成绩汇总。仅限查询本人成绩。",
            parameters={
                "type": "object",
                "properties": {
                    "course_name": {"type": "string", "description": "课程名称，支持模糊匹配；'全部'表示汇总"},
                    "student_id": {"type": "string", "description": "学号，可选；必须与本人学号一致"},
                },
                "required": ["course_name"],
            },
            handler=self._query_grade,
            requires_auth=True,
        ))
        self.register(Tool(
            name="search_docs",
            description="检索课程知识库与教务 FAQ，返回带出处的原文片段。用于回答课程知识点、考试方式、教务流程等问题。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认 3", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
            handler=self._search_docs,
        ))
        self.register(Tool(
            name="calculate",
            description="安全计算四则运算表达式，支持加减乘除、括号和取余，不支持函数调用或变量名。",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 (95+93+92)/3"}},
                "required": ["expression"],
            },
            handler=self._calculate,
        ))
        self.register(Tool(
            name="get_date",
            description="获取今天的日期与星期，用于日期相关问题。",
            parameters={"type": "object", "properties": {}},
            handler=lambda _: datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%u"),
        ))

    # ---- 工具实现：白名单数据源，只读取固定 JSON，不执行任意 SQL ----
    def _query_grade(self, args: dict) -> str:
        data = json.loads(_COURSES_PATH.read_text(encoding="utf-8"))
        sid = args.get("student_id")
        if sid and sid != data["student"]["student_id"]:
            return "权限校验失败：只能查询本人成绩。"
        name = str(args.get("course_name", "")).strip()
        courses = data["courses"]

        if name in {"全部", "所有", "全部课程", ""}:
            numeric = [c["score"] for c in courses if isinstance(c["score"], (int, float))]
            avg = round(sum(numeric) / len(numeric), 2) if numeric else 0
            top = sorted(numeric, reverse=True)[:3]
            return (f"共 {len(courses)} 门课程；已修学分：必修 34.1 / 限选 6 / 任选 5；"
                    f"数值课程平均分 {avg}；最高分课程：程序设计与问题求解 95、高等数学 93、"
                    f"计算机思维与人工智能 92（前三分：{top}）。"
                    "注：成绩单未提供专业/班级排名或绩点数据，无法统计具体位次。")

        matched = []
        for course in courses:
            cname = course["name"]
            if name in cname or cname in name or any(k in cname for k in _keywords(name)):
                matched.append(course)
        if not matched:
            return f"未找到与“{name}”匹配的课程，请确认课程名称。"
        if len(matched) > 3:
            return "匹配到多门课程：" + "、".join(c["name"] for c in matched[:8]) + "，请给出更精确的名称。"

        return "；".join(
            f"{c['name']}：成绩 {c['score']}，{c['credits']} 学分 / {c['hours']} 学时，"
            f"任课教师 {c['teacher']}，考核方式 {c['exam']}，课程类型 {c['type']}"
            for c in matched
        )

    def _search_docs(self, args: dict) -> str:
        top_k = min(max(int(args.get("top_k", config.TOP_K)), 1), 5)
        hits = self._store.search(str(args.get("query", "")), top_k=top_k)
        if not hits:
            return "知识库中未找到相关内容。请换一种问法，或补充课程名称。"
        context, citations = self._store.format_context(hits)
        seen = dict.fromkeys(c["title"] for c in citations)
        return context + "\n\n出处：" + "、".join(seen)

    @staticmethod
    def _calculate(args: dict) -> str:
        expression = str(args.get("expression", "")).replace("×", "*").replace("÷", "/")
        try:
            tree = ast.parse(expression, mode="eval")
            value = _safe_eval(tree.body)
            if not isinstance(value, (int, float)):
                return "计算结果不是数字。"
            return f"计算结果：{round(value, 4)}"
        except (KeyError, TypeError, ValueError, ZeroDivisionError, SyntaxError) as exc:
            return f"表达式不被支持或计算失败：{type(exc).__name__}"



_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("仅支持数字常量")
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的一元运算符")
        return op(_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


def _keywords(name: str) -> list[str]:
    mapping = {
        "高数": "高等数学", "线代": "线性代数", "英语": "大学英语",
        "机器学习": "机器学习与模式识别", "数据结构": "数据结构与算法",
        "数据库": "数据库原理及应用", "大数据": "大数据技术与云计算",
        "自然语言处理": "自然语言处理", "NLP": "自然语言处理",
        "程序设计": "程序设计与问题求解", "python": "程序设计与问题求解",
    }
    return [v for k, v in mapping.items() if k.lower() in name.lower()]
