"""离线评测：60 条标注问题，真实计算路由准确率、答案可用率、首字响应与分类表现。

运行：python evaluation/run_eval.py
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import create_agent  # noqa: E402
from src import config  # noqa: E402

QUESTIONS = ROOT / "evaluation" / "questions.json"
REPORT_JSON = ROOT / "evaluation" / "report.json"
REPORT_MD = ROOT / "evaluation" / "REPORT.md"


def usable_answer(item: dict, result) -> bool:
    text = result.answer.strip()
    if item.get("should_refuse"):
        return not result.success or "没有找到相关依据" in text or "无法回答" in text
    if not text:
        return False
    if "没有找到相关依据" in text or "请求失败" in text:
        return False
    if result.intent in {"calculator", "date"}:
        return result.success
    if len(text) < 8:
        return False
    return result.success or result.intent in {"query_grade", "calculator", "date"}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))], 2)


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    agent = create_agent()
    records = []

    for item in questions:
        result = agent.chat(item["question"])
        records.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "expected_intent": item["expected_intent"],
            "actual_intent": result.intent,
            "route_ok": result.intent == item["expected_intent"],
            "answer_usable": usable_answer(item, result),
            "should_refuse": bool(item.get("should_refuse")),
            "success": result.success,
            "citations": len(result.citations),
            "first_token_ms": result.first_token_ms,
            "latency_ms": result.latency_ms,
            "answer": result.answer[:200],
        })

    total = len(records)
    route_ok = sum(r["route_ok"] for r in records)
    answerable = [r for r in records if not r["should_refuse"]]
    refusals = [r for r in records if r["should_refuse"]]
    usable = sum(r["answer_usable"] for r in answerable)
    refused = sum(r["answer_usable"] for r in refusals)
    first_tokens = [r["first_token_ms"] for r in records]
    latencies = [r["latency_ms"] for r in records]

    by_category: dict[str, dict] = {}
    for r in records:
        bucket = by_category.setdefault(r["category"], {"total": 0, "route_ok": 0, "usable": 0})
        bucket["total"] += 1
        bucket["route_ok"] += int(r["route_ok"])
        bucket["usable"] += int(r["answer_usable"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": agent.mode,
        "model": "离线 Mock（规则决策 + 抽取式生成）" if agent.mode == "mock" else "DeepSeek API",
        "total_questions": total,
        "knowledge_chunks": len(agent.store.documents),
        "intent_route_accuracy": round(route_ok / total * 100, 2),
        "answer_usable_rate": round(usable / len(answerable) * 100, 2),
        "refuse_rate": round(refused / len(refusals) * 100, 2) if refusals else 100.0,
        "first_token_ms_avg": round(statistics.mean(first_tokens), 2),
        "first_token_ms_p95": percentile(first_tokens, 0.95),
        "latency_ms_avg": round(statistics.mean(latencies), 2),
        "latency_ms_p95": percentile(latencies, 0.95),
        "by_category": by_category,
    }

    REPORT_JSON.write_text(json.dumps({"summary": summary, "records": records},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 课程助手 Agent 评测报告",
        "",
        f"生成时间：{summary['generated_at']}　｜　运行模式：{summary['model']}",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 | 口径 |",
        "| --- | --- | --- |",
        f"| 意图路由准确率 | **{summary['intent_route_accuracy']}%** | {route_ok}/{total} 条路由到预期工具 |",
        f"| 答案可用率 | **{summary['answer_usable_rate']}%** | {usable}/{len(answerable)} 条有依据问题返回有效答案 |",
        f"| 无依据拒答率 | **{summary['refuse_rate']}%** | {refused}/{len(refusals)} 条无依据问题正确拒答 |",
        f"| 首字响应平均 / P95 | {summary['first_token_ms_avg']} ms / {summary['first_token_ms_p95']} ms | 从输入到生成首个内容片段 |",
        f"| 端到端延迟平均 / P95 | {summary['latency_ms_avg']} ms / {summary['latency_ms_p95']} ms | 完整一句话响应 |",
        f"| 知识库片段数 | {summary['knowledge_chunks']} | 课程文档按段落切分 + FAQ 条目 |",
        "",
        "## 分类表现",
        "",
        "| 类别 | 题数 | 路由正确 | 答案可用 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, bucket in by_category.items():
        lines.append(f"| {name} | {bucket['total']} | {bucket['route_ok']} | {bucket['usable']} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 评测集见 `evaluation/questions.json`，逐条结果见 `evaluation/report.json`，可重复运行本脚本核对。",
        f"- 相似度阈值 {config.SIMILARITY_THRESHOLD}：低于该值视为无依据并触发拒答，不硬凑答案。",
        "- 当前为离线 Mock 模式，指标反映检索与编排链路质量；切换 DEEPSEEK_API_KEY 后可用同一评测集复测真实模型。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
