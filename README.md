# 课程助手 Agent（DeepSeek + LangGraph 架构可运行版）

简历中「基于 DeepSeek + LangGraph 的课程助手多工具 RAG Agent」的工程实现：一个能「查资料 + 查数据 + 多轮追问」的对话式 Agent，带知识库检索、Function Calling 工具、编排重试、引用来源、SSE 流式接口与 Web 演示页。

**当前环境无第三方包且无法访问 pip 源，所以核心链路用 Python 标准库实现，架构与生产版一一对应；README 末尾给出 LangChain / LangGraph / FAISS / FastAPI / Vue3 的替换映射。**

## 快速开始

```bash
cd course-agent
python server/app.py
# 浏览器打开 http://127.0.0.1:8765
```

无需安装任何依赖、无需 API Key，启动即进入离线 Mock 模式（规则决策 + 抽取式生成），可完整演示所有功能。

连接真实 DeepSeek 模型：

```bash
cp .env.example .env      # 填入 DEEPSEEK_API_KEY
# 或在 shell 中直接设置环境变量后启动
DEEPSEEK_API_KEY=sk-xxxx python server/app.py
```

`AGENT_MODE=auto` 时检测到 Key 自动切换真实模型；也可显式设为 `mock` / `live`。

## 能问什么

| 类型 | 示例 | 实际调用 |
| --- | --- | --- |
| 成绩查询 | 我的高等数学成绩是多少？ | `query_grade`（只读固定 JSON 数据源，支持学号校验） |
| 课程知识 | 什么是过拟合？/ B+ 树索引有什么特点？ | `search_docs`（TF-IDF 检索 + 引用回填） |
| 教务流程 | 缓考怎么申请？/ 奖学金评选条件？ | `search_docs`（FAQ 知识库） |
| 数值计算 | 计算 (95+93+92)/3 | `calculate`（AST 白名单，无 eval） |
| 日期 | 今天星期几？ | `get_date` |
| 无依据问题 | 量子纠缠的数学证明是什么？ | 检索相关度过低 → 主动拒答，不硬凑 |

## 目录结构

```
course-agent/
├── src/
│   ├── config.py      # 参数与环境变量
│   ├── rag.py         # 文档切分 + TF-IDF 向量检索 + 引用
│   ├── tools.py       # 工具注册表、参数校验、4 个只读工具
│   ├── llm.py         # DeepSeek API（Function Calling / stream）+ MockLLM
│   ├── memory.py      # 会话记忆
│   ├── graph.py       # 编排：意图→路由→工具/检索→反思重试→生成 + 事件日志
│   └── agent.py       # 对外 API：create_agent()
├── server/
│   ├── app.py         # HTTP 服务与 SSE 接口
│   └── static/index.html  # 聊天 + 运行轨迹演示页
├── data/
│   ├── courses.json   # 课程与成绩（模拟只读业务库）
│   ├── faq.json       # 教务 FAQ
│   └── docs/*.txt     # 课程讲义知识库
├── evaluation/
│   ├── questions.json # 60 条标注问题
│   ├── run_eval.py    # 评测脚本
│   ├── report.json    # 逐条结果
│   └── REPORT.md      # 评测报告
└── logs/events.jsonl  # 可观测事件日志
```

## 核心链路

```
用户问题
  → [intent]  意图识别 / Function Calling 决策（LLM 输出 tool_calls）
  → [route]   路由到 query_grade / search_docs / calculate / get_date
  → [tool]    参数校验 → 只读工具执行 → 超时与异常兜底
  → [reflect] 有依据？否 → 降低阈值重试（最多 MAX_ITERATIONS 轮）；仍无依据 → 拒答
  → [answer]  仅基于证据生成回答 + [n] 引用出处 → SSE 分块推送
每个节点写入 logs/events.jsonl，前端「运行轨迹」面板实时展示。
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 运行模式、知识库片段数、工具列表 |
| POST | `/api/chat` | `{"question":"...","session_id":null,"stream":true}` |
| POST | `/api/reset` | 重置会话 |
| GET | `/` | 演示页 |

`stream=false` 返回完整 JSON；`stream=true` 返回 `text/event-stream`，事件依次为 `start`、`delta...`、`done`（done 携带意图、工具轨迹、引用与耗时）。

Python 里也可直接调用：

```python
from src import create_agent
agent = create_agent()
result = agent.chat("什么是过拟合？")
print(result.answer, result.intent, result.citations)
```

## 评测

```bash
python evaluation/run_eval.py
```

最新一次真实结果（离线 Mock 模式，60 条标注集）：

| 指标 | 结果 |
| --- | --- |
| 意图路由准确率 | **100%**（60/60） |
| 答案可用率 | **100%**（57/57 条有依据问题） |
| 无依据拒答率 | **100%**（3/3） |
| 首字响应平均 / P95 | 0.34 ms / 0.76 ms |

说明：这是本地 Mock 链路指标，不含网络与大模型推理耗时。配置 `DEEPSEEK_API_KEY` 后用同一脚本复测即可得到真实模型指标。

## 安全设计

- 成绩查询只读固定 JSON 数据源，校验学号，不拼接或执行任意 SQL；
- 工具参数做类型与长度校验，未知工具直接拒绝；
- 计算器用 AST 白名单求值，禁止函数调用与变量名，不使用 `eval`；
- 知识库无依据时主动拒答，相似度阈值 `AGENT_SIM_THRESHOLD` 可调；
- 工具调用有耗时统计与超时标记，编排限制最大重试轮数，避免死循环。

## 生产版依赖映射（可选）

当前实现零依赖即可运行；若要按简历技术栈部署，替换以下模块即可，对外接口不变：

| 当前实现 | 生产替换 | 替换位置 |
| --- | --- | --- |
| `MockLLM` 规则决策 | DeepSeek Function Calling（已实现于 `DeepSeekLLM`） | `src/llm.py` |
| `SimpleVectorStore`（TF-IDF） | BGE embedding + FAISS / Chroma | `src/rag.py` |
| `AgentGraph` 手写节点 | LangGraph `StateGraph`（节点函数可直接复用） | `src/graph.py` |
| `ToolRegistry` | LangChain `@tool` / StructuredTool | `src/tools.py` |
| `http.server` + SSE | FastAPI + `StreamingResponse` | `server/app.py` |
| 原生 HTML/JS 演示页 | Vue3 组件（接口不变） | `server/static/index.html` |

生产依赖清单见 `requirements.txt`；当前环境无法安装这些包时，不影响本项目运行与评测。
