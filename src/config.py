"""项目配置：所有可调参数集中在这里，支持环境变量覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
LOG_DIR = ROOT / "logs"
EVAL_DIR = ROOT / "evaluation"

# auto: 有 DEEPSEEK_API_KEY 走真实模型，否则离线 Mock
MODE = os.getenv("AGENT_MODE", "auto").lower()
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
REQUEST_TIMEOUT = float(os.getenv("AGENT_REQUEST_TIMEOUT", "30"))

# RAG 参数
TOP_K = int(os.getenv("AGENT_TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("AGENT_SIM_THRESHOLD", "0.12"))

# 编排参数
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "2"))
TOOL_TIMEOUT = float(os.getenv("AGENT_TOOL_TIMEOUT", "5"))

# 前端服务
HOST = os.getenv("AGENT_HOST", "127.0.0.1")
PORT = int(os.getenv("AGENT_PORT", "8765"))


def use_live_model() -> bool:
    if MODE == "live":
        return True
    if MODE == "mock":
        return False
    return bool(API_KEY)
