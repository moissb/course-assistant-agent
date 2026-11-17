"""标准库 HTTP 服务：对话接口 + SSE 流式输出 + 静态演示页。

生产版可直接换成 FastAPI（相同路由与响应结构），前端无需改动。
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent import create_agent  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Handler(BaseHTTPRequestHandler):
    agent = None

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _headers(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = (STATIC_DIR / "index.html").read_bytes()
            self._headers("text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/health":
            agent = self.__class__.agent
            self._json({"ok": True, "mode": agent.mode, "docs": len(agent.store.documents),
                        "tools": agent.registry.names()})
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._json({})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()
        agent = self.__class__.agent

        if path == "/api/reset":
            sid = body.get("session_id")
            if sid:
                agent.reset(sid)
            self._json({"ok": True, "session_id": agent.new_session()})
            return

        if path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return

        question = str(body.get("question", "")).strip()
        if not question:
            self._json({"error": "question 不能为空"}, 400)
            return
        sid = body.get("session_id") or agent.new_session()
        stream = bool(body.get("stream", True))

        try:
            result = agent.chat(question, sid)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"Agent 执行失败：{type(exc).__name__}: {exc}"}, 500)
            return

        payload = {
            "answer": result.answer,
            "intent": result.intent,
            "thought": result.thought,
            "success": result.success,
            "session_id": result.session_id,
            "citations": result.citations,
            "tool_trace": result.tool_trace,
            "events": result.events,
            "latency_ms": result.latency_ms,
            "first_token_ms": result.first_token_ms,
        }

        if not stream:
            self._json(payload)
            return

        # SSE：先推送元信息，再分块推送回答，最后推送完整轨迹
        self._headers("text/event-stream; charset=utf-8")
        self.end_headers()

        def send(obj: dict) -> None:
            chunk = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        try:
            send({"event": "start", "session_id": sid, "intent": result.intent})
            for delta in agent.stream(result):
                send({"delta": delta})
            send({"event": "done", "result": payload})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    Handler.agent = create_agent()
    server = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    mode = Handler.agent.mode
    print(f"Course Assistant Agent running at http://{config.HOST}:{config.PORT}")
    print(f"mode={mode}  docs={len(Handler.agent.store.documents)}  tools={Handler.agent.registry.names()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()


if __name__ == "__main__":
    main()
