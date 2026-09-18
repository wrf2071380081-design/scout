"""scout Console：本地 Web 控制台的 HTTP 服务与 API。

**为什么用标准库而不是 FastAPI。**

控制台是"演示与调试"用的，不是生产服务。引入 FastAPI + uvicorn 会让
`pip install -e .` 多拖十几个依赖，而 scout 的核心卖点之一就是
**克隆下来、不需要任何 API key、5 分钟能跑**——为一个演示界面破坏这条约束不值得。

所以这里用 ``http.server`` 手写一个三十行的路由：
- ``GET  /``                  控制台页面（单文件静态资源）
- ``GET  /api/health``        语料与模型状态
- ``POST /api/ask``           问答（pipeline / multiagent 两种模式）
- ``POST /api/action``        发起一个带副作用的动作（触发 HITL 审批）
- ``GET  /api/pending``       待审批列表
- ``POST /api/approve``       审批（可改参数）
- ``GET  /api/ablation``      消融结论摘要

单线程处理请求是刻意的：离线实现下每一步都是毫秒级，串行反而让 trace 时间干净。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Settings, get_settings
from .evaluation.dataset import corpus_fingerprint
from .evaluation.runner import load_corpus
from .llm.base import LLMResponse, ToolCall
from .llm.scripted import default_client
from .multiagent import MultiAgentOrchestrator
from .rag.pipeline import RAGPipeline, build_index
from .hitl import (
    HumanDecision,
    InMemoryCheckpointStore,
    ResumableAgent,
    TimeoutPolicy,
)
from .tools.actions import build_action_tools
from .tools.builtin import build_default_tools
from .tools.registry import ToolRegistry

WEB_DIR = Path(__file__).resolve().parent / "web"

_FALLBACK_CORPUS = [
    ("指南.md", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全等6个部分。"),
    ("年报.md", "公司2025年实现营业收入同比增长12%，研发投入占比达到8.5%。"),
]


class _ActionScriptLLM:
    """控制台"动作通道"用的确定性模型客户端。

    它的行为极简：**没有工具结果就发起一次发邮件调用；拿到工具结果就结束。**
    这样控制台的 HITL 演示每次都能稳定触发审批，不依赖启发式模型会不会调工具。

    这是**演示专用**的替身（test double），不是系统能力：
    它证明的是"审批→恢复→不重复副作用"这条链路，不是模型有多聪明。
    """

    @property
    def model_name(self) -> str:
        return "scripted-action"

    def complete(self, request: Any) -> Any:
        has_tool_result = any(getattr(m, "role", "") == "tool" for m in request.messages)
        if has_tool_result:
            return LLMResponse(content="已将周报发送给老板。")
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    name="send_email",
                    arguments={"to": "boss@example.com", "subject": "本周周报", "body": "本周进展见附件。"},
                )
            ]
        )


class ConsoleApp:
    """控制台的运行状态：索引、流水线、待审批的运行。"""

    def __init__(self, corpus_dir: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.documents: list[tuple[str, str]] = []
        if corpus_dir:
            path = Path(corpus_dir)
            if path.exists():
                self.documents = load_corpus(path)
        if not self.documents:
            self.documents = list(_FALLBACK_CORPUS)
        self.index = build_index(self.documents, settings=self.settings)
        self.llm = default_client()
        self.pipeline = RAGPipeline(self.index, self.llm, settings=self.settings)
        self.orchestrator = MultiAgentOrchestrator(self.pipeline, self.llm, settings=self.settings)
        self.store = InMemoryCheckpointStore()
        self.action_agent = ResumableAgent(
            _ActionScriptLLM(),
            ToolRegistry([*build_default_tools(), *build_action_tools()]),
            store=self.store,
            timeout_policy=TimeoutPolicy.REJECT,
            approval_deadline_seconds=600.0,
        )
        self.pending: dict[str, Any] = {}
        self._lock = threading.Lock()

    # —— 问答 ——

    def ask(self, question: str, mode: str = "pipeline") -> dict[str, Any]:
        if mode == "multiagent":
            result = self.orchestrator.answer(question)
            units = result.units
            payload: dict[str, Any] = {
                "mode": "multiagent",
                "answer": result.answer,
                "outcome": result.outcome,
                "plan": result.plan,
                "coverage_gaps": result.coverage_gaps,
                "subresults": [item.to_dict() for item in result.subresults],
                "meta": result.meta,
                "trace": result.trace.to_dict() if result.trace else None,
            }
        else:
            result = self.pipeline.answer(question)
            units = result.units
            payload = {
                "mode": "pipeline",
                "answer": result.answer,
                "outcome": result.outcome,
                "meta": result.meta,
                "trace": result.trace.to_dict() if result.trace else None,
            }
        payload["evidence"] = [
            {
                "chunk_id": unit.chunk.chunk_id,
                "filename": unit.chunk.filename,
                "score": round(unit.score, 4),
                "level": unit.chunk.level,
                "text": unit.context_text[:400],
            }
            for unit in units[:8]
        ]
        if result.grounding is not None:
            payload["grounding"] = {
                "verdict": result.grounding.verdict.value,
                "support_rate": round(result.grounding.support_rate, 4),
                "coverage": round(result.grounding.coverage, 4),
                "reason": result.grounding.reason,
            }
        return payload

    # —— 动作与审批 ——

    def start_action(self, question: str) -> dict[str, Any]:
        outcome = self.action_agent.start(question)
        with self._lock:
            self.pending[outcome.run_id] = outcome
        return self._run_payload(outcome)

    def _run_payload(self, outcome: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": outcome.run_id,
            "status": outcome.status.value,
            "answer": outcome.answer,
            "meta": outcome.meta,
            "request": outcome.request.to_dict() if outcome.request else None,
        }
        return payload

    def list_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for run_id, outcome in self.pending.items():
                if outcome.status.value != "awaiting_approval":
                    continue
                items.append(
                    {
                        "run_id": run_id,
                        "request": outcome.request.to_dict() if outcome.request else None,
                    }
                )
            return items

    def approve(
        self,
        run_id: str,
        *,
        approved: bool,
        edited_arguments: dict[str, Any] | None = None,
        decided_by: str = "console-reviewer",
    ) -> dict[str, Any]:
        with self._lock:
            outcome = self.pending.get(run_id)
        if outcome is None or outcome.request is None:
            raise KeyError(run_id)
        decision = HumanDecision(
            request_id=outcome.request.request_id,
            approved=approved,
            edited_arguments=edited_arguments,
            decided_by=decided_by,
        )
        resumed = self.action_agent.resume(run_id, decision)
        with self._lock:
            self.pending[run_id] = resumed
        return self._run_payload(resumed)

    def health(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.index.chunks),
            "corpus_fingerprint": corpus_fingerprint(self.documents)[:16],
            "model": self.llm.model_name,
            "offline": True,
        }


class ConsoleHandler(BaseHTTPRequestHandler):
    app: ConsoleApp = None  # type: ignore[assignment]

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # 允许跨源调用：页面可能是从磁盘（file://）或其他端口打开的，
        # 没有这几个头，浏览器会直接拦掉请求——现象就是"拿不到响应"。
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # 浏览器提前断开（比如用户切走了页面）不该在服务端留下噪声栈。
            pass

    def _json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib 接口名
        """CORS 预检。POST + application/json 会先发 OPTIONS，不回就整个请求失败。"""

        self._send(204, b"")

    def do_GET(self) -> None:  # noqa: N802 - stdlib 接口名
        try:
            self._route_get()
        except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都必须变成 JSON，而不是空响应
            self._json({"error": "internal", "detail": f"{type(exc).__name__}: {exc}"[:300]}, 500)

    def _route_get(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            html = WEB_DIR / "index.html"
            if not html.exists():
                self._json({"error": "ui_missing", "path": str(html)}, 500)
                return
            self._send(200, html.read_bytes(), "text/html")
            return
        if path == "/api/health":
            self._json(self.app.health())
            return
        if path == "/api/pending":
            self._json({"pending": self.app.list_pending()})
            return
        if path == "/api/ablation":
            self._json(_ablation_summary())
            return
        self._json({"error": "not_found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib 接口名
        try:
            self._route_post()
        except Exception as exc:  # noqa: BLE001 - 同上：不能让异常穿透成空响应
            self._json({"error": "internal", "detail": f"{type(exc).__name__}: {exc}"[:300]}, 500)

    def _route_post(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "invalid_json"}, 400)
            return
        if path == "/api/ask":
            self._json(self.app.ask(str(payload.get("question", "")), str(payload.get("mode", "pipeline"))))
        elif path == "/api/action":
            self._json(self.app.start_action(str(payload.get("question", ""))))
        elif path == "/api/approve":
            try:
                self._json(
                    self.app.approve(
                        str(payload.get("run_id", "")),
                        approved=bool(payload.get("approved", False)),
                        edited_arguments=payload.get("edited_arguments"),
                    )
                )
            except KeyError:
                self._json({"error": "unknown_run", "run_id": payload.get("run_id")}, 404)
        else:
            self._json({"error": "not_found", "path": path}, 404)

    def log_message(self, fmt: str, *args: Any) -> None:
        # 控制台默认静音：本地调试不该被请求日志刷屏。
        return


def _ablation_summary() -> dict[str, Any]:
    """消融结论摘要（相对全量基线的 MRR 变化，单位 pp）。"""

    return {
        "baseline_mrr": 68.6,
        "deltas_mrr": {
            "baseline_dense_only": -41.9,
            "-rerank": -16.2,
            "-rewrite": -1.1,
            "-route": 2.6,
            "-merge": 0.0,
            "-gate": 0.0,
            "merge_replace": 6.8,
        },
        "note": "19 条样本上的方向性结论；负值表示关掉该模块后变差。",
    }


def serve(corpus_dir: str | Path | None = None, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """启动控制台。阻塞运行，Ctrl-C 退出。"""

    app = ConsoleApp(corpus_dir)
    ConsoleHandler.app = app
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    print(f"scout console: http://{host}:{port}  (语料 {len(app.documents)} 篇，离线模式)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["ConsoleApp", "ConsoleHandler", "serve"]
