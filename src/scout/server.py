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
from .runtime.intent import ACTION_KEYWORDS
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

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        settings: Settings | None = None,
        *,
        embed_backend: str = "auto",
    ) -> None:
        self.settings = settings or get_settings()
        self.documents: list[tuple[str, str]] = []
        if corpus_dir:
            path = Path(corpus_dir)
            if path.exists():
                self.documents = load_corpus(path)
        if not self.documents:
            self.documents = list(_FALLBACK_CORPUS)
        self.index = build_index(self.documents, settings=self.settings, embed_backend=embed_backend)
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

    # —— 结果构造（/api/ask 与 /api/stream 共用） ——

    def _pipeline_payload(self, result: Any) -> dict[str, Any]:
        units = result.units
        payload: dict[str, Any] = {
            "mode": "pipeline",
            "answer": result.answer,
            "outcome": result.outcome,
            "meta": result.meta,
            "trace": result.trace.to_dict() if result.trace else None,
        }
        payload["evidence"] = _evidence_payload(units)
        if result.grounding is not None:
            payload["grounding"] = _grounding_payload(result.grounding)
        return payload

    def _multiagent_payload(self, result: Any) -> dict[str, Any]:
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
        payload["evidence"] = _evidence_payload(units)
        if result.grounding is not None:
            payload["grounding"] = _grounding_payload(result.grounding)
        return payload

    def ask(self, question: str, mode: str = "pipeline") -> dict[str, Any]:
        if mode == "multiagent":
            return self._multiagent_payload(self.orchestrator.answer(question))
        return self._pipeline_payload(self.pipeline.answer(question))

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
        semantic = self.index.embedder.name != "hashing-256" and not self.index.embedder.name.startswith("hashing")
        return {
            "documents": len(self.documents),
            "chunks": len(self.index.chunks),
            "corpus_fingerprint": corpus_fingerprint(self.documents)[:16],
            "model": self.llm.model_name,
            "embedder": self.index.embedder.name,
            "semantic_retrieval": bool(semantic),
            "offline": not self.settings.llm.configured,
            # 把动作关键词下发给前端：前端要判断"这条请求该走审批通道"，
            # 而它没法 import Python 常量。**下发而不是让前端各写一份**——
            # 两份词表迟早会漂移，而漂移的后果是"该审批的请求被当普通问答执行了"。
            "action_keywords": list(ACTION_KEYWORDS),
            "intent_enabled": bool(self.settings.intent.enabled),
        }


# —— 结果构造共享帮手 ——


def _evidence_payload(units: Any) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": unit.chunk.chunk_id,
            "filename": unit.chunk.filename,
            "score": round(unit.score, 4),
            "level": unit.chunk.level,
            "text": unit.context_text[:400],
        }
        for unit in (units or [])[:8]
    ]


def _grounding_payload(report: Any) -> dict[str, Any]:
    return {
        "verdict": report.verdict.value,
        "support_rate": round(report.support_rate, 4),
        "coverage": round(report.coverage, 4),
        "reason": report.reason,
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
        elif path == "/api/stream":
            self._stream(str(payload.get("question", "")), str(payload.get("mode", "pipeline")))
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


    def _stream(self, question: str, mode: str) -> None:
        """流式问答（SSE）。

        事件真实的来自流水线的阶段钩子——retrieve / grade / rewrite / sanitize /
        generate / grounding / done——不是前端伪造的进度条。每个事件末尾的
        ``result`` 携带完整答案，与 ``/api/ask`` 的返回结构一致。
        """

        chunk = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\n"
            "X-Accel-Buffering: no\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        try:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        def send(event: str, payload: dict[str, Any]) -> bool:
            try:
                self.wfile.write(
                    f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        if not send("stage", {"stage": "queued", "mode": mode}):
            return

        stages: list[dict[str, Any]] = []

        def on_stage(stage: str, payload: dict[str, Any]) -> None:
            stages.append({"stage": stage, **payload})
            send("stage", {"stage": stage, **payload})

        tokens = {"count": 0, "chars": 0}

        def on_token(piece: str) -> None:
            """token 级增量下发。这是"首 token 延迟"真正被感知到的地方——
            阶段事件告诉你它在忙什么，token 事件让你马上看到字。"""

            tokens["count"] += 1
            tokens["chars"] += len(piece)
            send("token", {"text": piece})

        try:
            if mode == "multiagent":
                result = self.app.orchestrator.answer(question, on_stage=on_stage)
                payload = self.app._multiagent_payload(result)  # noqa: SLF001 - 同一个模块内部
            else:
                result = self.app.pipeline.answer(question, on_stage=on_stage, on_token=on_token)
                payload = self.app._pipeline_payload(result)  # noqa: SLF001
            payload["stream"] = {"tokens": tokens["count"], "chars": tokens["chars"]}
            send("result", payload)
        except Exception as exc:  # noqa: BLE001 - 流式中断不能把异常留给某个写死的连接
            send("error", {"error": type(exc).__name__, "detail": str(exc)[:300]})


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


def serve(
    corpus_dir: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    embed_backend: str = "auto",
) -> None:
    """启动控制台。阻塞运行，Ctrl-C 退出。

    启动时会先打印进度再构建索引——索引构建需要十几秒，
    没有提示的话用户会以为命令卡死了。
    """

    print(f"正在加载语料并构建索引：{corpus_dir or '（内置小语料）'}（向量器={embed_backend}）", flush=True)
    app = ConsoleApp(corpus_dir, embed_backend=embed_backend)
    ConsoleHandler.app = app
    print(
        f"语料就绪：{len(app.documents)} 篇 / {len(app.index.chunks)} 块"
        f"｜模型 {app.llm.model_name}｜向量器 {app.index.embedder.name}",
        flush=True,
    )

    # 端口被占用时顺延，而不是直接崩——本地同时开两个实例很常见。
    server = None
    last_error: Exception | None = None
    for candidate in range(port, port + 10):
        try:
            server = ThreadingHTTPServer((host, candidate), ConsoleHandler)
            port = candidate
            break
        except OSError as exc:
            last_error = exc
            continue
    if server is None:
        raise SystemExit(f"无法绑定 {host}:{port}～{port + 9} 之间的端口：{last_error}")

    print(f"scout console 已启动 →  http://{host}:{port}", flush=True)
    print("（请用浏览器打开上面的地址；不要在编辑器里直接预览 HTML 文件，那样连不上 API）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        server.server_close()


__all__ = ["ConsoleApp", "ConsoleHandler", "serve"]
