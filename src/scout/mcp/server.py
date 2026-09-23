"""MCP（Model Context Protocol）服务器：把 scout 的检索能力暴露成标准工具。

**为什么自己实现而不装官方 SDK。**

官方 `mcp` SDK 当然更完整。但 scout 的约束是**零新增依赖、克隆即跑**，
而 MCP 在 stdio 上的核心其实很小：**换行分隔的 JSON-RPC 2.0**。
真正需要的只有四个方法：

- ``initialize`` / ``notifications/initialized`` —— 握手
- ``tools/list`` —— 声明能力
- ``tools/call`` —— 执行

所以这里手写了一个约 150 行的实现，换来的是：**任何装了 scout 的环境都能
直接当一个 MCP server 用**，不需要再拉一条依赖链。

::: 设计上值得说的两点

1. **工具描述里写清"这个工具会返回带出处的证据"。** MCP 客户端（Agent）
   是否引用证据，取决于工具描述怎么说。把溯源要求写进工具契约，
   比在 Agent 提示词里反复强调更可靠。
2. **搜索工具返回的是结构化证据，不是一大段文本。** 每个结果带
   ``chunk_id`` / ``filename`` / ``score``，让上层 Agent 能做归因和裁剪。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..llm.scripted import default_client
from ..rag.pipeline import RAGPipeline, build_index
from ..evaluation.runner import load_corpus
from ..trace import Trace

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "scout", "version": "1.0.0"}


class MCPServer:
    """极简 MCP 服务器（stdio，换行分隔 JSON-RPC 2.0）。"""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    # —— 工具定义 ——

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "scout_search",
                "description": (
                    "在长文档知识库中检索证据。返回带出处的结构化证据列表"
                    "（chunk_id / 来源文件 / 相关度 / 原文片段）。"
                    "回答时必须引用检索到的证据编号，不得凭空作答。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索查询"},
                        "top_k": {"type": "integer", "default": 5, "description": "返回条数"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scout_ask",
                "description": (
                    "对长文档知识库提问，返回带证据引用的答案；"
                    "证据不足时会明确说明无法回答，而不是编造。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "问题"}},
                    "required": ["question"],
                },
            },
            {
                "name": "scout_stats",
                "description": "返回知识库规模（文档数 / 块数），用于判断语料是否就绪。",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    # —— 工具执行 ——

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "scout_search":
            return self._search(str(arguments.get("query", "")), int(arguments.get("top_k", 5)))
        if name == "scout_ask":
            return self._ask(str(arguments.get("question", "")))
        if name == "scout_stats":
            return self._stats()
        return {"content": [{"type": "text", "text": f"未知工具：{name}"}], "isError": True}

    def _search(self, query: str, top_k: int) -> dict[str, Any]:
        trace = Trace(question=query)
        units, _meta, _grade = self.pipeline.collect(query, trace)
        if not units:
            return {"content": [{"type": "text", "text": "没有检索到相关证据。"}], "isError": False}
        items = []
        for index, unit in enumerate(units[: max(1, top_k)], start=1):
            items.append(
                f"[{index}] 来源：{unit.chunk.filename}（{unit.chunk.chunk_id}，score {unit.score:.3f}）\n"
                f"{unit.context_text[:600]}"
            )
        return {"content": [{"type": "text", "text": "\n\n".join(items)}], "isError": False}

    def _ask(self, question: str) -> dict[str, Any]:
        result = self.pipeline.answer(question)
        suffix = ""
        if result.grounding is not None:
            suffix = f"\n\n[归因] {result.grounding.verdict.value} · 支撑率 {result.grounding.support_rate:.0%}"
        return {
            "content": [{"type": "text", "text": (result.answer or "（无答案）") + suffix}],
            "isError": False,
        }

    def _stats(self) -> dict[str, Any]:
        index = self.pipeline.index
        payload = {
            "documents": len({chunk.document_id for chunk in index.chunks}),
            "chunks": len(index.chunks),
            "model": self.pipeline.llm.model_name,
        }
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": False}

    # —— JSON-RPC ——

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """处理一条 JSON-RPC 消息。通知类消息返回 None（不回复）。"""

        method = str(message.get("method", ""))
        request_id = message.get("id")
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "notifications/initialized" or method.startswith("notifications/"):
            return None
        if method == "tools/list":
            return self._result(request_id, {"tools": self.tools()})
        if method == "tools/call":
            params = message.get("params") or {}
            payload = self.call_tool(str(params.get("name", "")), dict(params.get("arguments") or {}))
            return self._result(request_id, payload)
        if method == "ping":
            return self._result(request_id, {})
        return self._error(request_id, -32601, f"不支持的方法：{method}")

    @staticmethod
    def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_mcp(corpus_dir: str | Path, settings: Settings | None = None) -> None:
    """以 stdio 方式运行 MCP 服务器。阻塞，直到 stdin 关闭。"""

    effective = settings or get_settings()
    path = Path(corpus_dir)
    # 图片同样可入：语料加载统一走抽取入口（抽不到会记进 skipped 而非静默跳过）
    from ..data import build_extractor_from_settings

    skipped: list[str] = []
    documents = (
        load_corpus(
            path,
            image_extractor=build_extractor_from_settings(effective),
            skip_log=skipped,
        )
        if path.exists()
        else []
    )
    for item in skipped:
        print(f"[scout-mcp] 跳过：{item}", file=sys.stderr)
    if not documents:
        documents = [("README.md", "空语料：请用 --corpus 指定文档目录。")]
    pipeline = RAGPipeline(build_index(documents, settings=effective), default_client(), settings=effective)
    server = MCPServer(pipeline)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(
                json.dumps(MCPServer._error(None, -32700, "解析失败：不是合法 JSON"), ensure_ascii=False) + "\n"
            )
            sys.stdout.flush()
            continue
        response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


__all__ = ["MCPServer", "run_mcp"]
