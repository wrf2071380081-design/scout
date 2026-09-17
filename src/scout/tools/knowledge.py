"""知识检索工具。

把 :class:`~scout.rag.pipeline.RAGPipeline` 的检索能力暴露成一个 Agent 工具。

设计要点：

- **工具只负责"找证据"，不负责"作答"**。答案是 Agent 自己生成的，
  这样 Agent 才能把检索结果与其他工具（日期、计算器）的输出一起组织进最终回答。
  如果工具直接返回成品答案，Agent 就退化成了一个 router。
- **返回给模型的文本带证据编号**（``[1]`` ``[2]``），编号是后续归因校验的锚点。
- **结构化数据不进 prompt**：chunk_id、分数、耗时放在 ``ToolResult.data``，
  模型只看文本，避免元数据污染上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import ErrorCode
from ..llm.scripted import tokenize
from ..prompts import format_evidence
from ..rag.merge import EvidenceUnit
from ..rag.pipeline import RAGPipeline
from ..trace import Trace
from .registry import ToolResult, ToolSpec

KNOWLEDGE_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 2,
            "maxLength": 500,
            "description": "检索查询。应当是一个具体、可检索的问句或关键词组合，而不是整段对话。",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "期望返回的证据条数，默认 8。",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass
class KnowledgeSearchState:
    """一次运行内的检索状态。

    用于两件事：一是给 Agent 提供"已经检索过什么"的信息（避免重复检索），
    二是把每一轮的检索结果累积下来，供最终生成与归因校验使用。
    """

    queries: list[str] = field(default_factory=list)
    units: list[EvidenceUnit] = field(default_factory=list)
    metas: list[dict[str, Any]] = field(default_factory=list)

    def add(self, query: str, units: list[EvidenceUnit], meta: dict[str, Any]) -> None:
        self.queries.append(query)
        self.units.extend(units)
        self.metas.append(meta)

    def dedupe_units(self) -> list[EvidenceUnit]:
        """按 chunk_id 去重，保留先出现的（即分数更高的那次）。"""

        seen: set[str] = set()
        ordered: list[EvidenceUnit] = []
        for unit in self.units:
            if unit.chunk.chunk_id in seen:
                continue
            seen.add(unit.chunk.chunk_id)
            ordered.append(unit)
        return ordered

    @property
    def repeated_queries(self) -> list[str]:
        return [query for query in self.queries if self.queries.count(query) > 1]


class KnowledgeSearchTool:
    """封装为可注册工具的检索能力。"""

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        trace_factory: Callable[[], Trace | None] | None = None,
        state: KnowledgeSearchState | None = None,
        max_calls: int = 4,
    ) -> None:
        self.pipeline = pipeline
        self.trace_factory = trace_factory or (lambda: None)
        self.state = state or KnowledgeSearchState()
        self.max_calls = max_calls

    # —— 供 Agent 读取 ——

    @property
    def call_count(self) -> int:
        return len(self.state.queries)

    @property
    def has_evidence(self) -> bool:
        return bool(self.state.units)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_search",
            description=(
                "在本地知识库中检索与查询相关的资料，返回带编号的证据片段。"
                "需要事实依据时使用它，不要凭记忆回答。"
                "如果第一次结果不足，可以换一个更精确或更概括的查询再检索一次，"
                "但不要重复提交同一个查询。"
            ),
            parameters=KNOWLEDGE_SEARCH_SCHEMA,
            handler=self,
        )

    # —— 执行 ——

    def __call__(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        top_k = int(arguments.get("top_k") or 8)

        if not query:
            return ToolResult.failure("查询不能为空", code=ErrorCode.TOOL_INVALID_ARGUMENTS)

        normalized = " ".join(query.split())
        if normalized in self.state.queries:
            return ToolResult.failure(
                f"该查询已经检索过：{normalized}。请换一个不同的查询，或基于已有证据作答。",
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
                repeated=True,
                known_queries=self.state.queries,
            )
        if self.call_count >= self.max_calls:
            return ToolResult.failure(
                f"检索次数已达上限（{self.max_calls} 次），请基于现有证据作答。",
                code=ErrorCode.BUDGET_EXCEEDED,
                known_queries=self.state.queries,
            )

        base_trace = self.trace_factory()
        trace = base_trace if base_trace is not None else Trace(question=query)
        units, meta, grade = self.pipeline.collect(query, trace)

        if not units:
            self.state.add(normalized, [], meta)
            return ToolResult.success(
                "知识库中没有检索到与该查询相关的内容。可以换一个查询再试，"
                "或告知用户当前知识库未涵盖该主题。",
                query=normalized,
                hit_count=0,
                route=grade.route,
                **{key: value for key, value in meta.items() if key.startswith("retrieval_")},
            )

        trimmed = units[:top_k]
        self.state.add(normalized, trimmed, meta)
        packed = self.state.dedupe_units()
        content = format_evidence(packed)
        if grade.route in {"clarify", "no_knowledge"}:
            content += (
                "\n\n[提示] 证据覆盖不足，如果无法基于以上材料回答，请明确说明，不要推测。"
            )

        return ToolResult.success(
            content,
            query=normalized,
            hit_count=len(trimmed),
            total_evidence=len(packed),
            grade_relevance=grade.relevance,
            grade_answerable=grade.answerable,
            route=grade.route,
            evidence_ids=[unit.chunk.chunk_id for unit in trimmed],
            source_filenames=sorted({unit.chunk.filename for unit in trimmed}),
        )

    # —— 便捷查询 ——

    def keyword_footprint(self, question: str) -> float:
        """已获取证据对问题的词法覆盖度，供自愈编排判断"是否该再检索一次"。"""

        if not self.state.units:
            return 0.0
        query_tokens = {token for token in tokenize(question) if len(token) > 1}
        if not query_tokens:
            return 0.0
        evidence_tokens: set[str] = set()
        for unit in self.state.units:
            evidence_tokens |= set(tokenize(unit.context_text))
        return len(query_tokens & evidence_tokens) / len(query_tokens)


__all__ = ["KNOWLEDGE_SEARCH_SCHEMA", "KnowledgeSearchState", "KnowledgeSearchTool"]
