"""子 Agent：**隔离的上下文窗口**。

::: 先把误解纠正过来

多智能体不是"再加一个角色"。对一个长文档 RAG 系统来说，
单 Agent 的真正瓶颈是**上下文窗口**：

- 同一份上下文里塞进三份文档的检索、评价、推理，
  一份文档的错误推理会污染另一份的判断——**上下文污染**
- 一份长文档的完整轨迹会挤掉另一份的证据——**上下文过载**
- 一旦中间某步走错，整个上下文都被带偏——**错误级联**

子 Agent 的真正价值不在于"多个人"，而在于**隔离**：

1. **独立的上下文窗口。** 每个子 Agent 有自己的消息、自己的证据、自己的轨迹，
   互不可见。A 文档的判断不会被 B 文档的噪声干扰。
2. **独立的预算与失败边界。** 一个子 Agent 超时/失败，
   不会拖垮整个运行——它只是在自己的结果里标注"我没完成"。
3. **压缩后的上行。** 主 Agent 看到的不是子 Agent 的完整轨迹，
   而是它**压缩过的结论 + 支撑 span**。这本身就是上下文工程：
   主上下文只装"已经验证过的事实"，而不是"原始证据 + 推理过程"。

这条设计里最关键的约束是：**子 Agent 不能再生子 Agent。**
允许的嵌套最多一层。多智能体失控最常见的原因不是模型不行，
而是嵌套深度不受限——深度每加一层，成本、延迟、出错面都乘上一个系数，
而边际收益急剧递减。把它写死在代码里，比靠提示词约束要可靠得多。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..context.compress import EvidenceCompressor
from ..rag.merge import EvidenceUnit
from ..rag.pipeline import RAGPipeline
from ..trace import Trace


@dataclass(slots=True)
class SubResult:
    """一个子 Agent 交付给主 Agent 的东西。

    **注意这里没有什么**：没有子 Agent 的完整消息历史，没有它的完整 trace，
    没有原始检索证据。主 Agent 拿到的是**压缩后的结论 + 支撑 span**——
    这是"上下文隔离"能兑现承诺的前提：上行数据必须比下行小。
    """

    subquestion: str
    conclusion: str = ""
    """压缩后的事实结论（几句话）。这是上行的主体。"""

    supporting_spans: list[dict[str, str]] = field(default_factory=list)
    """支撑结论的证据 span（压缩后），每项 {chunk_id, filename, text}。
    这些 span 会被主 Agent 的归因门控继续引用——出处不能丢。"""

    units: list[EvidenceUnit] = field(default_factory=list, repr=False)
    """压缩后的证据单元（**运行时专用，不进 to_dict**）。
    编排器用它做合并与归因校验。序列化视图用 ``supporting_spans``。"""

    status: str = "completed"
    """completed | insufficient_evidence | no_knowledge | failed | timeout | depth_exceeded"""

    confidence: float = 0.0
    """子 Agent 自评：grounding 支撑率与覆盖率的组合。0~1。"""

    error_code: str = ""
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """是否带回了可用证据。

        **注意判据**：是"有没有找到证据"，不是"能不能独立回答"。
        子 Agent 的职责是"检索 + 压缩"自己的那片证据，
        "能不能答"是编排器在**合成边界**统一判定的事——
        如果要求每个子 Agent 各自过一遍充分性门控，
        一个本该贡献部分证据的子问题也会被整条丢掉。
        """

        return bool(self.units)

    def to_dict(self, *, preview_chars: int = 80) -> dict[str, Any]:
        return {
            "subquestion": self.subquestion,
            "conclusion": self.conclusion[:preview_chars],
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "span_count": len(self.supporting_spans),
            "error_code": self.error_code or None,
            "latency_ms": round(self.latency_ms, 1),
            "meta": self.meta,
        }


class SubAgent:
    """一个隔离的子 Agent。

    :param pipeline: 共享的 RAG 流水线。**索引必须共享**（检索对象是同一个语料库），
        但每次调用 :meth:`run` 都是一次**独立的上下文**——新的 trace、
        新的证据打包、新的归因校验。
    :param compressor: 证据压缩器。子 Agent 用它把证据压成 span 再上行，
        而不是把原始块整段递给主 Agent。
    :param max_depth: 允许的最大嵌套深度。**默认 1**（子 Agent 不能再生子 Agent）。
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        compressor: EvidenceCompressor | None = None,
        max_depth: int = 1,
        deadline_seconds: float = 60.0,
    ) -> None:
        self.pipeline = pipeline
        self.compressor = compressor or EvidenceCompressor()
        self.max_depth = max_depth
        self.deadline_seconds = deadline_seconds

    def run(self, subquestion: str, *, depth: int = 0) -> SubResult:
        """在隔离的上下文里回答一个子问题。

        :param depth: 当前嵌套深度。超过 ``max_depth`` 直接返回失败，
            不会尝试"更深一层"——嵌套失控是多智能体系统最常见的死法。
        """

        started = time.perf_counter()

        if depth > self.max_depth:
            return SubResult(
                subquestion=subquestion,
                status="depth_exceeded",
                error_code="multiagent_depth_exceeded",
                meta={"depth": depth, "max_depth": self.max_depth},
            )

        result = SubResult(subquestion=subquestion)
        try:
            trace = Trace(question=subquestion)
            # 只做检索与评分（collect 不含生成、不过门控）：
            # "能不能答"是编排器在合成边界统一判定的事，不在子 Agent 这一层。
            units, meta, grade = self.pipeline.collect(subquestion, trace)
            result.latency_ms = (time.perf_counter() - started) * 1000.0

            if not units:
                result.status = "no_knowledge"
                result.confidence = 0.0
                result.meta["reason"] = "子问题在知识库中没有相关证据"
                return result

            # 压缩证据：从完整块里抽出支撑结论的 span，而不是把整块递上去。
            compressed, compression = self.compressor.compress(
                subquestion, units, budget_chars=None
            )
            result.units = list(compressed)
            result.supporting_spans = [
                {
                    "chunk_id": unit.chunk.chunk_id,
                    "filename": unit.chunk.filename,
                    "text": unit.context_text,
                }
                for unit in compressed
            ]
            # completed = 评分认为可独立回答；partial = 只有部分证据。
            # 两种状态都**保留 span 供合成使用**——只有完全没证据才算失败。
            result.status = "completed" if grade.answerable else "partial"
            result.confidence = min(1.0, (grade.relevance + grade.coverage) / 2.0)
            result.error_code = ""

            # 结论：评分认为能答才生成（离线启发式）；否则留空但保留 span。
            if grade.answerable:
                result.conclusion = self.pipeline._generate(subquestion, compressed, trace)  # noqa: SLF001 - 同包协作
            result.meta.update(
                {
                    "grade_relevance": round(grade.relevance, 4),
                    "grade_coverage": round(grade.coverage, 4),
                    "grade_answerable": grade.answerable,
                    "grade_route": grade.route,
                    "evidence_count": len(units),
                    **compression.to_meta(),
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001 - 子 Agent 失败必须被隔离，不能拖垮主 Agent
            result.status = "failed"
            result.error_code = f"subagent_error:{type(exc).__name__}"
            result.latency_ms = (time.perf_counter() - started) * 1000.0
            result.meta["exception"] = str(exc)[:200]
            return result


__all__ = ["SubAgent", "SubResult"]
