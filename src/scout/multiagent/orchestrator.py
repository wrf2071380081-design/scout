"""多智能体编排：规划 → 扇出 → 合成。

::: 什么时候才该用多智能体

不是"问题复杂就上多智能体"。值得扇出的**只有三类**：

- **跨文档**——要同时取到两份以上文档的证据，单路 Top-K 必然漏一边
- **多对象对比**——只命中一方就给出片面结论
- **长问题 / 多要点**——一个上下文装不下所有要点的判断过程

这三类的共性是：问题**自然能拆成几个互不依赖的子问题**。
拆不开的（单实体、单事实、定义）不该扇出——那样只会凭空多花 N 倍成本。

::: 三个硬约束

1. **上下文隔离。** 主 Agent 只看到子 Agent 压缩后的结论与 span，
   不是它的完整轨迹。这是隔离能兑现承诺的前提：上行数据必须比下行小。
2. **失败隔离。** 一个子 Agent 失败，主 Agent 继续，并在答案里**如实标注覆盖缺口**——
   而不是装作没看到，更不是替它编一个结论。
3. **嵌套最多一层。** 子 Agent 不能再生子 Agent。嵌套深度写死在代码里，
   比靠提示词约束可靠得多。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..llm.base import ChatMessage, LLMClient, LLMRequest
from ..prompts import SUBQUESTION_PROMPT, SubQuestions
from ..rag.merge import EvidenceUnit
from ..rag.pipeline import PipelineConfig, RAGPipeline
from ..trace import StepKind, Trace
from ..verify.grounding import GroundingReport, verify_answer
from .subagent import SubAgent, SubResult


@dataclass(slots=True)
class MultiAgentResult:
    """多智能体运行的结果。"""

    question: str
    answer: str = ""
    outcome: str = "no_knowledge"
    plan: list[str] = field(default_factory=list)
    subresults: list[SubResult] = field(default_factory=list)
    units: list[EvidenceUnit] = field(default_factory=list, repr=False)
    """最终用于生成答案的证据单元（合并去重后，或单路回退时的证据）。
    **运行时专用，不进 to_dict**；序列化视图用 subresults[].supporting_spans。"""
    grounding: GroundingReport | None = None
    coverage_gaps: list[str] = field(default_factory=list)
    """没能完成的子问题。**如实标注**，是失败隔离对外的承诺。"""
    meta: dict[str, Any] = field(default_factory=dict)
    trace: Trace | None = None

    def to_dict(self, *, preview_chars: int = 80) -> dict[str, Any]:
        return {
            "question": self.question,
            "outcome": self.outcome,
            "answer": self.answer[:preview_chars],
            "plan": self.plan,
            "subresults": [item.to_dict(preview_chars=preview_chars) for item in self.subresults],
            "coverage_gaps": self.coverage_gaps,
            "grounding_verdict": self.grounding.verdict.value if self.grounding else None,
            "meta": self.meta,
        }


class MultiAgentOrchestrator:
    """规划 → 扇出 → 合成。

    :param pipeline: 共享的 RAG 流水线（索引共享）。
    :param max_subagents: 一次最多扇出多少个子 Agent。默认 4，
        再多的并行子问题通常意味着问题没被正确拆分。
    :param max_depth: 子 Agent 允许的最大嵌套深度。**默认 1**。
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        llm: LLMClient,
        *,
        max_subagents: int = 4,
        max_depth: int = 1,
        deadline_seconds: float = 120.0,
        settings: Settings | None = None,
        config: PipelineConfig | None = None,
        compressor=None,
    ) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.max_subagents = max_subagents
        self.max_depth = max_depth
        self.deadline_seconds = deadline_seconds
        self.settings = settings or get_settings()
        self.config = config or PipelineConfig()
        self.compressor = compressor

    # —— 主流程 ——

    def answer(self, question: str, on_stage: Any = None) -> MultiAgentResult:
        """对一个问题做多智能体问答。

        ``on_stage`` 为可选阶段回调 ``(stage, payload)``，边界与流水线一致：
        plan / fanout / synthesize / grounding / done。
        """

        def emit(stage: str, payload: dict[str, Any]) -> None:
            if on_stage is not None:
                try:
                    on_stage(stage, payload)
                except Exception:  # noqa: BLE001 - 观察者故障绝不打断主流程
                    pass

        started = time.perf_counter()
        trace = Trace(question=question)
        result = MultiAgentResult(question=question, trace=trace)

        with trace.step("plan", StepKind.PLAN, question=question) as step:
            plan = self._plan(question)
            step.outputs = {"subquestion_count": len(plan), "plan": plan}
        result.plan = plan
        emit("plan", {"subquestions": len(plan)})

        if len(plan) <= 1:
            # 拆不开就不该扇出：直接走单路，不浪费成本。
            with trace.step("single_fallback", StepKind.PLAN) as step:
                single = self.pipeline.answer(question)
                step.outputs = {"outcome": single.outcome}
            result.outcome = single.outcome
            result.answer = single.answer
            result.grounding = single.grounding
            result.units = list(single.units)
            result.meta["mode"] = "single_fallback"
            result.meta["reason"] = "问题无法拆分，回退到单路流水线"
            emit("done", {"outcome": result.outcome, "mode": "single_fallback"})
            return result

        with trace.step("fanout", StepKind.RETRIEVE, count=len(plan)) as step:
            subresults = self._fanout(plan)
            step.outputs = {
                "completed": sum(1 for item in subresults if item.ok),
                "failed": sum(1 for item in subresults if not item.ok),
            }
        result.subresults = subresults
        emit(
            "fanout",
            {
                "total": len(subresults),
                "completed": sum(1 for item in subresults if item.ok),
                "failed": sum(1 for item in subresults if not item.ok),
            },
        )

        merged_units = self._merge_evidence(subresults)
        if not merged_units:
            result.outcome = "no_knowledge"
            result.answer = "根据现有资料无法回答该问题。"
            result.meta["mode"] = "multiagent"
            result.meta["reason"] = "所有子 Agent 均未获得可用证据"
            return result
        result.units = merged_units

        with trace.step("synthesize", StepKind.GENERATE, units=len(merged_units)) as step:
            answer = self._generate(question, merged_units, trace)
            step.outputs = {"answer_chars": len(answer)}
        emit("synthesize", {"chars": len(answer), "units": len(merged_units)})

        grounding = verify_answer(
            question,
            answer,
            merged_units,
            settings=self.settings.verify if self.config.sufficiency_gate else None,
            document_frequency=self.pipeline.idf_context()["document_frequency"],
            corpus_size=int(self.pipeline.idf_context()["corpus_size"] or 0),
        )
        result.grounding = grounding
        result.answer = answer
        result.outcome = grounding.verdict.value if grounding is not None else "pass"
        result.coverage_gaps = [item.subquestion for item in subresults if not item.ok]
        emit(
            "grounding",
            {"verdict": grounding.verdict.value, "support_rate": round(grounding.support_rate, 3)},
        )
        result.meta["mode"] = "multiagent"
        result.meta.update(
            {
                "subagents_total": len(subresults),
                "subagents_completed": sum(1 for item in subresults if item.ok),
                "subagents_failed": sum(1 for item in subresults if not item.ok),
                "merged_evidence": len(merged_units),
                "duration_ms": (time.perf_counter() - started) * 1000.0,
                "grounding_support_rate": round(grounding.support_rate, 4) if grounding else 0.0,
                "grounding_coverage": round(grounding.coverage, 4) if grounding else 0.0,
            }
        )
        if result.coverage_gaps:
            # 失败隔离对外的承诺：如实告诉调用方哪些子问题没覆盖到。
            result.meta["coverage_gaps_detail"] = [
                {
                    "subquestion": item.subquestion,
                    "status": item.status,
                    "error_code": item.error_code,
                }
                for item in subresults
                if not item.ok
            ]
        emit("done", {"outcome": result.outcome, "mode": "multiagent"})
        return result

    # —— 规划 ——

    def _plan(self, question: str) -> list[str]:
        """把问题拆成子问题。拆不出多个就返回原问题（调用方回退单路）。"""

        if not self.config.complexity_routing:
            return [question]
        response = self.llm.complete(
            LLMRequest(
                messages=[ChatMessage(role="user", content=SUBQUESTION_PROMPT.format(question=question))],
                schema=SubQuestions,
                task="subquestions",
                context={"question": question},
            )
        )
        try:
            parsed = response.parse(SubQuestions)
        except Exception:  # noqa: BLE001 - 分解失败时退化为单路，属于允许的降级
            return [question]
        questions = [item.strip() for item in parsed.questions if item.strip()][: self.max_subagents]
        return questions if len(questions) >= 2 else [question]

    # —— 扇出 ——

    def _fanout(self, plan: Sequence[str]) -> list[SubResult]:
        """对每个子问题跑一个隔离的子 Agent。

        **串行执行是刻意的。** 并行子 Agent 是工程优化，不是语义优化；
        先把"隔离与合成"做对，再加并行。串行让 trace 可顺序审计、
        让离线实现完全确定——这两件事在评测里是硬要求。
        """

        results: list[SubResult] = []
        for subquestion in plan:
            subagent = SubAgent(
                self.pipeline,
                compressor=self.compressor,
                max_depth=self.max_depth,
                deadline_seconds=self.deadline_seconds,
            )
            results.append(subagent.run(subquestion, depth=0))
        return results

    # —— 合成 ——

    def _merge_evidence(self, subresults: Sequence[SubResult]) -> list[EvidenceUnit]:
        """把各子 Agent 压缩后的证据合并去重。

        **出处不丢**：每条 span 都来自原始 chunk_id，归因门控与归因校验
        仍然能追到它是哪一份文档。
        """

        seen: set[str] = set()
        merged: list[EvidenceUnit] = []
        for index, item in enumerate(subresults):
            if not item.ok:
                continue
            for unit in item.units:
                if unit.chunk.chunk_id in seen:
                    continue
                seen.add(unit.chunk.chunk_id)
                merged.append(unit)
        # 按分数排序，保证最好的证据在前
        merged.sort(key=lambda unit: -unit.score)
        return merged

    def _generate(self, question: str, units: list[EvidenceUnit], trace: Trace) -> str:
        """从合并后的压缩证据生成答案。复用流水线的生成逻辑（同包）。"""

        return self.pipeline._generate(question, units, trace)  # noqa: SLF001 - 同包协作


__all__ = ["MultiAgentOrchestrator", "MultiAgentResult"]
