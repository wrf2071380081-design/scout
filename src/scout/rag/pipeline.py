"""RAG 流水线与消融开关。

这个模块的设计目标只有一个：**让每一个模块都能被单独打开和关闭**。

原因很直接——如果代码里不存在"关掉 Auto-merge"的开关，那么
"Auto-merge 到底有没有用"这个问题就永远只能靠直觉回答。
本项目的差异化恰恰建立在这类问题的**可测量**上，所以开关是一等公民：

===================  ==================================================
``complexity_routing``  复杂度判定 → 简单问题单路检索 / 复杂问题子问题分解并行检索
``retrieval_mode``      混合 / 仅稠密 / 仅稀疏
``merge_mode``          Auto-merge 的 replace / expand / off
``rerank_enabled``      是否重排
``rewrite_enabled``     是否允许缺陷触发的查询改写
``sufficiency_gate``    是否启用证据充分性门控（关掉后拒答率必然归零）
``injection_defense``   是否启用检索侧消毒与归因门控
===================  ==================================================

时序上有一个刻意的约束：**改写最多一次，重生成最多一次**。
无界循环是 Agent 系统最常见的失控方式，而"再多试一次"的边际收益
远低于它的成本与延迟代价。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..config import Settings, get_settings
from ..errors import InsufficientEvidenceError, NoKnowledgeError
from ..llm.base import ChatMessage, LLMClient, LLMRequest
from ..llm.scripted import default_client
from ..prompts import (
    ANSWER_PROMPT,
    COMPLEXITY_PROMPT,
    GRADE_PROMPT,
    SUBQUESTION_PROMPT,
    ComplexityPlan,
    GradePlan,
    SubQuestions,
    format_evidence,
    pack_evidence,
)
from ..trace import StepKind, Trace
from ..verify.grounding import GroundingReport, Verdict, verify_answer
from ..verify.sanitize import neutralize, sanitize_text
from .index import HybridIndex, RetrievalMode, ScoredChunk
from .merge import EvidenceUnit, MergeMode, MergeOutcome, auto_merge
from .ranking import LexicalReranker, RerankOutcome, default_reranker
from .rewrite import RewriteAdvisor, RewritePlan, apply_plan

ANSWERED = "answered"
NO_KNOWLEDGE = "no_knowledge"
INSUFFICIENT = "insufficient_evidence"
CLARIFY = "clarify"

ABSTENTION_ANSWER = "根据现有资料无法回答该问题。"
CLARIFY_ANSWER = "当前问题缺少必要的限定条件，请补充具体对象、时间范围或场景后我再尝试回答。"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """流水线的消融开关组合。

    ``label()`` 用于在评测报告里给每种组合起一个稳定短名，
    这样消融表可以直接用配置名做行标题。
    """

    complexity_routing: bool = True
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    merge_mode: MergeMode = MergeMode.EXPAND
    merge_threshold: int = 2
    rerank_enabled: bool = True
    rewrite_enabled: bool = True
    sufficiency_gate: bool = True
    injection_defense: bool = True
    max_rewrites: int = 1
    max_regenerations: int = 1

    def label(self) -> str:
        parts: list[str] = []
        parts.append("route" if self.complexity_routing else "noroute")
        parts.append(self.retrieval_mode.value)
        parts.append(f"merge-{self.merge_mode.value}")
        parts.append("rerank" if self.rerank_enabled else "norerank")
        parts.append("rewrite" if self.rewrite_enabled else "norewrite")
        parts.append("gate" if self.sufficiency_gate else "nogate")
        return "+".join(parts)

    def with_overrides(self, **kwargs: Any) -> PipelineConfig:
        return replace(self, **kwargs)


@dataclass(slots=True)
class PipelineResult:
    """一次问答的完整产出。"""

    question: str
    answer: str
    outcome: str
    units: list[EvidenceUnit] = field(default_factory=list)
    grounding: GroundingReport = field(default_factory=GroundingReport)
    trace: Trace | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""

    @property
    def evidence_ids(self) -> list[str]:
        return [unit.chunk.chunk_id for unit in self.units]

    @property
    def source_filenames(self) -> list[str]:
        seen: list[str] = []
        for unit in self.units:
            if unit.chunk.filename not in seen:
                seen.append(unit.chunk.filename)
        return seen

    def to_dict(self, *, include_evidence_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "outcome": self.outcome,
            "error_code": self.error_code or None,
            "evidence_ids": self.evidence_ids,
            "source_filenames": self.source_filenames,
            "evidence_count": len(self.units),
            **self.grounding.to_meta(),
            **self.meta,
        }
        if include_evidence_text:
            payload["evidence"] = [unit.to_dict() for unit in self.units]
        return payload


# 惰性导入：``scout.runtime`` 会在导入期反向依赖 ``scout.rag.embed``，
# 顶层导入会形成包级循环。放在函数内既断开循环，也符合"流式是可选能力"的语义。
def _collect_stream(*args: Any, **kwargs: Any):
    from ..runtime.stream import collect_stream

    return collect_stream(*args, **kwargs)


def _stream_supports(client: Any) -> bool:
    from ..runtime.stream import supports_streaming

    return supports_streaming(client)


# 与 runtime.intent.ACTION_LABEL 保持同值的字面量。这里不直接 import 是为了避免
# 包级循环导入；测试里有一条断言专门校验两者一致，防止哪天改了一边忘了另一边。
_ACTION_LABEL = "动作请求"


class RAGPipeline:
    """可配置、可观测、可消融的 RAG 流水线。"""
    def __init__(
        self,
        index: HybridIndex,
        llm: LLMClient | None = None,
        *,
        settings: Settings | None = None,
        config: PipelineConfig | None = None,
        funnel: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or PipelineConfig(
            merge_mode=MergeMode.EXPAND if self.settings.retrieval.auto_merge_enabled else MergeMode.OFF,
            merge_threshold=self.settings.retrieval.auto_merge_threshold,
            rerank_enabled=self.settings.retrieval.rerank_enabled,
        )
        self.index = index
        self.llm: LLMClient = llm or default_client()
        # 意图漏斗：只在配置显式打开时构建。
        # 复用索引已有的向量器（第二层语义判定不需要额外模型），
        # 完全没配就保持 None —— 一个"没有词典的空漏斗"只会浪费一次判定。
        self.funnel = funnel
        if self.funnel is None and self.settings.intent.enabled:
            from ..runtime.intent import build_knowledge_funnel

            self.funnel = build_knowledge_funnel(
                embedder=getattr(index, "embedder", None),
                llm=self.llm,
                semantic_threshold=self.settings.intent.semantic_threshold,
                margin_threshold=self.settings.intent.margin_threshold,
                sensitive_labels=self.settings.intent.sensitive_labels,
            )
        self.reranker = default_reranker(
            self.settings.retrieval.rerank_backend,
            candidate_limit=max(self.settings.retrieval.top_k * 4, 20),
            min_score=self.settings.retrieval.rerank_min_score,
            model_name=self.settings.retrieval.rerank_model,
        )
        stats = index.corpus_stats()
        self.advisor = RewriteAdvisor(
            vocabulary=index.vocabulary,
            document_frequency=index.document_frequency,
        )
        # 语料级统计缓存：充分性判定与评分都要用到稀有度加权，
        # 每次调用现算会重复遍历全部文档，这里在构造期取一次。
        #
        # 注意 corpus_size 的口径必须与 document_frequency 一致：
        # BM25 是在**叶子块**上拟合的，所以 df 的统计基数是 chunk 数而不是文档数。
        # 早期版本这里传了文档数（40 而 df 可达 988），
        # 导致 df > N、IDF 变负、覆盖率算出 -83 这种荒谬值。
        self._document_frequency = index.document_frequency
        self._corpus_size = len(index.chunks) or 1
        self._corpus_stats = stats

    def idf_context(self) -> dict[str, Any]:
        """语料级稀有度统计，供充分性判定与评分器使用。

        公开这个方法而不是让调用方去翻 ``_document_frequency``，
        是因为 Agent 侧（``scout.agent.loop``）也需要在收尾校验时用到它。
        """

        return {
            "document_frequency": self._document_frequency,
            "corpus_size": self._corpus_size,
        }

    # —— 检索阶段 ——

    def _plan_queries(self, question: str, trace: Trace) -> list[str]:
        if not self.config.complexity_routing:
            return [question]
        with trace.step("complexity", StepKind.PLAN, question=question) as step:
            response = self.llm.complete(
                LLMRequest(
                    messages=[ChatMessage(role="user", content=COMPLEXITY_PROMPT.format(question=question))],
                    schema=ComplexityPlan,
                    task="complexity",
                    context={"question": question},
                )
            )
            plan = response.parse(ComplexityPlan)
            step.outputs = {"complexity": plan.complexity, "reason": plan.reason}
        if plan.complexity == "simple":
            return [question]

        with trace.step("subquestions", StepKind.PLAN, question=question) as step:
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
            except Exception:  # noqa: BLE001 - 分解失败时退化为单路检索，属于允许的降级
                parsed = SubQuestions(questions=[])
            questions = [item for item in parsed.questions if item.strip()][:4]
            step.outputs = {"count": len(questions)}
        return questions or [question]

    def _search(self, query: str, trace: Trace) -> list[ScoredChunk]:
        with trace.step("retrieve", StepKind.RETRIEVE, query=query) as step:
            result = self.index.search(
                query,
                top_k=self.settings.retrieval.top_k * 2,
                mode=self.config.retrieval_mode,
            )
            step.outputs = result.to_meta()
        return result.hits

    @staticmethod
    def _dedupe(hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """按 chunk_id 去重，保留更高分，且保持首次出现的顺序。"""

        best: dict[str, ScoredChunk] = {}
        order: list[str] = []
        for hit in hits:
            key = hit.chunk.chunk_id
            if key not in best:
                best[key] = hit
                order.append(key)
            elif hit.score > best[key].score:
                best[key] = hit
        return [best[key] for key in order]

    def _postprocess(
        self,
        query: str,
        hits: list[ScoredChunk],
        trace: Trace,
    ) -> tuple[list[EvidenceUnit], MergeOutcome, RerankOutcome | None]:
        """Auto-merge → 重排。两者的顺序固定：先合并再重排。"""

        units = [EvidenceUnit(chunk=hit.chunk, score=hit.score) for hit in hits]
        with trace.step("auto_merge", StepKind.MERGE, units=len(units)) as step:
            units, merge_outcome = auto_merge(
                units,
                self.index.catalog,
                mode=self.config.merge_mode,
                threshold=self.config.merge_threshold,
            )
            step.outputs = merge_outcome.to_meta()

        rerank_outcome: RerankOutcome | None = None
        if self.config.rerank_enabled:
            with trace.step("rerank", StepKind.RERANK, candidates=len(units)) as step:
                # 重排只认**检索身份**（叶子块原文），不认包装后的 context_text：
                # EXPAND 的意图是"检索最小单元仍是叶子块，父块文本只是陪嫁"，
                # 如果拿父块全文去做词法重排，EXPAND 就会被自己稀释，
                # 甚至比不上干净的小块——这是消融实验里真实暴露过的问题。
                rerank_outcome = self.reranker.rerank(
                    query,
                    [(unit.chunk.chunk_id, unit.chunk.text) for unit in units],
                    enabled=True,
                )
                score_map = dict(rerank_outcome.ordered)
                order = [identifier for identifier, _ in rerank_outcome.ordered]
                by_id = {unit.chunk.chunk_id: unit for unit in units}
                for identifier in order:
                    unit = by_id.get(identifier)
                    if unit is not None:
                        unit.score = score_map.get(identifier, unit.score)
                units = [by_id[identifier] for identifier in order if identifier in by_id]
                step.outputs = rerank_outcome.to_meta()
        return units, merge_outcome, rerank_outcome

    def _grade(self, question: str, units: list[EvidenceUnit], trace: Trace) -> GradePlan:
        packed, _dropped = pack_evidence(units, budget_chars=self.settings.retrieval.grader_evidence_chars)
        with trace.step("grade", StepKind.GRADE, candidates=len(units), packed=len(packed)) as step:
            response = self.llm.complete(
                LLMRequest(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=GRADE_PROMPT.format(
                                question=question,
                                evidence=format_evidence(packed) or "（无证据）",
                            ),
                        )
                    ],
                    schema=GradePlan,
                    task="grade",
                    context={"question": question, "evidence_count": len(packed), **self.idf_context()},
                )
            )
            try:
                plan = response.parse(GradePlan)
            except Exception:  # noqa: BLE001 - 评分失败按"可作答"处理，交由充分性门控兜底
                plan = GradePlan(relevance=0.0, coverage=0.0, answerable=bool(units), route="answer")
            step.outputs = plan.model_dump()
        return plan

    # —— 消毒 ——

    def _sanitize_units(self, units: list[EvidenceUnit], trace: Trace) -> list[EvidenceUnit]:
        if not self.config.injection_defense or not self.settings.verify.sanitize_enabled:
            return units
        with trace.step("sanitize", StepKind.VERIFY, units=len(units)) as step:
            suspicious = 0
            removed = 0
            for unit in units:
                report = sanitize_text(unit.context_text)
                if report.removed_html_blocks or report.removed_invisible or report.stripped_tags:
                    removed += report.removed_html_blocks + report.stripped_tags
                if report.suspicious:
                    suspicious += 1
                    unit.context_text = neutralize(report.text)
                elif report.text:
                    unit.context_text = report.text
            step.outputs = {
                "sanitize_suspicious_units": suspicious,
                "sanitize_stripped_blocks": removed,
            }
            step.metrics = {"injection_signals": suspicious}
        return units

    # —— 生成 ——

    def _generate(
        self,
        question: str,
        units: list[EvidenceUnit],
        trace: Trace,
        *,
        strict: bool = False,
        on_token: Any = None,
    ) -> str:
        packed, _dropped = pack_evidence(units, budget_chars=self.settings.retrieval.evidence_budget_chars)
        prompt = ANSWER_PROMPT.format(question=question, evidence=format_evidence(packed) or "（无证据）")
        if strict:
            prompt = (
                "注意：上一版回答中存在无法被证据支撑的结论。"
                "这一版必须做到每一句话都能在证据里找到出处，无法支撑的内容直接省略。\n\n" + prompt
            )
        with trace.step("generate", StepKind.GENERATE, evidence=len(packed), strict=strict) as step:
            request = LLMRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                task="answer",
                context={"question": question, "evidence_count": len(packed)},
            )
            if on_token is not None and _stream_supports(self.llm):
                # 有流式能力且调用方要流：逐片吐出，同时收齐完整文本。
                # 流式与阻塞走**同一段提示词与同一条证据**，
                # 否则"流式"会悄悄变成另一个系统，两边结果对不上。
                outcome = _collect_stream(self.llm, request, on_token=on_token)
                content = outcome.text
                step.metrics = outcome.usage.to_dict()
            else:
                response = self.llm.complete(request)
                content = response.content
                step.metrics = response.usage.to_dict()
            step.outputs = {"answer_chars": len(content)}
        return content.strip()

    # —— 检索编排（流水线与 Agent 工具共用同一段实现） ——

    def collect(
        self,
        question: str,
        trace: Trace,
        *,
        original_question: str | None = None,
    ) -> tuple[list[EvidenceUnit], dict[str, Any], GradePlan]:
        """检索 → 合并 → 重排 → 评分（**不含生成**）。

        Agent 的 ``knowledge_search`` 工具与流水线的单次问答都走这里。
        共用一段编排是刻意的：如果工具侧另写一套检索，两边就会随时间漂移，
        评测拿到的好成绩也不再代表 Agent 的真实表现。
        """

        queries = self._plan_queries(question, trace)
        all_hits: list[ScoredChunk] = []
        for query in queries:
            all_hits.extend(self._search(query, trace))
        hits = self._dedupe(all_hits)

        meta: dict[str, Any] = {
            "retrieval_subquery_count": len(queries),
            "retrieval_unique_candidates": len(hits),
            "auto_merge_mode": self.config.merge_mode.value,
        }
        if not hits:
            return [], meta, GradePlan(relevance=0.0, coverage=0.0, answerable=False, route="no_knowledge")

        units, merge_outcome, rerank_outcome = self._postprocess(queries[0], hits, trace)
        meta.update(merge_outcome.to_meta())
        if rerank_outcome is not None:
            meta.update(rerank_outcome.to_meta())

        grade = self._grade(original_question or question, units, trace)
        meta.update(
            {
                "grade_relevance": grade.relevance,
                "grade_coverage": grade.coverage,
                "grade_answerable": grade.answerable,
                "grade_route": grade.route,
            }
        )
        return units, meta, grade

    # —— 主入口 ——

    def answer(self, question: str, on_stage: Any = None, on_token: Any = None) -> PipelineResult:
        """执行一次完整问答。

        ``on_stage`` 是可选的阶段回调：签名 ``(stage, payload)``，在每个
        真实的工作边界上被调用一次——retrieve / grade / rewrite / sanitize /
        generate / grounding / done。Web 控制台与评测的 SSE 输出就靠它，
        而不是靠对边界的猜测。**阶段钩子必须落在真实断点上，伪造的阶段比没有更糟。**

        ``on_token`` 是可选的增量回调：签名 ``(piece: str)``。
        给到它且底层客户端支持流式时，生成阶段改为逐片产出（首 token 延迟大幅下降）；
        客户端不支持时自动退化为一次性返回——调用方不需要写两个分支。
        """

        def emit(stage: str, payload: dict[str, Any]) -> None:
            if on_stage is not None:
                try:
                    on_stage(stage, payload)
                except Exception:  # noqa: BLE001 - 观察者故障绝不应该中断主流程
                    pass

        trace = Trace(question=question)
        result = PipelineResult(question=question, answer="", outcome=NO_KNOWLEDGE, trace=trace)
        result.meta["config_label"] = self.config.label()

        # —— 意图前置：决定"要不要检索" ——
        # 这一步的价值不只是分类准确率，而是**它能在检索之前短路**：
        # 闲聊与越界问题根本不需要检索与生成，省下的是真金白银与一次往返延迟。
        if self.funnel is not None:
            intent = self.funnel.classify(question)
            result.meta["intent"] = intent.to_dict()
            emit("intent", {k: v for k, v in intent.to_dict().items() if k != "candidates"})
            short_circuit = tuple(self.settings.intent.short_circuit_labels)
            if intent.needs_clarification:
                options = self.funnel.clarify_options(intent)
                result.outcome = CLARIFY
                result.answer = CLARIFY_ANSWER + (f"可选：{'、'.join(options)}" if options else "")
                result.error_code = InsufficientEvidenceError.default_code.value
                emit("done", {"outcome": result.outcome, "intent_short_circuit": "clarify"})
                return result
            if intent.label in short_circuit:
                result.outcome = NO_KNOWLEDGE
                result.answer = f"这个问题不属于本知识库的覆盖范围（识别为「{intent.label}」），未执行检索。"
                emit("done", {"outcome": result.outcome, "intent_short_circuit": intent.label})
                return result
            if intent.label == _ACTION_LABEL:
                # 动作类请求不在这里执行：标记出来交给带审批的通道。
                # 检索照常进行（动作往往需要先查清楚对象），但调用方必须看得见这个标记。
                result.meta["intent_route"] = "action"

        units, meta, grade = self.collect(question, trace)
        result.meta.update(meta)
        emit("retrieve", {"units": len(units)})
        emit("grade", {"answerable": grade.answerable, "route": grade.route, "coverage": round(grade.coverage, 3)})

        if not units:
            result.outcome = NO_KNOWLEDGE
            result.answer = ABSTENTION_ANSWER
            result.error_code = NoKnowledgeError.default_code.value
            emit("done", {"outcome": result.outcome})
            return result

        if self.config.rewrite_enabled and grade.route == "rewrite":
            with trace.step("rewrite_diagnose", StepKind.REWRITE, question=question) as step:
                report = self.advisor.diagnose(question)
                plan: RewritePlan = self.advisor.plan(question, report, llm=self.llm)
                step.outputs = {"method": plan.method.value, "reason": plan.reason, **report.to_meta()}
            result.meta["rewrite_triggered"] = plan.triggers
            result.meta["rewrite_method"] = plan.method.value
            result.meta["rewrite_reason"] = plan.reason
            if plan.triggers:
                emit("rewrite", {"method": plan.method.value, "reason": plan.reason})
                rewritten = apply_plan(question, plan)
                second_units, second_meta, second_grade = self.collect(
                    rewritten, trace, original_question=question
                )
                if second_units:
                    units, grade = second_units, second_grade
                    result.meta.update(second_meta)
                    result.meta["rewrite_improved"] = bool(second_grade.answerable)

        units = self._sanitize_units(units, trace)
        result.units = units
        emit("sanitize", {"units": len(units)})

        if grade.route == "clarify" or grade.ambiguous:
            result.outcome = CLARIFY
            result.answer = CLARIFY_ANSWER
            result.error_code = InsufficientEvidenceError.default_code.value
            emit("done", {"outcome": result.outcome})
            return result

        if not grade.answerable:
            result.outcome = INSUFFICIENT
            result.answer = ABSTENTION_ANSWER
            result.error_code = InsufficientEvidenceError.default_code.value
            emit("done", {"outcome": result.outcome})
            return result

        answer = self._generate(question, units, trace, on_token=on_token)
        emit("generate", {"chars": len(answer), "regenerations": 0})
        grounding: GroundingReport = verify_answer(
            question,
            answer,
            units,
            settings=self.settings.verify if self.config.sufficiency_gate else None,
            document_frequency=self._document_frequency,
            corpus_size=self._corpus_size,
        )
        if not self.config.sufficiency_gate:
            grounding.verdict = Verdict.PASS
            grounding.reason = "gate_disabled"

        regenerations = 0
        while (
            grounding.verdict is Verdict.REGENERATE
            and regenerations < self.config.max_regenerations
        ):
            regenerations += 1
            emit("regenerate", {"attempt": regenerations})
            with trace.step("regenerate", StepKind.GENERATE, attempt=regenerations) as step:
                answer = self._generate(question, units, trace, strict=True)
                step.outputs = {"answer_chars": len(answer)}
            grounding = verify_answer(
                question,
                answer,
                units,
                settings=self.settings.verify,
                document_frequency=self._document_frequency,
                corpus_size=self._corpus_size,
            )

        result.grounding = grounding
        result.meta["regenerations"] = regenerations
        emit("grounding", {"verdict": grounding.verdict.value, "support_rate": round(grounding.support_rate, 3)})

        if self.config.sufficiency_gate and grounding.verdict is Verdict.ABSTAIN:
            result.outcome = INSUFFICIENT
            result.answer = ABSTENTION_ANSWER
            result.error_code = InsufficientEvidenceError.default_code.value
            emit("done", {"outcome": result.outcome})
            return result

        result.answer = answer
        result.outcome = ANSWERED
        result.meta["trace_durations_ms"] = trace.kind_durations()
        result.meta["total_duration_ms"] = trace.total_duration_ms()
        emit("done", {"outcome": result.outcome, "total_ms": trace.total_duration_ms()})
        return result

    def answer_or_raise(self, question: str) -> PipelineResult:
        """与 :meth:`answer` 相同，但把「无知识 / 证据不足」抛成 typed 异常。

        供需要区分"语义结果"与"故障"的调用方使用（例如评测运行器要统计
        各类结果的分布，而 Agent 循环要把无知识渲染成对用户友好的措辞）。
        """

        result = self.answer(question)
        if result.outcome == NO_KNOWLEDGE:
            raise NoKnowledgeError("retrieval returned no candidates for the question")
        if result.outcome == INSUFFICIENT:
            raise InsufficientEvidenceError(
                "retrieved evidence does not sufficiently cover the question",
                missing=[str(item) for item in result.meta.get("missing", [])] if result.meta else [],
            )
        return result


def build_index(
    documents: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    embedder: Any = None,
    embed_backend: str = "auto",
    milvus: bool | None = None,
    corpus_key: str = "",
    strict: bool | None = None,
) -> HybridIndex:
    """便捷构造：``documents`` 为 ``[(filename, text), ...]``。

    评测脚本与测试都走这个入口，保证"索引构建方式"只有一处定义——
    否则不同脚本各建一次索引，报告之间就不可比。

    ``embedder`` 显式传入时优先；否则按 ``embed_backend`` 解析
    （``auto`` 会在装了 fastembed 时启用本地语义向量模型）。
    注意：单元测试传的是显式 embedder 或默认 ``auto``——
    而 ``auto`` 在没有 fastembed 的环境里退回哈希向量器，因此测试保持确定性。

    :param milvus: ``None`` 时读配置（``SCOUT_MILVUS_ENABLED``）；
        ``True`` 强制走 Milvus 后端，``False`` 强制内存。
        **两种后端必须由同一个入口构造**，否则"两种后端的指标可对比"
        这件事就没有保障——它们可能连分块方式都不一样。
    :param corpus_key: 语料身份，用于派生 Milvus 集合名。留空时用语料内容指纹。
    :param strict: 是否拒绝从 Milvus 降级到内存。出评测报告时应当为 True。
    """

    from .embed import default_embedder

    effective = settings or get_settings()
    active_embedder = embedder or default_embedder(effective, backend=embed_backend)
    use_milvus = effective.milvus.enabled if milvus is None else milvus
    index: HybridIndex
    if use_milvus:
        from .milvus_store import MilvusHybridIndex

        # 惰性导入：evaluation 包会反向依赖 rag，顶层导入会形成包级循环
        from ..evaluation.dataset import corpus_fingerprint

        index = MilvusHybridIndex(
            embedder=active_embedder,
            settings=effective.retrieval,
            milvus=effective.milvus,
            corpus_key=corpus_key or corpus_fingerprint(documents),
            strict=strict,
        )
    else:
        index = HybridIndex(embedder=active_embedder, settings=effective.retrieval)

    for position, (filename, text) in enumerate(documents):
        index.add_document(
            text,
            document_id=f"doc-{position:03d}",
            document_version="v1",
            filename=filename,
        )
    index.build()
    return index


__all__ = [
    "ABSTENTION_ANSWER",
    "ANSWERED",
    "CLARIFY",
    "CLARIFY_ANSWER",
    "INSUFFICIENT",
    "NO_KNOWLEDGE",
    "PipelineConfig",
    "PipelineResult",
    "RAGPipeline",
    "build_index",
]
