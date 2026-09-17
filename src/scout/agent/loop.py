"""ReAct 风格的 Agent 循环。

**与"tool-calling loop"的区别。**

很多自称 Agent 的实现其实是固定流程：先检索 → 再打分 → 再生成，中间夹一次
条件判断，然后叫它 Agent。真正的差别在于**控制流由模型在每一步决定**：
要不要检索、检索什么、够不够、要不要换个工具、什么时候停下。

但"让模型自己决定"和"系统可控"是一对矛盾。这个循环用四道约束把矛盾按住：

============  ==================================================================
步数预算      ``max_steps`` —— 停在有限步内，不依赖模型自己收敛
工具预算      ``max_tool_calls`` —— 单次运行的工具调用总数上限
重复抑制      ``max_repeated_tool_calls`` —— 同参数重复调用不再执行，回灌纠偏提示
墙钟 deadline 绝对截止时间 —— 超时立即终止，而不是让请求无限拉长
============  ==================================================================

另外，**工具失败不中断循环**：失败被归一成可读的 observation 回灌给模型，
由它决定修正参数还是换路子；只有当自愈编排判定"该停止"时才终止。
这是"能自愈"与"一错就崩"的分界线。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..config import AgentSettings, Settings, get_settings
from ..errors import BudgetExceededError, ErrorCode, ProviderError, ScoutError
from ..llm.base import ChatMessage, LLMClient, LLMRequest, TokenUsage, ToolCall
from ..prompts import AGENT_SYSTEM_PROMPT, format_evidence, pack_evidence
from ..trace import StepKind, Trace
from ..verify.grounding import GroundingReport, Verdict, verify_answer
from ..orchestrator.healing import RecoveryAction, SelfHealingOrchestrator
from ..tools.registry import ToolRegistry, ToolResult

ANSWERED = "answered"
BUDGET_STOPPED = "budget_exceeded"
NO_ANSWER = "no_answer"
PROVIDER_TIMEOUT = "provider_timeout"
"""模型服务不可用时的语义化结果。必须和"不知道"区分，
否则系统会把"服务挂了"误导成"知识库没有"——完全不同的两个问题。"""


@dataclass(slots=True)
class ObservedCall:
    """一次工具调用的观测记录。

    ``ok`` 给了默认值，是为了让调用点可以先构造记录、再在执行后回填结果——
    这样即使执行路径中途抛错，也仍然有一条可审计的调用记录留在 step 里。
    """

    name: str
    arguments: dict[str, Any]
    ok: bool = False
    error_code: str = ""
    content_preview: str = ""
    repeated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "error_code": self.error_code or None,
            "content_preview": self.content_preview[:160],
            "repeated": self.repeated,
        }


@dataclass(slots=True)
class AgentStep:
    """循环中的一步。"""

    index: int
    thought: str = ""
    calls: list[ObservedCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought[:200],
            "calls": [call.to_dict() for call in self.calls],
        }


@dataclass(slots=True)
class AgentRunResult:
    """一次 Agent 运行的结果。"""

    question: str
    answer: str
    outcome: str
    steps: list[AgentStep] = field(default_factory=list)
    trace: Trace | None = None
    grounding: GroundingReport = field(default_factory=GroundingReport)
    usage: TokenUsage = field(default_factory=TokenUsage)
    meta: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.calls) for step in self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def failed_call_count(self) -> int:
        return sum(1 for step in self.steps for call in step.calls if not call.ok)

    @property
    def repeated_call_count(self) -> int:
        return sum(1 for step in self.steps for call in step.calls if call.repeated)

    def to_dict(self, *, include_steps: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "outcome": self.outcome,
            "error_code": self.error_code or None,
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "failed_call_count": self.failed_call_count,
            "repeated_call_count": self.repeated_call_count,
            **self.usage.to_dict(),
            **self.grounding.to_meta(),
            **self.meta,
        }
        if self.trace is not None:
            payload["trace_durations_ms"] = self.trace.kind_durations()
            payload["total_duration_ms"] = self.trace.total_duration_ms()
        if include_steps:
            payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


class ToolAgent:
    """带预算、自愈与归因校验的工具调用 Agent。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        settings: Settings | None = None,
        agent_settings: AgentSettings | None = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        orchestrator: SelfHealingOrchestrator | None = None,
        memory: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget = agent_settings or self.settings.agent
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.orchestrator = orchestrator or SelfHealingOrchestrator()
        self.memory = memory
        self._current_trace: Trace | None = None

    # —— 供工具读取当前 trace（避免把 trace 塞进工具参数） ——

    @property
    def current_trace(self) -> Trace | None:
        return self._current_trace

    # —— 主循环 ——

    def run(self, question: str, *, history: list[ChatMessage] | None = None) -> AgentRunResult:
        trace = Trace(question=question)
        self._current_trace = trace
        self.orchestrator.reset()

        result = AgentRunResult(question=question, answer="", outcome=NO_ANSWER, trace=trace)
        result.meta["agent_max_steps"] = self.budget.max_steps
        result.meta["agent_max_tool_calls"] = self.budget.max_tool_calls

        deadline = time.monotonic() + self.budget.deadline_seconds
        messages: list[ChatMessage] = [ChatMessage(role="system", content=self._compose_system(question))]
        messages.extend(history or [])
        messages.append(ChatMessage(role="user", content=question))

        call_signatures: Counter[str] = Counter()

        for step_index in range(1, self.budget.max_steps + 1):
            if time.monotonic() > deadline:
                result.outcome = BUDGET_STOPPED
                result.error_code = ErrorCode.BUDGET_EXCEEDED.value
                result.answer = "抱歉，本次处理超出了时间预算，未能给出完整回答。"
                result.meta["stop_reason"] = "deadline_exceeded"
                break

            try:
                with trace.step("agent_step", StepKind.PLAN, index=step_index) as step:
                    response = self.llm.complete(
                        LLMRequest(
                            messages=messages,
                            tools=self.registry.schemas(),
                            task="decide",
                            context={"question": question, "step": step_index},
                            deadline=deadline,
                        )
                    )
                    result.usage.input_tokens += response.usage.input_tokens
                    result.usage.output_tokens += response.usage.output_tokens

                    record = AgentStep(index=step_index, thought=response.content.strip()[:200])

                    if not response.has_tool_calls:
                        if response.content.strip():
                            result.answer = response.content.strip()
                            result.outcome = ANSWERED
                            result.meta["stop_reason"] = "model_final_answer"
                        else:
                            result.meta["stop_reason"] = "empty_final_answer"
                        step.outputs = {"final": True, "chars": len(response.content)}
                        result.steps.append(record)
                        break

                    messages.append(
                        ChatMessage(role="assistant", content=response.content, tool_calls=list(response.tool_calls))
                    )

                    for call in response.tool_calls:
                        observation, observed = self._execute(
                            call,
                            call_signatures=call_signatures,
                            messages=messages,
                            deadline=deadline,
                        )
                        record.calls.append(observed)
                        messages.append(
                            ChatMessage(role="tool", content=observation, name=call.name, tool_call_id=call.id)
                        )

                    step.outputs = {"calls": [item.to_dict() for item in record.calls]}
                    result.steps.append(record)
            except ScoutError as exc:
                # Provider 或任何 typed 错误穿透到这里时，必须先归类：
                # 它可以是一次可恢复的临时故障，也可以是必须终止的硬错误。
                # 静默上升等于让主调用栈去猜这是什么——这是不能接受的。
                signal = self.orchestrator.classify(exc, stage="decide")
                decision = self.orchestrator.decide(signal)
                if decision.action is RecoveryAction.RETRY and exc.retryable:
                    self.orchestrator.consume(signal, decision)
                    self.orchestrator.mark_recovered(signal, decision)
                    result.meta.setdefault("recoveries", []).append(decision.to_dict())
                    continue
                self.orchestrator.mark_failed(signal, decision)
                result.outcome = PROVIDER_TIMEOUT if isinstance(exc, ProviderError) else NO_ANSWER
                result.error_code = exc.code.value
                result.answer = "抱歉，模型服务当前不可用，请稍后重试。" if isinstance(exc, ProviderError) else "抱歉，处理过程中发生错误，未能完成任务。"
                result.meta["stop_reason"] = "typed_failure"
                break
        else:
            # for-else：步数用尽仍未产出最终答案。
            result.outcome = BUDGET_STOPPED
            result.error_code = ErrorCode.BUDGET_EXCEEDED.value
            result.answer = "抱歉，本次推理超出了步数预算，未能给出完整回答。"
            result.meta["stop_reason"] = "max_steps_exhausted"

        # 归因校验：只有拿到证据的运行才需要校验。
        self._finalize_grounding(question, result)

        result.meta["failure_taxonomy"] = self.orchestrator.failure_taxonomy()
        result.meta["trace_durations_ms"] = trace.kind_durations()
        result.meta["total_duration_ms"] = trace.total_duration_ms()
        self._current_trace = None
        return result

    # —— 单次工具调用 ——

    def _execute(
        self,
        call: ToolCall,
        *,
        call_signatures: Counter[str],
        messages: list[ChatMessage],
        deadline: float,
    ) -> tuple[str, ObservedCall]:
        observed = ObservedCall(name=call.name, arguments=dict(call.arguments))
        signature = f"{call.name}:{sorted(call.arguments.items())}"

        if time.monotonic() > deadline:
            observed.error_code = ErrorCode.BUDGET_EXCEEDED.value
            observed.content_preview = "deadline exceeded before tool execution"
            return "已达到时间预算，工具不再执行。请基于已有信息作答或说明无法回答。", observed

        if call_signatures[signature] >= self.budget.max_repeated_tool_calls:
            observed.ok = False
            observed.repeated = True
            observed.error_code = ErrorCode.TOOL_INVALID_ARGUMENTS.value
            observed.content_preview = "repeated call suppressed"
            return (
                f"检测到重复调用：{call.name} 使用完全相同的参数已经执行过"
                f"{call_signatures[signature]} 次。请换一个参数或换一个工具；"
                "如果信息已经足够，请直接作答。",
                observed,
            )

        if sum(call_signatures.values()) >= self.budget.max_tool_calls:
            observed.error_code = ErrorCode.BUDGET_EXCEEDED.value
            observed.content_preview = "tool budget exhausted"
            return (
                f"工具调用总数已达上限（{self.budget.max_tool_calls} 次），"
                "请基于现有信息直接作答。",
                observed,
            )

        call_signatures[signature] += 1

        with (self._current_trace.step("tool", StepKind.TOOL, name=call.name, arguments=call.arguments)
              if self._current_trace
              else _nullcontext()) as step:
            outcome: ToolResult = self.registry.call(call.name, call.arguments)
            if step is not None:
                step.outputs = {
                    "ok": outcome.ok,
                    "error_code": outcome.error_code or None,
                    "chars": len(outcome.content),
                }

        observed.ok = outcome.ok
        observed.error_code = outcome.error_code
        observed.content_preview = outcome.content

        if outcome.ok:
            return outcome.to_observation(), observed

        # 失败：交给自愈编排决定是回灌纠偏还是停止。
        signal = self.orchestrator.classify(
            ScoutError(
                outcome.content,
                code=ErrorCode(outcome.error_code) if outcome.error_code else ErrorCode.TOOL_EXECUTION_FAILED,
                retryable=outcome.retryable,
                details=dict(outcome.data),
            ),
            stage="tool",
        )
        decision = self.orchestrator.decide(signal)
        if decision.action is RecoveryAction.STOP:
            self.orchestrator.mark_failed(signal, decision)
            return (
                f"{outcome.to_observation()}\n[系统] 该失败不可恢复（{decision.reason}）。"
                "请基于已有信息作答，或如实说明无法完成。",
                observed,
            )
        self.orchestrator.consume(signal, decision)
        self.orchestrator.mark_recovered(signal, decision)
        if self._current_trace is not None:
            with self._current_trace.step("recover", StepKind.RECOVER, action=decision.action.value) as step:
                step.outputs = decision.to_dict()
        return f"{outcome.to_observation()}\n[恢复提示] {decision.hint}", observed

    # —— 提示与收尾 ——

    def _compose_system(self, question: str) -> str:
        parts = [self.system_prompt]
        if self.memory is not None:
            excerpt = self.memory.recall_for_prompt(question)
            if excerpt:
                parts.append(excerpt)
        return "\n".join(parts)

    def _finalize_grounding(self, question: str, result: AgentRunResult) -> None:
        evidence = self._collect_evidence()
        if not evidence:
            # 没有证据的运行不做接地校验：可能是纯计算/日期类问题。
            result.grounding.verdict = Verdict.PASS
            result.grounding.reason = "no_evidence_required"
            return
        units, _dropped = pack_evidence(
            evidence, budget_chars=self.settings.retrieval.evidence_budget_chars
        )
        weights = self._corpus_weights()
        report = verify_answer(
            question,
            result.answer,
            units,
            settings=self.settings.verify,
            document_frequency=weights.get("document_frequency"),
            corpus_size=int(weights.get("corpus_size") or 0),
        )
        if report.verdict is Verdict.ABSTAIN and result.outcome == ANSWERED:
            # 证据不足却给了自信答案——这是最需要被拦下的情况。
            result.outcome = "insufficient_evidence"
            result.error_code = ErrorCode.INSUFFICIENT_EVIDENCE.value
        result.grounding = report

    def _corpus_weights(self) -> dict[str, Any]:
        """从知识检索工具取回语料级稀有度统计（充分性判定需要）。"""

        tool = self.registry.get("knowledge_search")
        if tool is None:
            return {}
        pipeline = getattr(tool.handler, "pipeline", None)
        if pipeline is None or not hasattr(pipeline, "idf_context"):
            return {}
        return pipeline.idf_context()

    def _collect_evidence(self):
        """从知识检索工具里取回累积的证据。"""

        tool = self.registry.get("knowledge_search")
        if tool is None:
            return []
        state = getattr(tool.handler, "state", None)
        if state is None:
            return []
        return state.dedupe_units()

    def evidence_preview(self, limit: int = 3) -> str:
        """给调试与测试用的证据摘要。"""

        units = self._collect_evidence()
        if not units:
            return ""
        return format_evidence(units[:limit])


class _nullcontext:
    """占位上下文：没有 trace 时不记录，但保持代码结构一致。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


__all__ = [
    "ANSWERED",
    "BUDGET_STOPPED",
    "NO_ANSWER",
    "PROVIDER_TIMEOUT",
    "AgentRunResult",
    "AgentStep",
    "ObservedCall",
    "ToolAgent",
]
