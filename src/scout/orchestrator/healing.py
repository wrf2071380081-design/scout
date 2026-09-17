"""失败分类与自愈编排。

**为什么需要这一层。**

Agent 的失败很少是"模型不会"，而大多是**运行时状态问题**：
网络超时、工具参数写错、上下文过期、证据互相冲突、重试死循环。
这些问题的共同点是——它们**可以被识别，也可以被对策**，
但如果没有一层专门负责这件事，它们就会以"答案不对"的形式暴露给用户，
而运维在监控上什么也看不到。

本模块做三件事：

1. **分类**（:meth:`SelfHealingOrchestrator.classify`）：把任意异常归一成
   :class:`FailureSignal`，带稳定 code、阶段、是否可重试。
2. **决策**（:meth:`SelfHealingOrchestrator.decide`）：由失败类型推导恢复动作。
   策略是显式表，不是散落在各处的 if-else——这样"系统会怎么恢复"是可审计的。
3. **抑制**（:meth:`SelfHealingOrchestrator._fingerprint`）：同一个失败指纹
   在预算内只允许恢复有限次。**这是防止死循环的关键**：没有它，
   "重试一次"会在嵌套调用里被放大成几十次外部请求。

一个重要的区分：``NO_KNOWLEDGE`` 与 ``INSUFFICIENT_EVIDENCE`` **不是故障**，
是合法的语义结果。它们同样会走到这里，但走的是"如何体面地告知用户"这条路，
而不是"如何恢复"。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..errors import ErrorCode, FailureRecord, ProviderError, ScoutError


class RecoveryAction(str, Enum):
    """恢复动作。"""

    RETRY = "retry"
    """同参数重试。仅对可重试的瞬时故障，且受重试预算约束。"""

    REPROMPT = "reprompt"
    """把失败原因回灌给模型，让它修正参数或换一个工具。"""

    FALLBACK = "fallback"
    """切换到降级通道（如混合检索 → 仅稠密）。"""

    REWRITE = "rewrite"
    """改写查询后重新检索。"""

    CLARIFY = "clarify"
    """向用户澄清，而不是继续猜。"""

    ABSTAIN = "abstain"
    """明确拒答。这是**正确行为**，不是失败。"""

    STOP = "stop"
    """放弃并抛出 typed 错误，让调用方感知故障。"""


# 失败类型 → 优先恢复动作。表是显式的，便于审查与测试。
_POLICY: dict[ErrorCode, tuple[RecoveryAction, ...]] = {
    ErrorCode.PROVIDER_TIMEOUT: (RecoveryAction.RETRY, RecoveryAction.FALLBACK, RecoveryAction.STOP),
    ErrorCode.PROVIDER_CONNECTION: (RecoveryAction.RETRY, RecoveryAction.FALLBACK, RecoveryAction.STOP),
    ErrorCode.PROVIDER_UNAVAILABLE: (RecoveryAction.RETRY, RecoveryAction.FALLBACK, RecoveryAction.STOP),
    ErrorCode.PROVIDER_RATE_LIMITED: (RecoveryAction.RETRY, RecoveryAction.FALLBACK, RecoveryAction.STOP),
    ErrorCode.PROVIDER_INVALID_RESPONSE: (RecoveryAction.REPROMPT, RecoveryAction.RETRY, RecoveryAction.STOP),
    ErrorCode.TOOL_INVALID_ARGUMENTS: (RecoveryAction.REPROMPT, RecoveryAction.STOP),
    ErrorCode.TOOL_NOT_FOUND: (RecoveryAction.REPROMPT, RecoveryAction.STOP),
    ErrorCode.TOOL_EXECUTION_FAILED: (RecoveryAction.REPROMPT, RecoveryAction.STOP),
    ErrorCode.BUDGET_EXCEEDED: (RecoveryAction.STOP,),
    ErrorCode.VALIDATION_FAILED: (RecoveryAction.STOP,),
    ErrorCode.INDEX_NOT_READY: (RecoveryAction.STOP, RecoveryAction.FALLBACK),
    ErrorCode.NO_KNOWLEDGE: (RecoveryAction.ABSTAIN,),
    ErrorCode.INSUFFICIENT_EVIDENCE: (RecoveryAction.REWRITE, RecoveryAction.CLARIFY, RecoveryAction.ABSTAIN),
}

# 语义结果：不是故障，不参与"故障率"统计。
SEMANTIC_CODES = frozenset({ErrorCode.NO_KNOWLEDGE, ErrorCode.INSUFFICIENT_EVIDENCE})


@dataclass(slots=True)
class FailureSignal:
    """归一化后的失败信号。"""

    code: ErrorCode
    stage: str
    retryable: bool
    attempt: int = 1
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_semantic(self) -> bool:
        return self.code in SEMANTIC_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "message": self.message[:200],
            "semantic": self.is_semantic,
        }


@dataclass(slots=True)
class RecoveryDecision:
    """恢复决策。"""

    action: RecoveryAction
    reason: str
    budget_cost: int = 1
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "budget_cost": self.budget_cost,
            "hint": self.hint,
        }


class SelfHealingOrchestrator:
    """失败分类与恢复决策。

    :param max_recoveries: 单次运行允许的恢复总次数
    :param max_retries_per_fingerprint: 同一失败指纹允许的恢复次数上限
    """

    def __init__(
        self,
        *,
        max_recoveries: int = 3,
        max_retries_per_fingerprint: int = 1,
        stages_with_fallback: tuple[str, ...] = ("retrieve", "rerank"),
    ) -> None:
        self.max_recoveries = max_recoveries
        self.max_retries_per_fingerprint = max_retries_per_fingerprint
        self.stages_with_fallback = stages_with_fallback
        self._fingerprints: Counter[str] = Counter()
        self._records: list[FailureRecord] = []
        self.recoveries_used = 0

    # —— 分类 ——

    def classify(self, error: BaseException | ScoutError, *, stage: str) -> FailureSignal:
        """把异常归一成 :class:`FailureSignal`。"""

        if isinstance(error, ScoutError):
            return FailureSignal(
                code=error.code,
                stage=stage,
                retryable=error.retryable,
                attempt=int(error.details.get("attempts", 1)) if isinstance(error, ProviderError) else 1,
                message=error.message,
                details=dict(error.details),
            )
        # 未预期的异常一律视为不可重试的执行失败——比起猜测，宁可快速失败。
        return FailureSignal(
            code=ErrorCode.TOOL_EXECUTION_FAILED,
            stage=stage,
            retryable=False,
            message=f"{type(error).__name__}: {error}",
            details={"unexpected": True},
        )

    # —— 决策 ——

    @staticmethod
    def _fingerprint(signal: FailureSignal) -> str:
        return f"{signal.stage}:{signal.code.value}:{signal.details.get('tool', '')}"

    def decide(self, signal: FailureSignal) -> RecoveryDecision:
        """由失败类型推导恢复动作。"""

        fingerprint = self._fingerprint(signal)
        attempts = self._fingerprints[fingerprint]

        if signal.is_semantic:
            # 语义结果不是故障，直接按策略走"体面告知"。
            action = _POLICY[signal.code][0]
            return RecoveryDecision(
                action=action,
                reason=f"semantic_result:{signal.code.value}",
                budget_cost=0,
                hint="这是合法结果，不要当作故障上报。",
            )

        if self.recoveries_used >= self.max_recoveries:
            self._record(signal, RecoveryAction.STOP, recovered=False)
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="recovery_budget_exhausted",
                budget_cost=0,
                hint=f"总恢复预算 {self.max_recoveries} 已用尽。",
            )

        if attempts >= self.max_retries_per_fingerprint:
            self._record(signal, RecoveryAction.STOP, recovered=False)
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason="fingerprint_retry_exhausted",
                budget_cost=0,
                hint=f"同一失败（{fingerprint}）已恢复 {attempts} 次，停止重试以避免死循环。",
            )

        candidates = _POLICY.get(signal.code, (RecoveryAction.STOP,))
        for action in candidates:
            if action is RecoveryAction.RETRY and not signal.retryable:
                continue
            if action is RecoveryAction.FALLBACK and signal.stage not in self.stages_with_fallback:
                continue
            hint = self._hint_for(action, signal)
            return RecoveryDecision(
                action=action,
                reason=f"policy_for:{signal.code.value}",
                hint=hint,
            )

        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason="no_applicable_policy",
            budget_cost=0,
        )

    @staticmethod
    def _hint_for(action: RecoveryAction, signal: FailureSignal) -> str:
        if action is RecoveryAction.REPROMPT:
            return (
                "把上面的错误原因原样告诉模型，要求它修正参数或改用其他工具；"
                "不要重复提交完全相同的调用。"
            )
        if action is RecoveryAction.FALLBACK:
            return "切换到降级通道继续，并在结果中标记已降级。"
        if action is RecoveryAction.REWRITE:
            return "用查询缺陷诊断改写查询后重试一次；仍不足则向用户澄清。"
        if action is RecoveryAction.CLARIFY:
            return "向用户说明缺少哪些信息，请求补充。"
        if action is RecoveryAction.ABSTAIN:
            return "明确告知无法回答，不要编造。"
        if action is RecoveryAction.RETRY:
            return f"第 {signal.attempt} 次尝试失败，可再试一次。"
        return ""

    # —— 记录与统计 ——

    def consume(self, signal: FailureSignal, decision: RecoveryDecision) -> None:
        """消耗恢复预算。调用方在执行恢复动作前调用。"""

        self.recoveries_used += decision.budget_cost
        if decision.action is not RecoveryAction.STOP:
            self._fingerprints[self._fingerprint(signal)] += 1

    def _record(self, signal: FailureSignal, action: RecoveryAction, *, recovered: bool) -> None:
        self._records.append(
            FailureRecord(
                code=signal.code,
                stage=signal.stage,
                retryable=signal.retryable,
                recovered=recovered,
                recovery_action=action.value,
                details=signal.details,
            )
        )

    def mark_recovered(self, signal: FailureSignal, decision: RecoveryDecision) -> None:
        self._record(signal, decision.action, recovered=True)

    def mark_failed(self, signal: FailureSignal, decision: RecoveryDecision) -> None:
        self._record(signal, decision.action, recovered=False)

    def failure_taxonomy(self) -> dict[str, Any]:
        """失败分类学统计。这是评测报告里的一等公民。"""

        by_code: Counter[str] = Counter()
        by_stage: Counter[str] = Counter()
        by_action: Counter[str] = Counter()
        recovered = 0
        for record in self._records:
            by_code[record.code.value] += 1
            by_stage[record.stage] += 1
            by_action[record.recovery_action] += 1
            if record.recovered:
                recovered += 1
        total = len(self._records)
        return {
            "failures_total": total,
            "failures_by_code": dict(sorted(by_code.items())),
            "failures_by_stage": dict(sorted(by_stage.items())),
            "failures_by_action": dict(sorted(by_action.items())),
            "failures_recovered": recovered,
            "recovery_success_rate": round(recovered / total, 4) if total else 0.0,
            "recoveries_used": self.recoveries_used,
        }

    def reset(self) -> None:
        """开始新的一次运行前清空状态。"""

        self._fingerprints.clear()
        self._records.clear()
        self.recoveries_used = 0


__all__ = [
    "SEMANTIC_CODES",
    "FailureSignal",
    "RecoveryAction",
    "RecoveryDecision",
    "SelfHealingOrchestrator",
]
