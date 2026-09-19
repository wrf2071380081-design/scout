"""LLM 调用预算管控：预扣 + 硬上限 + 结算。

**为什么它不该只是"事后对账"。** 预算管控常被写成"跑完看用了多少"。
但那样做只能告诉你"任务已经透支了"——钱已经花出去了。
真正的预算门，应该让任务**在预算耗尽的那一刻被截停**，
也就是每次真实调用发生之前先查、不足就类型化拒绝。

**并发下的两个致命坑（本模块语义性的打好基础）**：
1. **超扣 / Over-spend**：多个任务都先"读余额 → 判断 → 扣减"，
   在非原子的情况下并发执行，系统量消耗大于总 budget。
2. **占用不释放**：预算先扣下但任务失败，冻结的额度长期占着不放。
   这就是为什么"预扣 + 结算"两段流程缺一不可：
   冻结优先保证不超扣，结算保证失败时不漏费不漏额。

设计与这个工程内其它部分统一的错误语义：
超支抛出 :class:`scout.errors.BudgetExceededError`，
与 fallback 上层（Agent orchestrator 的自愈编排）联动，
之后的降规格就不靠 if 判断、靠类型化信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..errors import ErrorCode
from ..errors import BudgetExceededError
from .base import LLMClient, LLMRequest, LLMResponse


DEFAULT_MAX_OUT = 1500


def estimate_tokens(text: str) -> int:
    """一个 conservative 的中英混合 token 估算：每 1.6 字符算 1 token。

    这个估算在预算守卫里的角色是"**不低估**"——用于做超支保守检查。
    多估一点、少估一点是想被骗过。
    """

    if not text:
        return 0
    return max(1, int(len(text) / 1.6)) + 8


@dataclass(slots=True)
class Settlement:
    """预算结算结果。

    构建后的任何数字都能被去溯源：
    - frozen（pre-reserve）与 actual（真实 usage）分开;
    - remaining = frozen - actual 表示还剩多少冻结额度可供后续阶段用。
    """

    budget: int
    frozen: int
    used: int
    calls: int
    stopped_by_budget: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.frozen - self.used)

    def is_over(self) -> bool:
        return self.used > self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "frozen": self.frozen,
            "used": self.used,
            "calls": self.calls,
            "remaining": self.remaining,
            "stopped_by_budget": self.stopped_by_budget,
            "over_budget": self.is_over(),
        }


class BudgetedLLM:
    """包装一个 LLMClient，把"超支就停"打造成类型化的决策。

    用法：
        budgeted = BudgetedLLM(real_client, budget_tokens=30_000, reserve_estimate=20_000)
        response = budgeted.complete(request)          # 余额不足时会在这里
                                                        # 被类型化拒绝，根本不发电请求
        settlement = budgeted.settle()                 # 任务结束对账
    """

    def __init__(self, inner: LLMClient, budget_tokens: int, *, reserve_estimate: int | None = None) -> None:
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        self.inner = inner
        self.budget = budget_tokens
        # 预扣额度：默认为总预算（比较保守地冻结整条预算），真正的用量 running 在 settle 时结算。
        # 并发场景下这里应该走 Redis 原子预扣；单进程环境下直接冻结即可。
        self._frozen: int = min(budget_tokens, reserve_estimate if reserve_estimate is not None else budget_tokens)
        self._used = 0
        self._calls = 0
        self._stopped = False

    # —— 型代客眼 ——

    @property
    def model_name(self) -> str:
        return getattr(self.inner, "model_name", "unknown")

    # —— 估计一个请求会烧多少 token ——

    def _estimate(self, request: LLMRequest) -> int:
        """用一个保守估计估这次调用会烧掉多少 token。

        之所以用保守估计而不是精确值，是因为**预算守卫宁可少允许一次、
        也不能多放行一次**；它的角色是"守门员"，不是"计费员"。
        """

        prompt_chars = sum(len(message.content) for message in request.messages)
        estimate_in = max(1, int(prompt_chars / 1.6)) + 8
        if request.max_tokens and request.max_tokens > 0:
            estimate_out = min(int(request.max_tokens), DEFAULT_MAX_OUT)
        else:
            estimate_out = DEFAULT_MAX_OUT
        return estimate_in + estimate_out

    def complete(self, request: LLMRequest) -> LLMResponse:
        estimate = self._estimate(request)
        if self._used + estimate > self.budget:
            self._stopped = True
            raise BudgetExceededError(
                f"预算耗尽：已用 {self._used} tokens，本调用预估 {estimate} tokens，"
                f"上限 {self.budget} tokens。任务应在预算内收敛，或显式扩充预算。",
                code=ErrorCode.BUDGET_EXCEEDED,
                retryable=False,
            )

        response = self.inner.complete(request)
        self._calls += 1
        if response.usage is not None:
            self._used += int(response.usage.total)
        else:  # 模型没返回用量时，用保守估计兜底——宁可多估也不要漏费
            self._used += estimate
        return response

    # —— 结算 ——

    def settle(self) -> Settlement:
        """任务结束：按实际用量结算。

        工程语义：**冻结超额和实际用量必须分开存**——
        冻结是为了提前保障不超扣，结算是为了长期准确计线；
        混在一起会掩盖"是不是多点预算就能多跑一小步"的决策信息。
        """

        return Settlement(
            budget=self.budget,
            frozen=self._frozen,
            used=self._used,
            calls=self._calls,
            stopped_by_budget=self._stopped,
        )


__all__ = ["BudgetedLLM", "Settlement", "estimate_tokens"]
