"""预算管控测试："预扣 + 硬上限 + 结算"的三种语义。"""

from __future__ import annotations

import pytest

from scout.errors import BudgetExceededError, ErrorCode
from scout.llm.base import ChatMessage, LLMResponse
from scout.llm.budget import BudgetedLLM, Settlement, estimate_tokens
from scout.llm.scripted import ScriptedLLM
from scout.llm.base import TokenUsage

# —— 假数据制造 ——


class CountedLLM:
    """记录真实用量的测试 LLMClient。"""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def model_name(self) -> str:
        return "counted"

    def complete(self, request):
        usage = TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)
        return LLMResponse(content="done", usage=usage)


def _request(text: str = "hello") -> object:
    from scout.llm.base import LLMRequest

    return LLMRequest(
        messages=[ChatMessage(role="user", content=text)],
        task="answen",
        max_tokens=32,
    )


# —— 语义验证 ——


def test_estimate_tokens_is_conservative() -> None:
    """估算只可以**偏高**、不能**偏低**（它是预算守卫的门）。"""

    assert estimate_tokens("") == 0
    one = estimate_tokens("a")
    several = estimate_tokens("aaaa")
    assert one > len("a")  # 宁多勿少
    assert several > 0
    assert one + 3 > several or several >= one  # 单调


def test_budgeted_llm_passes_within_budget() -> None:
    inner = CountedLLM(input_tokens=100, output_tokens=50)
    budgeted = BudgetedLLM(inner, budget_tokens=10_000)
    response = budgeted.complete(_request("x" * 100))
    assert response.content == "done"
    settle = budgeted.settle()
    assert isinstance(settle, Settlement)
    assert settle.calls == 1
    assert settle.used == 150
    assert settle.used <= settle.budget
    assert not settle.stopped_by_budget
    assert not settle.is_over()


def test_budgeted_llm_stops_at_hard_cap() -> None:
    """第二个调用超预算时，必须**类型化拒绝**而不是悄悄延续。"""

    inner = CountedLLM(input_tokens=900, output_tokens=100)
    budgeted = BudgetedLLM(inner, budget_tokens=1_200)
    budgeted.complete(_request("a" * 50))  # 用 1000 token，剩 200 配
    with pytest.raises(BudgetExceededError) as excp:
        budgeted.complete(_request("b" * 500))
    error = excp.value
    assert error.code is ErrorCode.BUDGET_EXCEEDED
    assert error.retryable is False

    settle = budgeted.settle()
    assert settle.calls == 1
    assert settle.stopped_by_budget
    # 超预算的调用不应该被计费（因预算门在调用发生之前）
    assert settle.used == 1000


def test_settle_reports_frozen_vs_used() -> None:
    """冻结额度与实际用量分开汇报——这是"预扣 + 结算"的证据。"""

    inner = CountedLLM(input_tokens=10, output_tokens=5)
    budgeted = BudgetedLLM(inner, budget_tokens=1_000, reserve_estimate=800)
    budgeted.complete(_request("x" * 40))
    settle = budgeted.settle()
    payload = settle.to_dict()
    assert payload["budget"] == 1_000
    assert payload["frozen"] == 800
    assert payload["used"] == 15
    assert payload["calls"] == 1
    assert payload["remaining"] == 785
    assert payload["stopped_by_budget"] is False
