"""模型路由与成本核算：让"每一步该用哪个模型"成为一个可验证的决定。

**为什么需要路由。**
Agent 轨迹里大量的调用是低难度的：复杂度判定、证据打分、意图分类、查询改写。
它们的特点是**输出结构固定、判断局部**，用旗舰模型做是纯浪费；
而真正需要推理的只有"规划"与"最终生成"两步。
把这两类分开走不同模型，是 Agent 成本里最大的一根杠杆。

**但路由有一个必须守住的纪律：路由必须是显式且可观测的。**
"偷偷换小模型"会让问题变得不可解释——答案变差时，
你不知道是提示词差了、检索差了，还是路由把小模型用在了不该用的地方。
所以 :class:`ModelRouter` 做三件事：

1. 按任务名（白名单）与提示词长度做**确定性**判定，可复现；
2. 记录每一次路由决策到 `decisions`，进 trace；
3. 累计 :class:`CostLedger`，让"省了多少"变成数字，而不是感觉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..llm.base import LLMClient, LLMRequest

# 默认认为"轻"的任务：结构化判定类，输出短、无需跨步推理
LIGHT_TASKS = ("grade", "intent", "rewrite", "subquestions", "decide", "sanitize", "complexity")
# 默认认为"重"的任务：需要长上下文整合与推理
HEAVY_TASKS = ("answer", "synthesize", "plan", "report")


@dataclass(slots=True)
class RouteDecision:
    task: str
    model: str
    tier: str
    reason: str
    prompt_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "tier": self.tier,
            "reason": self.reason,
            "prompt_chars": self.prompt_chars,
        }


@dataclass(slots=True)
class CostLedger:
    """成本账本：**按模型分账**，而不是只记一个总数。

    只记总数看不出"路由有没有生效"——总数下降可能只是因为今天流量小。
    分账之后，"轻任务走了快模型"这件事才有证据。
    """

    calls: dict[str, int] = field(default_factory=dict)
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)
    routed_light: int = 0

    def record(self, model: str, input_tokens: int, output_tokens: int, *, light: bool) -> None:
        self.calls[model] = self.calls.get(model, 0) + 1
        self.input_tokens[model] = self.input_tokens.get(model, 0) + int(input_tokens)
        self.output_tokens[model] = self.output_tokens.get(model, 0) + int(output_tokens)
        if light:
            self.routed_light += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "input_tokens": dict(self.input_tokens),
            "output_tokens": dict(self.output_tokens),
            "routed_light": self.routed_light,
            "total_calls": sum(self.calls.values()),
        }


class ModelRouter:
    """双档路由：``fast``（轻）与 ``strong``（重）。

    :param light_tasks: 走 fast 的任务白名单。**白名单而不是黑名单**：
        新出现的任务名默认走强模型——路由错了多花钱，比路由错了降质量更容易接受。
    :param heavy_token_limit: 提示词超过这个长度就走强模型。
        长提示词通常意味着长上下文整合，小模型在这种输入上掉分明显。
    """

    def __init__(
        self,
        fast: LLMClient,
        strong: LLMClient,
        *,
        light_tasks: Sequence[str] = LIGHT_TASKS,
        heavy_token_limit: int = 6000,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.fast = fast
        self.strong = strong
        self.light_tasks = set(light_tasks)
        self.heavy_token_limit = heavy_token_limit
        # 任务级强制指定：给"某类任务必须走强模型"留一个显式的、可审计的开关
        self.overrides = dict(overrides or {})
        self.decisions: list[RouteDecision] = []
        self.ledger = CostLedger()
        self._active: RouteDecision | None = None

    def decide(self, request: LLMRequest) -> RouteDecision:
        prompt_chars = sum(len(message.content) for message in request.messages)
        task = request.task or "unknown"
        forced = self.overrides.get(task)
        if forced == "strong":
            return RouteDecision(task, self.strong.model_name, "strong", "override", prompt_chars)
        if forced == "fast":
            return RouteDecision(task, self.fast.model_name, "fast", "override", prompt_chars)
        if task in self.light_tasks and prompt_chars <= self.heavy_token_limit:
            return RouteDecision(task, self.fast.model_name, "fast", "light_task", prompt_chars)
        if task in HEAVY_TASKS:
            return RouteDecision(task, self.strong.model_name, "strong", "heavy_task", prompt_chars)
        if prompt_chars > self.heavy_token_limit:
            return RouteDecision(task, self.strong.model_name, "strong", "long_prompt", prompt_chars)
        return RouteDecision(task, self.strong.model_name, "strong", "default_strong", prompt_chars)

    def pick(self, request: LLMRequest) -> LLMClient:
        decision = self.decide(request)
        self._active = decision
        self.decisions.append(decision)
        return self.fast if decision.tier == "fast" else self.strong


class RoutedLLM:
    """把路由能力包装成一个 :class:`LLMClient`，可直接替换现有客户端。

    这样"上路由"对上层是零改动的——这是刻意的：路由属于运行时策略，
    不应该渗透到业务流程代码里。
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    @property
    def model_name(self) -> str:
        return f"router({self.router.fast.model_name}|{self.router.strong.model_name})"

    def complete(self, request: LLMRequest):
        decision = self.router.decide(request)
        self.router.decisions.append(decision)
        client = self.router.fast if decision.tier == "fast" else self.router.strong
        response = client.complete(request)
        usage = getattr(response, "usage", None)
        self.router.ledger.record(
            decision.model,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            light=decision.tier == "fast",
        )
        return response

    def report(self) -> dict[str, Any]:
        return {
            "ledger": self.router.ledger.to_dict(),
            "decisions": [decision.to_dict() for decision in self.router.decisions[-50:]],
            "light_ratio": (
                self.router.ledger.routed_light / len(self.router.decisions)
                if self.router.decisions
                else 0.0
            ),
        }


__all__ = [
    "CostLedger",
    "HEAVY_TASKS",
    "LIGHT_TASKS",
    "ModelRouter",
    "RouteDecision",
    "RoutedLLM",
]
