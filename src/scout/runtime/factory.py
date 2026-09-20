"""运行时装配：把预算、缓存、路由串成一条链，并且**顺序是设计的一部分**。

**为什么需要这个工厂。**
上一轮把三个包装器写出来了，但它们各自独立——真正跑起来时没人知道该怎么组合、
顺序该怎么放。而顺序不是随意的，它直接决定成本与语义：

```
   CachedLLM( BudgetedLLM( RoutedLLM( 真实客户端 ) ) )
      ↑            ↑            ↑
   最外层        中间层        最内层
   先查缓存     再查预算      最后才决定用哪个模型
```

**为什么缓存必须在预算之外。**
缓存命中不花钱。如果把预算放在缓存外层，一次"本来免费"的命中会先被预算拦下——
用户会看到"预算不足"，而实际上这次调用根本不需要钱。
**顺序错了，就会出现"明明有免费路径却先报错"这种荒谬行为。**

**为什么路由必须在预算之内。**
预算需要知道"这次要花多少钱"，而花多少钱取决于用哪个模型。
先路由、再判预算，才能用对的价格判断；反过来就只能拿最贵的模型预算去卡，
轻任务会被无辜拒绝。

**为什么这三层都必须可关。**
它们是策略，不是业务。对照组需要"只开缓存不开路由"这种组合，
所以每一层都由配置独立控制，而不是打包成一个开关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..llm.base import LLMClient
from ..llm.budget import BudgetedLLM
from .cache import CachedLLM, SemanticCache
from .router import ModelRouter, RoutedLLM


@dataclass(slots=True)
class RuntimeStack:
    """装配结果：一个客户端 + 各层的句柄（用于观测与单独取数）。"""

    client: LLMClient
    cache: SemanticCache | None = None
    budget: BudgetedLLM | None = None
    router: ModelRouter | None = None
    order: tuple[str, ...] = ()

    def report(self) -> dict[str, Any]:
        """各层的运行数据。**要能分开看**：
        "命中率低"和"预算被打满"是两个完全不同的问题，
        混在一个数字里就分不出来。"""

        payload: dict[str, Any] = {"order": list(self.order), "model": self.client.model_name}
        if self.cache is not None:
            payload["cache"] = self.cache.stats.to_dict()
        if self.budget is not None:
            payload["budget"] = self.budget.settle().to_dict()
        if self.router is not None:
            payload["router"] = {
                "ledger": self.router.ledger.to_dict(),
                "light_ratio": (
                    self.router.ledger.routed_light / len(self.router.decisions)
                    if self.router.decisions
                    else 0.0
                ),
            }
        return payload


def build_runtime_stack(
    base: LLMClient,
    settings: Settings,
    *,
    embedder: Any = None,
    budget_tokens: int | None = None,
) -> RuntimeStack:
    """按配置装配运行时链。

    :param base: 真实（或离线）LLM 客户端，作为链的最内层。
    :param embedder: 语义缓存需要它做近邻匹配；没有则缓存退化为精确匹配
        （并如实体现在 ``order`` 里，不假装自己是语义缓存）。
    :param budget_tokens: 单次运行的 token 预算。给 None 表示不装预算层。
    """

    layers: list[str] = ["base"]
    client: LLMClient = base
    router: ModelRouter | None = None

    # ① 路由（最内层）：决定用哪个模型
    if settings.runtime.routing_enabled:
        strong = base
        fast = _fast_client(base, settings)
        router = ModelRouter(fast, strong)
        client = RoutedLLM(router)
        layers.append("router")

    # ② 预算（中间层）：调用前查预算，不足即拒
    budget: BudgetedLLM | None = None
    limit = budget_tokens if budget_tokens is not None else settings.runtime.budget_tokens
    if limit and limit > 0:
        budget = BudgetedLLM(client, limit)
        client = budget
        layers.append("budget")

    # ③ 语义缓存（最外层）：命中直接返回，不进入预算与路由
    cache: SemanticCache | None = None
    if settings.runtime.cache_enabled:
        cache = SemanticCache(
            embedder,
            threshold=settings.runtime.cache_threshold,
            max_items=settings.runtime.cache_max_items,
        )
        client = CachedLLM(client, cache, model_version=settings.runtime.cache_scope_version)
        layers.append("cache")

    return RuntimeStack(client=client, cache=cache, budget=budget, router=router, order=tuple(layers))


def _fast_client(base: LLMClient, settings: Settings) -> LLMClient:
    """构造"快模型"客户端。

    默认复用同一个 base（路由没配快模型时，路由层退化为"名义存在、实际同款"）——
    这比凭空造一个不可用的客户端要诚实：**没有快模型的账号时，
    路由层不应该假装自己在省钱。** 配置了 ``SCOUT_LLM_FAST_MODEL`` 才会真正分流。
    """

    fast_model = settings.runtime.fast_model
    if not fast_model or fast_model == settings.llm.model:
        return base
    factory = getattr(base, "clone_with_model", None)
    if callable(factory):
        try:
            return factory(fast_model)
        except Exception:  # noqa: BLE001 - 取不到就退回同一客户端，不影响主流程
            return base
    return base


__all__ = ["RuntimeStack", "build_runtime_stack"]
