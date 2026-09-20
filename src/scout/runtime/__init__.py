"""运行时层：模型怎么调、状态放哪、钱怎么花、流怎么吐。

**这一层与检索层、数据层的边界。**
数据层负责"语料是什么样"，检索层负责"怎么找到证据"，
运行时层负责"调用怎么发生、状态怎么活下来、成本与延迟怎么守住"。

这一层的三条硬约束（也是它存在的理由）：

1. **策略与业务分离**：模型路由、语义缓存、预算守门都是**包装器**，
   对上完全同形（都实现 ``complete(request)``）。
   业务代码不需要知道今天有没有开路由——这让策略可以随时开关、做对照。
2. **决策必须留痕**：路由决策、缓存命中、预算冻结都要有结构化记录。
   没有留痕的优化无法证明自己有效，也无法在出问题时被摘除。
3. **降级要显式且清晰**：没有向量器就退化为精确匹配（而不是假装语义命中）、
   没有流式能力就一次性返回（而不是报错）。
"""

from __future__ import annotations

from .cache import CacheEntry, CacheStats, CachedLLM, SemanticCache, request_scope
from .intent import IntentFunnel, IntentResult, IntentTier
from .router import (
    HEAVY_TASKS,
    LIGHT_TASKS,
    CostLedger,
    ModelRouter,
    RoutedLLM,
    RouteDecision,
)
from .stream import (
    StreamOutcome,
    collect_stream,
    estimate_tokens,
    iter_tokens,
    sse_event,
    supports_streaming,
)

__all__ = [
    "CacheEntry",
    "CacheStats",
    "CachedLLM",
    "CostLedger",
    "HEAVY_TASKS",
    "IntentFunnel",
    "IntentResult",
    "IntentTier",
    "LIGHT_TASKS",
    "ModelRouter",
    "RouteDecision",
    "RoutedLLM",
    "SemanticCache",
    "StreamOutcome",
    "collect_stream",
    "estimate_tokens",
    "iter_tokens",
    "request_scope",
    "sse_event",
    "supports_streaming",
]
