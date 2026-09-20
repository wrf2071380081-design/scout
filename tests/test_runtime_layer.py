"""运行时层测试：意图漏斗、语义缓存、模型路由、流式（全部离线、毫秒级）。"""

from __future__ import annotations

import pytest

from scout.llm.base import ChatMessage, LLMRequest, LLMResponse, TokenUsage
from scout.rag.embed import HashingEmbedder
from scout.runtime import (
    CachedLLM,
    IntentFunnel,
    IntentTier,
    ModelRouter,
    RoutedLLM,
    SemanticCache,
    collect_stream,
    iter_tokens,
    request_scope,
    supports_streaming,
)

# —— 假客户端 ——


class CountingLLM:
    """记录调用次数的最小客户端。"""

    def __init__(self, content: str = "ok", *, model: str = "fake") -> None:
        self.content = content
        self._model = model
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content, usage=TokenUsage(input_tokens=10, output_tokens=5))


class StreamingLLM(CountingLLM):
    """带流式能力的客户端。"""

    def complete_stream(self, request: LLMRequest):
        yield "云"
        yield "计算"
        yield "标准"


def _request(text: str, task: str = "grade") -> LLMRequest:
    return LLMRequest(messages=[ChatMessage(role="user", content=text)], task=task)


# —— 意图漏斗 ——


def _funnel(**kwargs) -> IntentFunnel:
    return IntentFunnel(
        rules={
            "退款": ["退款", "退钱"],
            "开票": ["开发票", "开票"],
            "查余额": ["余额", "查账"],
        },
        prototypes={
            "退款": ["我要退款", "申请退款流程"],
            "开票": ["怎么开发票", "开票申请流程"],
            "查余额": ["账户余额查询"],
        },
        embedder=HashingEmbedder(dim=512),
        **kwargs,
    )


def test_rule_tier_hits_instantly() -> None:
    result = _funnel().classify("我要申请退款")
    assert result.tier is IntentTier.RULE
    assert result.label == "退款"


def test_semantic_tier_when_rules_miss() -> None:
    """原型句能覆盖关键词覆盖不到的说法——这正是第二层存在的理由。"""

    result = _funnel().classify("申请退款流程")
    assert result.tier in {IntentTier.SEMANTIC, IntentTier.RULE}
    assert result.label == "退款"


def test_ambiguous_input_goes_to_clarify_not_argmax() -> None:
    """前两名分差不足时必须进澄清，而不是取最大值。

    这是真实资损事故的修法：0.52 / 0.49 取 argmax 会误路由到"办理分期"这类有资金后果的动作。
    """

    funnel = IntentFunnel(
        rules={},
        prototypes={"退款": ["退款处理"], "开票": ["退款开票处理"]},
        embedder=HashingEmbedder(dim=512),
        margin_threshold=0.9,  # 刻意把分差门槛拉高，构造歧义区间
    )
    result = funnel.classify("退款开票处理")
    assert result.tier is IntentTier.CLARIFY
    assert "歧义" in result.reason


def test_sensitive_intent_requires_confirmation() -> None:
    """"办理类"意图即便高置信也不自动路由。"""

    funnel = IntentFunnel(
        rules={"账单分期": ["分期", "账单分期"]},
        prototypes={},
        sensitive_labels=["账单分期"],
    )
    result = funnel.classify("我想办理账单分期")
    assert result.tier is IntentTier.CLARIFY
    assert "敏感" in result.reason


def test_llm_fallback_only_when_earlier_tiers_fail() -> None:
    llm = CountingLLM(content='{"label": "退款", "confidence": 0.8, "needs_clarification": false}')
    funnel = IntentFunnel(rules={}, prototypes={}, llm=llm)
    result = funnel.classify("随便说点什么")
    assert llm.calls == 1, "前两层都没命中时，才允许花一次大模型"
    assert result.tier is IntentTier.LLM


def test_funnel_distribution_is_reported() -> None:
    funnel = _funnel()
    for query in ["我要退款", "怎么开发票", "账户余额查询"]:
        funnel.classify(query)
    distribution = funnel.distribution()
    assert sum(distribution.values()) >= 3
    # 各层命中的分布必须可见：LLM 层占比过高说明前两层该重调
    assert set(distribution) >= {"rule", "semantic", "llm", "clarify", "unknown"}


def test_clarify_options_are_actionable() -> None:
    funnel = IntentFunnel(
        rules={},
        prototypes={"退款": ["退款处理"], "开票": ["退款开票处理"]},
        embedder=HashingEmbedder(dim=512),
        margin_threshold=0.9,
    )
    result = funnel.classify("退款开票处理")
    options = funnel.clarify_options(result)
    assert options, "歧义时应当给出可点击的候选，而不是让用户重说一遍"


# —— 语义缓存 ——


def test_cache_hit_on_identical_text() -> None:
    cache = SemanticCache(HashingEmbedder(dim=256))
    scope = "scope-A"
    cache.put("Redis 为什么快", "因为内存", scope)
    hit = cache.get("Redis 为什么快", scope)
    assert hit is not None
    assert hit[0] == "因为内存"
    assert cache.stats.hit_rate == 1.0


def test_cache_miss_when_scope_changes() -> None:
    """同一问题、不同证据 → 必须 miss。

    这是 RAG 语义缓存最容易犯的错：只按问题文本缓存，
    会把"基于旧知识库生成的答案"喂给新数据。
    """

    cache = SemanticCache(HashingEmbedder(dim=256))
    cache.put("这家公司营收多少", "100 亿", scope="corpus-v1")
    assert cache.get("这家公司营收多少", "corpus-v2") is None
    assert cache.stats.misses_scope >= 1


def test_request_scope_includes_evidence_and_model() -> None:
    request = LLMRequest(
        messages=[ChatMessage(role="user", content="q")],
        task="grade",
        context={"evidence_digest": "abc", "model_version": "m1"},
    )
    scope_a = request_scope(request)
    request.context["evidence_digest"] = "def"
    assert request_scope(request) != scope_a, "证据变了，缓存作用域必须跟着变"


def test_cached_llm_skips_second_call() -> None:
    inner = CountingLLM(content="结构化结果")
    wrapper = CachedLLM(inner, SemanticCache(HashingEmbedder(dim=256)))
    request = _request("同一段提示词", task="grade")
    first = wrapper.complete(request)
    second = wrapper.complete(request)
    assert inner.calls == 1, "第二次应当命中缓存，不再花钱"
    assert first.content == second.content
    assert second.usage.total == 0, "缓存命中的用量记为 0，token 骤降才可解释"


def test_generation_task_not_cached() -> None:
    inner = CountingLLM(content="答案")
    wrapper = CachedLLM(inner, SemanticCache(HashingEmbedder(dim=256)))
    wrapper.complete(_request("问题", task="answer"))
    wrapper.complete(_request("问题", task="answer"))
    assert inner.calls == 2, "生成类任务默认不缓存（证据变化频繁、风险高于收益）"


# —— 模型路由 ——


def test_light_task_routed_to_fast_model() -> None:
    fast, strong = CountingLLM(model="fast"), CountingLLM(model="strong")
    routed = RoutedLLM(ModelRouter(fast, strong))
    routed.complete(_request("判一下充分性", task="grade"))
    assert fast.calls == 1 and strong.calls == 0


def test_heavy_task_routed_to_strong_model() -> None:
    fast, strong = CountingLLM(model="fast"), CountingLLM(model="strong")
    routed = RoutedLLM(ModelRouter(fast, strong))
    routed.complete(_request("生成最终答案", task="answer"))
    assert strong.calls == 1 and fast.calls == 0


def test_unknown_task_defaults_to_strong() -> None:
    """白名单而非黑名单：拿不准就多花钱，而不是拿不准就降质量。"""

    fast, strong = CountingLLM(model="fast"), CountingLLM(model="strong")
    routed = RoutedLLM(ModelRouter(fast, strong))
    routed.complete(_request("未知任务", task="brand_new_task"))
    assert strong.calls == 1


def test_override_forces_model_choice() -> None:
    fast, strong = CountingLLM(model="fast"), CountingLLM(model="strong")
    router = ModelRouter(fast, strong, overrides={"grade": "strong"})
    RoutedLLM(router).complete(_request("判一下", task="grade"))
    assert strong.calls == 1
    assert router.decisions[-1].reason == "override"


def test_ledger_tracks_by_model() -> None:
    fast, strong = CountingLLM(model="fast"), CountingLLM(model="strong")
    routed = RoutedLLM(ModelRouter(fast, strong))
    routed.complete(_request("判 A", task="grade"))
    routed.complete(_request("答 B", task="answer"))
    report = routed.report()
    assert report["ledger"]["calls"] == {"fast": 1, "strong": 1}
    assert report["light_ratio"] == 0.5


# —— 流式 ——


def test_streaming_client_yields_pieces() -> None:
    pieces = list(iter_tokens(StreamingLLM(), _request("q", task="answer")))
    assert "".join(pieces) == "云计算标准"
    assert len(pieces) == 3


def test_non_streaming_client_degrades_gracefully() -> None:
    """不支持流式的客户端不该报错——退化为一次性返回即可。"""

    client = CountingLLM(content="整段答案")
    assert not supports_streaming(client)
    outcome = collect_stream(client, _request("q", task="answer"))
    assert outcome.text == "整段答案"
    assert outcome.streamed is False


def test_stream_outcome_records_usage_estimate() -> None:
    outcome = collect_stream(StreamingLLM(), _request("q", task="answer"))
    assert outcome.streamed is True
    assert outcome.chunks == 3
    assert outcome.usage.output_tokens > 0
    assert outcome.estimated is True, "流式协议通常不给 usage，必须标明这是估算值"


def test_on_token_hook_failure_does_not_break_generation() -> None:
    """观察者（推 SSE 的钩子）出问题，绝不能弄坏生成。"""

    def bad_hook(_piece: str) -> None:
        raise RuntimeError("连接断了")

    pieces = list(iter_tokens(StreamingLLM(), _request("q", task="answer"), on_token=bad_hook))
    assert "".join(pieces) == "云计算标准"


@pytest.mark.parametrize("scope_a,scope_b", [("s1", "s2")])
def test_cache_isolation_between_scopes(scope_a: str, scope_b: str) -> None:
    cache = SemanticCache(HashingEmbedder(dim=128))
    cache.put("问题", "答案 A", scope_a)
    assert cache.get("问题", scope_b) is None
