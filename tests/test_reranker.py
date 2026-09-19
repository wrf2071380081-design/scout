"""重排器测试：词法基线与 cross-encoder 契约。"""

from __future__ import annotations

import pytest

from scout.errors import ProviderError
from scout.rag.ranking import (
    CrossEncoderReranker,
    LexicalReranker,
    default_reranker,
)


CANDIDATES = [
    ("a", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"),
    ("b", "低空经济标准体系涵盖低空航空器、起降设施与运行服务等领域。"),
    ("c", "Redis 通过内存存储和单线程事件循环实现高吞吐。"),
]


def test_default_reranker_respects_explicit_backend() -> None:
    assert isinstance(default_reranker("lexical"), LexicalReranker)


def test_cross_backend_raises_when_unavailable(monkeypatch) -> None:
    """``cross`` 是显式请求：装不上必须报错，不能静默降级。"""

    class Broken(CrossEncoderReranker):
        def __init__(self, **_kwargs) -> None:
            raise ProviderError("no fastembed", retryable=False)

    monkeypatch.setattr("scout.rag.ranking.CrossEncoderReranker", Broken)
    with pytest.raises(ProviderError):
        default_reranker("cross")


def test_auto_backend_falls_back_to_lexical(monkeypatch) -> None:
    class Broken(CrossEncoderReranker):
        def __init__(self, **_kwargs) -> None:
            raise ImportError("missing")

    monkeypatch.setattr("scout.rag.ranking.CrossEncoderReranker", Broken)
    assert isinstance(default_reranker("auto"), LexicalReranker)


def test_lexical_reranker_contract() -> None:
    reranker = LexicalReranker(candidate_limit=10)
    outcome = reranker.rerank("云计算标准体系结构包括哪几个部分？", CANDIDATES)
    assert outcome.applied
    # 与问题主题相关的 a 应当排在最前
    assert outcome.ordered[0][0] == "a"


def test_reranker_disabled_and_empty_are_typed() -> None:
    reranker = LexicalReranker()
    assert reranker.rerank("q", [], enabled=True).skipped_reason == "no_candidates"
    assert reranker.rerank("q", CANDIDATES, enabled=False).applied is False


@pytest.mark.slow
def test_cross_encoder_separates_relevance_better_than_lexical() -> None:
    """cross-encoder 的核心价值：**相关与不相关的分距更大**。

    这是"粗排管快、重排管准"这句话的可验证版本。
    标记 slow：需要约 1GB 权重，默认不跑（``pytest -m slow``）。
    """

    try:
        cross = CrossEncoderReranker(candidate_limit=len(CANDIDATES))
        cross._ensure_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"重排模型不可用：{type(exc).__name__}")

    question = "云计算标准体系结构包括哪几个部分？"
    lex_outcome = LexicalReranker(candidate_limit=len(CANDIDATES)).rerank(question, CANDIDATES)
    cross_outcome = cross.rerank(question, CANDIDATES)

    lex_map = dict(lex_outcome.ordered)
    cross_map = dict(cross_outcome.ordered)

    # 两者都应把 a 排最前，但 cross-encoder 给相关/不相关的分距应该明显更大
    assert cross_outcome.ordered[0][0] == "a"
    lex_gap = lex_map["a"] - lex_map["b"]
    cross_gap = cross_map["a"] - cross_map["b"]
    assert cross_gap > lex_gap, f"cross-encoder 相关/不相关的分距应大于词法重排（{cross_gap} vs {lex_gap}）"
