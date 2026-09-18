"""向量器测试：离线哈希基线与本地语义模型的契约。"""

from __future__ import annotations

import pytest

from scout.rag.embed import (
    HashingEmbedder,
    LocalEmbedder,
    cosine,
    default_embedder,
    embedder_status,
)


def test_default_embedder_respects_explicit_backend() -> None:
    assert isinstance(default_embedder(backend="hashing"), HashingEmbedder)


def test_default_embedder_local_raises_when_unavailable(monkeypatch) -> None:
    """``local`` 是显式请求：装不上就必须报错，不能静默退回词法向量。

    静默降级会让"我用了语义向量"这个结论变得不可解释——
    而这不是可用性问题，是**结论有效性问题**。
    """

    import scout.rag.embed as module
    from scout.errors import ProviderError

    class Broken(LocalEmbedder):
        def __init__(self, *_args, **_kwargs) -> None:
            raise ProviderError("fastembed 不可用", retryable=False)

    monkeypatch.setattr(module, "LocalEmbedder", Broken)
    with pytest.raises(ProviderError):
        module.default_embedder(backend="local")


def test_default_embedder_auto_falls_back_to_hashing(monkeypatch) -> None:
    """``auto`` 允许降级：没装 fastembed 时仍要能用。"""

    import scout.rag.embed as module

    class Broken(LocalEmbedder):
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("no fastembed")

    monkeypatch.setattr(module, "LocalEmbedder", Broken)
    assert isinstance(module.default_embedder(backend="auto"), HashingEmbedder)


def test_embedder_status_reports_semantic_flag() -> None:
    status = embedder_status()
    assert set(status) >= {"configured_backend", "resolved_backend", "semantic", "local_available"}
    # conftest 把后端钉成 hashing，因此这里不应报告语义向量
    assert status["resolved_backend"] == "hashing"
    assert status["semantic"] is False


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed(["同一段文本"])[0]
    second = HashingEmbedder(dim=64).embed(["同一段文本"])[0]
    assert first == second
    assert pytest.approx(sum(value * value for value in first), rel=1e-6) == 1.0


@pytest.mark.slow
def test_local_embedder_semantic_separation() -> None:
    """本地语义模型的核心契约：**相近句的分高于无关句的分**。

    这是"语义向量"与"词法向量"的分界线，也是这个模型值得引入的全部理由。
    标记 slow：需要约 90MB 权重，默认不跑（``pytest -m slow``）。
    """

    try:
        embedder = LocalEmbedder()
        vectors = embedder.embed(
            ["Redis 为什么这么快", "Redis 高性能的原因是什么", "今天天气怎么样"]
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"本地向量模型不可用：{type(exc).__name__}")

    related = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    assert related > unrelated, f"相近句 {related:.3f} 应高于无关句 {unrelated:.3f}"
    assert embedder.dim > 0
