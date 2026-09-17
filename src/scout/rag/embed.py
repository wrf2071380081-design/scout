"""文本向量化。

默认实现是**无依赖的哈希向量器**（signed hashing trick），原因是：

1. 整个仓库克隆下来不装任何模型权重、不联网就能跑通端到端评测；
2. 单元测试的断言必须稳定，模型下载会引入不确定性；
3. 向量器本身是**可替换接口**——生产上把 :class:`HashingEmbedder` 换成
   BGE / GTE / 任意 HTTP embedding 服务即可，检索链路其余部分不用动。

诚实说明：哈希向量器用的是词法信号（非语义），它的检索质量显著低于真正的
语义向量模型。本项目里它的角色是"让流程可复现的基线"，不是"能打的检索器"。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable, Protocol, Sequence

from ..errors import ErrorCode, ProviderError
from ..llm.scripted import tokenize


class Embedder(Protocol):
    """向量器协议。"""

    @property
    def dim(self) -> int:  # pragma: no cover - 协议声明
        ...

    @property
    def name(self) -> str:  # pragma: no cover - 协议声明
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - 协议声明
        ...


def l2_normalize(vector: Iterable[float]) -> list[float]:
    """L2 归一化。归一化之后内积即余弦相似度。"""

    values = list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（不做归一化也能算，但归一化后更快）。"""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class HashingEmbedder:
    """带符号的哈希向量器（signed hashing trick）。

    用 sublinear TF（``1 + log tf``）抑制高频词，并用哈希结果的最高位决定符号，
    使不同词项在同一桶上相互抵消而非简单叠加，缓解哈希碰撞带来的系统性偏差。
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def _vectorize(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = [0.0] * self._dim
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        normalized = l2_normalize(vector)
        self._cache[text] = normalized
        return normalized

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]


class OpenAICompatEmbedder:
    """对接 OpenAI 兼容 ``/embeddings`` 接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1024,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        batch_size: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._dim = dim
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.batch_size = max(1, batch_size)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        import requests

        payload = {"model": self.model, "input": batch}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - 归一化为 typed error
                if attempt >= self.max_attempts:
                    raise ProviderError(
                        f"embedding request failed: {exc}",
                        code=ErrorCode.PROVIDER_UNAVAILABLE,
                        retryable=True,
                        provider=self.model,
                        operation="embedding",
                        attempts=attempt,
                    ) from exc
                continue

            if not response.ok:
                raise ProviderError(
                    "embedding provider rejected the request",
                    code=(
                        ErrorCode.PROVIDER_RATE_LIMITED
                        if response.status_code == 429
                        else ErrorCode.PROVIDER_UNAVAILABLE
                    ),
                    retryable=response.status_code == 429 or response.status_code >= 500,
                    provider=self.model,
                    operation="embedding",
                    details={"status": response.status_code},
                )

            try:
                data = response.json().get("data") or []
            except ValueError as exc:
                raise ProviderError(
                    "embedding provider returned a non-JSON body",
                    code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                    retryable=True,
                    provider=self.model,
                    operation="embedding",
                ) from exc

            if len(data) != len(batch):
                raise ProviderError(
                    "embedding response count mismatch",
                    code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                    retryable=True,
                    provider=self.model,
                    operation="embedding",
                    details={"expected": len(batch), "received": len(data)},
                )

            vectors: list[list[float]] = []
            for item in data:
                vector = [float(value) for value in item.get("embedding") or []]
                if not vector:
                    raise ProviderError(
                        "embedding response contains an empty vector",
                        code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                        retryable=False,
                        provider=self.model,
                        operation="embedding",
                    )
                vectors.append(l2_normalize(vector))
            return vectors

        raise ProviderError(  # pragma: no cover - 循环内已抛
            "embedding batch produced no vectors",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.model,
            operation="embedding",
        )


__all__ = ["Embedder", "HashingEmbedder", "OpenAICompatEmbedder", "cosine", "l2_normalize"]
