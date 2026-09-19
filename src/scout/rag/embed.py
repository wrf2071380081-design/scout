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


class LocalEmbedder:
    """本地 ONNX 向量模型（通过 fastembed 加载，无需 torch、无需 API key）。

    默认用 ``BAAI/bge-small-zh-v1.5``：中文优化、512 维、约 90MB，
    在 CPU 上足够快。首次使用时下载权重（可配合 ``HF_ENDPOINT=https://hf-mirror.com``）。

    **为什么用 fastembed 而不是 sentence-transformers**：
    后者会拖进 torch（CPU 版也要数百 MB 到 GB 级），
    而 fastembed 走 ONNX Runtime，安装体积小一个数量级，CPU 推理也更快。
    本项目只做推理、不训练，没有理由为它装一个深度学习框架。

    依赖是可选的：没装 fastembed 时 :func:`default_embedder` 会退回哈希向量器，
    而不是让整个包导入失败。
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
    _DIM_BY_MODEL = {
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-small-en": 384,
        "jinaai/jina-embeddings-v2-base-zh": 768,
        "intfloat/multilingual-e5-large": 1024,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    }

    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache_dir: str = "", threads: int | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or None
        self.threads = threads
        self._model = None
        self._dim = self._DIM_BY_MODEL.get(model_name, 0)

    # —— 惰性加载：导入 scout 时不应触发模型下载 ——

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise ProviderError(
                "未安装 fastembed，无法使用本地向量模型。安装：pip install fastembed",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=False,
                provider=self.model_name,
                operation="embedding",
            ) from exc
        kwargs: dict[str, object] = {"model_name": self.model_name}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.threads:
            kwargs["threads"] = self.threads
        try:
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 下载/加载失败都归为 provider 不可用
            raise ProviderError(
                f"加载本地向量模型失败（{self.model_name}）：{exc}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=True,
                provider=self.model_name,
                operation="embedding",
            ) from exc
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = [l2_normalize([float(value) for value in vector]) for vector in model.embed(list(texts))]
        if vectors and self._dim != len(vectors[0]):
            # 以实际输出为准：模型元数据可能与预期不符，静默用错维度比报错更糟。
            self._dim = len(vectors[0])
        return vectors


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


def default_embedder(settings: object | None = None, *, backend: str = "auto") -> Embedder:
    """按配置选择向量器。

    ``backend``：
    - ``auto``（默认）：装了 fastembed 就用本地语义模型，否则退回哈希向量器。
    - ``local``：强制本地语义模型（未安装则抛可读错误，而不是静默降级）。
    - ``openai``：走 OpenAI 兼容 ``/embeddings`` 接口（需要 base_url）。
    - ``hashing``：强制离线哈希向量器（评测基线与单元测试用）。

    设计原则：**降级要显式**。``auto`` 允许降级但会通过 :func:`embedder_status`
    把"你现在跑的是哪一种"暴露给调用方；``local``/``openai`` 则不降级——
    因为用户明确要求了真实模型，静默退回词法向量会让实验结果变得不可解释。
    """

    from ..config import get_settings

    effective = settings or get_settings()
    embedding = getattr(effective, "embedding", None)
    resolved = backend if backend != "auto" else getattr(embedding, "backend", "auto")

    if resolved == "hashing":
        return HashingEmbedder()

    if resolved == "openai" or (embedding is not None and getattr(embedding, "base_url", "")):
        return OpenAICompatEmbedder(
            base_url=getattr(embedding, "base_url", ""),
            api_key=getattr(embedding, "api_key", ""),
            model=getattr(embedding, "model", "text-embedding-3-small"),
            dim=int(getattr(embedding, "dim", 1024) or 1024),
        )

    if resolved in {"auto", "local"}:
        model_name = getattr(embedding, "model", "") or LocalEmbedder.DEFAULT_MODEL
        try:
            return LocalEmbedder(model_name)
        except Exception:  # noqa: BLE001 - auto 模式下允许降级
            if resolved == "local":
                raise
            return HashingEmbedder()

    return HashingEmbedder()


def embedder_status(settings: object | None = None) -> dict[str, object]:
    """报告向量器能力现状。供 ``scout doctor`` 与评测报告的 environment 段使用。"""

    from ..config import get_settings

    effective = settings or get_settings()
    embedding = getattr(effective, "embedding", None)
    backend = getattr(embedding, "backend", "auto")
    model = getattr(embedding, "model", "") or LocalEmbedder.DEFAULT_MODEL
    local_available = False
    try:
        # 用 find_spec 而不是 import：只确认"装没装"，不把 onnxruntime 等
        # 原生库加载进进程。否则一个状态检查就能把进程拖进
        # 原生库卸载崩溃（Windows 退出时 0xC0000409）这种本不属于它的问题里。
        import importlib.util

        local_available = importlib.util.find_spec("fastembed") is not None
        local_detail = "fastembed 已安装" if local_available else "未安装 fastembed（pip install fastembed 即可启用本地语义向量）"
    except Exception:  # noqa: BLE001
        local_detail = "未安装 fastembed（pip install fastembed 即可启用本地语义向量）"

    resolved = backend
    if backend == "auto":
        resolved = "local" if local_available else "hashing"
    return {
        "configured_backend": backend,
        "resolved_backend": resolved,
        "local_model": model,
        "local_available": local_available,
        "local_detail": local_detail,
        "remote_base_url": getattr(embedding, "base_url", "") or "",
        "semantic": resolved in {"local", "openai"},
    }


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "LocalEmbedder",
    "OpenAICompatEmbedder",
    "cosine",
    "default_embedder",
    "embedder_status",
    "l2_normalize",
]
