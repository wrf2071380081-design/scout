"""语义缓存：把"问过的问题"变成"不再花钱的问题"。

**为什么必须是语义缓存，而不是字符串缓存。**
用户不会把同一个问题一字不差地问两遍。"Redis 为什么快" 与 "为什么 Redis 性能这么高"
在精确匹配下是两次调用、两次计费。语义缓存把 query 向量化后做近邻匹配，
命中相似度阈值即复用——这才是真实流量下的命中率来源。

**但语义缓存有一个 RAG 特有的陷阱，必须显式防住：**
同一个问题，在**不同证据**下要给出不同答案。
"这份年报的营收是多少" 在换了知识库版本之后，语义完全相同、答案必须不同。
所以缓存项必须带 **scope（作用域）** ——通常是证据指纹 + 模型版本：
**scope 变了就是 miss，哪怕问题一模一样。**
只按问题文本缓存，是这类实现最常见的严重错误：它会把旧答案喂给新数据。

**阈值为什么定得偏高（默认 0.93）。**
缓存命中是"直接返回，不再校验"，它的错误代价是**静默给出错答案**；
而 miss 的代价只是多花一次调用。两者不对称，所以阈值要偏保守。
宁可少命中，也不能命中错。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..llm.base import LLMClient, LLMRequest, LLMResponse, TokenUsage
from ..rag.embed import Embedder, cosine


def _text_of(request: LLMRequest) -> str:
    return "\n".join(message.content for message in request.messages)


def request_scope(request: LLMRequest, *, model_version: str = "") -> str:
    """作用域：决定"两个请求能不能共用一份缓存"。

    组成：任务名 + 模型版本 + 显式提供的上下文指纹。
    刻意**不包含完整提示词**——提示词里含温度、格式说明等噪声，
    把它们纳入会让本该命中的请求全部 miss。
    """

    context = request.context or {}
    fingerprint = str(context.get("evidence_digest") or context.get("corpus_fingerprint") or "")
    payload = {
        "task": request.task,
        "model_version": model_version or context.get("model_version", ""),
        "evidence": fingerprint,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(slots=True)
class CacheEntry:
    query: str
    vector: list[float]
    value: str
    scope: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass(slots=True)
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses_scope: int = 0
    misses_similarity: int = 0
    puts: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "misses_scope": self.misses_scope,
            "misses_similarity": self.misses_similarity,
            "puts": self.puts,
            "evictions": self.evictions,
        }


class SemanticCache:
    """近邻语义缓存（进程内实现）。

    生产上这一层应当换成 Redis + 向量索引（键空间按 scope 分片）；本实现的接口
    （``get`` / ``put`` / ``stats``）刻意保持与那种实现同形，替换时上层无感。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.93,
        max_items: int = 512,
        ttl_seconds: float = 24 * 3600,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_items = max(1, max_items)
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, list[CacheEntry]] = {}
        self.stats = CacheStats()

    def _scope_bucket(self, scope: str) -> list[CacheEntry]:
        return self._entries.setdefault(scope, [])

    def get(self, query: str, scope: str) -> tuple[str, float] | None:
        """返回 ``(值, 相似度)`` 或 None。"""

        self.stats.lookups += 1
        bucket = self._entries.get(scope)
        if not bucket:
            self.stats.misses_scope += 1
            return None
        if self.embedder is None:
            # 没有向量器就退化为精确匹配：不假装自己是语义缓存。
            for entry in bucket:
                if entry.query == query:
                    entry.hits += 1
                    self.stats.hits += 1
                    return entry.value, 1.0
            self.stats.misses_similarity += 1
            return None

        now = time.time()
        query_vector = self.embedder.embed([query])[0]
        best: CacheEntry | None = None
        best_score = 0.0
        for entry in list(bucket):
            if now - entry.created_at > self.ttl_seconds:
                bucket.remove(entry)
                self.stats.evictions += 1
                continue
            score = cosine(query_vector, entry.vector)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.threshold:
            best.hits += 1
            self.stats.hits += 1
            return best.value, best_score
        self.stats.misses_similarity += 1
        return None

    def put(self, query: str, value: str, scope: str) -> None:
        bucket = self._scope_bucket(scope)
        vector = self.embedder.embed([query])[0] if self.embedder is not None else []
        bucket.append(CacheEntry(query=query, vector=vector, value=value, scope=scope))
        self.stats.puts += 1
        # 超容量按 LRU 近似淘汰：先清过期，再按命中次数与年龄的组合排序淘汰
        total = sum(len(items) for items in self._entries.values())
        if total > self.max_items:
            self._evict()

    def _evict(self) -> None:
        candidates: list[tuple[float, str, CacheEntry]] = []
        now = time.time()
        for scope, bucket in self._entries.items():
            for entry in bucket:
                age = now - entry.created_at
                # 分数越高越值得留：命中多、最近用过
                score = entry.hits * 10.0 - age / 3600.0
                candidates.append((score, scope, entry))
        candidates.sort(key=lambda item: item[0])
        for _score, scope, entry in candidates[: max(1, self.max_items // 10)]:
            bucket = self._entries.get(scope)
            if bucket and entry in bucket:
                bucket.remove(entry)
                self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._entries.values())


class CachedLLM:
    """给任意 :class:`LLMClient` 套一层语义缓存。

    只缓存**确定性任务**（grade / rewrite / intent 这类）：它们对同一输入
    期望同一输出。生成类任务（answer）默认不缓存——答案的温度不总是 0，
    且它依赖的证据变化更频繁，缓存收益低、风险高。
    """

    def __init__(
        self,
        inner: LLMClient,
        cache: SemanticCache,
        *,
        cacheable_tasks: Sequence[str] = ("grade", "rewrite", "intent", "subquestions", "decide"),
        model_version: str = "",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.cacheable_tasks = set(cacheable_tasks)
        self.model_version = model_version
        self.saved_calls = 0

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def complete(self, request: LLMRequest) -> LLMResponse:
        if request.task not in self.cacheable_tasks:
            return self.inner.complete(request)

        scope = request_scope(request, model_version=self.model_version)
        query = _text_of(request)
        hit = self.cache.get(query, scope)
        if hit is not None:
            value, _score = hit
            self.saved_calls += 1
            # 命中时用量记为 0：这样 token 统计骤降是可解释的
            # （"缓存命中"而不是"模型少干活了"），而不是一个需要猜的异常。
            return LLMResponse(content=value, usage=TokenUsage())

        response = self.inner.complete(request)
        if response.content:
            self.cache.put(query, response.content, scope)
        return response


__all__ = ["CacheEntry", "CacheStats", "CachedLLM", "SemanticCache", "request_scope"]
