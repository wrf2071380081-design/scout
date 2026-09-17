"""混合索引。

把稠密向量检索与 BM25 稀疏检索封装成同一个可配置的检索器，支持按通道开关——
这是消融实验的基础设施：**要能回答"去掉 Hybrid 会掉多少分"，代码里就必须存在
一个能一键关掉 Hybrid 的开关**，而不是靠改代码分支。

索引单元固定为**叶子块**（三级结构中的 L3）。父块与中间层只作为合并阶段的
上下文来源，不参与向量与词法索引——这样"检索粒度"和"生成粒度"始终是解耦的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..config import RetrievalSettings
from .bm25 import BM25Index
from .chunking import Chunk, ChunkCatalog, ChunkSet, split_document
from .embed import Embedder, HashingEmbedder, cosine
from .ranking import reciprocal_rank_fusion


class RetrievalMode(str, Enum):
    """检索通道组合。消融时逐个关闭。"""

    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"


@dataclass(slots=True)
class ScoredChunk:
    """带分数的检索结果。"""

    chunk: Chunk
    score: float
    channel: str = "fused"

    @property
    def identity(self) -> dict[str, Any]:
        return self.chunk.identity


@dataclass(slots=True)
class SearchResult:
    """一次检索的完整产出，含做归因分析所需的元信息。"""

    hits: list[ScoredChunk] = field(default_factory=list)
    mode: str = RetrievalMode.HYBRID.value
    channel_sizes: dict[str, int] = field(default_factory=dict)
    candidate_k: int = 0
    degraded_code: str = ""
    duration_ms: float = 0.0

    def to_meta(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.mode,
            "retrieval_channel_sizes": dict(self.channel_sizes),
            "retrieval_candidate_k": self.candidate_k,
            "retrieval_degraded_code": self.degraded_code or None,
            "retrieval_hit_count": len(self.hits),
            "retrieval_duration_ms": round(self.duration_ms, 3),
        }


class HybridIndex:
    """内存混合索引。

    .. note::
       这是**单机内存实现**，目的是让评测与消融可以完全离线复现。
       生产环境应把 :meth:`search` 换成向量数据库（Milvus / Qdrant / pgvector）
       的原生混合检索；接口形状保持不变，上层流水线与评测代码无需改动。
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        settings: RetrievalSettings | None = None,
        channel_weights: dict[str, float] | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings()
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.channel_weights = dict(channel_weights or {})
        self.chunks: list[Chunk] = []
        self.chunk_sets: dict[str, ChunkSet] = {}
        self.catalog = ChunkCatalog()
        self._vectors: list[list[float]] = []
        self._bm25 = BM25Index()
        self._dirty = True
        self._vector_index: dict[str, list[float]] = {}

    # —— 写入 ——

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add_document(
        self,
        text: str,
        *,
        document_id: str,
        document_version: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkSet:
        """切分并索引一篇文档。"""

        chunk_set = split_document(
            text,
            document_id=document_id,
            document_version=document_version,
            filename=filename,
            settings=None,
            metadata=metadata or {},
        )
        self.add_chunk_set(chunk_set)
        return chunk_set

    def add_chunk_set(self, chunk_set: ChunkSet) -> None:
        """把一个已有分块结果加入索引（幂等：同 id 覆盖）。"""

        self.catalog.add(chunk_set)
        if not chunk_set.levels:
            self.chunk_sets[chunk_set.document_id] = chunk_set
            return
        leaves = chunk_set.leaves()
        existing = {chunk.chunk_id for chunk in self.chunks}
        new_leaves = [chunk for chunk in leaves if chunk.chunk_id not in existing]
        if new_leaves:
            self.chunks.extend(new_leaves)
            self._dirty = True
        self.chunk_sets[chunk_set.document_id] = chunk_set

    @property
    def vocabulary(self) -> set[str]:
        """词表。查询改写的错别字/别名检测依赖它。"""

        self.build()
        return set(self._bm25._document_frequency.keys())  # noqa: SLF001 - 同包内共享倒排统计

    @property
    def document_frequency(self) -> dict[str, int]:
        """词项 → 文档数。用于区分"具体词"与"泛词"。"""

        self.build()
        return dict(self._bm25._document_frequency)  # noqa: SLF001

    def build(self) -> None:
        """构建向量与倒排统计。写入后必须调用（或由 :meth:`search` 惰性触发）。"""

        if not self._dirty:
            return
        self._bm25.fit([chunk.text for chunk in self.chunks])
        pending = [chunk for chunk in self.chunks if chunk.chunk_id not in self._vector_index]
        if pending:
            for chunk, vector in zip(pending, self.embedder.embed([c.text for c in pending]), strict=False):
                self._vector_index[chunk.chunk_id] = vector
        self._vectors = [self._vector_index[chunk.chunk_id] for chunk in self.chunks]
        self._dirty = False

    def get_chunk_set(self, document_id: str) -> ChunkSet | None:
        return self.chunk_sets.get(document_id)

    # —— 单通道检索 ——

    def _dense_ranked(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.build()
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = [
            (chunk.chunk_id, cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self._vectors, strict=False)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def _sparse_ranked(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.build()
        ranked = self._bm25.search(query, top_k=limit)
        return [(self.chunks[index].chunk_id, score) for index, score in ranked]

    # —— 检索主入口 ——

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> SearchResult:
        """执行检索。

        :param candidate_k: 单通道召回池大小。默认 ``top_k × candidate_multiplier``，
            之所以要大于 ``top_k``，是因为融合/RRF 需要足够的候选空间才能体现价值。
        """

        started = time.perf_counter()
        effective_top_k = top_k or self.settings.top_k
        pool = candidate_k or effective_top_k * max(self.settings.candidate_multiplier, 1)

        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        rankings: dict[str, list[str]] = {}
        channel_sizes: dict[str, int] = {}

        if mode in (RetrievalMode.HYBRID, RetrievalMode.DENSE_ONLY):
            dense = self._dense_ranked(query, pool)
            rankings["dense"] = [identifier for identifier, _ in dense]
            channel_sizes["dense"] = len(dense)
        if mode in (RetrievalMode.HYBRID, RetrievalMode.SPARSE_ONLY):
            sparse = self._sparse_ranked(query, pool)
            rankings["sparse"] = [identifier for identifier, _ in sparse]
            channel_sizes["sparse"] = len(sparse)

        if not rankings:
            return SearchResult(
                mode=mode.value,
                candidate_k=pool,
                degraded_code="NO_CHANNEL_ENABLED",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        # 只有一个通道时，直接用该通道的原始分数。
        if len(rankings) == 1:
            channel = next(iter(rankings))
            raw = self._dense_ranked(query, pool) if channel == "dense" else self._sparse_ranked(query, pool)
            hits = [
                ScoredChunk(chunk=by_id[identifier], score=score, channel=channel)
                for identifier, score in raw
                if identifier in by_id
            ][:effective_top_k]
            return SearchResult(
                hits=hits,
                mode=mode.value,
                channel_sizes=channel_sizes,
                candidate_k=pool,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        weights = {name: self.channel_weights.get(name, 1.0) for name in rankings}
        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k, weights=weights, top_k=effective_top_k)
        hits = [
            ScoredChunk(chunk=by_id[identifier], score=score, channel="fused")
            for identifier, score in fused
            if identifier in by_id
        ]
        return SearchResult(
            hits=hits,
            mode=mode.value,
            channel_sizes=channel_sizes,
            candidate_k=pool,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def corpus_stats(self) -> dict[str, Any]:
        self.build()
        by_document: dict[str, int] = {}
        for chunk in self.chunks:
            by_document[chunk.filename] = by_document.get(chunk.filename, 0) + 1
        return {
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "leaf_chunks": len(self.chunks),
            "documents": len(self.chunk_sets),
            "leaves_by_document": dict(sorted(by_document.items())),
        }

    def chunk_sets_snapshot(self) -> Sequence[ChunkSet]:
        return list(self.chunk_sets.values())


__all__ = ["HybridIndex", "RetrievalMode", "ScoredChunk", "SearchResult"]
