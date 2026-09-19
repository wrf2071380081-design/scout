"""检索层：分块、索引、融合、合并、改写、流水线。"""

from __future__ import annotations

from .bm25 import BM25Index
from .chunking import Chunk, ChunkCatalog, ChunkSet, split_document
from .embed import Embedder, HashingEmbedder, OpenAICompatEmbedder, cosine, l2_normalize
from .index import HybridIndex, RetrievalMode, ScoredChunk, SearchResult
from .merge import EvidenceUnit, MergeMode, MergeOutcome, auto_merge
from .pipeline import (
    ABSTENTION_ANSWER,
    ANSWERED,
    CLARIFY,
    INSUFFICIENT,
    NO_KNOWLEDGE,
    PipelineConfig,
    PipelineResult,
    RAGPipeline,
    build_index,
)
from .ranking import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankOutcome,
    default_reranker,
    lexical_score,
    reciprocal_rank_fusion,
)
from .rewrite import (
    DefectReport,
    QueryDefect,
    RewriteAdvisor,
    RewriteMethod,
    RewritePlan,
    apply_plan,
)

__all__ = [
    "ABSTENTION_ANSWER",
    "ANSWERED",
    "BM25Index",
    "CLARIFY",
    "Chunk",
    "ChunkCatalog",
    "ChunkSet",
    "DefectReport",
    "Embedder",
    "EvidenceUnit",
    "HashingEmbedder",
    "HybridIndex",
    "INSUFFICIENT",
    "LexicalReranker",
    "CrossEncoderReranker",
    "MergeMode",
    "MergeOutcome",
    "NO_KNOWLEDGE",
    "OpenAICompatEmbedder",
    "PipelineConfig",
    "PipelineResult",
    "QueryDefect",
    "RAGPipeline",
    "RerankOutcome",
    "RetrievalMode",
    "RewriteAdvisor",
    "RewriteMethod",
    "RewritePlan",
    "ScoredChunk",
    "SearchResult",
    "apply_plan",
    "auto_merge",
    "build_index",
    "cosine",
    "default_reranker",
    "l2_normalize",
    "lexical_score",
    "reciprocal_rank_fusion",
    "split_document",
]
