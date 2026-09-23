"""基础数据层：入库、版面还原、去重、版本与血缘。

边界很清楚：**这一层的输出是可检索的语料，不是答案**。
它不做检索、不做生成、不调模型（版面启发式除外，那是纯文本处理）。
把边界守住的价值在于——语料质量问题的排查永远只在这一层内，
不用去翻检索与生成的代码。
"""

from __future__ import annotations

from .dedup import (
    ContentIndex,
    DedupReport,
    DocumentVersion,
    NearDuplicateIndex,
    VersionRegistry,
    content_hash,
    deduplicate,
    hamming,
    simhash,
)
from .extract import (
    GLMOCRExtractor,
    ExtractedText,
    ExtractionOutcome,
    OllamaOCRExtractor,
    PlainTextExtractor,
    StubTextExtractor,
    VLMTextExtractor,
    build_extractor,
    extract_document,
    load_documents_with_extraction,
)
from .layout import (
    LayoutReport,
    detect_columns,
    merge_cross_page_tables,
    reading_order_ratio,
    reorder_columns,
    restore_layout,
)
from .pipeline import IngestReport, Section, ingest, split_sections

__all__ = [
    "ContentIndex",
    "DedupReport",
    "DocumentVersion",
    "ExtractedText",
    "GLMOCRExtractor",
    "OllamaOCRExtractor",
    "ExtractionOutcome",
    "IngestReport",
    "LayoutReport",
    "NearDuplicateIndex",
    "PlainTextExtractor",
    "Section",
    "StubTextExtractor",
    "VLMTextExtractor",
    "VersionRegistry",
    "build_extractor",
    "content_hash",
    "deduplicate",
    "detect_columns",
    "extract_document",
    "hamming",
    "ingest",
    "load_documents_with_extraction",
    "merge_cross_page_tables",
    "reading_order_ratio",
    "reorder_columns",
    "restore_layout",
    "simhash",
    "split_sections",
]
