"""三级父子分块。

解决的核心问题：**检索粒度与生成粒度是一对矛盾**。

- 小块（叶子）向量表达聚焦，召回准，但把"它增长了 12%"这种句子单独切出来，
  主语就丢了。
- 大块（父块）上下文完整，但向量被稀释，召回不准。

父子分块把这个矛盾拆开：**用叶子块做检索目标，命中后按父块边界合并回更大的上下文**。
三级结构是为了支持两级合并（L3→L2→L1），让"命中密度"决定合并到哪一层——
同一父块下命中 2 个以上叶子块，说明这里确实是一个信息密集区，值得展开。

另一条可选路线是 late chunking（先整篇编码再切分）或上下文增强分块
（给每块前置一句 LLM 生成的定位说明）。那两条路要额外引入长上下文向量模型
或每块一次 LLM 调用，成本高一个量级；本实现保留三级结构作为默认，
并在 :mod:`scout.rag.pipeline` 里把分块策略做成可替换的接口，便于做消融对照。

**注意单位**：这里全部使用**字符**而不是 token。中英混排时 token 估算误差可达 ±40%，
而分块边界的稳定性对召回的影响远大于"精确对齐 token 预算"带来的收益。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import ChunkSettings

_SEPARATORS: tuple[str, ...] = ("\n\n", "。", "！", "？", "\n", "，", "、", "．", ". ", " ", "")
_WHITESPACE = re.compile(r"[ \t\u3000]+")


def _normalize(text: str) -> str:
    """折叠空白，保留换行结构。"""

    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _split_recursive(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """按分隔符优先级递归切分，尽量在自然边界断开。"""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = ""
    remaining: tuple[str, ...] = ()
    for position, candidate in enumerate(separators):
        if candidate == "":
            break
        if candidate in text:
            separator = candidate
            remaining = separators[position + 1 :]
            break

    if not separator:
        return [text[index : index + size] for index in range(0, len(text), size) if text[index : index + size].strip()]

    chunks: list[str] = []
    buffer = ""
    for part in text.split(separator):
        candidate = part if not buffer else f"{buffer}{separator}{part}"
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer.strip():
            chunks.append(buffer)
        buffer = ""
        if len(part) > size:
            chunks.extend(_split_recursive(part, size, remaining))
        else:
            buffer = part
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给相邻块加重叠，缓解"边界切断语义"。"""

    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        tail = previous[-overlap:]
        result.append(f"{tail}{current}" if tail else current)
    return result


@dataclass(slots=True)
class Chunk:
    """一个检索单元。"""

    chunk_id: str
    document_id: str
    document_version: str
    filename: str
    level: int
    text: str
    index: int
    parent_chunk_id: str | None = None
    start_offset: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> dict[str, Any]:
        """对外暴露的证据身份。**不含正文**，用于 trace 与评测报告。"""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "filename": self.filename,
            "level": self.level,
            "content_hash": self.content_hash,
            "characters": len(self.text),
        }

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self.identity,
            "parent_chunk_id": self.parent_chunk_id,
            "index": self.index,
        }
        if include_text:
            payload["text"] = self.text
        return payload


@dataclass(slots=True)
class ChunkSet:
    """一篇文档的三级块集合。

    ``by_id`` / ``children_of`` 两个索引在构造后一次性建立，
    避免在检索热路径上反复线性扫描（父块合并会频繁按 parent_id 反查）。
    """

    document_id: str
    document_version: str
    filename: str
    levels: dict[int, list[Chunk]] = field(default_factory=dict)
    parent_of: dict[str, str] = field(default_factory=dict)
    by_id: dict[str, Chunk] = field(default_factory=dict)
    children_of: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.by_id and self.children_of:
            return
        for chunk in self.all_chunks():
            self.by_id.setdefault(chunk.chunk_id, chunk)
            if chunk.parent_chunk_id:
                self.parent_of[chunk.chunk_id] = chunk.parent_chunk_id
                self.children_of.setdefault(chunk.parent_chunk_id, []).append(chunk.chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return [chunk for level in sorted(self.levels) for chunk in self.levels[level]]

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def children(self, chunk_id: str) -> list[Chunk]:
        return [self.by_id[item] for item in self.children_of.get(chunk_id, []) if item in self.by_id]

    def leaves(self) -> list[Chunk]:
        return list(self.levels.get(max(self.levels), [])) if self.levels else []

    def ancestors(self, chunk_id: str) -> list[Chunk]:
        """自下而上返回祖先链（最近的父块在前）。"""

        chain: list[Chunk] = []
        cursor = self.parent_of.get(chunk_id)
        seen: set[str] = set()
        while cursor and cursor not in seen:
            seen.add(cursor)
            parent = self.by_id.get(cursor)
            if parent is None:
                break
            chain.append(parent)
            cursor = self.parent_of.get(cursor)
        return chain

    def stats(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "levels": {str(level): len(items) for level, items in sorted(self.levels.items())},
            "total_chunks": len(self.all_chunks()),
        }


def split_document(
    text: str,
    *,
    document_id: str,
    document_version: str,
    filename: str,
    settings: ChunkSettings | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChunkSet:
    """把一篇文档切成三级父子块。

    :param metadata: 附加到每个块的元数据（如 page / section），会原样带入 Evidence 身份
    """

    effective = settings or ChunkSettings()
    normalized = _normalize(text)
    if not normalized.strip():
        return ChunkSet(document_id=document_id, document_version=document_version, filename=filename)

    levels: dict[int, list[Chunk]] = {}
    parents: list[Chunk] = []

    plan = (
        (1, effective.level1_size, effective.level1_overlap),
        (2, effective.level2_size, effective.level2_overlap),
        (3, effective.level3_size, effective.level3_overlap),
    )
    extra = dict(metadata or {})

    for level, size, overlap in plan:
        produced: list[Chunk] = []
        if level == 1:
            pieces = _apply_overlap(_split_recursive(normalized, size, _SEPARATORS), overlap)
            for index, piece in enumerate(pieces):
                produced.append(
                    Chunk(
                        chunk_id=_chunk_id(document_version, filename, level, index),
                        document_id=document_id,
                        document_version=document_version,
                        filename=filename,
                        level=level,
                        text=piece,
                        index=index,
                        parent_chunk_id=None,
                        metadata=dict(extra),
                    )
                )
        else:
            for parent in parents:
                pieces = _apply_overlap(_split_recursive(parent.text, size, _SEPARATORS), overlap)
                for index, piece in enumerate(pieces):
                    produced.append(
                        Chunk(
                            chunk_id=_chunk_id(document_version, filename, level, len(produced)),
                            document_id=document_id,
                            document_version=document_version,
                            filename=filename,
                            level=level,
                            text=piece,
                            index=index,
                            parent_chunk_id=parent.chunk_id,
                            metadata=dict(extra),
                        )
                    )
        levels[level] = produced
        parents = produced

    parent_of = {chunk.chunk_id: chunk.parent_chunk_id for chunk in levels.get(3, []) if chunk.parent_chunk_id}
    return ChunkSet(
        document_id=document_id,
        document_version=document_version,
        filename=filename,
        levels=levels,
        parent_of=parent_of,
    )


def _chunk_id(document_version: str, filename: str, level: int, index: int) -> str:
    return f"{document_version}::{filename}::l{level}::{index}"


@dataclass(slots=True)
class ChunkCatalog:
    """跨文档的块目录。

    **为什么需要它**：一次检索的候选可能来自不同文档，而父块合并需要
    "给定 chunk_id，找到它的父块"。如果只传单个 :class:`ChunkSet`，
    跨文档的候选会静默地拿不到父块——这种 bug 不会报错，只会让合并率变低，
    非常难发现。用一个聚合目录把这件事一次做对。
    """

    parent_of: dict[str, str] = field(default_factory=dict)
    by_id: dict[str, Chunk] = field(default_factory=dict)
    children_of: dict[str, list[str]] = field(default_factory=dict)
    document_of: dict[str, str] = field(default_factory=dict)

    def add(self, chunk_set: ChunkSet) -> None:
        for chunk in chunk_set.all_chunks():
            self.by_id[chunk.chunk_id] = chunk
            self.document_of[chunk.chunk_id] = chunk_set.document_id
            if chunk.parent_chunk_id:
                self.parent_of[chunk.chunk_id] = chunk.parent_chunk_id
                self.children_of.setdefault(chunk.parent_chunk_id, []).append(chunk.chunk_id)

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def ancestors(self, chunk_id: str) -> list[Chunk]:
        chain: list[Chunk] = []
        seen: set[str] = set()
        cursor = self.parent_of.get(chunk_id)
        while cursor and cursor not in seen:
            seen.add(cursor)
            parent = self.by_id.get(cursor)
            if parent is None:
                break
            chain.append(parent)
            cursor = self.parent_of.get(cursor)
        return chain

    @property
    def size(self) -> int:
        return len(self.by_id)


def chunk_texts(chunks: Iterable[Chunk]) -> list[str]:
    """便捷函数：取出文本列表（向量化时用）。"""

    return [chunk.text for chunk in chunks]


__all__ = ["Chunk", "ChunkCatalog", "ChunkSet", "chunk_texts", "split_document"]
