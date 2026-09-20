"""去重与增量更新：把"同一份内容"和"新版本内容"区分开。

**去重为什么是数据层的核心而不是优化项。**
知识库里同一段话出现三次，检索 Top-K 就会被它占满——用户看到的是
"三条一模一样的结果"，而真正需要的第四条根本没进榜。
这不是"浪费存储"，是**直接吃掉召回**。所以去重发生在入库之前。

两种粒度，缺一不可：

- :class:`ContentIndex` **精确去重**：内容哈希（sha256）完全一致的块直接丢弃。
  它解决的是"同一文件被重复导入""PDF 与 Markdown 双份"。
- :func:`simhash` + :class:`NearDuplicateIndex` **近似去重**：把文本压成 64 位指纹，
  汉明距离 ≤ 阈值即判定为近重复。它解决的是"同一段话改动几个字/换了标点"
  以及"年报里的模板化段落"。

**为什么用 SimHash 而不是"向量相似度"去重**：去重要求**确定性、可解释、极低成本**
（入库是离线批处理，可能有百万级块）。向量近邻需要索引与阈值调参，
而 SimHash 是位运算，天然带"距离"这个可解释量（差几位）。
向量适合"语义相关"，SimHash 适合"几乎一样"——两者用途不同，不该混用。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+")

_HASH_BITS = 64


def _tokens(text: str) -> list[str]:
    """切词：中文按 2-gram、英文按词。

    纯按字符会丢失局部语序信息，纯按词又对中文不友好。
    2-gram 是中文近似去重里的常用折中：改动 1 个字会影响 2 个 gram，
    既能感知改动、又不至于过于敏感。
    """

    pieces = _TOKEN.findall(text.lower())
    grams: list[str] = []
    for piece in pieces:
        if piece.isascii():
            grams.append(piece)
        else:
            if len(piece) <= 2:
                grams.append(piece)
            else:
                grams.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return grams


def content_hash(text: str) -> str:
    """内容指纹：归一化空白后取 sha256 前 16 位。

    归一化是刻意的——同一个 PDF 两次解析出来的换行/空格分布可能不同，
    不去归一化就"哈希不等但内容其实是同一份"。
    """

    normalized = re.sub(r"\s+", " ", text or "").strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def simhash(text: str, *, bits: int = _HASH_BITS) -> int:
    """64 位 SimHash 指纹。"""

    grams = _tokens(text)
    if not grams:
        return 0
    weights = [0] * bits
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for index in range(bits):
            weights[index] += 1 if (value >> index) & 1 else -1
    fingerprint = 0
    for index, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= 1 << index
    return fingerprint


def hamming(left: int, right: int) -> int:
    """两个 SimHash 之间的汉明距离（差几位）。"""

    return bin(left ^ right).count("1")


@dataclass(slots=True)
class DedupReport:
    """去重报告：**丢弃必须可追溯**，否则"为什么这条文档没进库"永远查不清。"""

    total: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    kept: int = 0
    dropped: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "kept": self.kept,
            "dropped_sample": self.dropped[:20],
        }


class ContentIndex:
    """精确去重索引：内容哈希 → 首次出现的标识。"""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def add(self, identifier: str, text: str) -> bool:
        """返回 True 表示"是新的，可以入库"，False 表示重复。"""

        digest = content_hash(text)
        if digest in self._seen:
            return False
        self._seen[digest] = identifier
        return True

    def first(self, digest: str) -> str | None:
        return self._seen.get(digest)

    def __len__(self) -> int:
        return len(self._seen)


class NearDuplicateIndex:
    """近似去重索引：线性扫描 + 汉明距离。

    为什么不做 LSH 分桶：本项目的块规模在 10^4 量级，线性扫描（位运算）
    远快于引入 LSH 所需的分桶工程复杂度；**上 LSH 的门槛是 10^6，
    不是"看起来更先进"**。规模上来之后把 :meth:`add` 换成分段索引即可，
    接口不变。
    """

    def __init__(self, *, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold 必须 >= 1")
        self.threshold = threshold
        self._entries: list[tuple[int, str]] = []

    def add(self, identifier: str, text: str) -> tuple[bool, str | None, int]:
        """返回 ``(是否可入库, 撞到的重复项标识, 最小汉明距离)``。"""

        fingerprint = simhash(text)
        best: str | None = None
        best_distance = _HASH_BITS
        for known, known_id in self._entries:
            distance = hamming(fingerprint, known)
            if distance < best_distance:
                best_distance, best = distance, known_id
        if best is not None and best_distance <= self.threshold:
            return False, best, best_distance
        self._entries.append((fingerprint, identifier))
        return True, None, best_distance

    def __len__(self) -> int:
        return len(self._entries)


def deduplicate(
    items: Sequence[tuple[str, str]],
    *,
    near_threshold: int = 3,
    enable_near: bool = True,
) -> tuple[list[tuple[str, str]], DedupReport]:
    """对 ``(id, text)`` 序列做两段去重，返回保留项与报告。

    顺序：先精确、再近似。理由很实际——精确去重是 O(1) 且零误判，
    先把完全一样的干掉，能显著减少近似去重的扫描量。
    **便宜且确定的检查永远放在前面。**
    """

    report = DedupReport(total=len(items))
    exact = ContentIndex()
    near = NearDuplicateIndex(threshold=near_threshold)
    kept: list[tuple[str, str]] = []

    for identifier, text in items:
        if not exact.add(identifier, text):
            report.exact_duplicates += 1
            report.dropped.append({"id": identifier, "reason": "exact", "match": "", "distance": "0"})
            continue
        if enable_near:
            ok, match, distance = near.add(identifier, text)
            if not ok:
                report.near_duplicates += 1
                report.dropped.append(
                    {
                        "id": identifier,
                        "reason": "near",
                        "match": match or "",
                        "distance": str(distance),
                    }
                )
                continue
        kept.append((identifier, text))
        report.kept += 1

    return kept, report


@dataclass(slots=True)
class DocumentVersion:
    """文档版本记录。缓存 key 由它派生，实现"改了就自然失效"。"""

    doc_id: str
    version: int
    content_digest: str
    updated_at: float = 0.0
    deleted: bool = False

    def cache_key(self, *, model_version: str = "v1") -> str:
        """向量缓存键：``doc_id + 内容指纹 + 模型版本``。

        三者缺一不可：
        - 少了内容指纹 → 文档更新后仍命中旧向量（脏读）
        - 少了模型版本 → 换 embedding 模型后仍命中旧维度向量（维度不匹配或语义错位）
        把"什么算失效"编码进 key，比写一堆失效逻辑更不容易漏。
        """

        return f"{self.doc_id}:{self.content_digest}:{model_version}"


class VersionRegistry:
    """文档版本登记表：支持增量更新与软删除。"""

    def __init__(self) -> None:
        self._versions: dict[str, DocumentVersion] = {}
        self._history: dict[str, list[DocumentVersion]] = {}

    def upsert(self, doc_id: str, text: str, *, now: float = 0.0) -> tuple[DocumentVersion, bool]:
        """登记一版内容。返回 ``(版本记录, 内容是否变化)``。

        内容未变化就不升版本——否则"重复导入同一份文件"会把版本号刷爆，
        下游据此判断"是否要重建索引"的语义也就失效了。
        """

        digest = content_hash(text)
        current = self._versions.get(doc_id)
        if current is not None and current.content_digest == digest and not current.deleted:
            return current, False
        next_version = (current.version + 1) if current else 1
        record = DocumentVersion(
            doc_id=doc_id,
            version=next_version,
            content_digest=digest,
            updated_at=now,
        )
        self._versions[doc_id] = record
        self._history.setdefault(doc_id, []).append(record)
        return record, True

    def soft_delete(self, doc_id: str, *, now: float = 0.0) -> DocumentVersion | None:
        """软删除：不抹掉历史，只标记。

        保留历史的价值：审计、以及"删错了能恢复"。
        真正清理是另一条独立流程（定期 compaction），不该和业务删除混在一起。
        """

        current = self._versions.get(doc_id)
        if current is None:
            return None
        tombstone = DocumentVersion(
            doc_id=doc_id,
            version=current.version + 1,
            content_digest=current.content_digest,
            updated_at=now,
            deleted=True,
        )
        self._versions[doc_id] = tombstone
        self._history.setdefault(doc_id, []).append(tombstone)
        return tombstone

    def active(self) -> Iterator[DocumentVersion]:
        for record in self._versions.values():
            if not record.deleted:
                yield record

    def history(self, doc_id: str) -> list[DocumentVersion]:
        return list(self._history.get(doc_id) or [])

    def stale_keys(self, *, model_version: str = "v1", valid_keys: Iterable[str]) -> list[str]:
        """找出应该失效的向量缓存键。

        取差集而不是逐个比对：**"哪些键还该活着"比"哪些键已经死了"更好表达**，
        缓存系统按这个语义实现（Cache-Aside 的失效侧）也更不容易漏。
        """

        alive = set(valid_keys)
        return [
            record.cache_key(model_version=model_version)
            for record in self.active()
            if record.cache_key(model_version=model_version) not in alive
        ]


__all__ = [
    "ContentIndex",
    "DedupReport",
    "DocumentVersion",
    "NearDuplicateIndex",
    "VersionRegistry",
    "content_hash",
    "deduplicate",
    "hamming",
    "simhash",
]
