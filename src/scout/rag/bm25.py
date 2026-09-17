"""BM25 稀疏检索。

混合检索里稀疏一路的价值是**精确词法匹配**：专有名词、型号、条款号、
代码标识符、数字。这些正是稠密向量最容易"语义相近但张冠李戴"的地方。

实现为标准 BM25（``k1=1.5``、``b=0.75``），分词复用
:func:`scout.llm.scripted.tokenize` 的轻量方案。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from ..llm.scripted import tokenize


@dataclass(slots=True)
class BM25Index:
    """BM25 倒排统计。"""

    k1: float = 1.5
    b: float = 0.75
    _doc_terms: list[Counter[str]] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _document_frequency: Counter[str] = field(default_factory=Counter, repr=False)
    _average_length: float = 0.0

    def fit(self, documents: Sequence[str]) -> None:
        self._doc_terms = []
        self._doc_lengths = []
        self._document_frequency = Counter()
        for document in documents:
            counts = Counter(tokenize(document))
            self._doc_terms.append(counts)
            self._doc_lengths.append(sum(counts.values()))
            for term in counts:
                self._document_frequency[term] += 1
        total = len(documents)
        self._average_length = (
            sum(self._doc_lengths) / total if total else 0.0
        )

    @property
    def size(self) -> int:
        return len(self._doc_terms)

    def _idf(self, term: str) -> float:
        """BM25 的 idf，带 +0.5 平滑以避免负值。"""

        total = len(self._doc_terms)
        if total == 0:
            return 0.0
        frequency = self._document_frequency.get(term, 0)
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def score(self, query: str) -> list[float]:
        """返回查询对每个文档的 BM25 分数。"""

        if not self._doc_terms:
            return []
        query_terms = Counter(tokenize(query))
        scores = [0.0] * len(self._doc_terms)
        average = self._average_length or 1.0
        for term, query_frequency in query_terms.items():
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for position, counts in enumerate(self._doc_terms):
                term_frequency = counts.get(term, 0)
                if not term_frequency:
                    continue
                length = self._doc_lengths[position] or 1
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length / average)
                scores[position] += idf * (term_frequency * (self.k1 + 1)) / denominator * query_frequency
        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 ``(下标, 分数)``，按分数降序，仅保留正分。"""

        scores = self.score(query)
        ranked = [(index, value) for index, value in enumerate(scores) if value > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def idf_weight(document_frequency: dict[str, int] | None, corpus_size: int, term: str) -> float:
    """语料级 IDF 权重。词项越稀有，权重越高。

    **为什么这个函数值得单独存在。** 判断"证据是否足以作答"时，
    如果按词面覆盖率平均计权，会出现一个致命的误判：
    政策类语料里 "建设""标准" 这类词几乎每篇都出现，
    于是一个完全无法回答的问题（"火星殖民基地的建设标准是什么"）
    仅凭命中这两个词就能拿到 55% 的覆盖率，从而绕过拒答门控。

    稀有度加权修掉了这个问题：覆盖 "火星""殖民" 才应该得分，
    而覆盖 "建设""标准" 接近于不得分。这和 BM25 用 IDF 压低停用词的
    道理是同一个——只是这里用在**充分性判定**上，而不只是排序上。

    :param document_frequency: 词项 → 包含它的文档数。缺失时退化为 1.0（不做加权）
    :param corpus_size: 文档总数
    """

    if not document_frequency or corpus_size <= 0:
        return 1.0
    frequency = document_frequency.get(term, 0)
    if frequency <= 0:
        # 语料里从未出现：这不是"稀有"，而是"不存在"。
        # 给一个略高于 1 的权重，让"完全没覆盖到"这件事被更严厉地惩罚。
        return 1.5

    ratio = (corpus_size - frequency + 0.5) / (frequency + 0.5)
    if ratio <= 0:
        # df 大于 N 说明调用方把两个不同口径的基数传混了
        # （例如 df 按 chunk 统计、N 却按文档统计）。
        # 此处不抛异常，但把权重夹到正数下界：
        # 一个负数权重会让"覆盖率"变成荒谬的负值（实测出现过 -83）。
        return 1e-6
    return math.log(1.0 + ratio) + 0.1


__all__ = ["BM25Index", "idf_weight"]
