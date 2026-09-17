"""排序融合与重排。

两个组件：

:func:`reciprocal_rank_fusion`
    RRF。把多路召回结果按**排名**融合，而不是按分数。这一点很关键——
    稠密相似度（0~1 的余弦）与 BM25 分数（无上界）量纲完全不同，
    直接加权求和需要先做归一化，而归一化本身会引入新的偏差。
    RRF 只看排名，天然规避了跨路分数不可比的问题。

    这里额外提供 ``weights`` 参数，支持**加权 RRF**。默认不加权（标准 RRF），
    但把权重做成显式参数，是为了让"稠密:稀疏 = 0.3:0.7 到底哪个更好"
    这类问题可以被实验回答，而不是被拍脑袋决定。

:class:`LexicalReranker`
    重排的**可替换占位实现**。真正的跨编码器（cross-encoder）需要额外模型，
    这里用词法加权覆盖度替代，保证整条链路可以在无模型环境下跑通并做消融对照。
    接口与真实重排器一致（输入 query + 候选，输出分数），替换时上层不用改。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..llm.scripted import content_tokens, tokenize


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """倒数排名融合。

    :param rankings: ``{"dense": [id...], "sparse": [id...]}``，每个列表按相关性降序
    :param k: RRF 平滑常数。越大则靠后名次的差异被压得越平
    :param weights: 可选的通道权重，缺省视为 1.0
    :param top_k: 可选截断
    :return: ``(id, fused_score)`` 降序列表；同分时按 id 字典序，保证确定性
    """

    if k <= 0:
        raise ValueError("rrf k must be positive")
    fused: dict[str, float] = defaultdict(float)
    for channel, ordered_ids in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(channel, 1.0))
        if weight == 0.0:
            continue
        for rank, identifier in enumerate(ordered_ids, start=1):
            fused[identifier] += weight / (k + rank)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_k] if top_k is not None else ordered


def term_weight(term: str) -> float:
    """词项权重：越长越具体，因而信息量越大。

    "人工智能" 比 "的" 有信息量。这里用字符长度做代理，是一种
    无需全局统计即可计算的近似——真实系统应当换成语料级 IDF。
    """

    return float(max(len(term), 1))


def lexical_score(query: str, document: str) -> float:
    """词法相关性：加权词项覆盖度 + 短语命中加成。

    返回 0~1。设计目标是**可解释、可复现**，而不是逼近人类相关性判断。

    查询侧只保留实义词（功能词不计权），否则"为什么""怎么"这类词会把
    真实内容词的权重稀释掉，导致所有候选分数都趋同、排序失去区分度。
    """

    query_tokens = content_tokens(query) or set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(document))
    total_weight = sum(term_weight(term) for term in query_tokens)
    if total_weight <= 0:
        return 0.0
    covered = sum(term_weight(term) for term in query_tokens if term in document_tokens)
    coverage = covered / total_weight

    bonus = 0.0
    stripped = query.strip()
    if len(stripped) >= 4 and stripped in document:
        bonus = 0.1
    return min(coverage + bonus, 1.0)


@dataclass(slots=True)
class RerankOutcome:
    """重排结果。

    ``threshold_applied`` 单独记录，是因为"用阈值过滤候选"是一个**有召回风险**的
    动作——它提升精度，但可能直接把正确答案删掉。把它显式写进 trace，
    出问题时才能一眼看出是不是阈值惹的祸。
    """

    ordered: list[tuple[str, float]]
    applied: bool
    skipped_reason: str = ""
    threshold_applied: bool = False
    fallback_applied: bool = False
    error_code: str | None = None
    dropped_count: int = 0

    def to_meta(self) -> dict[str, object]:
        return {
            "rerank_applied": self.applied,
            "rerank_skip_reason": self.skipped_reason or None,
            "rerank_threshold_applied": self.threshold_applied,
            "rerank_fallback_applied": self.fallback_applied,
            "rerank_error_code": self.error_code,
            "rerank_dropped_count": self.dropped_count,
        }


class LexicalReranker:
    """词法重排器（跨编码器的可替换占位实现）。"""

    def __init__(self, *, candidate_limit: int = 50, min_score: float = 0.0) -> None:
        self.candidate_limit = max(candidate_limit, 1)
        self.min_score = min_score

    @property
    def name(self) -> str:
        return "lexical-reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        enabled: bool = True,
    ) -> RerankOutcome:
        """对 ``(id, text)`` 候选重排。

        只对前 ``candidate_limit`` 个候选真正打分，其余保持原相对顺序追加在末尾——
        这是有界的重排，避免候选池很大时被重排阶段拖垮延迟。
        """

        if not enabled:
            return RerankOutcome(
                ordered=[(identifier, 0.0) for identifier, _ in candidates],
                applied=False,
                skipped_reason="disabled",
            )
        if not candidates:
            return RerankOutcome(ordered=[], applied=False, skipped_reason="no_candidates")

        head = list(candidates[: self.candidate_limit])
        tail = list(candidates[self.candidate_limit :])
        scored = sorted(
            ((identifier, lexical_score(query, text)) for identifier, text in head),
            key=lambda item: (-item[1], item[0]),
        )

        if self.min_score > 0.0:
            kept = [(identifier, score) for identifier, score in scored if score >= self.min_score]
            dropped_count = len(scored) - len(kept)
        else:
            kept = scored
            dropped_count = 0

        ordered = kept + [(identifier, 0.0) for identifier, _ in tail]
        return RerankOutcome(
            ordered=ordered,
            applied=True,
            threshold_applied=self.min_score > 0.0,
            dropped_count=dropped_count,
        )


def normalize_scores(items: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """把分数线性归一到 0~1，便于跨通道比较（谨慎使用，会改变排序语义）。"""

    materialized = list(items)
    if not materialized:
        return []
    values = [value for _identifier, value in materialized]
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return [(identifier, 1.0) for identifier, _ in materialized]
    return [(identifier, (value - low) / span) for identifier, value in materialized]


__all__ = [
    "LexicalReranker",
    "RerankOutcome",
    "lexical_score",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "term_weight",
]
