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

from ..errors import ErrorCode, ProviderError
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


class CrossEncoderReranker:
    """真正的 cross-encoder 重排器（BAAI/bge-reranker-base，经由 fastembed ONNX）。

    与 :class:`LexicalReranker` 接口完全一致：输入 query + ``(id, text)`` 候选、
    输出 ``RerankOutcome``。上层 ``RAGPipeline._postprocess`` 不需要改任何代码。

    **为什么粗排之后必须接这一层**（散落在题库/面试里被反复问到的点）：
    向量检索是 ANN，Query 与 Doc 分开编码、快但缺少词级交互——
    语义相近但答非所问的"难负样本"会排上来；
    cross-encoder 把 ``[query, doc]`` 拼接过一遍完整注意力，
    因为它慢（每对都要过模型），所以只能吃粗排的 top-k——
    这正是"粗排管快、重排管准"的分工来源。

    模型侧刻意与向量模型配套：语料是中文，所以选 ``bge-reranker-base``
    （中英都可用、1GB 级、CPU 可跑）；英文/多语言语料可换
    ``jinaai/jina-reranker-v2-base-multilingual``。

    与向量器一样**惰性加载**：import scout 不触发下载；装不上就走
    :func:`default_reranker` 的 ``auto`` 降级（并在 trace 里可见）。
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        *,
        candidate_limit: int = 50,
        min_score: float = 0.0,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 8,
    ) -> None:
        self.candidate_limit = max(candidate_limit, 1)
        self.min_score = min_score
        self.model_name = model_name
        self.batch_size = max(batch_size, 1)
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise ProviderError(
                "未安装 fastembed，无法使用 cross-encoder 重排。安装：pip install fastembed",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=False,
                provider=self.model_name,
                operation="rerank",
            ) from exc
        try:
            self._model = TextCrossEncoder(model_name=self.model_name)
        except Exception as exc:  # noqa: BLE001 - 下载/加载失败都归为 provider 不可用
            raise ProviderError(
                f"加载重排模型失败（{self.model_name}）：{exc}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=True,
                provider=self.model_name,
                operation="rerank",
            ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        enabled: bool = True,
    ) -> RerankOutcome:
        """与 :meth:`LexicalReranker.rerank` 相同的契约：有界重排、阈值过滤、尾部队保持原序。"""

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

        # cross-encoder 每对都要过一次模型，batch 化控制吞吐与内存。
        model = self._ensure_model()
        scores: list[float] = []
        for start in range(0, len(head), self.batch_size):
            batch = head[start : start + self.batch_size]
            batch_scores = list(model.rerank(query, [text for _id, text in batch]))
            scores.extend(float(score) for score in batch_scores)

        scored = sorted(
            ((identifier, score) for (identifier, _text), score in zip(head, scores)),
            key=lambda item: (-item[1], item[0]),
        )

        if self.min_score > 0.0:
            # 重排器的 raw 分数量纲与词法分数不同（logits，不是 0~1），
            # 但"过滤掉分数为负或接近零的"这个语义仍然成立。
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


class LexicalReranker:
    """词法重排器（跨编码器的离线基线占位实现）。"""

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


def default_reranker(
    backend: str = "auto",
    *,
    candidate_limit: int = 50,
    min_score: float = 0.0,
    model_name: str = "",
):
    """按后端选择重排器。

    - ``auto``：装了 fastembed 就用 cross-encoder，否则退回词法重排
    - ``cross``：强制 cross-encoder（装不上就报错，不静默降级）
    - ``lexical``：强制词法重排（离线基线与单元测试）

    与向量器的选择逻辑一致：**显式要求真实模型的路径不做静默降级**——
    降级会让"我用了神经重排"这个结论无法解释。
    """

    if backend == "lexical":
        return LexicalReranker(candidate_limit=candidate_limit, min_score=min_score)
    if backend in {"auto", "cross"}:
        try:
            kwargs: dict[str, object] = {"candidate_limit": candidate_limit, "min_score": min_score}
            if model_name:
                kwargs["model_name"] = model_name
            return CrossEncoderReranker(**kwargs)
        except Exception:
            if backend == "cross":
                raise
            return LexicalReranker(candidate_limit=candidate_limit, min_score=min_score)
    return LexicalReranker(candidate_limit=candidate_limit, min_score=min_score)


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
    "CrossEncoderReranker",
    "LexicalReranker",
    "RerankOutcome",
    "default_reranker",
    "lexical_score",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "term_weight",
]
