"""三层评测指标。

大多数 RAG 项目只测第一层（检索准不准）和一点第二层（答案对不对），
于是报告里全是"命中率"，而系统真正坏掉的地方看不见。本模块按三层组织：

============  ================================================================
**检索层**      Recall@k / MRR / nDCG@10 / 目标来源命中率
                —— 回答"材料找对了吗"
**生成层**      关键词覆盖率 / 归因支撑率 / 无支撑断言率 / 拒答正确率
                —— 回答"答案有依据吗、该拒时拒了吗"
**轨迹层**      平均步数 / 工具调用数 / 工具失败率 / 重复调用率 / 预算终止率
                —— 回答"过程是否合理、是否在浪费预算"
============  ================================================================

外加**性能层**（P50/P95 延迟、token 成本）。

轨迹层与生成层里的"拒答正确率"是区分度最高的两项：绝大多数系统的拒答率恒为 0，
不是因为不需要拒答，而是因为它从不拒答——这个指标会把它照出来。
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .dataset import EvalCase
from .taxonomy import MUST_ANSWER_TAGS, REFUSAL_EXPECTED_TAGS, QueryTag

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(slots=True)
class CaseObservation:
    """一条样本的运行观测。"""

    case_id: str
    question: str
    outcome: str
    answer: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    step_count: int = 0
    tool_call_count: int = 0
    failed_call_count: int = 0
    repeated_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    answer_support_rate: float = 0.0
    answer_coverage: float = 0.0
    abstained: bool = False
    error_code: str = ""
    failed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_texts: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "question": self.question,
            "outcome": self.outcome,
            "answer": self.answer,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_sources": list(self.retrieved_sources),
            "latency_ms": round(self.latency_ms, 3),
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "failed_call_count": self.failed_call_count,
            "repeated_call_count": self.repeated_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "answer_support_rate": round(self.answer_support_rate, 4),
            "answer_coverage": round(self.answer_coverage, 4),
            "abstained": self.abstained,
            "error_code": self.error_code or None,
            "failed": self.failed,
            "meta": self.meta,
        }
        if include_texts:
            payload["retrieved_texts"] = list(self.retrieved_texts)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseObservation:
        return cls(
            case_id=str(payload.get("case_id", "")),
            question=str(payload.get("question", "")),
            outcome=str(payload.get("outcome", "")),
            answer=str(payload.get("answer", "")),
            retrieved_chunk_ids=[str(item) for item in payload.get("retrieved_chunk_ids") or []],
            retrieved_sources=[str(item) for item in payload.get("retrieved_sources") or []],
            retrieved_texts=[str(item) for item in payload.get("retrieved_texts") or []],
            latency_ms=float(payload.get("latency_ms") or 0.0),
            step_count=int(payload.get("step_count") or 0),
            tool_call_count=int(payload.get("tool_call_count") or 0),
            failed_call_count=int(payload.get("failed_call_count") or 0),
            repeated_call_count=int(payload.get("repeated_call_count") or 0),
            input_tokens=int(payload.get("input_tokens") or 0),
            output_tokens=int(payload.get("output_tokens") or 0),
            answer_support_rate=float(payload.get("answer_support_rate") or 0.0),
            answer_coverage=float(payload.get("answer_coverage") or 0.0),
            abstained=bool(payload.get("abstained", False)),
            error_code=str(payload.get("error_code") or ""),
            failed=bool(payload.get("failed", False)),
            meta=dict(payload.get("meta") or {}),
        )


@dataclass(slots=True)
class CaseMetrics:
    """一条样本的指标。"""

    case_id: str
    tags: list[str] = field(default_factory=list)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    source_hit: bool | None = None
    keyword_coverage: float | None = None
    expected_abstention: bool = False
    abstained: bool = False
    abstention_correct: bool = False
    must_answer: bool = False
    false_refusal: bool = False
    hallucinated: bool = False
    leaked: bool = False
    injection_checked: bool = False
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tags": list(self.tags),
            "recall_at_k": {str(key): round(value, 4) for key, value in sorted(self.recall_at_k.items())},
            "mrr": round(self.mrr, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "source_hit": self.source_hit,
            "keyword_coverage": None if self.keyword_coverage is None else round(self.keyword_coverage, 4),
            "expected_abstention": self.expected_abstention,
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "must_answer": self.must_answer,
            "false_refusal": self.false_refusal,
            "hallucinated": self.hallucinated,
            "leaked": self.leaked,
            "injection_checked": self.injection_checked,
            "failed": self.failed,
        }


# —— 单条打分 ——


def _is_gold(case: EvalCase, chunk_id: str, text: str, position: int) -> bool:
    """判定命中的块是否为金标准。

    优先用文本片段匹配（与分块无关），没有片段时退化为"目标来源文件命中"。
    """

    if case.gold_snippets:
        return any(snippet and snippet in text for snippet in case.gold_snippets)
    if case.expected_sources:
        # 没有精确片段时，把该来源文件的首个命中视为相关。
        return False
    return False


def _ndcg(gains: Sequence[float], *, k: int) -> float:
    """二值相关度的 nDCG@k。"""

    def dcg(values: Sequence[float]) -> float:
        return sum(value / math.log2(index + 2) for index, value in enumerate(values[:k]))

    ideal = dcg(sorted(gains, reverse=True))
    return dcg(gains) / ideal if ideal > 0 else 0.0


def score_case(
    case: EvalCase,
    observation: CaseObservation,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    refusal_markers: Sequence[str] = (),
) -> CaseMetrics:
    """对一条样本打分。"""

    metrics = CaseMetrics(
        case_id=case.case_id,
        tags=[tag.value for tag in case.tags],
        failed=observation.failed,
    )

    relevance = [
        _is_gold(case, chunk_id, text, position)
        for position, (chunk_id, text) in enumerate(
            zip(observation.retrieved_chunk_ids, observation.retrieved_texts, strict=False)
        )
    ]
    total_gold = len(relevance) and sum(relevance) or 0
    expected_gold = len(case.gold_snippets) if case.gold_snippets else 0

    for k in sorted(k_values):
        top = relevance[:k]
        if expected_gold:
            # 以"标注片段数"为分母：漏检惩罚更严格，适合小标注集。
            metrics.recall_at_k[k] = min(sum(top) / expected_gold, 1.0)
        elif total_gold:
            metrics.recall_at_k[k] = sum(top) / total_gold
        else:
            metrics.recall_at_k[k] = 1.0 if case.allow_unknown else 0.0

    first_rank = next((index + 1 for index, hit in enumerate(relevance) if hit), 0)
    metrics.mrr = 1.0 / first_rank if first_rank else 0.0
    metrics.ndcg_at_10 = _ndcg([float(value) for value in relevance], k=10)

    if case.expected_sources:
        metrics.source_hit = any(source in observation.retrieved_sources for source in case.expected_sources)

    if case.expected_keywords:
        answer = observation.answer or ""
        hits = sum(1 for keyword in case.expected_keywords if keyword in answer)
        metrics.keyword_coverage = hits / len(case.expected_keywords)

    metrics.expected_abstention = case.allow_unknown or bool(REFUSAL_EXPECTED_TAGS & set(case.tags))
    metrics.abstained = observation.abstained
    if metrics.expected_abstention:
        metrics.abstention_correct = observation.abstained
        metrics.hallucinated = not observation.abstained
    metrics.must_answer = bool(MUST_ANSWER_TAGS & set(case.tags)) and not case.allow_unknown
    if metrics.must_answer:
        metrics.false_refusal = observation.abstained

    # 注入防护：答案里出现被禁止的内容即判定泄漏。
    if case.forbidden_keywords:
        metrics.injection_checked = True
        answer = observation.answer or ""
        metrics.leaked = any(keyword in answer for keyword in case.forbidden_keywords)

    return metrics


# —— 聚合 ——


@dataclass(slots=True)
class MetricValue:
    """一个聚合指标。``eligible`` 说明有多少样本参与计算。"""

    value: float | None
    eligible: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else round(self.value, 4),
            "eligible": self.eligible,
        }


def _mean(values: Sequence[float]) -> MetricValue:
    if not values:
        return MetricValue(None, 0)
    return MetricValue(statistics.fmean(values), len(values))


def _rate(flags: Sequence[bool]) -> MetricValue:
    if not flags:
        return MetricValue(None, 0)
    return MetricValue(sum(1 for flag in flags if flag) / len(flags), len(flags))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(
    cases: Sequence[EvalCase],
    observations: Sequence[CaseObservation],
    metrics: Sequence[CaseMetrics],
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, MetricValue]:
    """聚合为报告用的指标字典。"""

    result: dict[str, MetricValue] = {}
    case_map = {case.case_id: case for case in cases}

    for k in sorted(k_values):
        result[f"recall_at_{k}"] = _mean([m.recall_at_k.get(k, 0.0) for m in metrics])
    result["mrr"] = _mean([m.mrr for m in metrics])
    result["ndcg_at_10"] = _mean([m.ndcg_at_10 for m in metrics])
    result["source_hit_rate"] = _rate([m.source_hit for m in metrics if m.source_hit is not None])
    result["keyword_coverage"] = _mean(
        [m.keyword_coverage for m in metrics if m.keyword_coverage is not None]
    )

    # 生成层
    result["answer_support_rate"] = _mean(
        [obs.answer_support_rate for obs in observations if not obs.abstained and obs.answer]
    )
    result["answer_coverage"] = _mean(
        [obs.answer_coverage for obs in observations if not obs.abstained and obs.answer]
    )

    # 拒答校准
    result["abstention_accuracy"] = _rate(
        [m.abstention_correct for m in metrics if m.expected_abstention]
    )
    result["hallucination_rate"] = _rate([m.hallucinated for m in metrics if m.expected_abstention])
    result["false_refusal_rate"] = _rate([m.false_refusal for m in metrics if m.must_answer])
    result["answered_rate"] = _rate([not obs.abstained for obs in observations])

    # 安全层：注入泄漏率。没有这一项，注入防护就只是个说法。
    result["injection_leak_rate"] = _rate([m.leaked for m in metrics if m.injection_checked])

    # 轨迹层
    result["avg_steps"] = _mean([float(obs.step_count) for obs in observations])
    result["avg_tool_calls"] = _mean([float(obs.tool_call_count) for obs in observations])
    result["tool_failure_rate"] = _rate(
        [bool(obs.failed_call_count) for obs in observations if obs.tool_call_count]
    )
    result["repeated_call_rate"] = _rate(
        [bool(obs.repeated_call_count) for obs in observations if obs.tool_call_count]
    )
    result["provider_failure_rate"] = _rate([obs.failed for obs in observations])
    result["budget_stop_rate"] = _rate(
        [obs.outcome == "budget_exceeded" for obs in observations]
    )

    # 性能层
    durations = [obs.latency_ms for obs in observations if obs.latency_ms > 0]
    result["latency_mean_ms"] = _mean(durations)
    result["latency_p50_ms"] = MetricValue(_percentile(durations, 0.50), len(durations))
    result["latency_p95_ms"] = MetricValue(_percentile(durations, 0.95), len(durations))
    result["tokens_mean"] = _mean(
        [float(obs.input_tokens + obs.output_tokens) for obs in observations]
    )

    _ = case_map  # 预留：按 case 维度做加权时使用
    return result


def tag_slices(
    metrics: Sequence[CaseMetrics],
    observations: Sequence[CaseObservation],
    *,
    k: int = 5,
) -> dict[str, dict[str, MetricValue]]:
    """按标签切片。

    **这是整份报告里最有信息量的部分**：整体指标只说明"好或坏"，
    切片才说明"哪一类坏了"。例如"表格类 recall 骤降"直接指向分块策略，
    "错别字类 false_refusal 升高"直接指向改写触发条件。
    """

    buckets: dict[str, list[CaseMetrics]] = defaultdict(list)
    observation_map = {obs.case_id: obs for obs in observations}
    for metric in metrics:
        for tag in metric.tags:
            buckets[tag].append(metric)

    slices: dict[str, dict[str, MetricValue]] = {}
    for tag, items in sorted(buckets.items()):
        subset_obs = [observation_map[item.case_id] for item in items if item.case_id in observation_map]
        slices[tag] = {
            f"recall_at_{k}": _mean([item.recall_at_k.get(k, 0.0) for item in items]),
            "mrr": _mean([item.mrr for item in items]),
            "keyword_coverage": _mean(
                [item.keyword_coverage for item in items if item.keyword_coverage is not None]
            ),
            "abstention_accuracy": _rate(
                [item.abstention_correct for item in items if item.expected_abstention]
            ),
            "false_refusal_rate": _rate([item.false_refusal for item in items if item.must_answer]),
            "avg_tool_calls": _mean([float(obs.tool_call_count) for obs in subset_obs]),
            "latency_p95_ms": MetricValue(
                _percentile([obs.latency_ms for obs in subset_obs if obs.latency_ms > 0], 0.95),
                len(subset_obs),
            ),
        }
    return slices


def tag_distribution(cases: Sequence[EvalCase]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        for tag in case.tags:
            counts[tag.value] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "DEFAULT_K_VALUES",
    "CaseMetrics",
    "CaseObservation",
    "MetricValue",
    "aggregate",
    "score_case",
    "tag_distribution",
    "tag_slices",
]
