"""参数自调：把"评测结果"翻译成"下一步改哪个旋钮"。

**为什么这层该存在，以及它的边界在哪。**
一半的工程时间花在调参上（阈值、top_k、chunk 大小、路由门槛）。
这些决定大多靠感觉，改完也没人记得为什么这么定。
:class:`ParamTuner` 把它变成**有依据、有留痕、可回滚**的建议。

**但边界必须划清：它给建议，不自动改。**
理由不是保守，而是数学：单次评测的样本量通常只有几十到几百，
在这个量级上 1-2pp 的差异完全可能是噪声。
自动应用这些差异，等于**用噪声做决策**——系统会来回震荡，
而且每次震荡都会被"新数据"再次合理化。
所以本模块的输出里有一个硬字段 :attr:`ParamSuggestion.auto_applicable`：

- 只有在样本量足够（默认 ≥ 200）且效应量足够（超过噪声带）时才为 True；
- 其余一律为 False，走人工确认。

**这条纪律的收益是长期的**：半年后回看每一次参数变更，
都能找到当时的证据与决策理由，而不是一句"当时调了一下"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(slots=True)
class ParamSuggestion:
    """一条参数建议。"""

    param: str
    current: Any
    proposed: Any
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    auto_applicable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "current": self.current,
            "proposed": self.proposed,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
            "confidence": round(self.confidence, 3),
            "auto_applicable": self.auto_applicable,
        }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


class ParamTuner:
    """按失败结构给参数建议。

    :param min_samples_for_auto: 允许自动应用的最小样本量。
        这条门槛不是"调出来的经验值"，而是统计常识：小样本上的比例差不可信。
    :param noise_band: 视为噪声的效应量下限（pp）。低于它一律不建议改动。
    """

    def __init__(self, *, min_samples_for_auto: int = 200, noise_band: float = 3.0) -> None:
        self.min_samples_for_auto = min_samples_for_auto
        self.noise_band = noise_band

    def suggest(self, metrics: Mapping[str, Any], *, current: Mapping[str, Any] | None = None) -> list[ParamSuggestion]:
        """输入一次评测的聚合指标，输出建议列表（按优先级排序）。

        ``metrics`` 期望包含（缺项会被跳过，不报错——**指标不全时不该乱建议**）：
        ``n_cases``、``abstain_rate``、``false_refusal_rate``、``hallucination_rate``、
        ``recall_at_5``、``cross_doc_recall``、``empty_retrieval_rate``、``cache_hit_rate``。
        """

        current = dict(current or {})
        total = int(metrics.get("n_cases") or 0)
        suggestions: list[ParamSuggestion] = []

        def value_of(key: str) -> float | None:
            """取指标值。**缺失与 0 必须区分**——

            把"没测这一项"当成 0，会凭空触发一堆建议：
            比如没报 Recall@5 就被当成 0%，系统会去建议扩大 top_k。
            这类"基于假数据的建议"比没有建议更糟，因为它看起来有理有据。
            """

            raw = metrics.get(key)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        def add(
            param: str,
            proposed: Any,
            rationale: str,
            evidence: Mapping[str, Any],
            confidence: float,
        ) -> None:
            base = current.get(param)
            if base == proposed:
                return
            auto = total >= self.min_samples_for_auto and confidence >= 0.7
            suggestions.append(
                ParamSuggestion(
                    param=param,
                    current=base,
                    proposed=proposed,
                    rationale=rationale,
                    evidence=dict(evidence),
                    confidence=confidence,
                    auto_applicable=auto,
                )
            )

        abstain = value_of("abstain_rate")
        false_refusal = value_of("false_refusal_rate")
        hallucination = value_of("hallucination_rate")
        recall5 = value_of("recall_at_5")
        cross_doc = value_of("cross_doc_recall")
        empty_retrieval = value_of("empty_retrieval_rate")
        cache_hit = value_of("cache_hit_rate")

        # 1. 误拒高 → 门控过严。这是最容易被忽视的一类"性能损失"：
        #    拒答率看起来"稳健"，实际是把能答的问题也拒了。
        if false_refusal is not None and false_refusal * 100 >= self.noise_band:
            current_cov = float(current.get("sufficiency_min_coverage") or 0.5)
            proposed_cov = round(max(0.2, current_cov - 0.1), 2)
            add(
                "sufficiency_min_coverage",
                proposed_cov,
                f"误拒率 {false_refusal:.1%} 偏高，建议下调充分性覆盖门槛，"
                "让有证据的问题能通过门控；同时观察幻觉率是否上升（两者要一起看）。",
                {"false_refusal_rate": false_refusal, "abstain_rate": abstain},
                confidence=min(0.9, 0.5 + false_refusal),
            )

        # 2. 幻觉高 → 门控过松或检索噪声大。方向与上一条相反，所以两条不可能同时自动生效
        #    （这是刻意的：**互相矛盾的自动变更必须由人裁决**）。
        if hallucination is not None and hallucination * 100 >= self.noise_band:
            current_cov = float(current.get("sufficiency_min_coverage") or 0.5)
            add(
                "sufficiency_min_coverage",
                round(min(0.9, current_cov + 0.1), 2),
                f"幻觉率 {hallucination:.1%} 偏高，建议上调充分性门槛并加强引证校验。",
                {"hallucination_rate": hallucination},
                confidence=min(0.9, 0.5 + hallucination),
            )

        # 3. 召回不足但检索有结果 → 候选池太小，扩 top_k 比换模型便宜得多
        if recall5 is not None and recall5 * 100 < 60 and (empty_retrieval or 0.0) < 0.2:
            current_k = int(current.get("top_k") or 8)
            add(
                "top_k",
                min(20, current_k + 4),
                f"Recall@5 {recall5:.1%} 偏低且检索非空（空检索率 {empty_retrieval or 0.0:.1%}），"
                "优先扩大候选池；先做这一步再考虑换 embedding 模型。",
                {"recall_at_5": recall5, "empty_retrieval_rate": empty_retrieval},
                confidence=0.65,
            )

        # 4. 空检索率高 → 这不是调参能解决的，是知识库覆盖问题。
        #    明确建议"别调参"，避免把数据问题误当成模型问题。
        if empty_retrieval is not None and empty_retrieval >= 0.2:
            suggestions.append(
                ParamSuggestion(
                    param="(不调参)",
                    current=None,
                    proposed="补充知识库覆盖 / 检查分块与解析",
                    rationale=(
                        f"空检索率 {empty_retrieval:.1%}：相当一部分问题在库里找不到任何证据。"
                        "这是数据覆盖或解析问题，**调检索参数不会有实质改善**——"
                        "把力气花在这里只会掩盖真正的缺口。"
                    ),
                    evidence={"empty_retrieval_rate": empty_retrieval},
                    confidence=0.8,
                    auto_applicable=False,
                )
            )

        # 5. 跨文档题差 → 单路 Top-K 结构性不成立，需要子问题分解（结构改动，不是参数）
        if cross_doc is not None and cross_doc * 100 < 40:
            suggestions.append(
                ParamSuggestion(
                    param="(结构改动)",
                    current="单路 Top-K",
                    proposed="启用子问题分解 + 多路召回合并",
                    rationale=(
                        f"跨文档类 Recall {cross_doc:.1%}：需要同时命中两份以上文档时，"
                        "单路 Top-K 会被单篇占满。这属于结构问题，调 k 无效。"
                    ),
                    evidence={"cross_doc_recall": cross_doc},
                    confidence=0.7,
                    auto_applicable=False,
                )
            )

        # 6. 缓存命中率低 → 大概率是 key 设计问题，而不是"没人问重复问题"
        if cache_hit is not None and 0 < cache_hit < 0.15:
            suggestions.append(
                ParamSuggestion(
                    param="cache_key_normalization",
                    current="原始文本哈希",
                    proposed="归一化（去停用词/标点/大小写）+ 语义近邻",
                    rationale=(
                        f"缓存命中率仅 {cache_hit:.1%}。这个数字通常不反映「没有重复问题」，"
                        "而是 key 太严格——先做归一化，再考虑调相似度阈值。"
                    ),
                    evidence={"cache_hit_rate": cache_hit},
                    confidence=0.6,
                    auto_applicable=False,
                )
            )

        suggestions.sort(key=lambda item: (-item.confidence, item.param))
        return suggestions

    def render(self, suggestions: Sequence[ParamSuggestion]) -> str:
        """渲染成 Markdown 片段，可直接贴进评测报告。"""

        if not suggestions:
            return "_本轮没有需要调整的参数（各项指标均在噪声带内）。_\n"
        lines = ["| 参数 | 当前 | 建议 | 置信度 | 自动生效 |", "|---|---|---|---|---|"]
        for item in suggestions:
            lines.append(
                f"| `{item.param}` | {item.current} | {item.proposed} | "
                f"{item.confidence:.2f} | {'是' if item.auto_applicable else '否（需人工确认）'} |"
            )
        lines.append("")
        lines.append("理由明细：")
        for item in suggestions:
            lines.append(f"- **{item.param}**：{item.rationale}")
        return "\n".join(lines) + "\n"


__all__ = ["ParamSuggestion", "ParamTuner"]
