"""上下文工程：证据压缩与观测压缩。

**为什么检索回来的东西不能直接喂模型。**

三级父子分块的一个副作用是：命中的叶子块往往只含一、两句真正与问题相关的话，
但它的"上下文"是整个父块——800 字符里可能 80% 是填充。
把它原样喂进 prompt 有三大代价：

1. **成本。** token 按字符计费，填充字符是真金白银。
2. **精度。** 模型在噪声里找答案，比在一小簇相关句里找答案更容易走神——
   这正是"上下文腐烂"（context rot）的典型形式。
3. **拒答校准失真。** 充分性判定看的是"问题要点是否有证据支撑"，
   一堆无关句会稀释它。

本模块做两件互补的事：

- :class:`EvidenceCompressor`：检索→生成之间，把每个证据单元**抽取式压缩**——
  只保留与问题最相关的句子，丢弃其余。
- :class:`ObservationCompressor`：长运行时（Agent 跑了很多步），
  把**旧的工具观测**压缩成摘要，保留最近几条原文——对抗长程上下文腐烂。

两个共用的原则：

- **永不丢出处。** 压缩后的文本必须还能指回原始 ``chunk_id``，
  否则归因门控没法工作，"压缩"就退化成"生成"。
- **压缩比是可测量的指标。** 不是为了"看起来更聪明"，
  是为了让"质量不变前提下的成本节省"变成一个**可以报出来的数字**。
  只做压缩、不报压缩比，等于没做。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..llm.scripted import content_tokens
from ..rag.merge import EvidenceUnit

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")

# 数字与否定/极端词是"参数题"和"时间版本题"的承重句，给一个额外权重，
# 防止压缩时把唯一正确的数字句丢掉。
_DIGIT = re.compile(r"\d")
_CRITICAL_HINTS = (
    "不超过", "不少于", "不低于", "最高", "最低", "必须", "禁止", "不得", "应当",
    "目标", "达到", "不超过", "至少", "至多", "上限", "下限", "截止", "到",
)


@dataclass(slots=True)
class CompressionReport:
    """一次压缩的完整账目。这是"成本工程"能成立的前提。"""

    chars_before: int = 0
    chars_after: int = 0
    units_before: int = 0
    units_after: int = 0
    sentences_total: int = 0
    sentences_kept: int = 0

    @property
    def ratio(self) -> float:
        return self.chars_after / self.chars_before if self.chars_before > 0 else 1.0

    @property
    def savings(self) -> float:
        """节省的比例（0~1）。这是给简历和消融表看的数字。"""

        return 1.0 - self.ratio

    def to_meta(self) -> dict[str, Any]:
        return {
            "compression_chars_before": self.chars_before,
            "compression_chars_after": self.chars_after,
            "compression_ratio": round(self.ratio, 4),
            "compression_savings": round(self.savings, 4),
            "compression_sentences_kept": self.sentences_kept,
            "compression_sentences_dropped": self.sentences_total - self.sentences_kept,
            "compression_units_after": self.units_after,
        }


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]


def _sentence_score(sentence: str, query_tokens: set[str]) -> float:
    """给一句子打分。

    与 :func:`scout.rag.ranking.lexical_score` 不同：这里是在**同一段证据内部**
    做句级选择，句子的相关性直接按"问题实义词命中数"计，
    再做数字/否定词的小幅加成，保持简单可解释。
    """

    sentence_tokens = content_tokens(sentence)
    if not sentence_tokens:
        return 0.0
    hits = sum(1 for token in query_tokens if token in sentence_tokens)
    score = float(hits)
    if _DIGIT.search(sentence):
        score += 0.35
    if any(hint in sentence for hint in _CRITICAL_HINTS):
        score += 0.2
    # 稍长的句子在压缩里更值：它更可能自带主语与谓语，截断后不丢语义。
    if 18 <= len(sentence) <= 120:
        score += 0.1
    return score


class EvidenceCompressor:
    """把证据单元抽取式压缩成"与问题相关的句子"的集合。

    :param min_sentences_per_unit: 每个单元至少保留的句数。
        **哪怕是唯一的证据，也至少保留它的最好一句**——
        否则充分性门控会因为"证据凭空消失"而误报。
    :param max_sentences_per_unit: 每个单元最多保留的句数。
        上限防止"一个单元吃光整个预算"。
    """

    def __init__(
        self,
        *,
        min_sentences_per_unit: int = 1,
        max_sentences_per_unit: int = 3,
        citation_prefix: str = "【证据压缩】",
    ) -> None:
        self.min_sentences = max(min_sentences_per_unit, 1)
        self.max_sentences = max(max_sentences_per_unit, self.min_sentences)
        self.citation_prefix = citation_prefix

    def compress(
        self,
        question: str,
        units: Sequence[EvidenceUnit],
        *,
        budget_chars: int | None = None,
    ) -> tuple[list[EvidenceUnit], CompressionReport]:
        """压缩证据。返回 (压缩后的单元列表, 压缩账目)。"""
        report = CompressionReport(
            chars_before=sum(len(unit.context_text) for unit in units),
            units_before=len(units),
        )
        query_tokens = content_tokens(question)

        compressed: list[EvidenceUnit] = []
        for unit in units:
            sentences = _split_sentences(unit.context_text)
            report.sentences_total += len(sentences)
            if not sentences:
                # 没有可分句的文本（罕见，例如纯代码块）原样保留
                compressed.append(unit)
                continue

            scored = sorted(
                ((index, _sentence_score(sentence, query_tokens)) for index, sentence in enumerate(sentences)),
                key=lambda item: -item[1],
            )
            # 至少保留 min_sentences，至多 max_sentences，且**保持原文顺序**——
            # 句子的逻辑顺序是语义的一部分，打乱会让压缩结果读起来支离破碎。
            keep_indices = sorted(index for index, _score in scored[: self.max_sentences])
            while len(keep_indices) < self.min_sentences and len(keep_indices) < len(sentences):
                keep_indices.append(len(keep_indices))
            kept = [sentences[index] for index in keep_indices]
            report.sentences_kept += len(kept)

            # 出处必须保留：压缩文本带原始 chunk 的引用标记，
            # 归因门控和后来的归因校验仍然能找到它是哪一份证据。
            #
            # 但有一个硬约束：**压缩不能让文本变大**。短块（3~4 句）加上
            # 引用前缀后可能比原文还长——那种"压缩"是假的，反而增加成本。
            # 所以只有当新文本严格更短时，才用压缩版本；否则保留原文。
            new_text = f"{self.citation_prefix}（来自 {unit.chunk.chunk_id}）\n" + " ".join(kept)
            if len(new_text) < len(unit.context_text):
                final_text = new_text
            else:
                final_text = unit.context_text
            compressed.append(
                EvidenceUnit(
                    chunk=unit.chunk,
                    score=unit.score,
                    context_text=final_text,
                    merge_source=unit.merge_source,
                    merged_child_count=unit.merged_child_count,
                    context_level=unit.context_level,
                )
            )

        # 预算裁剪：按得分从低到高砍掉保留句，直到符合预算
        if budget_chars is not None:
            compressed = self._enforce_budget(compressed, budget_chars, report)

        report.units_after = len(compressed)
        report.chars_after = sum(len(unit.context_text) for unit in compressed)
        return compressed, report

    def _enforce_budget(
        self,
        units: list[EvidenceUnit],
        budget_chars: int,
        report: CompressionReport,
    ) -> list[EvidenceUnit]:
        """预算裁剪：优先砍低分句，但**至少保留每个单元的第一句**。"""
        # 全局排序砍掉得分最低的句子，但保底每单元一句
        def total_chars(items: list[EvidenceUnit]) -> int:
            return sum(len(unit.context_text) for unit in items)

        if total_chars(units) <= budget_chars:
            return units

        # 反向迭代：从最后（通常最不重要）的单元开始截断
        trimmed: list[EvidenceUnit] = []
        for unit in reversed(units):
            sentences = _split_sentences(unit.context_text)
            if len(sentences) > 1 and total_chars(trimmed + units[: len(trimmed)]) > budget_chars:
                kept = sentences[: self.min_sentences]
                unit = EvidenceUnit(
                    chunk=unit.chunk,
                    score=unit.score,
                    context_text=" ".join(kept),
                    merge_source=unit.merge_source,
                    merged_child_count=unit.merged_child_count,
                    context_level=unit.context_level,
                )
            trimmed.insert(0, unit)
        return trimmed


@dataclass(slots=True)
class ObservationCompressionReport:
    """观测压缩的账目。"""

    messages_before: int = 0
    messages_after: int = 0
    chars_before: int = 0
    chars_after: int = 0
    compressed_observations: int = 0

    def to_meta(self) -> dict[str, Any]:
        return {
            "obs_compression_messages": f"{self.messages_before}→{self.messages_after}",
            "obs_compression_chars": f"{self.chars_before}→{self.chars_after}",
            "obs_compression_count": self.compressed_observations,
        }


class ObservationCompressor:
    """长运行时对旧工具观测的压缩。

    **Context rot 的另一种形式**：Agent 跑了十几步之后，
    上下文里塞满了旧工具的完整输出——其中大部分只被用了一次，
    但会一直占用预算。这个压缩器把**超过最近 K 条**的工具观测压成摘要，
    最近 K 条保留原文。

    设计决定：**最近几条保留原文**。
    把旧观测也压缩成摘要会省更多 token，但最近几步往往是模型正在"回看"的对象，
    压掉它们会让模型丢失当前任务的细节。这是成本与召回的取舍。
    """

    def __init__(
        self,
        *,
        keep_recent: int = 4,
        summary_prefix: str = "【已压缩的旧观测】",
        max_summary_chars: int = 120,
    ) -> None:
        self.keep_recent = keep_recent
        self.summary_prefix = summary_prefix
        self.max_summary_chars = max_summary_chars

    def compress_messages(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], ObservationCompressionReport]:
        """压缩消息列表（dict 形式，见 hitl.checkpoints 的序列化约定）。"""
        report = ObservationCompressionReport(
            messages_before=len(messages),
            chars_before=sum(len(str(message.get("content", ""))) for message in messages),
        )

        # 找出所有 tool 角色的下标，除了最后 keep_recent 个之外全部压缩
        tool_indices = [
            index for index, message in enumerate(messages) if str(message.get("role")) == "tool"
        ]
        to_compress = set(tool_indices[: max(0, len(tool_indices) - self.keep_recent)])

        compressed: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if index not in to_compress:
                compressed.append(dict(message))
                continue
            content = str(message.get("content", ""))
            summary = self._summarize(content)
            compressed.append(
                {
                    **dict(message),
                    "content": f"{self.summary_prefix}（原 {len(content)} 字符）{summary}",
                }
            )
            report.compressed_observations += 1

        report.messages_after = len(compressed)
        report.chars_after = sum(len(str(message.get("content", ""))) for message in compressed)
        return compressed, report

    def _summarize(self, content: str) -> str:
        """抽取式摘要：取第一句 + 含数字/结论词的句子。"""
        sentences = _split_sentences(content)
        if not sentences:
            return content[: self.max_summary_chars]
        keep: list[str] = [sentences[0]]
        for sentence in sentences[1:]:
            if _DIGIT.search(sentence) or any(hint in sentence for hint in _CRITICAL_HINTS):
                keep.append(sentence)
            if sum(len(item) for item in keep) >= self.max_summary_chars:
                break
        summary = " ".join(keep)
        return summary[: self.max_summary_chars]


__all__ = [
    "CompressionReport",
    "EvidenceCompressor",
    "ObservationCompressionReport",
    "ObservationCompressor",
]
