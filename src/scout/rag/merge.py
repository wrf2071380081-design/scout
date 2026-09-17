"""父块合并（Auto-merge）。

**为什么这个模块值得单独存在，以及它为什么可能是负贡献。**

经典 Auto-merging 的做法是"替换"：同一父块下命中 N 个以上叶子块时，
把这些叶子块**整体替换成父块**。直觉很合理——说明这一片是信息密集区，
给模型更大上下文。

但这里有一个被忽视的代价：**替换会破坏排序**。
叶子块之所以被命中，是因为它的向量与查询最接近；父块是多个叶子块的拼接，
向量是模糊的，而且文本长度可能大一个数量级。替换之后：

- 原本排第 1 的叶子块，变成了一个又长又泛的父块；
- 后续重排/打分阶段拿到的输入分布完全变了；
- 上下文预算被父块吃掉，能放进 prompt 的**独立证据条数反而变少**。

这正是本项目在自有长文档语料上观测到的现象：裸检索 MRR = 1.000，
跑完整流水线后降到 0.857。所以本模块提供**两种模式**，并且把两者的差异做成可测量：

``MergeMode.REPLACE``
    经典替换。保留它，是为了让"替换是否伤害排序"这个问题可以被实验证伪。

``MergeMode.EXPAND``（默认）
    只**附加**上下文，不替换检索单元。叶子块仍然是排序单位（精度不受损），
    但每个单元会带上最近的、满足密度阈值的那一级祖先文本作为
    ``context_text``，打包进 prompt 时使用。这样精度与上下文完整性同时拿到。

选择哪种模式不该靠直觉，应当由 :mod:`scout.evaluation` 的消融实验决定。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .chunking import Chunk, ChunkCatalog


class MergeMode(str, Enum):
    """合并策略。"""

    REPLACE = "replace"
    EXPAND = "expand"
    OFF = "off"


@dataclass(slots=True)
class EvidenceUnit:
    """一条证据单元。

    ``chunk`` 是**排序身份**（始终是被命中的那个块），
    ``context_text`` 是**送进 prompt 的文本**（可能来自祖先块）。
    把这两者分开，是 EXPAND 模式能同时保住精度和上下文的关键。
    """

    chunk: Chunk
    score: float
    context_text: str = ""
    merge_source: str = "leaf"
    merged_child_count: int = 0
    context_level: int = 0

    def __post_init__(self) -> None:
        if not self.context_text:
            self.context_text = self.chunk.text

    @property
    def identity(self) -> dict[str, object]:
        return {
            **self.chunk.identity,
            "merge_source": self.merge_source,
            "context_level": self.context_level,
            "context_characters": len(self.context_text),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity,
            "score": round(self.score, 6),
            "text": self.context_text,
        }


@dataclass(slots=True)
class MergeOutcome:
    """合并统计，用于 trace 与消融分析。"""

    mode: str
    enabled: bool
    applied: bool = False
    merged_groups: int = 0
    replaced_chunks: int = 0
    steps: int = 0
    context_growth_chars: int = 0
    units_before: int = 0
    units_after: int = 0

    def to_meta(self) -> dict[str, object]:
        return {
            "auto_merge_mode": self.mode,
            "auto_merge_enabled": self.enabled,
            "auto_merge_applied": self.applied,
            "auto_merge_merged_groups": self.merged_groups,
            "auto_merge_replaced_chunks": self.replaced_chunks,
            "auto_merge_steps": self.steps,
            "auto_merge_context_growth_chars": self.context_growth_chars,
            "auto_merge_units_before": self.units_before,
            "auto_merge_units_after": self.units_after,
        }


def _group_by_parent(units: Sequence[EvidenceUnit], catalog: ChunkCatalog) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for position, unit in enumerate(units):
        parent_id = catalog.parent_of.get(unit.chunk.chunk_id)
        if parent_id:
            groups[parent_id].append(position)
    return groups


def auto_merge(
    units: Sequence[EvidenceUnit],
    catalog: ChunkCatalog,
    *,
    mode: MergeMode = MergeMode.EXPAND,
    threshold: int = 2,
    max_levels: int = 2,
) -> tuple[list[EvidenceUnit], MergeOutcome]:
    """按同父块命中密度合并。

    :param units: 已排序的证据单元（顺序即相关性，函数不会打乱它）
    :param catalog: 跨文档块目录，用于按 chunk_id 反查父块
    :param threshold: 同一父块下命中多少个才触发合并
    :param max_levels: 最多向上合并几级（2 表示 L3→L2→L1）
    """

    outcome = MergeOutcome(mode=mode.value, enabled=mode is not MergeMode.OFF)
    outcome.units_before = len(units)
    outcome.units_after = len(units)
    if mode is MergeMode.OFF or not units or threshold < 2:
        return list(units), outcome

    current = list(units)
    total_growth = 0

    for _step in range(max_levels):
        groups = _group_by_parent(current, catalog)
        dense = {parent: positions for parent, positions in groups.items() if len(positions) >= threshold}
        if not dense:
            break

        if mode is MergeMode.EXPAND:
            for parent_id, positions in dense.items():
                parent = catalog.get(parent_id)
                if parent is None:
                    continue
                best_child_count = max(len(positions), 1)
                for position in positions:
                    unit = current[position]
                    # 只在能提供更多上下文时才展开，且不覆盖已经展开过的更高层上下文。
                    if len(parent.text) > len(unit.context_text):
                        total_growth += len(parent.text) - len(unit.context_text)
                        unit.context_text = parent.text
                        unit.context_level = parent.level
                        unit.merge_source = "parent_expanded"
                    unit.merged_child_count = best_child_count
                outcome.merged_groups += 1
            outcome.steps += 1
            outcome.applied = True
            # EXPAND 不改变单元数量与顺序，且已经取到满足密度的最高祖先，
            # 继续向上只会让上下文更泛，因此直接结束。
            break

        # REPLACE：把命中密度达标的子块整体换成父块，保持首次出现位置。
        replacements: dict[int, EvidenceUnit] = {}
        consumed: set[int] = set()
        for parent_id, positions in dense.items():
            parent = catalog.get(parent_id)
            if parent is None:
                continue
            best_score = max(current[position].score for position in positions)
            replacements[min(positions)] = EvidenceUnit(
                chunk=parent,
                score=best_score,
                context_text=parent.text,
                merge_source="parent_replaced",
                merged_child_count=len(positions),
                context_level=parent.level,
            )
            consumed.update(positions)
            outcome.merged_groups += 1
            outcome.replaced_chunks += len(positions) - 1

        rebuilt: list[EvidenceUnit] = []
        for position, unit in enumerate(current):
            if position in replacements:
                rebuilt.append(replacements[position])
            elif position in consumed:
                continue
            else:
                rebuilt.append(unit)
        current = rebuilt
        outcome.steps += 1
        outcome.applied = True

    outcome.units_after = len(current)
    outcome.context_growth_chars = total_growth
    return current, outcome


def to_units(pairs: Sequence[tuple[Chunk, float]]) -> list[EvidenceUnit]:
    """便捷函数：把 ``(chunk, score)`` 转成 :class:`EvidenceUnit`。"""

    return [EvidenceUnit(chunk=chunk, score=score) for chunk, score in pairs]


__all__ = ["EvidenceUnit", "MergeMode", "MergeOutcome", "auto_merge", "to_units"]
