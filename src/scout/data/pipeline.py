"""入库流水线：把原始文档变成"可检索、可追溯、可增量更新"的语料。

**这一层为什么该独立存在。**
上游是杂乱的原始文件（PDF 双栏、跨页表格、重复导入、持续更新），
下游是要求"每条证据都能追到来源与版本"的检索器。
把中间这段塞进"读文件"里，会得到一个谁都不敢改的 300 行函数；
抽成流水线之后，每一步都能单独测、单独换、单独观测。

固定的四步，顺序不可换：

1. **版面还原**（:mod:`scout.data.layout`）——先修阅读顺序，再谈别的。
   顺序错了，后面所有步骤都在错误的输入上工作。
2. **结构切分**——按标题层级切，而不是按固定字数切。
3. **两段去重**（:mod:`scout.data.dedup`）——先精确后近似。
4. **版本登记**（:class:`scout.data.dedup.VersionRegistry`)——产出缓存键与增量更新依据。

每一步都往 :class:`IngestReport` 里写数字。**没有报告的数据管线等于没有管线**：
语料质量是隐性变量，出了坏事却查不到是哪一步进来的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .dedup import VersionRegistry, deduplicate
from .layout import LayoutReport, reading_order_ratio, restore_layout

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = "。！？；.!?;"


@dataclass(slots=True)
class Section:
    """结构切分后的一个片段。"""

    doc_id: str
    index: int
    heading: str
    text: str
    level: int = 0

    def to_tuple(self) -> tuple[str, str]:
        return (f"{self.doc_id}#{self.index}", self.text)


@dataclass(slots=True)
class IngestReport:
    """入库报告。"""

    documents: int = 0
    sections: int = 0
    kept: int = 0
    dropped_exact: int = 0
    dropped_near: int = 0
    versions_created: int = 0
    low_quality: list[str] = field(default_factory=list)
    layout: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "sections": self.sections,
            "kept": self.kept,
            "dropped_exact": self.dropped_exact,
            "dropped_near": self.dropped_near,
            "versions_created": self.versions_created,
            "low_quality": self.low_quality[:20],
            "layout": dict(self.layout),
        }


def split_sections(doc_id: str, text: str, *, max_chars: int = 1800) -> list[Section]:
    """按标题层级 + 段落边界切分。

    两个约束同时作用：
    - **标题优先**：遇到标题一定断开，哪怕上一段还没到长度上限——
      把两节的内容混进一个块，检索到的证据就会"半对半错"。
    - **长度上限兜底**：没有标题的长文按句末标点就近断开，
      绝不在句子中间硬切（硬切会让"……因此该公司"这类残句进入索引）。
    """

    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = ""
    current_level = 0
    buffer: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal buffer, index
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                Section(doc_id=doc_id, index=index, heading=current_heading, text=body, level=current_level)
            )
            index += 1
        buffer = []

    for line in lines:
        match = _HEADING.match(line.strip())
        if match:
            flush()
            current_heading = match.group(2).strip()
            current_level = len(match.group(1))
            buffer.append(f"# {current_heading}" if current_level == 1 else f"## {current_heading}")
            continue
        buffer.append(line)
        if sum(len(item) for item in buffer) >= max_chars:
            # 只在句末断开；否则再攒一行（宁可超一点，也不切断句子）
            tail = buffer[-1].rstrip()
            if tail and (tail[-1] in _SENTENCE_END or line.strip() == ""):
                flush()
    flush()
    return sections


def ingest(
    documents: Sequence[tuple[str, str]],
    *,
    registry: VersionRegistry | None = None,
    near_threshold: int = 3,
    min_reading_order: float = 0.3,
    max_chars: int = 1800,
) -> tuple[list[Section], IngestReport, VersionRegistry]:
    """完整入库流水线。

    :param min_reading_order: 阅读顺序完好度低于此值的文档会被拒绝入库并记入报告。
        宁可少收一篇，也不要让错序文本污染整个检索库——
        **坏数据比没数据更难排查，因为它不报错。**
    """

    report = IngestReport(documents=len(documents))
    registry = registry or VersionRegistry()
    layout_totals = LayoutReport()
    pending: list[tuple[str, str]] = []

    for doc_id, raw in documents:
        restored, layout = restore_layout(raw)
        layout_totals.columns_detected = max(layout_totals.columns_detected, layout.columns_detected)
        layout_totals.lines_reordered += layout.lines_reordered
        layout_totals.tables_merged += layout.tables_merged
        layout_totals.headers_repeated += layout.headers_repeated
        layout_totals.page_breaks += layout.page_breaks

        quality = reading_order_ratio(restored)
        if quality < min_reading_order:
            report.low_quality.append(f"{doc_id} (reading_order={quality:.2f})")
            continue

        record, changed = registry.upsert(doc_id, restored)
        if changed:
            report.versions_created += 1

        for section in split_sections(doc_id, restored, max_chars=max_chars):
            pending.append(section.to_tuple())

    report.sections = len(pending)
    kept, dedup_report = deduplicate(pending, near_threshold=near_threshold)
    report.kept = dedup_report.kept
    report.dropped_exact = dedup_report.exact_duplicates
    report.dropped_near = dedup_report.near_duplicates
    report.layout = layout_totals.to_dict()

    sections: list[Section] = []
    for identifier, text in kept:
        doc_id, _, index = identifier.partition("#")
        sections.append(Section(doc_id=doc_id, index=int(index or 0), heading="", text=text, level=0))
    # 稳定顺序，便于对账与增量比对（去重会打乱原始顺序，排序保证可复现）
    sections.sort(key=lambda item: (item.doc_id, item.index))
    return sections, report, registry


__all__ = ["IngestReport", "Section", "ingest", "split_sections"]
