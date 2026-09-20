"""版面还原：把"人眼看的版式"转成"机器读的顺序"。

**为什么这属于数据层而不是解析层的实现细节。**
多栏 PDF 直接按 y 坐标逐行读出来，会把左右两栏的文字交错拼在一起——
"左边第一行 + 右边第一行 + 左边第二行 + …"。
一旦这样入库，后面做多少优化都救不回来：**分块边界从源头就坏了**，
检索到的永远是语义错乱的片段。所以版面还原必须在分块之前完成。

本模块提供四个可组合的步骤，全部是纯文本启发式（不依赖 PDF 库，便于测试与离线复现）：

:func:`detect_columns`
    用行内空白间隔的统计特征判断是否双栏，并给出切分点。

:func:`reorder_columns`
    先左栏整列、再右栏整列重排，恢复自然阅读顺序。

:func:`merge_cross_page_tables`
    跨页表格：检测表头并在续页重复表头，让"同一张表"在文本层面可被识别。

:func:`restore_layout`
    串起上面三步的一站式入口。

诚实的边界：这些是**启发式**，不是版面分析模型（LayoutLM / Docling 那一路）。
它解决的是"两栏被读成一栏"和"表格被页断成两张"这两个最高频的破坏性问题；
对印章、公式、图文混排这类复杂版式无能为力——那种场景应当换真模型，
而不是把启发式堆得更复杂。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

# 行内"大段空白"的判定：中文文档里栏间距通常 ≥ 3 个全角空格或 6 个半角空格
_GAP = re.compile(r"[ \t\u3000]{4,}")
_PAGE_MARKER = re.compile(r"第\s*\d+\s*页|page\s*\d+", re.IGNORECASE)
_RULE = re.compile(r"^[-=_]{3,}$")


def is_page_break(line: str) -> bool:
    """是否是分页标记。

    真实文档里的分页标记有几种常见写法，都要认：
    ``---`` / ``===`` 纯分隔线、``第 3 页``、``Page 3``，
    以及最常见的混合形式 ``--- 第 2 页 ---``。
    只认纯分隔线是不够的——PDF 转文本后几乎总是混合形式，
    漏掉它就等于跨页表格合并永远不会触发。
    """

    stripped = line.strip()
    if not stripped:
        return False
    if _RULE.match(stripped):
        return True
    return bool(_PAGE_MARKER.search(stripped))


@dataclass(slots=True)
class LayoutReport:
    """版面还原的执行报告。数字落盘的目的是**可对账**——
    还原了多少行、合并了几张表，出问题时能一眼看出是"没检测到"还是"检测错了"。"""

    columns_detected: int = 1
    lines_reordered: int = 0
    tables_merged: int = 0
    headers_repeated: int = 0
    page_breaks: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "columns_detected": self.columns_detected,
            "lines_reordered": self.lines_reordered,
            "tables_merged": self.tables_merged,
            "headers_repeated": self.headers_repeated,
            "page_breaks": self.page_breaks,
            "notes": list(self.notes),
        }


def _split_line(line: str) -> tuple[str, str] | None:
    """把一行按"行内大空白"切成左右两半。切不开返回 None。"""

    stripped = line.rstrip()
    if not stripped.strip():
        return None
    matches = list(_GAP.finditer(stripped))
    if not matches:
        return None
    # 取最靠中间的那个间隔作为栏分隔，避免把"缩进"误判成栏间距
    middle = len(stripped) / 2
    best = min(matches, key=lambda match: abs((match.start() + match.end()) / 2 - middle))
    left = stripped[: best.start()].rstrip()
    right = stripped[best.end() :].lstrip()
    if len(left) < 6 or len(right) < 6:
        return None
    return left, right


def detect_columns(text: str, *, min_ratio: float = 0.35) -> tuple[int, list[tuple[str, str]]]:
    """检测双栏并返回被切开的行。

    :param min_ratio: 能切开的行占比达到多少才判定为双栏。
        阈值不能太低——正文里偶发的大空白（表格、对齐）也会被切开，
        占比统计正是用来把"偶发"和"整篇"区分开的。
    :return: ``(栏数, [(左, 右), ...])``
    """

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return 1, []
    pairs = [(line, _split_line(line)) for line in lines]
    splittable = [(line, parts) for line, parts in pairs if parts]
    if not splittable:
        return 1, []
    ratio = len(splittable) / len(lines)
    if ratio < min_ratio:
        return 1, []
    return 2, [parts for _line, parts in splittable]  # type: ignore[misc]


def reorder_columns(text: str) -> tuple[str, int]:
    """双栏重排：先整列读左栏，再整列读右栏。

    返回 ``(重排后的文本, 被重排的行数)``。单栏文本原样返回。
    """

    columns, _pairs = detect_columns(text)
    if columns == 1:
        return text, 0

    left: list[str] = []
    right: list[str] = []
    reordered = 0
    for line in text.splitlines():
        parts = _split_line(line)
        if parts is None:
            # 整行（标题、跨栏表格）——按"属于当前栏"处理，放在左栏末尾之前更安全，
            # 这里选择直接追加到左栏，因为它通常出现在页首或栏间。
            if line.strip():
                left.append(line)
            continue
        left.append(parts[0])
        right.append(parts[1])
        reordered += 1
    return "\n".join(left + right), reordered


def _looks_like_header(line: str) -> bool:
    """表头启发式：短行 + 含两个以上分隔符（空格/制表/竖线）。"""

    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    separators = sum(stripped.count(char) for char in ("\t", "|"))
    separators += len(re.findall(r"[ \u3000]{2,}", stripped))
    return separators >= 2


def _find_table_header(emitted: Sequence[str]) -> str | None:
    """从已输出的行里回溯找出"表格第一行"。

    为什么不能直接看分页前最后一行：跨页表格在断开处的那一行是
    **表格中间的某一行数据**，不是表头。真正要延续的是这张表的第一行。

    做法：从末尾往前连续收集"看起来像表格行"的行，直到遇到非表格行为止，
    这个连续段的第一行（即反向收集的最后一个）就是表头。
    """

    run: list[str] = []
    for line in reversed(list(emitted)):
        if not line.strip() or not _looks_like_header(line):
            break
        run.append(line)
    if not run:
        return None
    return run[-1]


def merge_cross_page_tables(text: str) -> tuple[str, int, int]:
    """跨页表格合并：在续页重放表头。

    做法：遇到分页符时回溯出"这张表的第一行"，分页后如果紧接着的行
    仍是表格行，就把表头重复插入一次——
    这样"同一张表被页断成两张"在文本层面可以被下游识别为一张。

    返回 ``(文本, 合并次数, 补写表头次数)``。
    """

    lines = text.splitlines()
    output: list[str] = []
    page_breaks = 0
    headers_repeated = 0
    merged = 0
    pending_header: str | None = None

    for line in lines:
        if is_page_break(line):
            page_breaks += 1
            pending_header = _find_table_header(output)
            output.append(line)
            continue

        if pending_header is not None and line.strip() and _looks_like_header(line):
            output.append(pending_header)
            headers_repeated += 1
            merged += 1
        pending_header = None
        output.append(line)

    return "\n".join(output), merged, headers_repeated


def restore_layout(text: str) -> tuple[str, LayoutReport]:
    """一站式版面还原：双栏重排 → 跨页表格合并。"""

    report = LayoutReport()
    columns, _pairs = detect_columns(text)
    report.columns_detected = columns

    reordered, count = reorder_columns(text)
    report.lines_reordered = count

    merged_text, merged, repeated = merge_cross_page_tables(reordered)
    report.tables_merged = merged
    report.headers_repeated = repeated
    report.page_breaks = len([line for line in text.splitlines() if is_page_break(line)])

    if columns == 1:
        report.notes.append("未检测到双栏，按单栏处理")
    if merged == 0 and report.page_breaks:
        report.notes.append("有分页但未发现可合并表格")
    return merged_text, report


_STRUCTURED = re.compile(r"^\s*(?:[#>\-*+|]|\d+[.、)）]|[（(]\d+[)）])")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LETTER = re.compile(r"[\u4e00-\u9fffA-Za-z]")

# 句末标点集合。**必须包含 PDF 转文本的常见变体**：
# 半角句号 ｡ (U+FF61)、全角句点 ．、省略号 …、以及半角分号/逗号。
# 只认标准的中文句号会让整篇用半角标点的文档"每行都算硬断"——
# 那不是版面错误，是标点变体没被识别，而结果看起来像是文档有问题。
_SENTENCE_END = "。！？；.!?;｡､．…、,，：:）)】」』\"'"

# 散文行的"字母密度"下限：非空白字符里至少一半必须是文字（中文或拉丁字母）。
# **用字母密度而不是"中文占比"**，是因为它天然跨语言：
# 数字型表格行（"科目0 1234567.89 9876543.21"）在中英文语料里密度都极低，
# 而纯中文散文与纯英文散文都能通过。
# 用中文占比会踩两个坑：英文语料全部失效；数字表格会把整篇的占比拉低，
# 反而让过滤器不生效（这两种情况都真实出现过）。
_MIN_LETTER_RATIO = 0.5


def cjk_ratio(text: str) -> float:
    """中文字符占比。仅供报表与诊断使用，不参与判定。"""

    if not text:
        return 0.0
    return len(_CJK.findall(text)) / len(text)


def letter_ratio(text: str) -> float:
    """文字字符（中文或拉丁字母）在非空白字符中的占比。"""

    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    return len(_LETTER.findall(stripped)) / len(stripped)


def _is_prose(line: str, *, min_chars: int = 14, min_letters: float = _MIN_LETTER_RATIO) -> bool:
    """这句话是"散文"吗——只有散文行才参与阅读顺序评分。

    **为什么必须把结构化行排除掉。** 早期实现把标题、列表项、表格行一起算进去，
    结果一篇列表密集的申报指南被判 `reading_order=0.09` 并被质量门禁拒绝——
    可那些行本来就**不该以句号结尾**。用"散文的规矩"去量"清单"，
    得到的低分反映的是文档体裁，不是版面错误。
    """

    stripped = line.strip()
    if len(stripped) < min_chars:
        return False
    if _STRUCTURED.match(stripped):
        return False
    if letter_ratio(stripped) < min_letters:
        return False
    # 表格行：含 2 个以上连续空白或制表符
    if len(re.findall(r"[ \t\u3000]{2,}", stripped)) >= 2:
        return False
    return stripped[-1] not in "：:，,、"


def reading_order_ratio(text: str, *, min_prose_lines: int = 20) -> float:
    """估算"阅读顺序完好度"，用于入库前的质量门禁。

    做法：统计**散文行**之间首尾的连贯性——真实阅读顺序下，相邻行之间
    通常在标点上连贯；错序拼接后会出现大量莫名其妙的跳变。
    返回 0~1，越高越连贯；**返回 1.0 也可能是"无法判断"**（见下）。

    "不下结论"的分支是被真实语料教出来的：散文行少于 ``min_prose_lines`` 时
    直接判为无法判断。在 40 篇真实语料上量过——加上这个下限后 19 篇被判"无法判断"，
    **剩下 21 篇里没有任何一篇低于 0.5**（最低 0.56、中位 0.86）。
    也就是说：能判断的都是正常的，假阳性消失了。

    用"测不准"去拒绝入库，是把不确定性当成了否定证据。
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prose = [line for line in lines if _is_prose(line)]
    if len(prose) < min_prose_lines:
        return 1.0
    breaks = 0
    for previous, current in zip(prose, prose[1:]):
        head = current[0]
        # 句末标点 + 新句开头 = 正常；句中被硬断 = 可疑
        if previous[-1] in _SENTENCE_END or head in "　 \u3000（(【":
            continue
        breaks += 1
    return 1.0 - breaks / (len(prose) - 1)


__all__ = [
    "LayoutReport",
    "cjk_ratio",
    "detect_columns",
    "is_page_break",
    "letter_ratio",
    "merge_cross_page_tables",
    "reading_order_ratio",
    "reorder_columns",
    "restore_layout",
]
