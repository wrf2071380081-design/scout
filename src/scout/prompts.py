"""提示词与结构化输出 schema。

集中放一处，是为了让"模型输出契约"可以被单独审查和版本化——
提示词散落在各处时，改一个字段名就可能让某个下游解析悄悄失败。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注，避免 prompts ↔ rag 循环导入
    from .rag.merge import EvidenceUnit


# —— 结构化输出契约 ——


class ComplexityPlan(BaseModel):
    """复杂度判定。决定走单路检索还是子问题分解。"""

    complexity: Literal["simple", "complex"] = Field(description="问题复杂度")
    reason: str = Field(default="", description="判定依据")


class GradePlan(BaseModel):
    """证据评估。这是"要不要改写""要不要拒答"的决策输入。"""

    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="最高相关性")
    coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="对问题要点的覆盖度")
    answerable: bool = Field(default=False, description="证据是否足以作答")
    ambiguous: bool = Field(default=False, description="问题本身是否歧义")
    route: Literal["answer", "rewrite", "clarify", "no_knowledge"] = Field(
        default="answer", description="下一步动作"
    )
    missing: list[str] = Field(default_factory=list, description="缺失的信息点")


class SubQuestions(BaseModel):
    """复杂问题的子问题分解。"""

    questions: list[str] = Field(default_factory=list, max_length=4)


# —— 提示词 ——

COMPLEXITY_PROMPT = (
    "判断下面这个问题应当被当作「简单问题」还是「复杂问题」。\n"
    "简单：答案集中在单一文档的连续片段内，单次检索可覆盖。\n"
    "复杂：需要跨文档、多跳推理、对比多个对象，或包含多个并列子问题。\n"
    '只输出 JSON：{{"complexity": "simple"|"complex", "reason": "简短依据"}}\n\n'
    "问题：{question}"
)

SUBQUESTION_PROMPT = (
    "把下面的复杂问题拆成 2~4 个可以独立检索的子问题。"
    "每个子问题应当指向一个独立的信息点，避免互相包含。\n"
    '只输出 JSON：{{"questions": ["...", "..."]}}\n\n'
    "问题：{question}"
)

GRADE_PROMPT = (
    "你在为一次检索做证据评估。请严格依据给出的证据材料判断，不要借助你自己的知识。\n"
    "评估四项：\n"
    "- relevance：最相关的证据与问题主题的匹配程度（0~1）\n"
    "- coverage：证据对问题各个要点的覆盖程度（0~1）\n"
    "- answerable：仅凭这些证据，能否给出一个可靠回答\n"
    "- ambiguous：问题本身是否缺少必要限定（如指代不明、缺少时间范围）\n"
    "再给出 route：answer（可直接作答）/ rewrite（需改写查询再检索）/"
    "clarify（需向用户澄清）/ no_knowledge（确实没有相关资料）\n"
    "只输出 JSON。\n\n"
    "问题：{question}\n\n"
    "证据：\n{evidence}"
)

ANSWER_PROMPT = (
    "你是严谨的知识库问答助手。**只能依据下面提供的证据作答。**\n"
    "规则：\n"
    "1. 每个结论后面必须标注来源编号，形如 [1]、[2]。\n"
    "2. 证据不足以支撑某个结论时，明确说明「现有资料未涵盖」，不要推测。\n"
    "3. 如果全部证据都无法回答问题，直接回答「根据现有资料无法回答该问题」，不要编造。\n"
    "4. 不要引用证据编号之外的内容。\n\n"
    "问题：{question}\n\n"
    "证据：\n{evidence}"
)

AGENT_SYSTEM_PROMPT = (
    "你是一个可以使用工具的知识库助手。\n"
    "工作方式：\n"
    "1. 需要事实依据时，先调用 knowledge_search 检索；不要凭记忆回答。\n"
    "2. 拿到检索结果后判断是否足够；不足时可以换一个更精确的查询再检索一次。\n"
    "3. 需要日期或算术时调用对应工具，不要自己心算。\n"
    "4. 最终回答必须标注来源编号，形如 [1]；资料不足时明确说明。\n"
    "5. 不要重复调用同一个查询超过一次。\n"
)


def format_evidence(units: Sequence[EvidenceUnit]) -> str:
    """把证据单元格式化成带编号的文本块。

    编号是整个系统的引用锚点：模型输出里的 ``[1]`` 会被
    :mod:`scout.verify.grounding` 用来做归因校验。
    """

    blocks: list[str] = []
    for position, unit in enumerate(units, start=1):
        chunk = unit.chunk
        header = f"[{position}] 来源：{chunk.filename} | 位置：{chunk.chunk_id}"
        if unit.merge_source != "leaf":
            header += f" | 已合并上下文（层级 L{unit.context_level}）"
        blocks.append(f"{header}\n{unit.context_text}")
    return "\n\n".join(blocks)


def pack_evidence(units: Sequence[EvidenceUnit], *, budget_chars: int) -> tuple[list[EvidenceUnit], int]:
    """按**已有排序**装包证据，直到接近预算上限。

    关键点：这里**不重排**。排序是检索与重排阶段的职责，打包阶段重排会让
    "检索排序"与"最终送入模型的顺序"不一致，实验归因时就说不清是哪一步起了作用。

    :return: ``(选中的证据, 被截断的条数)``
    """

    selected: list[EvidenceUnit] = []
    used = 0
    for unit in units:
        cost = len(unit.context_text) + 120  # 头部与编号的粗略开销
        if selected and used + cost > budget_chars:
            continue
        if not selected and cost > budget_chars:
            # 单条就超预算：截断它，而不是返回空证据。
            truncated = unit.context_text[: max(budget_chars - 120, 200)]
            unit.context_text = truncated + "\n…[证据已按预算截断]"
            selected.append(unit)
            used += len(unit.context_text) + 120
            break
        selected.append(unit)
        used += cost
    return selected, max(len(units) - len(selected), 0)


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "ANSWER_PROMPT",
    "COMPLEXITY_PROMPT",
    "GRADE_PROMPT",
    "SUBQUESTION_PROMPT",
    "ComplexityPlan",
    "GradePlan",
    "SubQuestions",
    "format_evidence",
    "pack_evidence",
]
