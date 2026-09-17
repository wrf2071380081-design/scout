"""查询改写：从"阈值触发"改成"缺陷触发"。

**为什么重写这个模块。**

主流实现（包括很多开源项目）用相关性阈值来触发改写：Top-K 平均分低于 0.6 就触发。
这个设计有一个致命的工程缺陷——**阈值在真实语料上几乎永远不满足**。
实测中大量系统的改写触发率是 0，也就是说这段代码从未执行过，却一直挂在
架构图上当作卖点。而且阈值本身无法回答"改写什么"：相关性低有很多种原因，
错别字、别名、问得太泛、问得太窄，需要的改写方向完全不同。

本模块换一个判据：**先诊断查询自身的缺陷，再由缺陷决定改写方向**。

====================  ==========================  ====================================
缺陷类型              判据                         对应改写
====================  ==========================  ====================================
``TYPO``             查询词不在语料词表内，但存在     词项改写：替换为 1 编辑距离内的
                     1 编辑距离内的语料词             最近语料词
``ALIAS``            大量查询词不在语料词表中         语义扩展（HyDE）：先生成一段
                     （术语体系不匹配）               假设性文档，用文档去检索
``TOO_NARROW``       查询包含过多低文档频率的         退步（Step-back）：抽象到
                     具体词（型号 / 编号 / 长实体）   概念层再检索
``TOO_BROAD``        查询很短且全部是高文档频率       收缩（Specify）：补上领域限定
                     的泛词                           或要求澄清
====================  ==========================  ====================================

这个设计的好处是可测量：每个缺陷类型都可以构造对应的难例集，
"触发率"和"改写增益"都变成可观测的数字，而不是一个永远不触发的阈值。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from ..llm.base import ChatMessage, LLMClient, LLMRequest
from ..llm.scripted import tokenize
from pydantic import BaseModel, Field


class QueryDefect(str, Enum):
    """查询缺陷类型。"""

    TYPO = "typo"
    ALIAS = "alias"
    TOO_NARROW = "too_narrow"
    TOO_BROAD = "too_broad"


class RewriteMethod(str, Enum):
    """改写方式。"""

    NONE = "none"
    TERM_FIX = "term_fix"
    STEP_BACK = "step_back"
    HYDE = "hyde"
    SPECIFY = "specify"


class RewritePlan(BaseModel):
    """一次改写计划。"""

    method: RewriteMethod = Field(description="改写方式")
    reason: str = Field(default="", description="触发原因")
    defects: list[str] = Field(default_factory=list, description="检测到的缺陷")
    step_back_question: str = Field(default="", max_length=300, description="退步问题")
    hyde_document: str = Field(default="", max_length=1200, description="假设性文档")
    term_fixes: dict[str, str] = Field(default_factory=dict, description="词项替换")
    specified_query: str = Field(default="", max_length=300, description="收缩后的查询")

    @property
    def triggers(self) -> bool:
        return self.method is not RewriteMethod.NONE


@dataclass(slots=True)
class DefectReport:
    """缺陷诊断结果。"""

    defects: list[QueryDefect] = field(default_factory=list)
    out_of_vocabulary: list[str] = field(default_factory=list)
    rare_terms: list[str] = field(default_factory=list)
    generic_terms: list[str] = field(default_factory=list)
    vocabulary_size: int = 0

    @property
    def triggered(self) -> bool:
        return bool(self.defects)

    def to_meta(self) -> dict[str, object]:
        return {
            "rewrite_defects": [item.value for item in self.defects],
            "rewrite_oov_count": len(self.out_of_vocabulary),
            "rewrite_rare_count": len(self.rare_terms),
            "rewrite_vocabulary_size": self.vocabulary_size,
        }


def edit_distance_at_most_one(left: str, right: str) -> bool:
    """判断两个字符串的编辑距离是否 ≤ 1（早退，不做完整 DP）。"""

    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    index = 0
    while index < len(left) and left[index] == right[index]:
        index += 1
    if len(left) == len(right):
        return left[index + 1 :] == right[index + 1 :]
    return left[index:] == right[index + 1 :]


class RewriteAdvisor:
    """查询缺陷诊断器。

    :param vocabulary: 语料词表。为空时退化为"只按长度与频率启发式"，
        仍可用，但错别字与别名检测会失效。
    :param document_frequency: 词项 → 出现的文档数。用于区分"具体词"与"泛词"。
    """

    def __init__(
        self,
        *,
        vocabulary: Iterable[str] | None = None,
        document_frequency: Mapping[str, int] | None = None,
        oov_ratio_threshold: float = 0.4,
        narrow_rare_threshold: int = 2,
        broad_max_tokens: int = 3,
        broad_df_ratio: float = 0.3,
    ) -> None:
        self.vocabulary: set[str] = set(vocabulary or ())
        self.document_frequency: Counter[str] = Counter(document_frequency or {})
        self.oov_ratio_threshold = oov_ratio_threshold
        self.narrow_rare_threshold = narrow_rare_threshold
        self.broad_max_tokens = broad_max_tokens
        self.broad_df_ratio = broad_df_ratio

    def _correct(self, term: str, *, max_candidates: int = 2) -> list[str]:
        """在词表里找编辑距离 ≤ 1 的近邻。

        **只对长度 ≥ 2 的 token 生效。** 单字之间的编辑距离 1 匹配几乎全是误报：
        "内" 能匹配到 "内存"，"绩" 能匹配到 "成绩"，把本来完全正确的查询改坏。
        错别字纠正的收益来自多字词（"管里"→"管理"），而不是单字。
        """

        if len(term) < 2 or not self.vocabulary or term in self.vocabulary:
            return []
        candidates: list[str] = []
        for candidate in self.vocabulary:
            if len(candidate) < 2 or abs(len(candidate) - len(term)) > 1:
                continue
            if edit_distance_at_most_one(term, candidate):
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    break
        return candidates

    def diagnose(self, query: str) -> DefectReport:
        """诊断查询缺陷。**这是整个改写链路的触发判据。**"""

        tokens = [token for token in tokenize(query) if token.strip()]
        unique = list(dict.fromkeys(tokens))
        report = DefectReport(vocabulary_size=len(self.vocabulary))
        if not unique:
            return report

        oov = [token for token in unique if self.vocabulary and token not in self.vocabulary]
        total_documents = max(sum(self.document_frequency.values()), 1)
        rare = [
            token
            for token in unique
            if 0 < self.document_frequency.get(token, 0) <= self.narrow_rare_threshold
        ]
        generic = [
            token
            for token in unique
            if self.document_frequency.get(token, 0) / total_documents >= self.broad_df_ratio
        ]

        report.out_of_vocabulary = oov
        report.rare_terms = rare
        report.generic_terms = generic

        # 错别字：存在长度 ≥2 且编辑距离 ≤1 的语料词。
        # 只看多字词——单字的编辑距离 1 匹配几乎全是误报。
        if oov and any(self._correct(token) for token in oov):
            report.defects.append(QueryDefect.TYPO)

        # 别名/术语不匹配：**按多字词统计**不在词表的比例。
        # 不能用全部 token：中文单字数量多，会把比例无条件推高，
        # 让每个查询都被判成"术语不匹配"并触发一次多余的 HyDE。
        concept_tokens = [token for token in unique if len(token) >= 2]
        oov_concepts = [token for token in concept_tokens if self.vocabulary and token not in self.vocabulary]
        oov_ratio = len(oov_concepts) / len(concept_tokens) if concept_tokens else 0.0
        if self.vocabulary and concept_tokens and oov_ratio >= self.oov_ratio_threshold:
            report.defects.append(QueryDefect.ALIAS)

        # 过窄：介词之外大量稀有具体词（型号、编号、长实体名）。
        if len(rare) >= 3 or (len(rare) >= 2 and len(unique) <= 8):
            report.defects.append(QueryDefect.TOO_NARROW)

        # 过泛：查询极短且全部是高频泛词。
        if len(unique) <= self.broad_max_tokens and len(generic) == len(unique):
            report.defects.append(QueryDefect.TOO_BROAD)

        return report

    def plan(
        self,
        query: str,
        report: DefectReport,
        *,
        llm: LLMClient | None = None,
        allow_llm: bool = True,
    ) -> RewritePlan:
        """由缺陷推导改写方案。

        优先级：错别字 > 别名 > 过窄 > 过泛。
        错别字最便宜（纯词项替换，不调模型），所以排最前。
        """

        if not report.triggered:
            return RewritePlan(method=RewriteMethod.NONE, reason="no_defect", defects=[])

        defects = [item.value for item in report.defects]

        if QueryDefect.TYPO in report.defects:
            fixes: dict[str, str] = {}
            for token in report.out_of_vocabulary:
                candidates = self._correct(token)
                if candidates:
                    fixes[token] = candidates[0]
            if fixes:
                return RewritePlan(
                    method=RewriteMethod.TERM_FIX,
                    reason="out_of_vocabulary_with_near_match",
                    defects=defects,
                    term_fixes=fixes,
                )

        if QueryDefect.ALIAS in report.defects:
            return RewritePlan(
                method=RewriteMethod.HYDE,
                reason="terminology_mismatch",
                defects=defects,
                hyde_document=self._hyde_document(query, llm, allow_llm),
            )

        if QueryDefect.TOO_NARROW in report.defects:
            return RewritePlan(
                method=RewriteMethod.STEP_BACK,
                reason="over_specific_query",
                defects=defects,
                step_back_question=self._step_back_question(query, llm, allow_llm),
            )

        return RewritePlan(
            method=RewriteMethod.SPECIFY,
            reason="over_generic_query",
            defects=defects,
            specified_query=f"{query}（请限定具体场景、对象或时间范围）",
        )

    # —— 改写内容生成（可用模型，也可纯规则） ——

    def _hyde_document(self, query: str, llm: LLMClient | None, allow_llm: bool) -> str:
        if llm is not None and allow_llm:
            response = llm.complete(
                LLMRequest(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                "请写一段 2~3 句的假设性文档，用于检索。"
                                "它应当说明该主题常用的术语体系、可能出现在哪类文档的哪一节。"
                                "只输出这段文档本身。\n\n用户问题：" + query
                            ),
                        )
                    ],
                    task="hyde",
                    max_tokens=300,
                    context={"question": query},
                )
            )
            if response.content.strip():
                return response.content.strip()
        return f"针对「{query}」，相关文档通常在其定义、适用条件与典型做法等章节展开说明。"

    def _step_back_question(self, query: str, llm: LLMClient | None, allow_llm: bool) -> str:
        if llm is not None and allow_llm:
            response = llm.complete(
                LLMRequest(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                "把下面的具体问题抽象成一个更概括的概念性问题，"
                                "去掉具体实体名、型号、编号与时间条件。只输出这个问题本身。\n\n"
                                + query
                            ),
                        )
                    ],
                    task="step_back",
                    max_tokens=120,
                    context={"question": query},
                )
            )
            if response.content.strip():
                return response.content.strip()
        # 规则兜底：剥离数字 / 拉丁串 / 引号内容。
        import re

        stripped = re.sub(r"[A-Za-z0-9][A-Za-z0-9\-_.:：]{1,}", "", query)
        stripped = re.sub(r"[《》「」“”\"'（）()]", "", stripped)
        stripped = re.sub(r"\s+", "", stripped)
        stem = stripped or query
        return f"{stem}的基本原理、适用条件与通用做法是什么"


def apply_plan(query: str, plan: RewritePlan) -> str:
    """把改写方案落成实际检索用的查询串。"""

    if plan.method is RewriteMethod.NONE:
        return query
    if plan.method is RewriteMethod.TERM_FIX:
        corrected = query
        for original, replacement in plan.term_fixes.items():
            corrected = corrected.replace(original, replacement)
        return corrected
    if plan.method is RewriteMethod.STEP_BACK:
        return f"{query}\n退步问题：{plan.step_back_question}"
    if plan.method is RewriteMethod.HYDE:
        return f"{query}\n假设性答案文档：{plan.hyde_document}"
    if plan.method is RewriteMethod.SPECIFY:
        return plan.specified_query
    return query


__all__ = [
    "DefectReport",
    "QueryDefect",
    "RewriteAdvisor",
    "RewriteMethod",
    "RewritePlan",
    "apply_plan",
    "edit_distance_at_most_one",
]
