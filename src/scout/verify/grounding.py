"""证据接地校验与拒答校准。

**这个模块针对的是一个反直觉、但已被实验证明的问题：**

> RAG 整体上提升了下游性能，却**降低了模型适时说"不知道"的能力**。
> 当上下文不充分时，额外注入的信息不但没帮忙，反而增加了幻觉倾向。

原因不难理解：给模型一段"看起来相关"的材料，它会倾向于从中拼出一个答案，
而不是承认材料不够。结果是系统在**最需要拒答的时候最自信**。

所以本模块做三件事：

1. **归因校验**（:func:`check_grounding`）：把答案拆成 claim，逐个检查是否被
   它引用的证据 span 支撑。没有任何一句话被支撑的答案不允许通过。
2. **充分性判定**（:func:`estimate_sufficiency`）：用问题侧信息需求的覆盖率
   估计"证据够不够"，而不是用检索分数——分数高只说明"最像"，不说明"够用"。
3. **拒答校准**（:func:`verify_answer`）：综合前两者给出显式动作
   （通过 / 重生成 / 拒答 / 澄清），把"要不要说不知道"从一个模型行为
   变成一个可以在评测里统计的**决策**。

这样"拒答率"才是一个有意义的指标——大多数系统的拒答率恒为 0，
不是因为它从不需要拒答，而是因为它从来不拒答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..config import VerifySettings
from ..llm.scripted import content_tokens
from .sanitize import attribution_gate

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注，避免 verify ↔ rag 循环导入
    from ..rag.merge import EvidenceUnit

_CITATION = re.compile(r"\[(\d+)]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?\n])")

# 明确的拒答表达。用于统计"该拒时是否真的拒了"。
ABSTENTION_MARKERS: tuple[str, ...] = (
    "无法回答",
    "无法确定",
    "没有相关",
    "未涵盖",
    "资料不足",
    "证据不足",
    "知识库中暂无",
    "没有找到",
    "没有检索到",
    "不能确定",
    "不足以回答",
    "未提及",
)


class Verdict(str, Enum):
    """校验动作。"""

    PASS = "pass"
    REGENERATE = "regenerate"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"


@dataclass(slots=True)
class Claim:
    """一条待核验的断言。"""

    text: str
    citations: list[int] = field(default_factory=list)
    support_score: float = 0.0
    supported: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text[:160],
            "citations": list(self.citations),
            "support_score": round(self.support_score, 4),
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass(slots=True)
class GroundingReport:
    """接地校验结果。"""

    claims: list[Claim] = field(default_factory=list)
    suppressed_sentences: int = 0
    coverage: float = 0.0
    focus_coverage: float = 0.0
    focus_missing: list[str] = field(default_factory=list)
    sufficient: bool = False
    abstained: bool = False
    verdict: Verdict = Verdict.PASS
    reason: str = ""

    @property
    def support_rate(self) -> float:
        if not self.claims:
            return 0.0
        return sum(1 for claim in self.claims if claim.supported) / len(self.claims)

    @property
    def unsupported_claims(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.supported]

    def to_meta(self) -> dict[str, object]:
        return {
            "grounding_claim_count": len(self.claims),
            "grounding_support_rate": round(self.support_rate, 4),
            "grounding_unsupported_count": len(self.unsupported_claims),
            "grounding_coverage": round(self.coverage, 4),
            "grounding_focus_coverage": round(self.focus_coverage, 4),
            "grounding_focus_missing": list(self.focus_missing[:8]),
            "grounding_sufficient": self.sufficient,
            "grounding_abstained": self.abstained,
            "grounding_suppressed_sentences": self.suppressed_sentences,
            "grounding_verdict": self.verdict.value,
            "grounding_reason": self.reason,
        }


def is_abstention(answer: str) -> bool:
    """答案是否构成一次明确拒答（仅看文本）。"""

    return any(marker in answer for marker in ABSTENTION_MARKERS)


# 语义上等价于"没有给出断言"的运行结果。``clarify``（请求澄清）
# 与 ``insufficient_evidence``（明确拒答）都应当计入"正确拒答"，
# 否则系统越是老老实实请求澄清，指标反而越难看——这会激励错误的行为。
ABSTENTION_OUTCOMES: frozenset[str] = frozenset({"no_knowledge", "insufficient_evidence", "clarify"})


def abstained_from(answer: str, outcome: str = "") -> bool:
    """综合文本与运行结果判断是否属于"未断言"。"""

    return outcome in ABSTENTION_OUTCOMES or is_abstention(answer)


def _content_tokens(text: str) -> set[str]:
    """有实义的 token。复用统一的停用词表，避免各处口径不一致。"""

    return content_tokens(text)


def extract_claims(answer: str) -> list[Claim]:
    """把答案拆成 claim，并抽出各自引用的证据编号。"""

    claims: list[Claim] = []
    for raw in _SENTENCE_SPLIT.split(answer):
        sentence = raw.strip()
        if not sentence:
            continue
        citations = [int(item) for item in _CITATION.findall(sentence)]
        claims.append(Claim(text=sentence, citations=citations))
    return claims


@dataclass(slots=True)
class SufficiencyEstimate:
    """充分性估计的两个维度。

    **为什么需要两个维度。** 只看向量化的"加权覆盖率"会漏掉一类致命情况：

    政策类语料里 "建设""标准" 几乎每篇都出现，而 "火星""殖民" 一次都没有。
    问题"火星殖民基地的建设标准是什么"的覆盖率会被前半段撑到 0.6 以上，
    从而绕过拒答门控开开心心地编造一个答案。

    原因是覆盖率把所有词一视同仁：命中常见词和命中稀有词得到同样的信用。
    所以再引入一个维度——**焦点词覆盖率**：

    - ``coverage``：全部实义词的加权覆盖（回答"材料大致对题吗"）
    - ``focus_coverage``：多字词/拉丁词这类**具体概念词**的未加权覆盖
      （回答"问题真正在问的那个东西找到了吗"）

    两者必须同时达标才算证据充分。``focus_coverage`` 用未加权计数是刻意的：
    加权会让跨词边界的伪双字组（例如"建**设标**准"切出的 "设标"，
    在语料中 df=1 因而 IDF 极高）主导分子，反而把噪声放大。
    """

    coverage: float = 0.0
    focus_coverage: float = 0.0
    focus_total: int = 0
    focus_covered: int = 0
    focus_missing: list[str] = field(default_factory=list)

    def sufficient(self, *, coverage_threshold: float, focus_threshold: float) -> bool:
        if self.focus_total == 0:
            # 查询里没有具体概念词（例如"这个指南说了什么"），
            # 这种情况本来就该走澄清而不是判定充分。
            return False
        return self.coverage >= coverage_threshold and self.focus_coverage >= focus_threshold

    def to_meta(self) -> dict[str, object]:
        return {
            "sufficiency_coverage": round(self.coverage, 4),
            "sufficiency_focus_coverage": round(self.focus_coverage, 4),
            "sufficiency_focus_missing": list(self.focus_missing[:8]),
        }


def estimate_sufficiency(
    question: str,
    units: list[EvidenceUnit],
    *,
    document_frequency: dict[str, int] | None = None,
    corpus_size: int = 0,
    focus_limit: int = 12,
) -> SufficiencyEstimate:
    """估计证据对问题信息需求的覆盖情况。

    **必须做稀有度加权**，否则政策语料里 "建设""标准" 这类高频词
    会把一个完全无法回答的问题撑到过半覆盖率。

    这里对 :func:`scout.rag.bm25.idf_weight` 采用函数内延迟导入，
    是为了打断 ``rag.pipeline → verify.grounding → rag.bm25`` 的循环依赖。
    """

    # 延迟导入以打断循环依赖：rag 包在 __init__ 里就会 import pipeline。
    from ..rag.bm25 import idf_weight

    query_tokens = _content_tokens(question)
    if not query_tokens:
        return SufficiencyEstimate()

    evidence_tokens: set[str] = set()
    for unit in units:
        evidence_tokens |= _content_tokens(unit.context_text)

    def weight(token: str) -> float:
        return float(max(len(token), 1)) * idf_weight(document_frequency, corpus_size, token)

    total = sum(weight(token) for token in query_tokens)
    coverage = (
        sum(weight(token) for token in query_tokens if token in evidence_tokens) / total
        if total > 0
        else 0.0
    )

    # 焦点词：多字词与拉丁词（单字太粗，"火"能命中不代表"火星"存在）。
    focus_tokens = sorted(
        (token for token in query_tokens if len(token) >= 2),
        key=lambda token: (-len(token), token),
    )[:focus_limit]
    focus_covered = [token for token in focus_tokens if token in evidence_tokens]
    focus_missing = [token for token in focus_tokens if token not in evidence_tokens]

    return SufficiencyEstimate(
        coverage=coverage,
        focus_coverage=(len(focus_covered) / len(focus_tokens)) if focus_tokens else 0.0,
        focus_total=len(focus_tokens),
        focus_covered=len(focus_covered),
        focus_missing=focus_missing,
    )


def check_grounding(
    answer: str,
    units: list[EvidenceUnit],
    *,
    min_support: float = 0.3,
) -> GroundingReport:
    """逐 claim 校验归因支撑度。"""

    report = GroundingReport()
    evidence_blocks = {index: unit.context_text for index, unit in enumerate(units, start=1)}
    report.claims = extract_claims(answer)
    for claim in report.claims:
        if not claim.citations:
            claim.supported = _implicitly_supported(claim.text, evidence_blocks)
            claim.reason = "no_citation" if claim.supported else "no_citation_no_overlap"
            continue
        support = "".join(evidence_blocks.get(number, "") for number in claim.citations)
        if not support:
            claim.supported = False
            claim.reason = "citation_out_of_range"
            continue
        claim_tokens = _content_tokens(claim.text)
        if not claim_tokens:
            claim.supported = True
            claim.support_score = 1.0
            continue
        overlap = len(claim_tokens & _content_tokens(support)) / len(claim_tokens)
        claim.support_score = overlap
        claim.supported = overlap >= min_support
        claim.reason = "supported" if claim.supported else "low_overlap"

    # 归因门控：把无支撑的长句删掉，避免它们进入最终输出。
    if report.unsupported_claims:
        gated, suppressed = attribution_gate(answer, evidence_blocks, min_overlap=min_support)
        report.suppressed_sentences = suppressed
        if suppressed:
            report.verdict = Verdict.REGENERATE
            report.reason = "unsupported_claims_suppressed"
    return report


def _implicitly_supported(sentence: str, evidence_blocks: dict[int, str]) -> bool:
    """无引用编号的句子：短句放行，长句要求与任一证据有足够重叠。"""

    if len(sentence) <= 12:
        return True
    tokens = _content_tokens(sentence)
    if not tokens:
        return True
    for block in evidence_blocks.values():
        if len(tokens & _content_tokens(block)) / len(tokens) >= 0.5:
            return True
    return False


def verify_answer(
    question: str,
    answer: str,
    units: list[EvidenceUnit],
    *,
    settings: VerifySettings | None = None,
    document_frequency: dict[str, int] | None = None,
    corpus_size: int = 0,
) -> GroundingReport:
    """完整的校验与决策。

    决策优先级（顺序不能换）：

    1. 没有证据 → 拒答
    2. 覆盖不足 → 拒答（哪怕模型已经给了一个自信的答案）
    3. 答案自己就在拒答 → 通过（这是期望行为，不是失败）
    4. 有 claim 不被支撑 → 重生成
    5. 否则通过
    """

    effective = settings or VerifySettings()
    report = check_grounding(answer, units, min_support=0.3)
    estimate = estimate_sufficiency(
        question,
        units,
        document_frequency=document_frequency,
        corpus_size=corpus_size,
    )
    report.coverage = estimate.coverage
    report.focus_coverage = estimate.focus_coverage
    report.focus_missing = list(estimate.focus_missing)
    report.sufficient = estimate.sufficient(
        coverage_threshold=effective.sufficiency_min_coverage,
        focus_threshold=effective.sufficiency_min_focus_coverage,
    )
    report.abstained = is_abstention(answer)

    if not units:
        report.verdict = Verdict.ABSTAIN
        report.reason = "no_evidence"
        return report
    if not report.sufficient:
        report.verdict = Verdict.ABSTAIN
        report.reason = (
            f"coverage_below_threshold:{report.coverage:.3f}"
            if report.coverage < effective.sufficiency_min_coverage
            else f"focus_coverage_below_threshold:{report.focus_coverage:.3f}"
        )
        return report
    if report.abstained:
        report.verdict = Verdict.PASS
        report.reason = "model_abstained_as_expected"
        return report
    if report.verdict is Verdict.REGENERATE:
        return report
    report.verdict = Verdict.PASS
    report.reason = "grounded"
    return report


__all__ = [
    "ABSTENTION_MARKERS",
    "ABSTENTION_OUTCOMES",
    "Claim",
    "GroundingReport",
    "SufficiencyEstimate",
    "Verdict",
    "abstained_from",
    "check_grounding",
    "estimate_sufficiency",
    "extract_claims",
    "is_abstention",
    "verify_answer",
]
