"""裁判 vs 归因校验：两种评分放在一起看，而不是二选一。

**为什么要做这个对照，而不是直接换成 LLM 裁判。**
两种评分器的失效模式完全不同：

| | 归因校验（词法） | LLM 裁判（语义） |
|---|---|---|
| 依据 | 答案句子能否在证据里找到支撑 | 语义上是否切题、忠实、简洁 |
| 成本 | 零（纯计算） | 一次模型调用 |
| 确定性 | 完全可复现 | 有随机性，需要去偏 |
| 盲区 | 换个说法就判不出支撑 | 位置/长度偏见、可能自洽地判错 |

**它们不一致的样本才是最有价值的东西。**
一致 → 互证；不一致 → 要么 rubric 有问题、要么阈值有问题，
这正是需要人去校准的地方。所以本模块的核心产物不是"谁更准"，
而是 :attr:`ComparisonResult.disagreements` 这份清单。

注意：这里刻意**不做"哪个更好"的结论**。要下那个结论需要人工标注做锚点，
而当前没有——给一个没有锚点的结论，等于用裁判给自己打分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..llm.base import LLMClient
from ..rag.chunking import Chunk
from ..rag.merge import EvidenceUnit
from ..verify.grounding import Verdict, verify_answer
from .judge import JudgeCalibration, LLMJudge


@dataclass(slots=True)
class ComparisonCase:
    """对照样本：问题 + 答案 + 证据 + 这条答案"应该"是什么样。

    ``expect`` 是**人工给的期望标签**（可选）：有了它才能算"谁判对了"。
    没有它的样本只能用来观察一致率，不能用来评级。
    """

    question: str
    answer: str
    evidence: str
    expect: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "expect": self.expect,
        }


@dataclass(slots=True)
class ComparisonResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    agreement: float = 0.0
    n: int = 0
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement": round(self.agreement, 4),
            "calibration": dict(self.calibration),
            "rows": self.rows,
            "disagreements": self.disagreements,
        }


def _units_from_text(evidence: str) -> list[EvidenceUnit]:
    """把一段纯文本包装成证据单元，供词法归因校验使用。

    这一步是把"对照"做起来的必要代价：``verify_answer`` 期望的是检索结果对象，
    而对照场景里我们手上只有文本。**不因此就去改动 verify_answer 的签名**——
    生产路径的类型契约不该为了一个对照脚本而放宽。
    """

    chunk = Chunk(
        chunk_id="cmp#0",
        document_id="cmp",
        document_version="v1",
        filename="comparison",
        level=3,
        text=evidence,
        index=0,
    )
    return [EvidenceUnit(chunk=chunk, score=1.0)]


def compare_judges(
    client: LLMClient,
    cases: Sequence[ComparisonCase],
    *,
    judge: LLMJudge | None = None,
    verdict_threshold: float = 0.6,
) -> ComparisonResult:
    """对每条样本同时跑词法归因与 LLM 裁判，输出一致率与分歧清单。

    :param verdict_threshold: 裁判分（0-5 归一化后）达到多少算"通过"。
        取 0.6 对应 rubric 里的"及格线"（3/5）——**这个阈值要显式写出来**，
        因为它直接决定一致率，藏在代码里的话一致率就没法被解释。
    """

    judge = judge or LLMJudge(client)
    result = ComparisonResult()
    calibration = JudgeCalibration()

    for case in cases:
        units = _units_from_text(case.evidence)
        report = verify_answer(case.question, case.answer, units)
        lexical_pass = report.verdict is Verdict.PASS

        judge_verdict = judge.score(case.question, case.answer, case.evidence)
        judge_pass = (judge_verdict.overall / 5.0) >= verdict_threshold

        agree = lexical_pass == judge_pass
        row = {
            "question": case.question,
            "verdict": report.verdict.value,
            "lexical_pass": lexical_pass,
            "judge_overall": round(judge_verdict.overall, 2),
            "judge_pass": judge_pass,
            "agree": agree,
            "support_rate": round(report.support_rate, 3),
            "judge_rationale": judge_verdict.rationale,
            "expect": case.expect,
        }
        result.rows.append(row)
        if not agree:
            result.disagreements.append(row)
        if case.expect:
            # 有期望标签时才做校准：把裁判分与"人工认为该通过/不通过"对齐
            calibration.add(judge_verdict.overall, 5.0 if case.expect == "pass" else 1.0)

    result.n = len(result.rows)
    result.agreement = (
        sum(1 for row in result.rows if row["agree"]) / result.n if result.n else 0.0
    )
    result.calibration = calibration.to_dict()
    return result


def demo_cases() -> list[ComparisonCase]:
    """内置对照样本：四种典型配对，覆盖两种评分器的差异区间。"""

    evidence = (
        "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"
        "到2027年，新制定云计算国家标准和行业标准30项以上。"
    )
    return [
        ComparisonCase(
            question="云计算标准体系结构包括哪几个部分？",
            answer="包括基础、技术、服务、应用、管理和安全六个部分。",
            evidence=evidence,
            expect="pass",
        ),
        ComparisonCase(
            question="到2027年有什么量化目标？",
            answer="新制定国家标准和行业标准30项以上。",
            evidence=evidence,
            expect="pass",
        ),
        ComparisonCase(
            question="云计算标准体系结构包括哪几个部分？",
            answer="包括六个部分，分别是基础、技术、服务、应用、管理和安全。",  # 同义改写
            evidence=evidence,
            expect="pass",
        ),
        ComparisonCase(
            question="到2027年有什么量化目标？",
            answer="到2027年要制定1000项标准，并覆盖全部行业。",  # 编造
            evidence=evidence,
            expect="fail",
        ),
    ]


__all__ = ["ComparisonCase", "ComparisonResult", "compare_judges", "demo_cases"]
