"""LLM-as-Judge：用模型做评分器，并**正面处理它自身的两类偏见**。

为什么需要它：检索指标（Recall@k）能测有没有找对证据，
但"答案有没有编造""答得切不切题"只能用语义判断。
词法指标（BLEU/ROUGE 那一路）在这类任务上几乎无意义——
答案换个说法意思相同，词重叠却接近零。

**但把模型当裁判有三个已知的坑，本模块逐个处理：**

1. **位置偏见**：pairwise 比较时，裁判倾向选先出现的那一个。
   → :meth:`LLMJudge.compare` 强制**跑两遍、交换顺序、取平均**。
   单遍结果只记录不采信。
2. **长度偏见**：更长的答案显得更"充实"。
   → 分项 rubric 里显式包含 ``conciseness``，并在提示词里写明
   "内容相同的情况下更短的答案得分不低于更长的"。把偏见写进评分标准，
   比事后统计校正更直接。
3. **自我偏好**：裁判会偏爱与自己同源的模型输出。
   → 记录 :attr:`JudgeVerdict.judge_model`，跨模型对比时**必须用第三方模型**
   （本项目用 Kimi 评判 scout 的输出，不做同源自评）。

另有一条纪律：**裁判本身也要被评测**（judge eval）。
:class:`JudgeCalibration` 用少量人工标注做锚点，
把裁判分数与人类分数对齐；对不齐就调 rubric，而不是直接采信分数。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field

from ..errors import ProviderError
from ..llm.base import ChatMessage, LLMClient, LLMRequest, TokenUsage


class _RubricScore(BaseModel):
    """单项评分。"""

    score: int = Field(default=0, ge=0, le=5, description="0-5 分")
    reason: str = Field(default="", description="一句话依据，必须引用答案或材料中的具体内容")


class _JudgeOutput(BaseModel):
    """裁判的结构化输出契约。"""

    faithfulness: _RubricScore = Field(default_factory=_RubricScore, description="忠实度：答案是否只依据给定材料")
    relevance: _RubricScore = Field(default_factory=_RubricScore, description="相关性：是否直接回答了问题")
    conciseness: _RubricScore = Field(default_factory=_RubricScore, description="简洁度：是否有冗余复述")
    verdict: str = Field(default="", description="一句话总评")


class _PairwiseOutput(BaseModel):
    winner: str = Field(default="tie", description="A / B / tie")
    reason: str = Field(default="", description="判定依据")


@dataclass(slots=True)
class JudgeVerdict:
    """一次评分的完整结果。"""

    overall: float
    scores: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    rationale: str = ""
    judge_model: str = ""
    passes: int = 1
    order_swapped: bool = False
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "scores": dict(self.scores),
            "reasons": dict(self.reasons),
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "passes": self.passes,
            "order_swapped": self.order_swapped,
            "parse_failed": self.parse_failed,
        }


RUBRIC_PROMPT = """你在为一个 RAG 问答系统做评分。请严格按下面三个维度打分（0-5 分），只输出 JSON。

评分维度：
1. faithfulness 忠实度：答案中的每个事实是否都能在【材料】中找到依据。
   编造、外推、或引入材料之外的知识，本项不得高于 2 分。
2. relevance 相关性：答案是否直接回应了【问题】。答非所问本项 0-1 分。
3. conciseness 简洁度：是否有与问题无关的冗余复述。
   **注意：在信息量相同的情况下，更短的答案得分不低于更长的答案。**

每个维度必须给出一句话 reason，且必须引用答案或材料中的具体片段——
"看起来不错"这类无依据的评价视为无效评分。

【问题】
{question}

【材料】
{evidence}

【答案】
{answer}
"""

PAIRWISE_PROMPT = """你在比较两个 RAG 系统对同一个问题的回答。只输出 JSON。

判断标准：忠实于材料 > 直接回应问题 > 简洁。三者冲突时按此优先级。

【问题】
{question}

【材料】
{evidence}

【回答 A】
{answer_a}

【回答 B】
{answer_b}
"""


class LLMJudge:
    """基于 rubric 的裁判。

    :param use_pairwise_debias: 是否对 pairwise 比较做双跑换序去偏（强烈建议开启）。
    :param seed: 交换顺序的随机种子。固定种子保证**同一批评测可复现**——
        评测本身不可复现的话，它的结论就没有资格被引用。
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        weights: dict[str, float] | None = None,
        use_pairwise_debias: bool = True,
        seed: int = 7,
        max_evidence_chars: int = 4000,
    ) -> None:
        self.llm = llm
        self.weights = weights or {"faithfulness": 0.5, "relevance": 0.35, "conciseness": 0.15}
        self.use_pairwise_debias = use_pairwise_debias
        self.random = random.Random(seed)
        self.max_evidence_chars = max_evidence_chars
        self.usage = TokenUsage()

    # —— 单答案评分 ——

    def score(self, question: str, answer: str, evidence: str) -> JudgeVerdict:
        prompt = RUBRIC_PROMPT.format(
            question=question,
            evidence=(evidence or "（无材料）")[: self.max_evidence_chars],
            answer=answer or "（空答案）",
        )
        try:
            response = self.llm.complete(
                LLMRequest(
                    messages=[ChatMessage(role="user", content=prompt)],
                    schema=_JudgeOutput,
                    task="judge",
                    context={"question": question},
                )
            )
        except ProviderError:
            # 裁判失败必须显式暴露：不能悄悄返回 0 分，
            # 那会把"评分器故障"混进"系统质量差"，污染整份评测结论。
            return JudgeVerdict(overall=0.0, judge_model=self._model(), parse_failed=True,
                                rationale="裁判调用失败")
        if response.usage is not None:
            self.usage_input_add(response.usage.input_tokens)
            self.usage_output_add(response.usage.output_tokens)

        try:
            parsed = response.parse(_JudgeOutput)
        except Exception:  # noqa: BLE001 - 解析失败按"无效评分"记录，而不是当 0 分
            return JudgeVerdict(overall=0.0, judge_model=self._model(), parse_failed=True,
                                rationale="裁判输出无法解析为结构化结果")

        scores = {
            "faithfulness": parsed.faithfulness.score,
            "relevance": parsed.relevance.score,
            "conciseness": parsed.conciseness.score,
        }
        reasons = {
            "faithfulness": parsed.faithfulness.reason,
            "relevance": parsed.relevance.reason,
            "conciseness": parsed.conciseness.reason,
        }
        total_weight = sum(self.weights.get(name, 0.0) for name in scores) or 1.0
        overall = sum(scores[name] * self.weights.get(name, 0.0) for name in scores) / total_weight
        return JudgeVerdict(
            overall=overall,
            scores=scores,
            reasons=reasons,
            rationale=parsed.verdict,
            judge_model=self._model(),
        )

    # —— 双答案比较（含位置去偏） ——

    def compare(self, question: str, answer_a: str, answer_b: str, evidence: str) -> dict[str, Any]:
        """比较两个答案。**双跑换序**，位置偏见因此被平均掉。"""

        first = self._one_comparison(question, answer_a, answer_b, evidence, swapped=False)
        if not self.use_pairwise_debias:
            return {"winner": first["winner"], "reason": first["reason"], "passes": 1, "swapped": False}

        second = self._one_comparison(question, answer_a, answer_b, evidence, swapped=True)
        # 第二次把 A/B 位置换过来问，所以要把它的结论映射回原始语义：
        # swapped 之后模型看到的"左边"其实是 B。
        mapped = {"A": "B", "B": "A", "tie": "tie"}.get(second["winner"], "tie")
        votes = [first["winner"], mapped]
        winner = votes[0] if votes[0] == votes[1] else "tie"
        return {
            "winner": winner,
            "reason": first["reason"],
            "second_reason": second["reason"],
            "passes": 2,
            "swapped": True,
            "inconsistent": votes[0] != votes[1],
        }

    def _one_comparison(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        evidence: str,
        *,
        swapped: bool,
    ) -> dict[str, str]:
        left, right = (answer_b, answer_a) if swapped else (answer_a, answer_b)
        prompt = PAIRWISE_PROMPT.format(
            question=question,
            evidence=(evidence or "（无材料）")[: self.max_evidence_chars],
            answer_a=left,
            answer_b=right,
        )
        try:
            response = self.llm.complete(
                LLMRequest(
                    messages=[ChatMessage(role="user", content=prompt)],
                    schema=_PairwiseOutput,
                    task="judge_pairwise",
                    context={"question": question},
                )
            )
            parsed = response.parse(_PairwiseOutput)
        except Exception:  # noqa: BLE001 - 比较失败按平局处理，但必须留痕（inconsistent/reason）
            return {"winner": "tie", "reason": "裁判调用或解析失败"}
        winner = (parsed.winner or "tie").strip().upper()
        if winner not in {"A", "B", "TIE"}:
            winner = "TIE"
        return {"winner": winner if winner != "TIE" else "tie", "reason": parsed.reason}

    # —— 用量 ——

    def usage_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }

    def usage_input_add(self, value: int) -> None:
        self.usage.input_tokens += int(value or 0)

    def usage_output_add(self, value: int) -> None:
        self.usage.output_tokens += int(value or 0)

    def _model(self) -> str:
        return getattr(self.llm, "model_name", "unknown")


@dataclass(slots=True)
class JudgeCalibration:
    """裁判校准：拿人工锚点对齐裁判分数。

    做法只有一步但很关键：**算裁判与人评的偏差，并如实展示**。
    偏差大于一格（1 分）就说明 rubric 需要重写，
    这时不能拿裁判分数去做模块级决策——不然你优化的是裁判的偏好，不是系统质量。
    """

    pairs: list[tuple[float, float]] = field(default_factory=list)

    def add(self, judge_score: float, human_score: float) -> None:
        self.pairs.append((float(judge_score), float(human_score)))

    @property
    def mean_bias(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(judge - human for judge, human in self.pairs) / len(self.pairs)

    def agreement(self, tolerance: float = 1.0) -> float:
        if not self.pairs:
            return 0.0
        hits = sum(1 for judge, human in self.pairs if abs(judge - human) <= tolerance)
        return hits / len(self.pairs)

    def trustable(self) -> bool:
        """是否可用作决策依据。**样本少于 10 条一律不可信**。"""

        return len(self.pairs) >= 10 and self.agreement() >= 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.pairs),
            "mean_bias": round(self.mean_bias, 3),
            "agreement_at_1": round(self.agreement(), 3),
            "trustable": self.trustable(),
        }


__all__ = ["JudgeCalibration", "LLMJudge", "JudgeVerdict"]
