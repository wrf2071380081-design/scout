"""数据飞轮：把线上 trace 变成下一轮的评测集与偏好数据。

**为什么"自进化"必须先有飞轮，而不是先有微调。**
很多人把"自进化"理解成"自动微调模型"。但微调需要高质量样本，
而高质量样本从哪来？——**从线上真实失败里挖**。
跳过飞轮直接微调，等于用一个没有反馈闭环的系统去改进自己。

飞轮的四个环节，缺一环就转不起来：

1. **采集**：线上每次问答都留下结构化 trace（已有：``scout.trace``）。
2. **挖掘**：从 trace 里自动挑出"值得进评测集"的样本——
   拒答的、重生成过的、低支撑率的、用户点踩的。
   :class:`FailureMiner` 做这件事，输出 :class:`CandidateCase`。
3. **标注**：候选不能直接进评测集（模型判错的样本里混着"这题本来就不该答"）。
   进人工/裁判复核队列，通过后才入库。
4. **回归**：新样本补齐后重跑评测，用于判断"这次改动到底有没有更好"。

**一个必须写下来的纪律：飞轮只允许让评测集变大，不允许改小。**
删样本（尤其是删掉自己答不好的样本）等于自己给自己放水，
是这类系统里最常见的自欺。所以 :class:`FlywheelLedger` 记录每条样本的
加入与永不删除，只允许"标记为作废并说明原因"。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class FailureReason(str, Enum):
    """样本进候选池的理由。**理由是分层的**——它决定了复核时该看什么。"""

    ABSTAINED = "abstained"          # 该答没答（门控过严）
    REGENERATED = "regenerated"      # 一次生成不达标，重生成过（质量边缘）
    LOW_SUPPORT = "low_support"      # 支撑率低（疑似幻觉）
    USER_NEGATIVE = "user_negative"  # 用户点踩（最强信号）
    EMPTY_EVIDENCE = "empty_evidence"  # 检索为空（可能知识库缺内容）
    ERROR = "error"                  # 运行期异常


@dataclass(slots=True)
class CandidateCase:
    """候选评测样本。"""

    case_id: str
    question: str
    reason: FailureReason
    outcome: str = ""
    answer: str = ""
    support_rate: float = 0.0
    retrieved: int = 0
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "reason": self.reason.value,
            "outcome": self.outcome,
            "answer": self.answer[:500],
            "support_rate": round(self.support_rate, 4),
            "retrieved": self.retrieved,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateCase:
        return cls(
            case_id=str(payload.get("case_id", "")),
            question=str(payload.get("question", "")),
            reason=FailureReason(str(payload.get("reason", "error"))),
            outcome=str(payload.get("outcome", "")),
            answer=str(payload.get("answer", "")),
            support_rate=float(payload.get("support_rate", 0.0) or 0.0),
            retrieved=int(payload.get("retrieved", 0) or 0),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            notes=str(payload.get("notes", "")),
        )


def candidate_id(question: str) -> str:
    """候选编号由问题文本派生：同一问题反复失败只算一条。

    这条规则避免了飞轮最常见的膨胀方式——同一道难题每天失败一次，
    一个月后评测集里全是它。**去重按语义，不按时间。**
    """

    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:12]
    return f"cand-{digest}"


def _field(item: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """按多个候选键名取值。

    **为什么需要它（一次真实的静默失败）。** 挖掘器最初只认 ``retrieved``
    和 ``support_rate`` 两个键，而观测日志写的是 ``retrieved_sources``
    与 ``answer_support_rate``。字段名不一致不会有任何报错：
    取值全为 None → 检索数量当 0 → 每条观测都被判成"空检索"并被跳过。
    结果是**飞轮转了一圈挖出 0 条候选，而报告上一切正常**。

    这就是"跑一圈"不可替代的原因：单测里两边各自都对，
    只有端到端接起来才会暴露字段错配。

    兼容多个键名的代价是可接受的；真正的风险是**只认一个名字还不报错**。
    """

    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return default


def _as_count(value: Any) -> int:
    """把"数量"或"列表"统一成 int。"""

    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class MineStats:
    scanned: int = 0
    mined: int = 0
    skipped_no_retrieval: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "mined": self.mined,
            "skipped_no_retrieval": self.skipped_no_retrieval,
            "by_reason": dict(self.by_reason),
        }


class FailureMiner:
    """从痕迹里挖候选样本。

    :param support_floor: 支撑率低于此值视为"疑似幻觉"。
    :param require_retrieval: 只有"检索到了东西却答错"才更值得进评测集；
        检索为空通常意味着知识库缺内容，那是数据问题，不该混进"系统能力"的评测集。
    """

    def __init__(
        self,
        *,
        support_floor: float = 0.6,
        require_retrieval: bool = True,
    ) -> None:
        self.support_floor = support_floor
        self.require_retrieval = require_retrieval
        self.stats = MineStats()

    def mine(self, observations: Iterable[dict[str, Any]]) -> list[CandidateCase]:
        """从观测记录里挖候选。

        ``observations`` 接受**观测日志的原样记录**（``CaseObservation.to_dict()``），
        字段名通过 :func:`_field` 做多键兼容——见它上面的说明，
        这里曾经因为键名不一致而静默挖出 0 条。
        """

        found: list[CandidateCase] = []
        seen: set[str] = set()
        for item in observations:
            self.stats.scanned += 1
            question = str(_field(item, "question", default="") or "").strip()
            if not question:
                continue
            reason = self._classify(item)
            if reason is None:
                continue
            if self.require_retrieval and reason is FailureReason.EMPTY_EVIDENCE:
                self.stats.skipped_no_retrieval += 1
                continue
            identifier = candidate_id(question)
            if identifier in seen:
                continue
            seen.add(identifier)
            found.append(
                CandidateCase(
                    case_id=identifier,
                    question=question,
                    reason=reason,
                    outcome=str(_field(item, "outcome", default="") or ""),
                    answer=str(_field(item, "answer", default="") or ""),
                    support_rate=float(_field(item, "answer_support_rate", "support_rate", default=0.0) or 0.0),
                    retrieved=_as_count(
                        _field(item, "retrieved", "retrieved_sources", "retrieved_chunk_ids")
                    ),
                    notes=str(_field(item, "notes", default="") or ""),
                )
            )
            self.stats.mined += 1
            self.stats.by_reason[reason.value] = self.stats.by_reason.get(reason.value, 0) + 1
        return found

    def _classify(self, item: dict[str, Any]) -> FailureReason | None:
        """判定这条观测"为什么值得进候选池"。

        判据顺序与理由（每一步的顺序都不是随意的）：

        1. **真正的运行异常**：只看 ``failed`` 与显式的 ``error`` 字段。
           **刻意不看 ``error_code``**——在本项目里 error_code 是"类型化结果"的标识，
           拒答（``insufficient_evidence``）本身就带 error_code。
           把它当异常信号，会让**所有拒答都被误判成崩溃**，而报告上看起来一切正常。
        2. 用户点踩：最强信号，优先于所有结构性判断。
        3. 检索为空：数据覆盖问题，默认不进"系统能力"评测集。
        4. 重生成过 / 支撑率低 / 拒答：按从强到弱排列。

        ``support_rate`` 为 0 时**不判低支撑**：拒答样本没有答案可校验，
        0 在这里的含义是"不适用"而不是"零支撑"。把两者混同会把拒答误报成幻觉。
        """

        if _field(item, "failed") is True or _field(item, "error"):
            return FailureReason.ERROR
        if _field(item, "user_feedback") == "negative":
            return FailureReason.USER_NEGATIVE
        retrieved = _as_count(_field(item, "retrieved", "retrieved_sources", "retrieved_chunk_ids"))
        if retrieved == 0:
            return FailureReason.EMPTY_EVIDENCE
        if _as_count(_field(item, "regenerations")) > 0:
            return FailureReason.REGENERATED
        support = float(_field(item, "answer_support_rate", "support_rate", default=0.0) or 0.0)
        if support and support < self.support_floor:
            return FailureReason.LOW_SUPPORT
        outcome = str(_field(item, "outcome", default="") or "")
        if outcome in {"insufficient_evidence", "no_knowledge", "clarify"}:
            return FailureReason.ABSTAINED
        return None


class ReviewQueue:
    """复核队列：候选样本在进评测集之前的必经关口。

    **为什么不能自动入库。** 模型判错的样本里混着"这题本来就不该答"
    （比如知识库里确实没有）。把它当"应该答对"的样本加进评测集，
    会系统性地惩罚一个**行为正确**的系统——门控会被调得越来越松，
    最后变成一个自信胡说八道的系统。所以自动挖掘只产出候选，
    入库必须经过一次显式判断。
    """

    def __init__(self) -> None:
        self.pending: dict[str, CandidateCase] = {}
        self.approved: list[CandidateCase] = []
        self.rejected: list[tuple[CandidateCase, str]] = []

    def enqueue(self, cases: Sequence[CandidateCase]) -> int:
        added = 0
        for case in cases:
            if case.case_id in self.pending or any(
                item.case_id == case.case_id for item in self.approved
            ):
                continue
            self.pending[case.case_id] = case
            added += 1
        return added

    def approve(self, case_id: str) -> CandidateCase | None:
        case = self.pending.pop(case_id, None)
        if case is not None:
            self.approved.append(case)
        return case

    def reject(self, case_id: str, reason: str) -> CandidateCase | None:
        """拒绝必须写原因。**没有原因的拒绝和删除没有区别**——
        三个月后没人知道当初为什么把它挡在外面。"""

        case = self.pending.pop(case_id, None)
        if case is not None:
            self.rejected.append((case, reason))
        return case

    def sample_pending(self, limit: int = 10) -> list[CandidateCase]:
        return list(self.pending.values())[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending": len(self.pending),
            "approved": len(self.approved),
            "rejected": len(self.rejected),
            "rejection_reasons": [reason for _case, reason in self.rejected][:20],
        }


@dataclass(slots=True)
class FlywheelTurn:
    """飞轮转一圈的记录。用来回答"这套东西到底有没有在改进"——
    如果连续几轮的 approve 都是 0，说明挖掘规则挑的东西不对，
    该修的是 :class:`FailureMiner`，不是继续加数据。"""

    turn: int
    mined: int
    approved: int
    rejected: int
    dataset_size: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "mined": self.mined,
            "approved": self.approved,
            "rejected": self.rejected,
            "dataset_size": self.dataset_size,
            "created_at": self.created_at,
        }


class FlywheelLedger:
    """飞轮账本：**只增不减**。"""

    def __init__(self) -> None:
        self.turns: list[FlywheelTurn] = []
        self.dataset_size = 0

    def record(self, *, mined: int, queue: ReviewQueue) -> FlywheelTurn:
        self.dataset_size += len(queue.approved)
        turn = FlywheelTurn(
            turn=len(self.turns) + 1,
            mined=mined,
            approved=len(queue.approved),
            rejected=len(queue.rejected),
            dataset_size=self.dataset_size,
        )
        self.turns.append(turn)
        queue.approved.clear()
        return turn

    def stalled(self, *, window: int = 3) -> bool:
        """连续 window 轮没有新增样本 → 挖掘策略失效，需要人工重定规则。"""

        recent = self.turns[-window:]
        return len(recent) >= window and all(turn.approved == 0 for turn in recent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "dataset_size": self.dataset_size,
            "stalled": self.stalled(),
        }


def export_preference_pairs(
    verdicts: Sequence[tuple[str, str, str]],
    *,
    path: str | None = None,
) -> list[dict[str, str]]:
    """把裁判的 pairwise 结论导成偏好对，供后续对齐使用。

    ``verdicts`` 为 ``(prompt, chosen, rejected)`` 三元组。
    导出成 JSONL 而不是专有格式：**让这份数据不绑定本项目的工具链**——
    数据比代码活得久。
    """

    pairs = [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        for prompt, chosen, rejected in verdicts
        if chosen and rejected and chosen != rejected
    ]
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return pairs


__all__ = [
    "CandidateCase",
    "FailureMiner",
    "FailureReason",
    "FlywheelLedger",
    "FlywheelTurn",
    "MineStats",
    "ReviewQueue",
    "candidate_id",
    "export_preference_pairs",
]
