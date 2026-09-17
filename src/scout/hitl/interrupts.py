"""中断点定义、风险分级、人工决策与副作用账本。

**先说这套设计要解决的真问题。**

"给 Agent 加个人工确认"听起来很简单——在工具调用前弹个框问一下就是了。
但一旦要求**中断可以跨进程、跨小时、跨重启存活**，问题立刻变形：

1. **不是所有操作都值得打扰人。** 查一下知识库也要审批的 Agent 没人会用。
   所以需要**风险分级**，而不是一刀切的确认机制。
2. **恢复时不能重复执行副作用。** 这是整件事里最难、也最容易被忽略的一点。
   运行在"已发出邮件"之后、"写下一句话"之前中断，恢复时如果按朴素方式重放，
   邮件会被再发一次。对外部世界来说，这不是 bug，这是事故。
3. **人可能会改参数再批准。** "批准，但把金额从 100 万改成 10 万"是真实需求，
   不是边角场景。审批结果必须能携带参数修改。
4. **没人审批时怎么办。** 审批请求可能永远等不到人。此时必须有明确的**失效策略**，
   而且默认必须是 fail-closed（安全侧），不能是"没人管就放行"。

本模块逐条对应：:class:`InterruptPolicy` 解决 1，:class:`SideEffectLedger`
解决 2，:class:`HumanDecision` 解决 3，:class:`TimeoutPolicy` 解决 4。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence


class RiskLevel(str, Enum):
    """操作风险等级。决定"要不要打扰人、打扰到什么程度"。"""

    SAFE = "safe"
    """只读或无副作用：自动放行，不产生任何中断。"""

    CONFIRM = "confirm"
    """有副作用但可撤销：需要确认，但可以配置为在可信环境自动批准。"""

    REQUIRED = "required"
    """不可逆或高风险：必须人工审批，不可自动放行。"""


_RANK = {RiskLevel.SAFE: 0, RiskLevel.CONFIRM: 1, RiskLevel.REQUIRED: 2}


def escalate(current: RiskLevel, suggested: RiskLevel) -> RiskLevel:
    """取两者中更高的一级。升级容易、降级需要显式声明，是刻意的方向性约束。"""

    return current if _RANK[current] >= _RANK[suggested] else suggested


@dataclass(slots=True)
class RiskAssessment:
    """一次风险评估的结果。带原因，因为审批人要看到"为什么找我"。"""

    level: RiskLevel
    reason: str = ""
    escalations: list[str] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return self.level is not RiskLevel.SAFE

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "reason": self.reason,
            "escalations": list(self.escalations),
        }


# 参数级升级规则：默认认为"工具是安全的"，直到参数告诉我们不是。
Escalator = Callable[[str, Mapping[str, Any]], str | None]

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("手机号", re.compile(r"\b1[3-9]\d{9}\b")),
    ("身份证号", re.compile(r"\b\d{17}[\dXx]\b")),
    ("银行卡号", re.compile(r"\b\d{16,19}\b")),
    ("邮箱地址", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
)

_EXTERNAL_HINTS = ("http://", "https://", "@", "外部", "第三方", "客户", "全体")
_DESTRUCTIVE_HINTS = ("删除", "清空", "drop", "truncate", "delete", "revoke", "下线")
_BULK_THRESHOLD_KEYS = ("count", "quantity", "amount", "金额", "数量", "条数", "人数", "user_count")


def pii_escalator(_name: str, arguments: Mapping[str, Any]) -> str | None:
    """参数里出现个人信息 → 至少升级到 CONFIRM。

    这类判断放在"参数层"而不是"工具层"，因为同一个工具既可以传公开数据
    也可以传个人信息——按工具名分级会漏掉后者。
    """

    blob = json.dumps(arguments, ensure_ascii=False, default=str)
    for label, pattern in _PII_PATTERNS:
        if pattern.search(blob):
            return f"参数中包含疑似{label}"
    return None


def external_recipient_escalator(_name: str, arguments: Mapping[str, Any]) -> str | None:
    """影响到外部的操作 → REQUIRED。内部操作出错可以道歉，对外发送出错无法收回。"""

    blob = json.dumps(arguments, ensure_ascii=False, default=str)
    if any(hint in blob for hint in _EXTERNAL_HINTS):
        return "参数指向外部收件人或外部地址"
    return None


def destructive_escalator(name: str, arguments: Mapping[str, Any]) -> str | None:
    """不可逆语义 → REQUIRED。"""

    blob = f"{name} {json.dumps(arguments, ensure_ascii=False, default=str)}".lower()
    for hint in _DESTRUCTIVE_HINTS:
        if hint in blob:
            return f"命中不可逆操作语义「{hint}」"
    return None


def bulk_escalator(_name: str, arguments: Mapping[str, Any]) -> str | None:
    """批量操作 → 至少 CONFIRM。单条出错是错误，批量出错是事故。"""

    for key in _BULK_THRESHOLD_KEYS:
        value = arguments.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 10:
            return f"批量操作（{key}={value}）"
    return None


DEFAULT_ESCALATORS: tuple[Escalator, ...] = (
    pii_escalator,
    external_recipient_escalator,
    destructive_escalator,
    bulk_escalator,
)


@dataclass(slots=True)
class ToolRiskProfile:
    """一个工具的基础风险画像。"""

    level: RiskLevel = RiskLevel.CONFIRM
    note: str = ""
    auto_approve: bool = False
    """可信环境下允许把 CONFIRM 降级为自动放行。
    REQUIRED 不受此开关影响——这是刻意的：不可逆操作不应该存在"免审开关"。"""


class InterruptPolicy:
    """中断策略。**声明式**，不是散落在工具实现里的 if。

    这样做的好处是"系统什么时候会打扰人"这件事可以被单独审查、单独测试、
    单独向合规解释——而如果中断条件散落在几十个工具里，没人能回答这个问题。

    :param profiles: 工具名 → 风险画像。未登记的工具按 ``default_profile`` 处理
        （默认 CONFIRM，即"未知的就有副作用"，fail-closed）。
    """

    def __init__(
        self,
        profiles: Mapping[str, ToolRiskProfile] | None = None,
        *,
        default_profile: ToolRiskProfile | None = None,
        escalators: Sequence[Escalator] | None = None,
    ) -> None:
        self.profiles: dict[str, ToolRiskProfile] = dict(profiles or {})
        self.default_profile = default_profile or ToolRiskProfile(
            level=RiskLevel.CONFIRM, note="未登记工具，按有副作用处理"
        )
        self.escalators: tuple[Escalator, ...] = tuple(
            DEFAULT_ESCALATORS if escalators is None else escalators
        )

    def register(self, name: str, profile: ToolRiskProfile) -> None:
        self.profiles[name] = profile

    def profile_for(self, name: str) -> ToolRiskProfile:
        return self.profiles.get(name, self.default_profile)

    def assess(self, name: str, arguments: Mapping[str, Any]) -> RiskAssessment:
        """评估一次调用的风险等级。"""

        profile = self.profile_for(name)
        level = profile.level
        escalations: list[str] = []

        for escalate_rule in self.escalators:
            reason = escalate_rule(name, arguments)
            if reason:
                escalations.append(reason)
                suspected = (
                    RiskLevel.REQUIRED
                    if escalate_rule in (external_recipient_escalator, destructive_escalator)
                    else RiskLevel.CONFIRM
                )
                level = escalate(level, suspected)

        if level is RiskLevel.CONFIRM and profile.auto_approve:
            return RiskAssessment(
                level=RiskLevel.SAFE,
                reason=f"CONFIRM 已按可信环境策略自动放行（{profile.note or name}）",
                escalations=escalations,
            )

        reason = profile.note or f"{name} 的风险等级为 {profile.level.value}"
        if escalations:
            reason = "；".join([reason, *escalations])
        return RiskAssessment(level=level, reason=reason, escalations=escalations)


# —— 中断请求与人工决策 ——


@dataclass(slots=True)
class InterruptRequest:
    """一次待处理的审批请求。

    这是要交给外部（前端 / 工单系统 / 审批人）的对象，
    所以它必须能安全地序列化并展示——**不包含模型上下文全文**，
    只包含"审批人做判断真正需要的信息"。
    """

    request_id: str
    run_id: str
    step: int
    call_index: int
    tool_name: str
    arguments: dict[str, Any]
    assessment: RiskAssessment
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None

    @property
    def expired(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        moment = time.time() if now is None else now
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "step": self.step,
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "risk_level": self.assessment.level.value,
            "risk_reason": self.assessment.reason,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "expired": self.deadline is not None and moment > self.deadline,
        }


@dataclass(slots=True)
class HumanDecision:
    """人工审批结果。

    ``edited_arguments`` 是刻意设计的：真实审批里"改一下再批准"远比
    "批准/拒绝"二元选择常见。支持它意味着审批不只是个开关，
    而是**一个可以修正模型输出的介入点**。
    """

    request_id: str
    approved: bool
    edited_arguments: dict[str, Any] | None = None
    comment: str = ""
    decided_by: str = "unknown"
    decided_at: float = field(default_factory=time.time)
    source: str = "human"
    """``human`` 表示真人；``timeout_policy`` 表示超时策略的自动决定。
    两者必须可区分——否则审计时无法回答"这次批准到底是谁做的"。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "edited_arguments": self.edited_arguments,
            "comment": self.comment,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HumanDecision:
        edited = payload.get("edited_arguments")
        return cls(
            request_id=str(payload.get("request_id", "")),
            approved=bool(payload.get("approved", False)),
            edited_arguments=dict(edited) if isinstance(edited, dict) else None,
            comment=str(payload.get("comment", "")),
            decided_by=str(payload.get("decided_by", "unknown")),
            decided_at=float(payload.get("decided_at", 0.0)),
            source=str(payload.get("source", "human")),
        )


class TimeoutPolicy(str, Enum):
    """审批超时后的默认动作。

    默认值必须是 ``REJECT``。理由很简单：**没有人做决定时，
    系统不应该替人做了那个更危险的默认选择。**
    自动放行只应该在明确知情的场景下被显式开启。
    """

    REJECT = "reject"
    """拒绝该调用，把拒绝理由回灌给模型让它换路子，运行继续。"""

    ABORT = "abort"
    """终止整个运行，状态停在待审批处。"""

    AUTO_APPROVE = "auto_approve"
    """自动批准。仅适用于"宁可错发也不能卡住"的低风险场景。"""

    def to_decision(self, request: InterruptRequest, *, now: float | None = None) -> HumanDecision:
        return HumanDecision(
            request_id=request.request_id,
            approved=self is TimeoutPolicy.AUTO_APPROVE,
            comment=f"审批超时，按 {self.value} 策略自动处理",
            decided_by=f"policy:{self.value}",
            decided_at=time.time() if now is None else now,
            source="timeout_policy",
        )


# —— 副作用账本 ——


@dataclass(slots=True)
class EffectRecord:
    """一条已执行的副作用记录。

    ``observation`` 保存的是**模型当时看到的那段完整文本**，
    而不是摘要。恢复时要用它原样回放——如果只存摘要，
    恢复后的上下文与首次执行时不一致，模型会因为"证据变了"而做出不同决策，
    这会让"断点恢复"退化成"从某个位置重新开始"，语义上完全不同。
    """

    key: str
    slot: str
    run_id: str
    step: int
    call_index: int
    tool_name: str
    arguments: dict[str, Any]
    compensator: str = ""
    compensator_payload: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    executed_at: float = field(default_factory=time.time)

    def to_dict(self, *, preview_chars: int | None = None) -> dict[str, Any]:
        observation = self.observation
        if preview_chars is not None and len(observation) > preview_chars:
            observation = observation[:preview_chars] + "…[已截断]"
        return {
            "key": self.key,
            "slot": self.slot,
            "run_id": self.run_id,
            "step": self.step,
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "compensator": self.compensator,
            "compensator_payload": self.compensator_payload,
            "observation": observation,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EffectRecord:
        return cls(
            key=str(payload.get("key", "")),
            slot=str(payload.get("slot", "")),
            run_id=str(payload.get("run_id", "")),
            step=int(payload.get("step", 0)),
            call_index=int(payload.get("call_index", 0)),
            tool_name=str(payload.get("tool_name", "")),
            arguments=dict(payload.get("arguments") or {}),
            compensator=str(payload.get("compensator", "")),
            compensator_payload=dict(payload.get("compensator_payload") or {}),
            observation=str(payload.get("observation", "")),
            executed_at=float(payload.get("executed_at", 0.0)),
        )


class SideEffectLedger:
    """副作用账本。**整个 HITL 机制里最关键的一块。**

    它解决的是"恢复后重复执行副作用"的问题。做法是把幂等性建立在
    **调用槽位**上，而不是参数上：

    - 槽位 = ``{run_id}:{step}:{call_index}``，表示"第 N 步的第 k 次调用"
    - 恢复时先查槽位：这个位置已经产生过副作用 → 直接跳过执行，回放结果
    - 参数变了（例如审批人改了金额）则 key 变化 → 视为一次新的副作用，照常执行

    为什么用槽位而不是纯参数哈希：恢复场景下，**"同一个位置"比"同样的参数"
    更能表达"这是同一次调用"**。模型完全可能在两次运行里生成参数略有差异的
    同一个意图（时间戳、措辞），纯参数哈希会漏判并重复执行。
    """

    def __init__(self, records: Iterable[EffectRecord] | None = None) -> None:
        self._records: list[EffectRecord] = list(records or ())

    @staticmethod
    def slot_of(run_id: str, step: int, call_index: int) -> str:
        return f"{run_id}:{step}:{call_index}"

    @staticmethod
    def key_of(slot: str, tool_name: str, arguments: Mapping[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {"tool": tool_name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:12]
        return f"{slot}:{digest}"

    # —— 查询 ——

    def entries(self) -> list[EffectRecord]:
        return list(self._records)

    def by_slot(self, slot: str) -> EffectRecord | None:
        for record in reversed(self._records):
            if record.slot == slot:
                return record
        return None

    def already_executed(self, slot: str) -> bool:
        """这个调用槽位是否已经产生过副作用。**恢复路径上的第一道检查。**"""

        return self.by_slot(slot) is not None

    # —— 写入 ——

    def record(
        self,
        *,
        run_id: str,
        step: int,
        call_index: int,
        tool_name: str,
        arguments: Mapping[str, Any],
        compensator: str = "",
        compensator_payload: Mapping[str, Any] | None = None,
        observation: str = "",
    ) -> EffectRecord:
        slot = self.slot_of(run_id, step, call_index)
        entry = EffectRecord(
            key=self.key_of(slot, tool_name, arguments),
            slot=slot,
            run_id=run_id,
            step=step,
            call_index=call_index,
            tool_name=tool_name,
            arguments=dict(arguments),
            compensator=compensator,
            compensator_payload=dict(compensator_payload or {}),
            observation=observation,
        )
        self._records.append(entry)
        return entry

    # —— 补偿 ——

    def compensation_plan(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        """生成补偿计划：把已执行的副作用**逆序**撤销。

        逆序是硬要求。正序撤销会违反依赖关系——如果先删除了主记录，
        再去删它的附属记录就会失败，甚至留下孤儿数据。
        """

        targets = [item for item in self._records if run_id is None or item.run_id == run_id]
        plan: list[dict[str, Any]] = []
        for record in reversed(targets):
            if not record.compensator:
                continue
            plan.append(
                {
                    "compensator": record.compensator,
                    "payload": record.compensator_payload,
                    "undoes": record.tool_name,
                    "slot": record.slot,
                }
            )
        return plan

    def irreversible(self, *, run_id: str | None = None) -> list[EffectRecord]:
        """列出**无法撤销**的副作用。这是审批人最该看到的信息。"""

        return [
            item
            for item in self._records
            if not item.compensator and (run_id is None or item.run_id == run_id)
        ]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._records]

    @classmethod
    def from_dicts(cls, payloads: Sequence[dict[str, Any]]) -> SideEffectLedger:
        return cls([EffectRecord.from_dict(item) for item in payloads])


__all__ = [
    "DEFAULT_ESCALATORS",
    "EffectRecord",
    "Escalator",
    "HumanDecision",
    "InterruptPolicy",
    "InterruptRequest",
    "RiskAssessment",
    "RiskLevel",
    "SideEffectLedger",
    "TimeoutPolicy",
    "ToolRiskProfile",
    "bulk_escalator",
    "destructive_escalator",
    "escalate",
    "external_recipient_escalator",
    "pii_escalator",
]
