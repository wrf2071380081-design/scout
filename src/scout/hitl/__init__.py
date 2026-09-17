"""HITL：人工介入（human-in-the-loop）。

四个公开部分：

- :mod:`~scout.hitl.checkpoints` —— 可序列化状态与 append-only 检查点存储
- :mod:`~scout.hitl.interrupts` —— 风险分级、审批请求/决策、超时策略、副作用账本
- :mod:`~scout.hitl.runtime` —— :class:`ResumableAgent`（中断/恢复/时间旅行）
"""

from __future__ import annotations

from .checkpoints import (
    AgentState,
    Checkpoint,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from .interrupts import (
    DEFAULT_ESCALATORS,
    EffectRecord,
    HumanDecision,
    InterruptPolicy,
    InterruptRequest,
    RiskAssessment,
    RiskLevel,
    SideEffectLedger,
    TimeoutPolicy,
    ToolRiskProfile,
    bulk_escalator,
    destructive_escalator,
    external_recipient_escalator,
    pii_escalator,
)
from .runtime import ResumableAgent, RunOutcome, RunStatus, build_default_policy

__all__ = [
    "DEFAULT_ESCALATORS",
    "AgentState",
    "Checkpoint",
    "CheckpointStore",
    "EffectRecord",
    "FileCheckpointStore",
    "HumanDecision",
    "InMemoryCheckpointStore",
    "InterruptPolicy",
    "InterruptRequest",
    "ResumableAgent",
    "RiskAssessment",
    "RiskLevel",
    "RunOutcome",
    "RunStatus",
    "SideEffectLedger",
    "TimeoutPolicy",
    "ToolRiskProfile",
    "build_default_policy",
    "bulk_escalator",
    "destructive_escalator",
    "external_recipient_escalator",
    "pii_escalator",
]
