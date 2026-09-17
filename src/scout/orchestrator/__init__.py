"""失败分类与自愈编排。"""

from __future__ import annotations

from .healing import (
    SEMANTIC_CODES,
    FailureSignal,
    RecoveryAction,
    RecoveryDecision,
    SelfHealingOrchestrator,
)

__all__ = [
    "SEMANTIC_CODES",
    "FailureSignal",
    "RecoveryAction",
    "RecoveryDecision",
    "SelfHealingOrchestrator",
]
