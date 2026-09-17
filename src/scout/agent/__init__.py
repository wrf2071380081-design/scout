"""Agent 层：ReAct 循环。

工具注册表位于 :mod:`scout.tools.registry`——工具的定义、参数校验与执行属于
工具层；Agent 循环只负责"决定调用什么"与"处理结果"。
"""

from __future__ import annotations

from .loop import (
    ANSWERED,
    BUDGET_STOPPED,
    NO_ANSWER,
    AgentRunResult,
    AgentStep,
    ObservedCall,
    ToolAgent,
)

__all__ = [
    "ANSWERED",
    "AgentRunResult",
    "AgentStep",
    "BUDGET_STOPPED",
    "NO_ANSWER",
    "ObservedCall",
    "ToolAgent",
]
