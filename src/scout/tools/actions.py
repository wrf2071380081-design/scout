"""带副作用的动作工具工厂 + 内存执行器。

**为什么把"副作用"做成显式概念。**

只读工具与写工具在很多框架里看起来一模一样——都是 ``handler(arguments)``。
但对人类介入（HITL）来说，这两类工具是天壤之别：

- 只读工具出错了，重试一次就行
- 写工具出错了，**世界已经被改变了**

所以本模块里的工具显式声明 ``side_effects=True`` 和 ``compensator``：
不只是给审批策略提供输入，也是把"这个动作能不能撤销"这件事写进工具的定义里。

配套的 :class:`InMemoryActionExecutor` 把每个动作落成一个带 id 的记录，
补偿动作用 id 把它撤销——这是测试与演示用的内存实现，
生产环境替换为真实适配器（SMTP / 工单系统 / 数据库）即可，**ToolSpec 不变**。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..errors import ErrorCode, ToolError
from .registry import ToolResult, ToolSpec


@dataclass(slots=True)
class ActionRecord:
    """一条已执行的动作。"""

    action_id: str
    tool: str
    arguments: dict[str, Any]
    done: bool = True
    compensated: bool = False
    compensator_payload: dict[str, Any] = field(default_factory=dict)


class ActionExecutor(Protocol):
    """动作执行器接口。隔离"动作语义"与"动作实现"——策略层只关心前者。"""

    def execute(self, tool: str, arguments: Mapping[str, Any]) -> ToolResult: ...

    def compensate(self, compensator: str, payload: Mapping[str, Any]) -> ToolResult: ...


class InMemoryActionExecutor:
    """内存动作执行器。每个动作有稳定 id，补偿动作用 id 撤销。"""

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []
        self._by_id: dict[str, ActionRecord] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # —— 动作实现 ——

    def execute(self, tool: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((tool, dict(arguments)))
        handler = getattr(self, f"_do_{tool}", None)
        if handler is None:
            raise ToolError(f"未知动作：{tool}", code=ErrorCode.TOOL_NOT_FOUND, tool=tool)
        return handler(dict(arguments))

    def compensate(self, compensator: str, payload: Mapping[str, Any]) -> ToolResult:
        self.calls.append((compensator, dict(payload)))
        handler = getattr(self, f"_undo_{compensator}", None)
        if handler is None:
            return ToolResult.failure(
                f"补偿动作 {compensator} 未实现",
                code=ErrorCode.TOOL_EXECUTION_FAILED,
                retryable=False,
                compensator=compensator,
            )
        return handler(dict(payload))

    # —— 查询（测试与演示用） ——

    def history(self) -> list[ActionRecord]:
        return list(self._records)

    def committed(self) -> list[ActionRecord]:
        return [record for record in self._records if record.done and not record.compensated]

    def _register(self, tool: str, arguments: dict[str, Any], *, text: str) -> ToolResult:
        action_id = f"act-{uuid.uuid4().hex[:10]}"
        record = ActionRecord(action_id=action_id, tool=tool, arguments=arguments)
        self._records.append(record)
        self._by_id[action_id] = record
        return ToolResult.success(text, action_id=action_id)

    def _mark_compensated(self, action_id: str, text: str) -> ToolResult:
        record = self._by_id.get(action_id)
        if record is None:
            return ToolResult.failure(
                f"找不到要撤销的动作 {action_id}",
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
            )
        record.compensated = True
        record.done = False
        return ToolResult.success(text, action_id=action_id)

    # —— 具体动作 ——

    def _do_send_email(self, arguments: dict[str, Any]) -> ToolResult:
        to = str(arguments.get("to", ""))
        subject = str(arguments.get("subject", ""))
        body = str(arguments.get("body", ""))
        if not to or not subject:
            raise ToolError(
                "send_email 需要 to 与 subject",
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
                tool="send_email",
            )
        return self._register(
            "send_email",
            arguments,
            text=f"已发送邮件至 {to}，主题《{subject}》。",
        )

    def _undo_recall_email(self, payload: dict[str, Any]) -> ToolResult:
        return self._mark_compensated(
            str(payload.get("undo_token", "")), "已召回邮件。"
        )

    def _do_create_ticket(self, arguments: dict[str, Any]) -> ToolResult:
        title = str(arguments.get("title", ""))
        if not title:
            raise ToolError(
                "create_ticket 需要 title",
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
                tool="create_ticket",
            )
        return self._register(
            "create_ticket", arguments, text=f"已创建工单《{title}》。"
        )

    def _undo_close_ticket(self, payload: dict[str, Any]) -> ToolResult:
        return self._mark_compensated(
            str(payload.get("undo_token", "")), "已关闭工单。"
        )

    def _do_delete_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments.get("path", ""))
        if not path:
            raise ToolError(
                "delete_file 需要 path",
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
                tool="delete_file",
            )
        return self._register(
            "delete_file", arguments, text=f"已删除 {path}。**此操作不可撤销。**"
        )

    # delete_file 故意没有 _undo_：它代表了"不可逆"那一类操作。


# —— 工具工厂 ——


def build_action_tools(executor: InMemoryActionExecutor | None = None) -> list[ToolSpec]:
    """构造一组带副作用声明的示例工具。

    每个工具都显式声明 ``side_effects`` 与 ``compensator``——
    这两个字段是 HITL 审批策略能自动推导风险等级的前提。

    :param executor: 可选的执行器。测试时传入自己的实例以便断言
        "副作用到底执行了几次"；不传则内部新建一个（只用于演示）。
    """

    executor = executor or InMemoryActionExecutor()

    def make_executor_handler(executor_instance: InMemoryActionExecutor, tool: str):
        def handler(arguments: dict[str, Any]) -> ToolResult:
            return executor_instance.execute(tool, arguments)

        return handler

    send_email = ToolSpec(
        name="send_email",
        description="发送邮件。有影响，但可通过召回邮件撤销。",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "minLength": 3, "description": "收件人邮箱"},
                "subject": {"type": "string", "minLength": 1, "description": "主题"},
                "body": {"type": "string", "description": "正文", "default": ""},
            },
            "required": ["to", "subject"],
            "additionalProperties": False,
        },
        handler=make_executor_handler(executor, "send_email"),
        side_effects=True,
        compensator="recall_email",
    )

    create_ticket = ToolSpec(
        name="create_ticket",
        description="创建工单。有影响，可通过关闭工单撤销。",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 3, "description": "工单标题"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "优先级",
                    "default": "medium",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        handler=make_executor_handler(executor, "create_ticket"),
        side_effects=True,
        compensator="close_ticket",
    )

    delete_file = ToolSpec(
        name="delete_file",
        description="删除文件。不可撤销，必须人工审批。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "description": "文件路径"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=make_executor_handler(executor, "delete_file"),
        side_effects=True,
        compensator="",
    )

    return [send_email, create_ticket, delete_file]


__all__ = [
    "ActionExecutor",
    "ActionRecord",
    "InMemoryActionExecutor",
    "build_action_tools",
]
