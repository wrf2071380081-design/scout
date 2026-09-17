"""内置工具：计算器与日期时间。

这两个工具存在的意义不是"功能"，而是**把模型的确定性缺陷挡在系统外**：
大模型的算术与日期推理错误率远高于它自认为的水平，而这两个任务恰好有
精确的确定性解法。让模型调用工具而不是"心算"，是性价比最高的可靠性改进之一。

顺带它们也是**幻觉探针**：如果 Agent 在具备计算器的情况下仍然直接给出算术结论，
说明它在"何时该用工具"的判断上出了问题——这类行为可以被评测统计出来
（见 :mod:`scout.evaluation.metrics` 的工具使用指标）。
"""

from __future__ import annotations

import ast
import datetime as _dt
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..errors import ErrorCode, ToolError
from .registry import ToolResult, ToolSpec

# —— 计算器 ——

CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "四则运算表达式，支持 + - * / // % ** 与括号。例如 (1200-980)/980*100。",
        }
    },
    "required": ["expression"],
    "additionalProperties": False,
}

_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_NORMALIZE = str.maketrans({"×": "*", "÷": "/", "（": "(", "）": ")", "－": "-", "＋": "+", " ": ""})


def _eval_node(node: ast.AST, depth: int = 0) -> float:
    if depth > 20:
        raise ToolError("表达式嵌套过深", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("表达式只能包含数字", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("不支持的运算符", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
        left, right = _eval_node(node.left, depth + 1), _eval_node(node.right, depth + 1)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("除数为零", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
            raise ToolError("幂运算超出允许范围", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
        return float(handler(left, right))
    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("不支持的一元运算符", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
        return float(handler(_eval_node(node.operand, depth + 1)))
    raise ToolError(
        "表达式包含不支持的语法结构",
        code=ErrorCode.TOOL_INVALID_ARGUMENTS,
        tool="calculator",
    )


def safe_eval(expression: str) -> float:
    """安全的算术求值。

    用 AST 白名单求值而不是 ``eval``：``eval`` 能执行任意代码，
    而工具参数是**模型生成的、来自不可信上下文的**，这条路径必须封死。
    """

    cleaned = expression.translate(_NORMALIZE).strip()
    if not cleaned:
        raise ToolError("表达式为空", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
    if len(cleaned) > 200:
        raise ToolError("表达式过长", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="calculator")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            f"表达式语法错误：{exc.msg}",
            code=ErrorCode.TOOL_INVALID_ARGUMENTS,
            tool="calculator",
        ) from exc
    return _eval_node(tree.body)


def calculator_tool(arguments: dict[str, Any]) -> ToolResult:
    expression = str(arguments.get("expression", ""))
    value = safe_eval(expression)
    # 整数结果不要显示成 4.0，避免模型误判精度。
    rendered = f"{value:.10g}"
    return ToolResult.success(f"{expression} = {rendered}", expression=expression, value=value)


# —— 日期时间 ——

DATETIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detail": {
            "type": "string",
            "enum": ["date", "datetime", "weekday"],
            "description": "date 只返回日期；datetime 返回日期与时间；weekday 返回星期几。",
        },
        "timezone": {
            "type": "string",
            "maxLength": 64,
            "description": "IANA 时区名，例如 Asia/Shanghai。默认 Asia/Shanghai。",
        },
        "offset_days": {
            "type": "integer",
            "minimum": -3650,
            "maximum": 3650,
            "description": "相对今天的偏移天数，用于计算前后日期。默认 0。",
        },
    },
    "required": [],
    "additionalProperties": False,
}

_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def datetime_tool(arguments: dict[str, Any]) -> ToolResult:
    detail = str(arguments.get("detail") or "date")
    timezone_name = str(arguments.get("timezone") or "Asia/Shanghai")
    offset_days = int(arguments.get("offset_days") or 0)

    try:
        tzinfo = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        # 时区库不可用或名称非法时退化为 UTC，而不是报错中断任务。
        tzinfo = _dt.timezone.utc
        timezone_name = "UTC"

    now = (_dt.datetime.now(tzinfo) + _dt.timedelta(days=offset_days)).replace(microsecond=0)
    weekday = _WEEKDAYS[now.weekday()]

    if detail == "weekday":
        content = f"{now.date().isoformat()} 是{weekday}（{timezone_name}）"
    elif detail == "datetime":
        content = f"{now.isoformat()} {weekday}（{timezone_name}）"
    else:
        content = f"{now.date().isoformat()} {weekday}（{timezone_name}）"

    return ToolResult.success(
        content,
        iso=now.isoformat(),
        date=now.date().isoformat(),
        weekday=weekday,
        timezone=timezone_name,
        offset_days=offset_days,
    )


def build_default_tools() -> list[ToolSpec]:
    """计算器与日期工具。"""

    return [
        ToolSpec(
            name="calculator",
            description=(
                "执行精确的算术运算。涉及数字计算时**必须**使用该工具，"
                "不要自己心算——模型的算术结果不可靠。"
            ),
            parameters=CALCULATOR_SCHEMA,
            handler=calculator_tool,
        ),
        ToolSpec(
            name="datetime",
            description=(
                "获取当前日期、时间或星期，也可以计算相对今天偏移若干天的日期。"
                "问题涉及「今天」「上个月」「距今多久」时必须调用。"
            ),
            parameters=DATETIME_SCHEMA,
            handler=datetime_tool,
        ),
    ]


__all__ = [
    "CALCULATOR_SCHEMA",
    "DATETIME_SCHEMA",
    "build_default_tools",
    "calculator_tool",
    "datetime_tool",
    "safe_eval",
]
