"""沙箱工具：在子进程里执行 Python。

::: 这个工具为什么被标成"必须人工审批"

它能执行任意 Python——这意味着它可以读写文件、发起网络请求、删除数据。
scout 的 HITL 策略会自动把它升级到 **REQUIRED**（不可补偿 → 必须人审、无免审开关），
因为工具声明里 ``side_effects=True`` 且 ``compensator=""``。

**这不是保守，是这类工具的应有待遇。** 一个能跑任意代码的 Agent 工具，
如果没有审批闸门，那整个系统的安全设计就是摆设。

::: 三层防护，各自都不够，叠起来才成立

1. **模式拦截**（`_DENY_PATTERNS`）：挡掉明显的危险调用。
   **它很容易被绕过**（字符串拼接、编码、反射都行），所以只能是第一层。
2. **子进程 + 超时 + 输出截断**：真正的边界——代码跑在独立进程里，
   超时即杀，输出超长即截断，主进程不受影响。
3. **人工审批**（HITL）：决定"这段代码该不该跑"的是人，不是系统。

1 和 2 是技术控制，3 是责任归属。**缺了 3，前两层只是延迟了事故。**
生产环境还需要第四层：容器/沙箱（Docker、gVisor、seccomp），本模块不涉及。
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from ..errors import ErrorCode, ToolError
from .registry import ToolResult, ToolSpec

_DENY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("删除文件", re.compile(r"\b(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir)\b")),
    ("系统调用", re.compile(r"\b(os\.system|subprocess\.(Popen|run|call)|pty\.spawn)\b")),
    ("动态执行", re.compile(r"\b(eval|exec|compile)\s*\(")),
    ("网络访问", re.compile(r"\b(urllib|requests|httpx|socket|http\.client)\b")),
    ("环境篡改", re.compile(r"\b(os\.environ|sys\.modules|__import__)\b")),
)

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_SECONDS = 5.0


def _screen(code: str) -> str | None:
    """返回命中的危险模式描述，没有命中则返回 None。"""

    for label, pattern in _DENY_PATTERNS:
        if pattern.search(code):
            return label
    return None


def run_python(
    code: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> str:
    """在子进程里执行一段 Python，返回 stdout+stderr 的截断文本。"""

    if not code or not code.strip():
        raise ToolError("python_exec 需要 code", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="python_exec")
    if len(code) > 8000:
        raise ToolError("代码过长（上限 8000 字符）", code=ErrorCode.TOOL_INVALID_ARGUMENTS, tool="python_exec")

    denied = _screen(code)
    if denied is not None:
        # 明确拒绝，而不是"跑了但没效果"——静默失败比显式失败危险得多。
        raise ToolError(
            f"已拦截：代码中包含{denied}相关调用。本工具仅用于纯计算类代码。",
            code=ErrorCode.TOOL_INVALID_ARGUMENTS,
            retryable=False,
            tool="python_exec",
        )

    try:
        completed = subprocess.run(  # noqa: S603 - 有意在子进程执行，超时与输出截断是边界
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"执行超时（{timeout_seconds}s）",
            code=ErrorCode.TOOL_EXECUTION_FAILED,
            retryable=False,
            tool="python_exec",
        ) from None

    output = (completed.stdout or "") + (completed.stderr or "")
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + "\n…[输出已截断]"
    prefix = f"退出码 {completed.returncode}"
    return f"{prefix}\n{output}".strip()


def python_exec_tool() -> ToolSpec:
    """构造 python_exec 工具。

    注意 ``side_effects=True`` + ``compensator=""``：
    这两个字段会让 HITL 策略把它判为 **REQUIRED**，即**必须人工审批且无免审开关**。
    """

    def handler(arguments: dict[str, Any]) -> ToolResult:
        code = str(arguments.get("code", ""))
        timeout = float(arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
        try:
            output = run_python(code, timeout_seconds=min(timeout, 15.0))
        except ToolError:
            raise
        return ToolResult.success(output, chars=len(output))

    return ToolSpec(
        name="python_exec",
        description=(
            "在受控子进程中执行一段 Python（纯计算用途）。"
            "禁用文件删除、系统调用、动态执行与网络访问；超时 5 秒，输出截断。"
            "**该工具会影响系统状态，需要人工审批。**"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "minLength": 1, "description": "要执行的 Python 代码"},
                "timeout_seconds": {"type": "number", "default": 5, "description": "超时（秒），上限 15"},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=handler,
        side_effects=True,
        compensator="",
    )


def build_sandbox_tools() -> list[ToolSpec]:
    """返回沙箱类工具（目前只有 python_exec）。"""

    return [python_exec_tool()]


__all__ = ["build_sandbox_tools", "python_exec_tool", "run_python"]
