"""工具注册表。

Agent 系统里有一类失败非常隐蔽：**模型把工具参数写错了，而系统照单全收**。
参数类型不对、必填项缺失、枚举值写成同义词——这些都不会抛异常，
只会让工具返回一个"看起来正常"的空结果，然后模型基于这个空结果继续推理。

所以注册表承担三件事：

1. **声明**：每个工具用 JSON Schema 描述自己的输入契约，这份 schema 会原样喂给模型。
2. **校验**：调用前做本地校验（类型 / 必填 / 枚举 / 范围），不合法就返回
   带明确错误码的 :class:`ToolResult`，让模型有依据去修正参数。
3. **归一**：所有异常都被归一成 typed 错误，绝不让工具内部异常穿透到 Agent 循环。

校验器是自己实现的（不用 ``jsonschema`` 依赖），支持 Agent 工具真正用得到的子集：
``type`` / ``required`` / ``enum`` / ``minimum`` / ``maximum`` / ``minLength`` / ``maxLength`` /
``items`` / ``properties``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..errors import ErrorCode, ToolError
from ..llm.base import ToolSchema


@dataclass(slots=True)
class ToolResult:
    """一次工具调用的结果。

    ``content`` 是给模型看的文本；``data`` 是结构化数据，**不进入 prompt**。
    把两者分开，是为了避免结构化元数据（评分、id、耗时）污染模型上下文。
    """

    ok: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    retryable: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, content=content, data=dict(data))

    @classmethod
    def failure(cls, content: str, *, code: ErrorCode, retryable: bool = False, **data: Any) -> ToolResult:
        return cls(
            ok=False,
            content=content,
            data=dict(data),
            error_code=code.value,
            retryable=retryable,
        )

    def to_observation(self) -> str:
        """转成喂回模型的 observation 文本。"""

        return self.content if self.ok else f"[工具执行失败:{self.error_code}] {self.content}"


ToolHandler = Callable[[Mapping[str, Any]], ToolResult]


@dataclass(slots=True)
class ToolSpec:
    """工具定义。

    ``side_effects`` / ``compensator`` 是 HITL 审批机制的声明性来源：
    策略层不需要知道"这个工具内部做了什么"，只要知道**它对世界有改变**、
    **以及能不能被撤销**，就足以决定要不要打扰人。
    把这两件事写在工具定义里，等于让"风险"跟着能力走，而不是靠人脑背。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    enabled: bool = True
    side_effects: bool = False
    """True 表示该工具会改变外部状态（写文件、发消息、建工单等），需要纳入审批范围。"""

    compensator: str = ""
    """撤销该工具副作用所用的补偿工具名（如 send_email → recall_email）。
    空串表示不可补偿——**不可补偿的工具一律走 REQUIRED 级审批**。"""

    def schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def validate(self, arguments: Mapping[str, Any]) -> list[str]:
        """返回参数问题列表；空列表表示合法。"""

        return validate_against_schema(arguments, self.parameters)


def validate_against_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """JSON Schema 子集校验。返回人类可读的问题列表。"""

    problems: list[str] = []
    expected_type = schema.get("type")

    def type_ok(candidate: Any) -> bool:
        return {
            "object": isinstance(candidate, dict),
            "array": isinstance(candidate, list),
            "string": isinstance(candidate, str),
            "integer": isinstance(candidate, int) and not isinstance(candidate, bool),
            "number": isinstance(candidate, (int, float)) and not isinstance(candidate, bool),
            "boolean": isinstance(candidate, bool),
            "null": candidate is None,
        }.get(str(expected_type), True)

    if expected_type and not type_ok(value):
        problems.append(f"{path}: 期望类型 {expected_type}，实际为 {type(value).__name__}")
        return problems

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: 取值必须是 {schema['enum']} 之一，实际为 {value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            problems.append(f"{path}: 长度不得小于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            problems.append(f"{path}: 长度不得超过 {schema['maxLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: 不得小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: 不得大于 {schema['maximum']}")

    if isinstance(value, dict):
        for required in schema.get("required", []) or []:
            if required not in value:
                problems.append(f"{path}: 缺少必填字段 {required!r}")
        properties = schema.get("properties") or {}
        for key, item in value.items():
            if key in properties:
                problems.extend(validate_against_schema(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path}: 不支持字段 {key!r}")

    if isinstance(value, list) and "items" in schema:
        for position, item in enumerate(value):
            problems.extend(validate_against_schema(item, schema["items"], f"{path}[{position}]"))

    return problems


class ToolRegistry:
    """工具集合。"""

    def __init__(self, tools: Iterable[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if not tool.name:
            raise ValueError("tool name must not be empty")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return [name for name, tool in self._tools.items() if tool.enabled]

    def schemas(self) -> list[ToolSchema]:
        """导出给模型的工具描述。"""

        return [tool.schema() for tool in self._tools.values() if tool.enabled]

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
        """校验并执行一次工具调用。

        **本方法不抛异常。** 所有失败都转成 ``ok=False`` 的 :class:`ToolResult`，
        因为 Agent 循环需要拿到可读的失败原因去决定下一步，而不是被中断。
        """

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                f"未注册的工具：{name}。可用工具：{', '.join(self.names())}",
                code=ErrorCode.TOOL_NOT_FOUND,
                available=self.names(),
            )
        if not tool.enabled:
            return ToolResult.failure(
                f"工具 {name} 当前不可用",
                code=ErrorCode.TOOL_NOT_FOUND,
            )

        payload = dict(arguments or {})
        problems = tool.validate(payload)
        if problems:
            return ToolResult.failure(
                "参数校验失败：" + "；".join(problems),
                code=ErrorCode.TOOL_INVALID_ARGUMENTS,
                problems=problems,
            )

        try:
            return tool.handler(payload)
        except ToolError as exc:
            return ToolResult.failure(
                exc.message,
                code=exc.code,
                retryable=exc.retryable,
                **exc.details,
            )
        except Exception as exc:  # noqa: BLE001 - 工具内部异常绝不允许穿透到 Agent 循环
            return ToolResult.failure(
                f"工具 {name} 执行异常：{type(exc).__name__}: {exc}",
                code=ErrorCode.TOOL_EXECUTION_FAILED,
                retryable=False,
                tool=name,
            )


__all__ = ["ToolRegistry", "ToolResult", "ToolSpec", "validate_against_schema"]
