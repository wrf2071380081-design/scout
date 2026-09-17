"""LLM 客户端抽象。

为什么自己定一层抽象，而不是直接调 SDK：

1. **可替换**。离线评测、单元测试、消融实验都需要一个不依赖网络的确定性实现
   （见 :class:`~scout.llm.scripted.HeuristicLLM`）。把 LLM 抽象成协议之后，
   「换模型」和「离线跑」都只是换一个实现。
2. **可记录**。所有调用都要进 trace（耗时、token、是否失败），
   在 SDK 外面包一层比在每个调用点手写埋点可靠。
3. **typed 失败**。网络异常必须被归一化成 :class:`~scout.errors.ProviderError`
   并携带稳定 code，否则上层无法做重试与降级决策。

``task`` 字段是这个抽象里唯一"非标准"的设计：它把调用意图（复杂度判断 / 相关性打分 /
改写 / 作答 / 校验）显式传给客户端。真实模型会忽略它，启发式实现则依赖它选择策略。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..errors import ProviderError, ErrorCode


@dataclass(slots=True)
class ChatMessage:
    """一条对话消息。"""

    role: str
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in self.tool_calls
            ]
        return payload


@dataclass(slots=True)
class ToolCall:
    """模型请求的一次工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            # 有些 Provider 不返回 id；自己补一个稳定值，便于与 tool 消息配对。
            self.id = f"call_{abs(hash((self.name, json.dumps(self.arguments, sort_keys=True, default=str)))) % 10**10}"


@dataclass(slots=True)
class ToolSchema:
    """暴露给模型的工具描述（JSON Schema）。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass(slots=True)
class LLMRequest:
    """一次模型调用请求。

    ``task`` 与 ``context`` 是给**实现方**看的结构化提示，真实模型会忽略它们：

    - ``task``：调用意图（complexity / grade / rewrite / answer / decide …）
    - ``context``：任务相关的结构化输入，例如 ``{"question": "Redis 为什么快？"}``

    为什么要专门加 ``context``：模板化提示词把问题、证据、约束拼成一大段文本后，
    任何基于词法分析的离线实现都无法可靠地从中还原出"原始问题"——
    它会把这整段模板都当成查询，导致覆盖率被稀释到接近零。
    显式传进来比事后用正则去猜要可靠得多。
    """

    messages: Sequence[ChatMessage]
    tools: Sequence[ToolSchema] | None = None
    schema: type[BaseModel] | None = None
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None
    deadline: float | None = None


@dataclass(slots=True)
class LLMResponse:
    """一次模型调用响应。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def parse(self, schema: type[BaseModel]) -> BaseModel:
        """把 ``content`` 解析成结构化对象。

        真实模型经常在 JSON 外面裹一层 ```json 代码块，这里统一剥离，
        解析失败则抛 typed :class:`ProviderError`（属于"响应畸形"，
        上层可以据此换一条恢复路径）。
        """

        text = self.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[: -3]
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        try:
            payload = json.loads(text)
            return schema.model_validate(payload)
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            raise ProviderError(
                f"model returned unparseable structured output: {exc}",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                provider="llm",
                operation="structured_output",
                details={"schema": schema.__name__, "content_preview": self.content[:200]},
            ) from exc


class LLMClient(Protocol):
    """LLM 客户端协议。"""

    @property
    def model_name(self) -> str:  # pragma: no cover - 协议声明
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - 协议声明
        ...


__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "ToolSchema",
]
