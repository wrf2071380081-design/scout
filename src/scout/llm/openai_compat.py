"""OpenAI 兼容的 LLM 客户端。

只依赖 ``requests``，因此可以对接任何暴露 ``/chat/completions`` 的服务
（OpenAI、Azure OpenAI、vLLM、Ollama、各类国产模型网关）。

重试语义的关键约束：**重试只有一个所有者**。
这里关闭调用方的自发重试循环之外的一切隐式重试——不在 SDK 层、不在 urllib3 层
重复设置 retry，避免"乘法重试"把一次请求放大成十几次外部调用并击穿 deadline。
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests

from ..errors import ErrorCode, ProviderError
from .base import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

_JSON_INSTRUCTION = (
    "\n\n只输出一个合法 JSON 对象，不要输出解释、不要使用 Markdown 代码块。"
    "字段缺失时按 schema 的默认值处理。"
)


class OpenAICompatLLM:
    """OpenAI 兼容协议的同步客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    # —— 请求构造 ——

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        messages = [message.to_wire() for message in request.messages]
        if request.schema is not None and messages:
            messages[-1] = {
                **messages[-1],
                "content": f"{messages[-1].get('content', '')}{_JSON_INSTRUCTION}",
            }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature if request.temperature is None else request.temperature,
        }
        if request.tools:
            payload["tools"] = [tool.to_wire() for tool in request.tools]
            payload["tool_choice"] = "auto"
        if request.schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _remaining_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.timeout_seconds
        return max(min(deadline - time.monotonic(), self.timeout_seconds), 0.05)

    # —— 错误归一化 ——

    @staticmethod
    def _classify(exc: Exception) -> ProviderError | None:
        if isinstance(exc, requests.exceptions.Timeout):
            return ProviderError(
                "llm request timed out",
                code=ErrorCode.PROVIDER_TIMEOUT,
                retryable=True,
                provider="llm",
                operation="chat",
            )
        if isinstance(exc, requests.exceptions.ConnectionError):
            return ProviderError(
                "llm connection failed",
                code=ErrorCode.PROVIDER_CONNECTION,
                retryable=True,
                provider="llm",
                operation="chat",
            )
        if isinstance(exc, requests.exceptions.RequestException):
            return ProviderError(
                f"llm transport error: {exc}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=True,
                provider="llm",
                operation="chat",
            )
        return None

    @staticmethod
    def _classify_status(status: int, body: str) -> ProviderError:
        preview = body[:300]
        if status == 429:
            return ProviderError(
                "llm rate limited",
                code=ErrorCode.PROVIDER_RATE_LIMITED,
                retryable=True,
                provider="llm",
                operation="chat",
                details={"status": status, "body_preview": preview},
            )
        if status >= 500:
            return ProviderError(
                "llm provider unavailable",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=True,
                provider="llm",
                operation="chat",
                details={"status": status, "body_preview": preview},
            )
        # 4xx 属于请求本身有问题（模型名错、schema 不被支持），重试没有意义。
        return ProviderError(
            "llm rejected the request",
            code=ErrorCode.PROVIDER_INVALID_RESPONSE,
            retryable=False,
            provider="llm",
            operation="chat",
            details={"status": status, "body_preview": preview},
        )

    # —— 响应解析 ——

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                "llm response contains no choices",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                provider="llm",
                operation="chat",
            )
        choice = choices[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                arguments = {"__raw__": raw_args}
            calls.append(
                ToolCall(name=str(function.get("name", "")), arguments=arguments, id=str(raw.get("id", "")))
            )
        usage_raw = data.get("usage") or {}
        return LLMResponse(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=int(usage_raw.get("prompt_tokens") or 0),
                output_tokens=int(usage_raw.get("completion_tokens") or 0),
            ),
            model=str(data.get("model") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    # —— 主入口 ——

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request)
        last_error: ProviderError | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(
                    self._endpoint(),
                    headers=self._headers(),
                    json=payload,
                    timeout=self._remaining_timeout(request.deadline),
                )
            except Exception as exc:  # noqa: BLE001 - 统一归一化后重抛
                classified = self._classify(exc)
                if classified is None:
                    raise
                last_error = classified
                if not classified.retryable or attempt >= self.max_attempts:
                    raise classified from exc
                time.sleep(min(0.5 * attempt, 2.0))
                continue

            if not response.ok:
                error = self._classify_status(response.status_code, response.text)
                last_error = error
                if not error.retryable or attempt >= self.max_attempts:
                    raise error
                time.sleep(min(0.5 * attempt, 2.0))
                continue

            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError(
                    "llm returned non-JSON body",
                    code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                    retryable=True,
                    provider="llm",
                    operation="chat",
                    details={"body_preview": response.text[:300]},
                ) from exc

            parsed = self._parse_response(data)
            return parsed

        if last_error is not None:  # pragma: no cover - 循环内已 raise
            raise last_error
        raise ProviderError(
            "llm call failed without a classified error",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider="llm",
            operation="chat",
        )


    # —— 流式 ——

    def complete_stream(self, request: LLMRequest) -> Iterator[str]:
        """流式生成，逐片产出增量文本。

        **三个实现要点，每个都在线上咬过人：**

        1. **SSE 分片不按行对齐。** 网络返回的 chunk 可能把一行 JSON 切成两半，
           所以必须自己维护缓冲区、按 ``\\n`` 切分，而不是"一次 read = 一条事件"。
        2. **不重试。** 流式已经吐出去的内容收不回来，中途重试会导致内容重复。
           失败就让调用方感知（抛 :class:`ProviderError`），由上层决定是否重来。
           这与 ``complete`` 的重试策略刻意不同——**语义不同，策略就不该一样**。
        3. **遇 ``[DONE]`` 或迭代结束都要干净收尾。** 有些网关不发 ``[DONE]``，
           不能依赖它来判断结束。
        """

        payload = self._payload(request)
        payload["stream"] = True
        # 流式下不请求 usage：多数网关最后一帧才补，反而容易卡住收尾
        payload.pop("response_format", None)

        try:
            response = requests.post(
                self._endpoint(),
                headers=self._headers(),
                json=payload,
                timeout=self._remaining_timeout(request.deadline),
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001 - 统一归一化
            classified = self._classify(exc)
            if classified is None:
                raise
            raise classified from exc

        if not response.ok:
            raise self._classify_status(response.status_code, response.text)

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            text = line.strip()
            if not text.startswith("data:"):
                continue
            data = text[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield str(piece)


def build_tool_payload(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """供测试与调试使用：把工具列表转成 wire 格式。"""

    return [tool.to_wire() for tool in tools]


__all__ = ["ChatMessage", "OpenAICompatLLM", "build_tool_payload"]
