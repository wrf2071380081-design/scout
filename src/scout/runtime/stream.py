"""token 级流式输出：把"首 token 延迟"变成用户能感知的东西。

**阶段级 SSE 与 token 级流式的区别，以及为什么两个都要有。**
阶段级（检索完了 / 生成完了）解决的是**可解释性**：用户知道系统在忙什么、
每一步花了多久。token 级解决的是**体感延迟**：TTFT 从"整段答案生成完"
降到"第一个字出现"。两者不互相替代，真实产品两个都给。

**流式的三个工程约束**（都会在生产里咬人）：

1. **不能因为流式而丢失用量统计。** 流式响应里 usage 常在最后一帧才给，
   甚至不给。所以 :func:`iter_tokens` 在流结束后要能拿到累计用量；
   拿不到时由调用方用估算兜底——**账要能对上，不能因为"是流式"就不记**。
2. **客户端断开要立刻停止上游生成。** 用户关页面之后继续烧 token 是纯浪费。
   本模块用生成器的 ``close()`` 语义实现：调用方不再迭代，上游请求即被中断。
3. **降级必须无损。** 客户端不支持流式时，退化为一次性返回整段——
   调用方不需要写两个分支。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence, runtime_checkable

from ..llm.base import LLMClient, LLMRequest, LLMResponse, TokenUsage


@runtime_checkable
class StreamingClient(Protocol):
    """支持流式的客户端协议。"""

    def complete_stream(self, request: LLMRequest) -> Iterator[str]: ...  # pragma: no cover


def supports_streaming(client: object) -> bool:
    return callable(getattr(client, "complete_stream", None))


@dataclass(slots=True)
class StreamOutcome:
    """一次流式生成的结果与账。"""

    text: str = ""
    chunks: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    streamed: bool = False
    estimated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "text_chars": len(self.text),
            "chunks": self.chunks,
            "streamed": self.streamed,
            "usage_estimated": self.estimated,
            "usage": self.usage.to_dict(),
        }


def estimate_tokens(text: str) -> int:
    """流式场景下的用量估算。中英混排按 1.6 字符/token 保守估。"""

    return max(1, int(len(text or "") / 1.6)) if text else 0


def estimate_tokens_from_chars(chars: int) -> int:
    """按字符数估算（提示词侧用）。"""

    return max(1, int(chars / 1.6)) + 8 if chars else 0


def iter_tokens(client: LLMClient, request: LLMRequest, on_token=None) -> Iterator[str]:
    """统一的取词接口：有流式就用流式，没有就一次性吐出来。

    ``on_token`` 是给观测层用的钩子（往里塞 SSE 推送即可），
    它抛异常绝不影响生成——观察者不该有能力弄坏主流程。
    """

    if supports_streaming(client):
        for piece in client.complete_stream(request):  # type: ignore[attr-defined]
            if not piece:
                continue
            if on_token is not None:
                try:
                    on_token(piece)
                except Exception:  # noqa: BLE001 - 观察者故障不打断生成
                    pass
            yield piece
        return

    response: LLMResponse = client.complete(request)
    content = response.content or ""
    if on_token is not None and content:
        try:
            on_token(content)
        except Exception:  # noqa: BLE001
            pass
    yield content


def collect_stream(client: LLMClient, request: LLMRequest, on_token=None) -> StreamOutcome:
    """把流收集成完整文本，同时保留用量与"是否真的流式"的证据。

    用量口径要诚实：流式协议通常在最后一帧才给 usage，有些服务干脆不给。
    所以这里统一用**估算值**并打上 ``estimated=True``，而不是伪造一个精确数字——
    下游按估算记账没问题，但**不能把估算值当成实测值展示**。
    """

    chunks: list[str] = []
    count = 0
    for piece in iter_tokens(client, request, on_token=on_token):
        chunks.append(piece)
        count += 1
    text = "".join(chunks)
    streamed = supports_streaming(client)
    prompt_chars = sum(len(message.content) for message in request.messages)
    return StreamOutcome(
        text=text,
        chunks=count,
        usage=TokenUsage(
            input_tokens=estimate_tokens_from_chars(prompt_chars),
            output_tokens=estimate_tokens(text),
        ),
        streamed=streamed,
        estimated=True,
    )


def sse_event(event: str, payload: dict) -> bytes:
    """把事件编码成 SSE 帧。集中在一处，避免各处手写格式写错。"""

    import json

    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


__all__ = [
    "StreamOutcome",
    "StreamingClient",
    "collect_stream",
    "estimate_tokens",
    "estimate_tokens_from_chars",
    "iter_tokens",
    "sse_event",
    "supports_streaming",
]
