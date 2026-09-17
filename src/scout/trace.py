"""结构化链路追踪。

设计原则（这几点是从生产事故里总结出来的，不是洁癖）：

1. **Trace 不是日志。** 它要能被序列化、被程序断言、被评测脚本消费，
   所以字段必须稳定、有类型，不能是拼给人看的字符串。
2. **不存正文，只存身份。** 记录 ``chunk_id`` / ``document_id`` / 长度 / 分数，
   不记录 ``chunk.text``。理由有两个：trace 会被写进数据库和评测报告，
   正文进去会撑爆体积；而原文可能含敏感内容，一旦落库就很难回收。
3. **不存密钥。** 递归过滤 key 名含 token / key / secret / password 的字段。
4. **每步都要有耗时。** 没有分阶段耗时，就无法回答"慢在检索还是慢在生成"。
"""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

_SENSITIVE_KEY_HINTS = ("token", "secret", "password", "passwd", "api_key", "apikey", "authorization")
_MAX_PREVIEW_CHARS = 200


class StepKind(str, Enum):
    """步骤类型。评测时按 kind 分组聚合耗时与失败率。"""

    PLAN = "plan"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    MERGE = "merge"
    REWRITE = "rewrite"
    GRADE = "grade"
    TOOL = "tool"
    MEMORY = "memory"
    GENERATE = "generate"
    VERIFY = "verify"
    RECOVER = "recover"
    HUMAN = "human"
    """人工介入（审批/编辑/复核）。单独成类是因为它的耗时不在系统内，
    把"等人"的时间混进"处理"的时间会让性能指标失真。——评测里按 kind 分组时
    应当把它单独列出来，或者默认排除。"""


def redact(value: Any) -> Any:
    """递归脱敏并截断长文本。

    - 键名命中敏感词 → 整体替换为 ``"***"``
    - 长字符串 → 截断并附长度标记
    - dict / list / tuple → 递归处理
    """

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(hint in lowered for hint in _SENSITIVE_KEY_HINTS):
                result[str(key)] = "***"
            else:
                result[str(key)] = redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if len(value) > _MAX_PREVIEW_CHARS:
            return f"{value[:_MAX_PREVIEW_CHARS]}…[+{len(value) - _MAX_PREVIEW_CHARS}]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:_MAX_PREVIEW_CHARS]


@dataclass(slots=True)
class TraceStep:
    """单个步骤。

    ``itertools`` 无法直接挂属性，所以这里用显式对象而不是生成器状态。
    """

    name: str
    kind: StepKind
    duration_ms: float = 0.0
    started_at: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_retryable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "duration_ms": round(self.duration_ms, 3),
            "inputs": redact(self.inputs),
            "outputs": redact(self.outputs),
            "metrics": redact(self.metrics),
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code
            payload["error_retryable"] = self.error_retryable
        return payload


@dataclass(slots=True)
class Trace:
    """一次运行的完整链路。"""

    question: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[TraceStep] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # —— 记录 ——

    @contextlib.contextmanager
    def step(self, label: str, kind: StepKind, **inputs: Any) -> Iterator[TraceStep]:
        """上下文管理器：自动计时，异常时记录 error_code 而不吞异常。

        第一个参数叫 ``label`` 而不是 ``name``，是为了让 ``**inputs`` 能安全接收
        名为 ``name`` 的字段（例如工具调用名 ``name="calculator"``）——
        否则会与形参冲突并抛 ``got multiple values for argument``。
        """

        item = TraceStep(name=label, kind=kind, started_at=time.perf_counter(), inputs=dict(inputs))
        try:
            yield item
        except Exception as exc:  # noqa: BLE001 - 这里必须记录所有异常再抛出
            code = getattr(exc, "code", None)
            item.error_code = getattr(code, "value", None) or type(exc).__name__
            item.error_retryable = bool(getattr(exc, "retryable", False))
            raise
        finally:
            item.duration_ms = (time.perf_counter() - item.started_at) * 1000.0
            self.steps.append(item)

    # —— 查询 ——

    def kind_durations(self) -> dict[str, float]:
        """按步骤类型汇总耗时（毫秒）。这是定位性能瓶颈的主要手段。"""

        totals: dict[str, float] = {}
        for item in self.steps:
            totals[item.kind.value] = totals.get(item.kind.value, 0.0) + item.duration_ms
        return {key: round(value, 3) for key, value in totals.items()}

    def total_duration_ms(self) -> float:
        return round(sum(item.duration_ms for item in self.steps), 3)

    def failures(self) -> list[TraceStep]:
        return [item for item in self.steps if item.error_code is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": redact(self.question),
            "total_duration_ms": self.total_duration_ms(),
            "kind_durations_ms": self.kind_durations(),
            "steps": [item.to_dict() for item in self.steps],
            "meta": redact(self.meta),
        }


__all__ = ["StepKind", "Trace", "TraceStep", "redact"]
