"""Typed 错误分类。

核心设计：**把「检索确实为空」和「检索失败了」严格分开**。

这是 RAG/Agent 系统里最常见也最致命的混淆——把 Provider 异常吞掉返回空列表，
下游就会把「服务挂了」渲染成「知识库中没有相关信息」，用户拿到一个自信的错误答案，
而运维在监控上看不到任何异常。本模块用类型系统强制区分：

===========================  ==========================================
``NoKnowledgeError``         检索健康、结果确实为空
``InsufficientEvidenceError`` 检索成功，但证据不足以支撑回答
``ProviderError``            外部依赖失败（网络 / 超时 / 限流 / 响应畸形）
``ToolError``                工具调用失败（参数非法 / 权限拒绝 / 执行异常）
``BudgetExceededError``      超出步数、token 或耗时预算
``ValidationError``          输入或数据契约不合法
===========================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """稳定错误码。对外暴露、写日志、做监控告警都依赖它，不要随文案改动。"""

    # —— 语义结果 ——
    NO_KNOWLEDGE = "NO_KNOWLEDGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    # —— Provider 故障 ——
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_CONNECTION = "PROVIDER_CONNECTION"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_INVALID_RESPONSE = "PROVIDER_INVALID_RESPONSE"

    # —— 工具与预算 ——
    TOOL_INVALID_ARGUMENTS = "TOOL_INVALID_ARGUMENTS"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    # —— 数据契约 ——
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INDEX_NOT_READY = "INDEX_NOT_READY"


class ScoutError(Exception):
    """所有 scout 异常的基类。

    :param message: 面向开发者的描述
    :param code: 稳定错误码
    :param retryable: 是否值得重试（决定自愈编排是否换一条恢复路径）
    :param details: 额外的结构化上下文，写入 trace；**不要放密钥或文档全文**
    """

    default_code: ErrorCode = ErrorCode.VALIDATION_FAILED
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return f"{type(self).__name__}(code={self.code.value!r}, retryable={self.retryable})"


class NoKnowledgeError(ScoutError):
    """检索健康且确实没有可用材料。

    这是**合法结果**，不是故障。Agent 应当据此诚实地回答「知识库中没有相关内容」，
    而不是编造答案。
    """

    default_code = ErrorCode.NO_KNOWLEDGE
    default_retryable = False


class InsufficientEvidenceError(ScoutError):
    """检索到了内容，但证据不足以支撑一个可靠回答。

    与 :class:`NoKnowledgeError` 的区别：这里有候选材料，只是覆盖不全、互相冲突，
    或关键前提缺失。正确行为是请求澄清或限定回答范围，而不是硬答。
    """

    default_code = ErrorCode.INSUFFICIENT_EVIDENCE
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        missing: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        details = dict(kwargs.pop("details", {}) or {})
        if missing:
            details["missing"] = list(missing)
        super().__init__(message, details=details, **kwargs)


class ProviderError(ScoutError):
    """外部依赖失败。

    :class:`ProviderError` 永远不应该被转成「无知识」。上层的自愈编排可以基于
    ``code`` 与 ``retryable`` 决定换用哪条恢复路径（重试 / 降级 / 换 Provider）。
    """

    default_code = ErrorCode.PROVIDER_UNAVAILABLE
    default_retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        operation: str = "",
        attempts: int = 1,
        **kwargs: Any,
    ) -> None:
        details = dict(kwargs.pop("details", {}) or {})
        details.update({"provider": provider, "operation": operation, "attempts": attempts})
        super().__init__(message, details=details, **kwargs)

    @property
    def provider(self) -> str:
        return str(self.details.get("provider", ""))

    @property
    def attempts(self) -> int:
        return int(self.details.get("attempts", 1))


class ToolError(ScoutError):
    """工具调用失败。"""

    default_code = ErrorCode.TOOL_EXECUTION_FAILED
    default_retryable = False

    def __init__(self, message: str, *, tool: str = "", **kwargs: Any) -> None:
        details = dict(kwargs.pop("details", {}) or {})
        details["tool"] = tool
        super().__init__(message, details=details, **kwargs)


class BudgetExceededError(ScoutError):
    """超出预算（步数 / token / 墙钟时间）。

    预算是 Agent 可控性的核心：没有预算的 Agent 会陷入无限循环，
    或者在一次请求里烧掉不可预期的费用。
    """

    default_code = ErrorCode.BUDGET_EXCEEDED
    default_retryable = False

    def __init__(self, message: str, *, budget: str = "", limit: float | None = None, **kw: Any):
        details = dict(kw.pop("details", {}) or {})
        details["budget"] = budget
        if limit is not None:
            details["limit"] = limit
        super().__init__(message, details=details, **kw)


class ValidationError(ScoutError):
    """数据或输入契约不合法。"""

    default_code = ErrorCode.VALIDATION_FAILED
    default_retryable = False


@dataclass(slots=True)
class FailureRecord:
    """一次失败的归一化记录，用于失败分类学统计。"""

    code: ErrorCode
    stage: str
    retryable: bool
    recovered: bool = False
    recovery_action: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "retryable": self.retryable,
            "recovered": self.recovered,
            "recovery_action": self.recovery_action,
            "details": self.details,
        }


__all__ = [
    "BudgetExceededError",
    "ErrorCode",
    "FailureRecord",
    "InsufficientEvidenceError",
    "NoKnowledgeError",
    "ProviderError",
    "ScoutError",
    "ToolError",
    "ValidationError",
]
