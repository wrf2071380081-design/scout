"""检查点：Agent 运行状态的可序列化快照与持久化。

===================  ==============================================================
**为什么不 pickle 对象图**

最朴素的做法是 ``pickle.dumps(agent)`` 把整个运行时对象图存下来。它能跑通 demo，
但在真实系统里几乎必然出问题：

1. **对代码变更零容忍。** 改一个字段名、调整一个类的位置，历史快照全部反序列化失败——
   而这正是"恢复一个三天前的运行"最需要的时刻。
2. **存进去了不该存的东西。** LLM 客户端、连接池、锁、回调闭包都会被一起序列化，
   要么失败，要么把密钥和内部实现细节写进磁盘。
3. **人无法审查或修改。** 一个二进制 blob 没法被运维看懂，也没法在恢复前
   手工修补一个字段——而"人工修正状态后续跑"恰恰是 HITL 场景的日常需求。

所以本模块坚持：**状态用数据（dict / list / 标量）表示，且只包含可 JSON 化的内容。**
这条约束会反过来影响运行时的设计——所有需要跨中断存活的东西，
必须在被放进状态时就转成数据。这个"不方便"是刻意的，它是可恢复性的前提。
===================  ==============================================================

另一个决定是**存储形态：append-only 的 JSONL**。

每次检查点追加一行，从不覆盖。这样：

- **时间旅行天然可用**：历史版本都还在，按 ``step`` 取即可
- **可审计**：谁能改历史？改不了，追加日志本身就是证据
- **崩溃安全**：写入是 O(1) 追加，不需要读-改-写，进程中途死掉最多丢最后一行

文件级持久化用 ``FileCheckpointStore``，每个 run 一个 ``<run_id>.jsonl``。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from ..errors import ErrorCode, ScoutError
from ..llm.base import ChatMessage, TokenUsage, ToolCall


# —— 消息与工具调用的序列化 ——


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": dict(call.arguments), "id": call.id}


def tool_call_from_dict(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=str(payload.get("name", "")),
        arguments=dict(payload.get("arguments") or {}),
        id=str(payload.get("id", "")),
    )


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [tool_call_to_dict(call) for call in message.tool_calls]
    return payload


def message_from_dict(payload: dict[str, Any]) -> ChatMessage:
    return ChatMessage(
        role=str(payload.get("role", "user")),
        content=str(payload.get("content", "")),
        name=str(payload.get("name", "")),
        tool_call_id=str(payload.get("tool_call_id", "")),
        tool_calls=[tool_call_from_dict(item) for item in payload.get("tool_calls") or []],
    )


def messages_to_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(message) for message in messages]


def messages_from_dicts(payloads: Sequence[dict[str, Any]]) -> list[ChatMessage]:
    return [message_from_dict(item) for item in payloads]


# —— 检查点 ——


@dataclass(slots=True)
class Checkpoint:
    """一次状态快照。"""

    run_id: str
    step: int
    state: dict[str, Any]
    label: str = ""
    created_at: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step": self.step,
            "label": self.label,
            "created_at": self.created_at,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Checkpoint:
        version = int(payload.get("schema_version", 1))
        if version != 1:
            # 版本不匹配时宁可明确失败，也不要"尽力解析"出一个半对的状态：
            # 半对的状态比没有状态更危险，它会让后续步骤基于错误的上下文继续跑。
            raise ScoutError(
                f"检查点 schema 版本 {version} 不受支持（当前支持 1）",
                code=ErrorCode.VALIDATION_FAILED,
            )
        return cls(
            run_id=str(payload.get("run_id", "")),
            step=int(payload.get("step", 0)),
            label=str(payload.get("label", "")),
            created_at=float(payload.get("created_at", 0.0)),
            schema_version=version,
            state=dict(payload.get("state") or {}),
        )

    def digest(self) -> str:
        """状态指纹。用于校验"我恢复的确实是那一份状态"。"""

        canonical = json.dumps(
            {"run_id": self.run_id, "step": self.step, "state": self.state},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CheckpointStore(Protocol):
    """检查点存储接口。"""

    def save(self, checkpoint: Checkpoint) -> None: ...

    def latest(self, run_id: str) -> Checkpoint | None: ...

    def at(self, run_id: str, step: int) -> Checkpoint | None: ...

    def history(self, run_id: str) -> list[Checkpoint]: ...

    def runs(self) -> list[str]: ...


class InMemoryCheckpointStore:
    """内存存储。测试与单进程演示用。"""

    def __init__(self) -> None:
        self._by_run: dict[str, list[Checkpoint]] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        self._by_run.setdefault(checkpoint.run_id, []).append(checkpoint)

    def latest(self, run_id: str) -> Checkpoint | None:
        items = self._by_run.get(run_id) or []
        return items[-1] if items else None

    def at(self, run_id: str, step: int) -> Checkpoint | None:
        for checkpoint in reversed(self._by_run.get(run_id) or []):
            if checkpoint.step == step:
                return checkpoint
        return None

    def history(self, run_id: str) -> list[Checkpoint]:
        return list(self._by_run.get(run_id) or [])

    def runs(self) -> list[str]:
        return sorted(self._by_run)


class FileCheckpointStore:
    """文件存储：每个 run 一个 append-only 的 JSONL。

    :param root: 存放目录。写入用 ``flush + fsync``，
        保证"进程被 kill 掉之后，已确认的中断点仍然可恢复"——
        对 HITL 场景这是硬要求：用户看到的是"等待审批"，
        如果这时进程崩了而状态丢了，那次审批就永远等不到结果。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        safe = "".join(char for char in run_id if char.isalnum() or char in "-_.")
        if not safe:
            raise ScoutError("非法的 run_id", code=ErrorCode.VALIDATION_FAILED)
        return self.root / f"{safe}.jsonl"

    def save(self, checkpoint: Checkpoint) -> None:
        path = self._path(checkpoint.run_id)
        line = json.dumps(checkpoint.to_dict(), ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def history(self, run_id: str) -> list[Checkpoint]:
        path = self._path(run_id)
        if not path.exists():
            return []
        items: list[Checkpoint] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(Checkpoint.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ScoutError):
                # 最后一行可能因崩溃写坏——跳过它比整体失败更合理，
                # 因为 append-only 日志的语义就是"前面写成功了"。
                continue
        return items

    def latest(self, run_id: str) -> Checkpoint | None:
        items = self.history(run_id)
        return items[-1] if items else None

    def at(self, run_id: str, step: int) -> Checkpoint | None:
        for checkpoint in reversed(self.history(run_id)):
            if checkpoint.step == step:
                return checkpoint
        return None

    def runs(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.jsonl"))


# —— 状态契约 ——


@dataclass(slots=True)
class AgentState:
    """可中断 Agent 的完整状态。

    **这是唯一需要跨中断存活的东西**，所以它只包含数据：
    ``messages`` 是 dict 列表而不是 :class:`ChatMessage` 对象，
    副作用记录是 dict 列表而不是对象——见模块开头对"为什么不 pickle"的说明。
    """

    run_id: str
    question: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    status: str = "running"
    answer: str = ""
    pending_step: dict[str, Any] = field(default_factory=dict)
    """进行中的那一步。中断发生在某个工具调用循环中间时，恢复必须从**那次调用**
    继续，而不是跳到下一步或让模型重跑一次。这个字段存的是：
    ``{step, tool_calls: [...], next_index, request: {...}}``。

    它的缺失是大多数"自研 HITL"实现里最隐蔽的 bug——
    看上去 resume 回来了，实际上把上一步的后半段整个吃掉了。"""
    decisions: list[dict[str, Any]] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "messages": list(self.messages),
            "step": self.step,
            "status": self.status,
            "answer": self.answer,
            "pending_step": dict(self.pending_step),
            "decisions": list(self.decisions),
            "effects": list(self.effects),
            "call_log": list(self.call_log),
            "meta": dict(self.meta),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentState:
        return cls(
            run_id=str(payload.get("run_id", "")),
            question=str(payload.get("question", "")),
            messages=[dict(item) for item in payload.get("messages") or []],
            step=int(payload.get("step", 0)),
            status=str(payload.get("status", "running")),
            answer=str(payload.get("answer", "")),
            pending_step=dict(payload.get("pending_step") or {}),
            decisions=[dict(item) for item in payload.get("decisions") or []],
            effects=[dict(item) for item in payload.get("effects") or []],
            call_log=[dict(item) for item in payload.get("call_log") or []],
            meta=dict(payload.get("meta") or {}),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
        )

    def token_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)


class RedisCheckpointStore:
    """Redis 存储：把状态从进程里搬出去。

    **为什么状态必须外置**（这是"算力受限/多副本"场景下的硬要求）：

    - 进程随时可能被调度、重启、扩缩容——状态放内存就跟着一起没了；
    - 多副本要共享同一份会话状态，才能把请求负载均衡到任意一台；
    - HITL 场景更狠：用户看到的是"等待审批"，如果这时进程重启而状态丢了，
      那次审批永远等不到结果，**会话永久卡死**。

    代价是多一跳网络往返；用连接池与管道把这一跳压到亚毫秒级即可接受。

    **存储形态与文件版保持一致**：同样用 append-only 列表
    （Redis LIST 的 ``RPUSH``），这样时间旅行、审计、"崩溃最多丢最后一行"
    这些性质在两个后端上是同一套语义——**换后端不该换语义**。

    :param url: redis://host:port/db，或直接注入 ``client``（测试用）。
    :param ttl_seconds: 过期时间。生产上要设——不然 run 会无限堆积。
    """

    SCHEMA = "scout:ckpt"

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        client: Any = None,
        prefix: str = SCHEMA,
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        if client is not None:
            self._client = client
        else:
            try:
                import redis  # noqa: PLC0415 - 可选依赖，用到才导入
            except ImportError as exc:  # pragma: no cover - 取决于环境
                raise ScoutError(
                    "未安装 redis 客户端，无法使用 RedisCheckpointStore。"
                    "安装：pip install redis（或把 CheckpointStore 换成 FileCheckpointStore）",
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                ) from exc
            self._client = redis.Redis.from_url(url, decode_responses=True)

    def _key(self, run_id: str) -> str:
        safe = "".join(char for char in run_id if char.isalnum() or char in "-_.")
        if not safe:
            raise ScoutError("非法的 run_id", code=ErrorCode.VALIDATION_FAILED)
        return f"{self.prefix}:{safe}"

    def save(self, checkpoint: Checkpoint) -> None:
        key = self._key(checkpoint.run_id)
        payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False, default=str)
        pipe = self._client.pipeline()
        pipe.rpush(key, payload)
        if self.ttl_seconds > 0:
            # 每次写入续期：活跃会话不该因为"最后写入了 7 天前"而消失
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def history(self, run_id: str) -> list[Checkpoint]:
        raw = self._client.lrange(self._key(run_id), 0, -1) or []
        items: list[Checkpoint] = []
        for line in raw:
            text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
            try:
                items.append(Checkpoint.from_dict(json.loads(text)))
            except (json.JSONDecodeError, ScoutError):
                # 与文件版同一约定：坏行跳过。append-only 的语义是"前面写成功了"。
                continue
        return items

    def latest(self, run_id: str) -> Checkpoint | None:
        raw = self._client.lindex(self._key(run_id), -1)
        if not raw:
            return None
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            return Checkpoint.from_dict(json.loads(text))
        except (json.JSONDecodeError, ScoutError):
            items = self.history(run_id)
            return items[-1] if items else None

    def at(self, run_id: str, step: int) -> Checkpoint | None:
        for checkpoint in reversed(self.history(run_id)):
            if checkpoint.step == step:
                return checkpoint
        return None

    def runs(self) -> list[str]:
        keys = self._client.keys(f"{self.prefix}:*") or []
        prefix_len = len(self.prefix) + 1
        return sorted(
            (key.decode("utf-8") if isinstance(key, bytes) else str(key))[prefix_len:] for key in keys
        )


def build_checkpoint_store(url: str = "", *, root: str | None = None) -> CheckpointStore:
    """按配置挑一个存储后端，**并且明确告诉调用方挑到了哪一个**。

    选择顺序：显式 Redis URL → 文件目录 → 内存。
    之所以不让它"自动尝试 Redis 失败再降级"：状态存储降级是**重大语义变化**
    （从"跨进程可恢复"变成"重启即丢"），必须由调用方显式决定，
    而不是被一个隐式 fallback 悄悄换掉。
    """

    if url:
        return RedisCheckpointStore(url)
    if root:
        return FileCheckpointStore(root)
    return InMemoryCheckpointStore()


__all__ = [
    "AgentState",
    "Checkpoint",
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "RedisCheckpointStore",
    "build_checkpoint_store",
    "message_from_dict",
    "message_to_dict",
    "messages_from_dicts",
    "messages_to_dicts",
    "tool_call_from_dict",
    "tool_call_to_dict",
]
