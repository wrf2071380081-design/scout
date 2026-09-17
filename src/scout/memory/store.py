"""三层记忆：工作记忆 / 情景记忆 / 语义记忆。

**为什么不是"再加一个向量库"。**

把历史对话塞进向量库、按相似度召回，这是最容易做的版本，也是最容易出问题的版本。
它有一个结构性缺陷：**向量相似度不区分"现在为真"和"曾经为真"**。
用户上个月说"我在做 Java"，这个月改口说"转 Go 了"，两条记忆的向量同样接近，
召回时谁排前面几乎随机。系统于是会自信地重复一个已经过期的事实——
这类失败在长周期对话里极其常见，而且用户一眼就能看出来。

本模块的核心设计是**把"失效"做成一等公民**：

- 每条记忆带有 ``valid_from`` / ``valid_until`` 的有效期窗口（bi-temporal 思路）
- 写入与已有事实冲突的新值时，旧事实**不是被删除，而是被标记失效**并指向继任者
- 召回默认**只返回当前有效**的记忆；要查历史必须显式声明
- 事实发生变更时，变更本身作为情景记忆保留下来，使"它什么时候变的"可追溯

三层各自的职责：

=============  ==========================  ==========================================
工作记忆        当前任务状态 + 最近若干轮对话   有界、可压缩；超出预算时做有损摘要（保事实丢过程）
情景记忆        带时间戳的具体事件              会过期；重复出现的情景会被**提升**为语义记忆
语义记忆        稳定的用户偏好与领域事实        带有效期窗口；支持冲突消解与历史追溯
=============  ==========================  ==========================================
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..config import MemorySettings
from ..llm.base import ChatMessage
from ..llm.scripted import tokenize


class MemoryKind(str, Enum):
    """记忆类型。"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class MemoryItem:
    """一条记忆。"""

    key: str
    kind: MemoryKind
    text: str
    scope: str = "default"
    subject: str = ""
    predicate: str = ""
    value: str = ""
    created_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)
    valid_until: float | None = None
    superseded_by: str = ""
    confidence: float = 1.0
    observations: int = 1
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def is_valid_at(self, when: float | None = None) -> bool:
        moment = time.time() if when is None else when
        if moment < self.valid_from:
            return False
        return self.valid_until is None or moment < self.valid_until

    @property
    def superseded(self) -> bool:
        return self.valid_until is not None

    def invalidate(self, *, when: float | None = None, successor: str = "") -> None:
        self.valid_until = time.time() if when is None else when
        self.superseded_by = successor

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "text": self.text,
            "scope": self.scope,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded_by": self.superseded_by or None,
            "confidence": self.confidence,
            "observations": self.observations,
            "source": self.source,
            "valid": self.is_valid_at(),
        }


@dataclass(slots=True)
class WorkingMemory:
    """工作记忆：最近若干轮对话 + 任务状态。

    超出 ``max_turns`` 时不是简单丢弃，而是做**有损压缩**：
    把被挤出的轮次压成一段摘要保留在 ``compacted`` 里。
    直接丢弃会让 Agent 忘掉前面已经确认过的决策，从而反复提问同一件事。
    """

    max_turns: int = 8
    turns: deque[tuple[str, str]] = field(default_factory=deque)
    compacted: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # deque 的 maxlen 必须在构造后显式绑定：dataclass 的 default_factory
        # 拿不到 max_turns，早期版本因此让容量参数完全失效（实测 8 轮上限
        # 在 max_turns=2 时仍保留 8 轮）。
        self.turns = deque(self.turns, maxlen=max(self.max_turns, 1))

    def add_turn(self, role: str, content: str) -> None:
        if len(self.turns) == self.turns.maxlen:
            evicted = self.turns[0]
            self.compacted.append(_summarize_turn(*evicted))
            if len(self.compacted) > self.max_turns:
                self.compacted = self.compacted[-self.max_turns :]
        self.turns.append((role, content))

    def as_messages(self) -> list[ChatMessage]:
        return [ChatMessage(role=role, content=content) for role, content in self.turns]

    def digest(self) -> str:
        if not self.compacted:
            return ""
        return "【早前对话摘要】\n" + "\n".join(f"- {item}" for item in self.compacted)

    def clear(self) -> None:
        self.turns.clear()
        self.compacted.clear()
        self.state.clear()


def _summarize_turn(role: str, content: str) -> str:
    """把一轮对话压成一句保留事实的摘要（启发式，不调模型）。"""

    prefix = "用户" if role == "user" else "助手"
    condensed = " ".join(content.split())
    if len(condensed) > 80:
        condensed = condensed[:80] + "…"
    return f"{prefix}：{condensed}"


class LayeredMemory:
    """三层记忆容器。

    :param scope: 隔离域（通常传 user_id 或 thread_id）。
        **跨作用域的记忆绝不互相召回**——这是最常见的记忆串味来源。
    """

    def __init__(self, *, settings: MemorySettings | None = None, scope: str = "default") -> None:
        self.settings = settings or MemorySettings()
        self.scope = scope
        self.working = WorkingMemory(max_turns=self.settings.working_max_turns)
        self._items: dict[str, MemoryItem] = {}
        self._facts: dict[tuple[str, str], str] = {}  # (subject, predicate) → 当前有效的 key
        self._counter = 0
        self._conflicts_resolved = 0
        self._expired = 0

    # —— 写入 ——

    def _next_key(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:05d}"

    def remember_episode(
        self,
        text: str,
        *,
        source: str = "",
        tags: Iterable[str] | None = None,
    ) -> MemoryItem:
        """记录一条情景记忆。"""

        item = MemoryItem(
            key=self._next_key("ep"),
            kind=MemoryKind.EPISODIC,
            text=text,
            scope=self.scope,
            source=source,
            tags=list(tags or ()),
            valid_until=time.time() + self.settings.default_ttl_seconds,
        )
        self._items[item.key] = item
        self._enforce_capacity(MemoryKind.EPISODIC, self.settings.episodic_max_items)
        return item

    def remember_fact(
        self,
        subject: str,
        value: str,
        *,
        predicate: str = "is",
        text: str = "",
        confidence: float = 1.0,
        source: str = "",
    ) -> tuple[MemoryItem, list[MemoryItem]]:
        """记录一条语义事实，并**消解与已有事实的冲突**。

        :return: ``(新事实, 被失效的旧事实列表)``

        冲突规则：
        - 同 ``(subject, predicate)`` 下有生效中的旧事实，且值不同 → 旧事实被标记失效，
          ``valid_until`` 置为当前时刻，``superseded_by`` 指向新事实；
        - 值相同 → 不新建，只增加 ``observations`` 计数（累积到阈值后可被提升为更高置信度）；
        - 值不同但旧事实已失效 → 直接新建，不动历史记录。
        """

        identity = (subject.strip(), predicate.strip())
        rendered = text or f"{subject} 的 {predicate} 是 {value}"
        existing_key = self._facts.get(identity)
        superseded: list[MemoryItem] = []

        if existing_key:
            existing = self._items.get(existing_key)
            if existing is not None and existing.is_valid_at() and existing.value == value:
                existing.observations += 1
                existing.confidence = min(existing.confidence + 0.05, 1.0)
                return existing, []
            if existing is not None and existing.is_valid_at():
                item = MemoryItem(
                    key=self._next_key("sem"),
                    kind=MemoryKind.SEMANTIC,
                    text=rendered,
                    scope=self.scope,
                    subject=identity[0],
                    predicate=identity[1],
                    value=value,
                    confidence=confidence,
                    source=source,
                )
                existing.invalidate(successor=item.key)
                superseded.append(existing)
                self._items[item.key] = item
                self._facts[identity] = item.key
                self._conflicts_resolved += 1
                # 变更本身作为情景记忆保留，使"什么时候变的"可追溯。
                self.remember_episode(
                    f"{subject} 的 {predicate} 由「{existing.value}」变更为「{value}」",
                    source=source,
                    tags=["fact_change"],
                )
                self._enforce_capacity(MemoryKind.SEMANTIC, self.settings.semantic_max_items)
                return item, superseded

        item = MemoryItem(
            key=self._next_key("sem"),
            kind=MemoryKind.SEMANTIC,
            text=rendered,
            scope=self.scope,
            subject=identity[0],
            predicate=identity[1],
            value=value,
            confidence=confidence,
            source=source,
        )
        self._items[item.key] = item
        self._facts[identity] = item.key
        self._enforce_capacity(MemoryKind.SEMANTIC, self.settings.semantic_max_items)
        return item, superseded

    def forget(self, key: str, *, reason: str = "") -> bool:
        item = self._items.pop(key, None)
        if item is None:
            return False
        item.invalidate()
        item.tags.append(f"forgotten:{reason}" if reason else "forgotten")
        return True

    # —— 读取 ——

    def recall(
        self,
        query: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 5,
        include_expired: bool = False,
        when: float | None = None,
    ) -> list[MemoryItem]:
        """按相关度召回记忆。

        默认 ``include_expired=False``——**已失效的事实不会出现在结果里**。
        这个默认值是本模块存在的主要理由。
        """

        wanted = set(kinds or (MemoryKind.SEMANTIC, MemoryKind.EPISODIC))
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items.values():
            if item.scope != self.scope or item.kind not in wanted:
                continue
            if not include_expired and not item.is_valid_at(when):
                continue
            overlap = len(query_tokens & set(tokenize(item.text)))
            if overlap == 0 and item.kind is MemoryKind.SEMANTIC:
                # 语义事实即便词面不重合也值得带出（例如偏好类事实），给一个低基线分。
                scored.append((0.05 * item.confidence, item))
                continue
            score = overlap / max(len(query_tokens), 1) + 0.2 * item.confidence
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].key))
        return [item for _score, item in scored[:limit]]

    def recall_for_prompt(self, query: str, *, limit: int = 5) -> str:
        """生成注入系统提示的记忆片段。"""

        items = self.recall(query, limit=limit)
        parts: list[str] = []
        digest = self.working.digest()
        if digest:
            parts.append(digest)
        if items:
            lines = []
            for item in items:
                marker = "事实" if item.kind is MemoryKind.SEMANTIC else "事件"
                lines.append(f"- [{marker}] {item.text}")
            parts.append("【相关记忆】\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def history_of(self, subject: str, predicate: str = "is") -> list[MemoryItem]:
        """返回某个事实的完整时间线（含已失效的版本），按生效时间排序。

        这是"事实更新"类评测的取证入口：先告诉系统 A、再改口 B，
        用这个方法可以直接检查它是否记录了变更、以及当前认为哪个值有效。
        """

        identity = (subject.strip(), predicate.strip())
        return sorted(
            (
                item
                for item in self._items.values()
                if (item.subject, item.predicate) == identity and item.kind is MemoryKind.SEMANTIC
            ),
            key=lambda item: item.valid_from,
        )

    def current_fact(self, subject: str, predicate: str = "is") -> MemoryItem | None:
        """返回当前有效的那个事实版本。"""

        for item in reversed(self.history_of(subject, predicate)):
            if item.is_valid_at():
                return item
        return None

    # —— 维护 ——

    def expire(self, *, now: float | None = None) -> int:
        """清理自然过期的情景记忆（保留语义事实的历史）。"""

        moment = time.time() if now is None else now
        removed = 0
        for key, item in list(self._items.items()):
            if item.kind is not MemoryKind.EPISODIC:
                continue
            if item.valid_until is not None and item.valid_until <= moment:
                del self._items[key]
                removed += 1
        self._expired += removed
        return removed

    def consolidate(self) -> int:
        """把重复出现的情景记忆提升为语义事实。

        这是"情景 → 语义"的固化：用户分三次说"回答简短点"，
        不应该留下三条互相竞争的情景记忆，而应该沉淀成一条稳定偏好。
        """

        groups: dict[str, list[MemoryItem]] = {}
        for item in self._items.values():
            if item.kind is MemoryKind.EPISODIC:
                groups.setdefault(item.text.strip(), []).append(item)
        promoted = 0
        for text, items in groups.items():
            valid = [item for item in items if item.is_valid_at()]
            if len(valid) < self.settings.consolidation_min_repeats:
                continue
            existing = next(
                (item for item in self._items.values() if item.kind is MemoryKind.SEMANTIC and item.text == text),
                None,
            )
            if existing is not None:
                existing.observations += len(valid)
                continue
            self.remember_fact(
                subject=text,
                value="confirmed",
                predicate="repeated_preference",
                text=text,
                confidence=0.8,
                source="consolidation",
            )
            promoted += 1
        return promoted

    def _enforce_capacity(self, kind: MemoryKind, limit: int) -> None:
        """超容时优先淘汰已失效、低置信、旧的条目。"""

        same_kind = [item for item in self._items.values() if item.kind is kind]
        if len(same_kind) <= limit:
            return
        ordered = sorted(
            same_kind,
            key=lambda item: (item.is_valid_at(), item.confidence, item.created_at),
        )
        for item in ordered[: len(same_kind) - limit]:
            if item.kind is MemoryKind.SEMANTIC and (item.subject, item.predicate) in self._facts:
                if self._facts[(item.subject, item.predicate)] == item.key:
                    continue
            self._items.pop(item.key, None)

    # —— 诊断 ——

    def stats(self) -> dict[str, Any]:
        counts = Counter(item.kind.value for item in self._items.values())
        return {
            "scope": self.scope,
            "items_total": len(self._items),
            "items_by_kind": dict(counts),
            "valid_items": sum(1 for item in self._items.values() if item.is_valid_at()),
            "superseded_items": sum(1 for item in self._items.values() if item.superseded),
            "conflicts_resolved": self._conflicts_resolved,
            "expired_items": self._expired,
            "working_turns": len(self.working.turns),
            "working_compacted": len(self.working.compacted),
        }


__all__ = ["LayeredMemory", "MemoryItem", "MemoryKind", "WorkingMemory"]
