"""三层记忆的测试。重点是**失效与冲突消解**，而不是存储。"""

from __future__ import annotations

import time

from scout.config import MemorySettings
from scout.memory.store import LayeredMemory, MemoryKind


def test_fact_update_supersedes_instead_of_overwriting() -> None:
    memory = LayeredMemory(scope="u1")
    old, _ = memory.remember_fact("用户主语言", "Java")
    new, superseded = memory.remember_fact("用户主语言", "Go")

    assert [item.key for item in superseded] == [old.key]
    assert old.superseded_by == new.key
    assert old.is_valid_at() is False
    assert new.is_valid_at() is True
    assert memory.current_fact("用户主语言").value == "Go"


def test_recall_excludes_superseded_by_default() -> None:
    """这是本模块存在的核心理由：过期事实不得出现在默认召回里。"""

    memory = LayeredMemory(scope="u1")
    memory.remember_fact("用户主语言", "Java")
    memory.remember_fact("用户主语言", "Go")

    texts = [item.text for item in memory.recall("用户主语言", kinds=[MemoryKind.SEMANTIC], limit=10)]
    assert texts, "当前有效事实应当被召回"
    assert all("Java" not in text for text in texts)

    with_history = [
        item.text
        for item in memory.recall("用户主语言", kinds=[MemoryKind.SEMANTIC], limit=10, include_expired=True)
    ]
    assert any("Java" in text for text in with_history)


def test_history_records_the_change_timeline() -> None:
    memory = LayeredMemory(scope="u1")
    memory.remember_fact("部署环境", "测试集群")
    memory.remember_fact("部署环境", "生产集群")

    history = memory.history_of("部署环境")
    assert len(history) == 2
    assert history[0].value == "测试集群"
    assert history[1].value == "生产集群"
    assert history[0].valid_until is not None
    assert history[1].valid_until is None

    # 事实变更本身应当作为情景记忆被保留下来，"什么时候变的"可追溯。
    episodes = [item for item in memory.recall("部署环境", kinds=[MemoryKind.EPISODIC], limit=10)]
    assert any("变更为" in item.text for item in episodes)


def test_same_value_reinforces_instead_of_duplicating() -> None:
    memory = LayeredMemory(scope="u1")
    first, _ = memory.remember_fact("回答风格", "简洁")
    again, superseded = memory.remember_fact("回答风格", "简洁")

    assert again.key == first.key
    assert superseded == []
    assert again.observations == 2


def test_scope_isolation() -> None:
    """跨作用域召回是最常见的记忆串味来源，必须被隔离。"""

    alice = LayeredMemory(scope="alice")
    bob = LayeredMemory(scope="bob")
    alice.remember_fact("主语言", "Java")
    bob.remember_fact("主语言", "Go")

    assert alice.current_fact("主语言").value == "Java"
    assert bob.current_fact("主语言").value == "Go"
    assert alice.recall("主语言", limit=5)[0].scope == "alice"


def test_episodic_expiry() -> None:
    settings = MemorySettings(default_ttl_seconds=1)
    memory = LayeredMemory(settings=settings, scope="u1")
    memory.remember_episode("用户问了一个问题")
    assert memory.expire(now=time.time() + 3600) == 1
    assert memory.recall("问题", kinds=[MemoryKind.EPISODIC], limit=5) == []


def test_consolidation_promotes_repeated_episodes() -> None:
    memory = LayeredMemory(settings=MemorySettings(consolidation_min_repeats=2), scope="u1")
    memory.remember_episode("用户希望回答简短")
    memory.remember_episode("用户希望回答简短")
    promoted = memory.consolidate()
    assert promoted == 1
    semantic = [item for item in memory.recall("回答简短", kinds=[MemoryKind.SEMANTIC], limit=5)]
    assert semantic


def test_working_memory_compacts_instead_of_dropping() -> None:
    memory = LayeredMemory(settings=MemorySettings(working_max_turns=2), scope="u1")
    for index in range(5):
        memory.working.add_turn("user", f"第 {index} 轮问题内容")
        memory.working.add_turn("assistant", f"第 {index} 轮回答内容")

    assert len(memory.working.turns) == 2
    assert memory.working.compacted, "被挤出的轮次必须被压缩保留，而不是静默丢弃"
    assert "早前对话摘要" in memory.working.digest()


def test_stats_report_conflicts() -> None:
    memory = LayeredMemory(scope="u1")
    memory.remember_fact("a", "1")
    memory.remember_fact("a", "2")
    stats = memory.stats()
    assert stats["conflicts_resolved"] == 1
    assert stats["superseded_items"] >= 1
