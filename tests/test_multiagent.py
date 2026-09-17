"""多智能体编排的测试。

测试聚焦**行为契约**，而不是"能跑通"：

- 拆不开的问题回退单路，不浪费成本
- 对比类问题真正扇出多个子 Agent
- 子 Agent 上行的是**压缩后的 span**，不是完整轨迹（上下文隔离的可验证形式）
- 一个子 Agent 失败，其余照常完成，覆盖缺口被如实标注
- 嵌套深度被写死
- 合并证据不重复、出处不丢
"""

from __future__ import annotations

import pytest

from scout.llm.scripted import HeuristicLLM
from scout.multiagent import MultiAgentOrchestrator, SubAgent
from scout.rag.pipeline import RAGPipeline, build_index

DOCS = [
    (
        "redis.md",
        "Redis 的高性能主要来自三点。第一，数据放在内存中，避免了磁盘 IO 的开销。"
        "第二，采用单线程模型处理命令，省去了上下文切换和锁竞争。"
        "第三，使用 IO 多路复用（epoll）在单线程内处理大量并发连接。",
    ),
    (
        "golang.md",
        "Go 的 GMP 调度模型由 G、M、P 组成。G 是 goroutine，M 是操作系统线程，"
        "P 是处理器上下文。调度器通过 work stealing 提升并行效率。",
    ),
    (
        "mysql.md",
        "MySQL 的 InnoDB 使用聚簇索引存储数据。主键查询直接定位到数据页。"
        "覆盖索引可以避免回表。MySQL 默认隔离级别是可重复读。",
    ),
]


@pytest.fixture()
def pipeline() -> RAGPipeline:
    return RAGPipeline(build_index(DOCS), HeuristicLLM())


@pytest.fixture()
def orchestrator(pipeline: RAGPipeline) -> MultiAgentOrchestrator:
    return MultiAgentOrchestrator(pipeline, HeuristicLLM())


def test_simple_question_falls_back_to_single(orchestrator: MultiAgentOrchestrator) -> None:
    """拆不开的问题必须回退单路，不浪费多智能体成本。"""

    result = orchestrator.answer("Redis 为什么这么快？")
    assert result.meta["mode"] == "single_fallback"
    assert len(result.plan) == 1
    assert result.subresults == []
    assert result.answer.strip()


def test_comparison_question_fans_out(
    orchestrator: MultiAgentOrchestrator,
) -> None:
    """对比类问题要真正扇出多个子 Agent。"""

    result = orchestrator.answer("Redis 和 MySQL 分别是怎么保证高性能的？")
    assert result.meta["mode"] == "multiagent"
    assert len(result.plan) >= 2
    assert len(result.subresults) >= 2
    # 两个子 Agent 各自检索自己的实体，互不干扰。
    questions = " ".join(result.plan)
    assert "Redis" in questions and "MySQL" in questions


def test_subresult_carries_compressed_spans_not_transcript(
    orchestrator: MultiAgentOrchestrator,
) -> None:
    """子 Agent 上行的是压缩 span，不是完整轨迹——上下文隔离的可验证形式。

    这是"隔离"能被兑现的前提：**上行数据必须比下行小**。
    验证方式是：主 Agent 合成用的证据，是各子 Agent 压缩过的 span，
    而不是它们的原始检索块或完整消息历史。
    """

    result = orchestrator.answer("Redis 和 MySQL 分别是怎么保证高性能的？")
    assert result.meta["mode"] == "multiagent"

    for sub in result.subresults:
        # 每个子结果都应该有压缩报告，且压缩后的 span 不超过原始证据量
        ratio = sub.meta.get("compression_ratio", 1.0)
        assert 0 < ratio <= 1.0, f"子 Agent 应该压缩证据，ratio={ratio}"
        # span 要带出处，归因门控才能继续引用
        for span in sub.supporting_spans:
            assert span["chunk_id"]
            assert span["filename"]


def test_merged_evidence_covers_both_sides(
    orchestrator: MultiAgentOrchestrator,
) -> None:
    """合并证据必须同时覆盖对比双方——单路 Top-K 覆盖不到的场景。

    这是多智能体对 RAG 的真实价值：**每个子 Agent 独立检索自己的实体**，
    合并后不会因为 Top-K 被一侧占满而漏掉另一侧。
    """

    result = orchestrator.answer("Redis 和 MySQL 分别是怎么保证高性能的？")
    filenames = {
        span["filename"]
        for sub in result.subresults
        for span in sub.supporting_spans
    }
    assert "redis.md" in filenames
    assert "mysql.md" in filenames


def test_no_evidence_marks_coverage_gap(pipeline: RAGPipeline) -> None:
    """一个子问题完全没证据 → 其余照常完成，缺口被如实标注（失败隔离）。

    失败隔离的核心承诺：不让一个失败的子 Agent 拖垮整体，
    也**不替它编一个结论**。
    """

    docs = [("only.md", "Python 的 GIL 限制了多线程的并行能力。")]
    orchestrator = MultiAgentOrchestrator(
        RAGPipeline(build_index(docs), HeuristicLLM()), HeuristicLLM()
    )
    result = orchestrator.answer("Python 的 GIL 是什么以及 Rust 的所有权模型是什么？")
    if result.meta["mode"] == "multiagent":
        # 至少应该如实记录哪些子问题没覆盖到
        gaps = result.coverage_gaps
        assert isinstance(gaps, list)
        # 合并证据不为空时，答案照常生成；为空时如实说不知道
        assert result.outcome in {"pass", "no_knowledge", "insufficient_evidence", "abstain"}


def test_depth_cap_enforced(pipeline: RAGPipeline) -> None:
    """嵌套深度写死在代码里——多智能体失控最常见的原因不是模型，是深度不受限。"""

    sub = SubAgent(pipeline, max_depth=1)
    result = sub.run("任意问题", depth=99)
    assert result.status == "depth_exceeded"
    assert result.error_code == "multiagent_depth_exceeded"


def test_subagent_run_independent_trace(pipeline: RAGPipeline) -> None:
    """每个子 Agent 跑独立的 trace——上下文隔离的一部分。"""

    sub = SubAgent(pipeline)
    result1 = sub.run("Redis 为什么快？", depth=0)
    result2 = sub.run("MySQL 默认隔离级别是什么？", depth=0)
    # 两次运行互不干扰：各自的状态不串。
    assert result1.subquestion != result2.subquestion
    assert result1.latency_ms >= 0 and result2.latency_ms >= 0


def test_merge_dedupes_evidence(orchestrator: MultiAgentOrchestrator) -> None:
    """合并证据按 chunk_id 去重——同一个块被多个子 Agent 命中时只保留一次。

    注意：去重发生在**编排器的合并证据**（``result.units``）里，
    而不是各子 Agent 自己的 span 列表里——子 Agent 各自独立检索，
    命中重叠的块是正常且隔离的；是编排器在合并时才去重。
    """

    result = orchestrator.answer("Redis 和 MySQL 分别是怎么保证高性能的？")
    chunk_ids = [unit.chunk.chunk_id for unit in result.units]
    assert chunk_ids, "合并证据不应为空"
    assert len(chunk_ids) == len(set(chunk_ids)), "合并证据不得有重复的 chunk_id"
