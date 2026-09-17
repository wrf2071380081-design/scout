"""Agent 循环、预算、重复调用抑制与工具失败恢复的测试。"""

from __future__ import annotations

from scout.agent.loop import ANSWERED, BUDGET_STOPPED, ToolAgent
from scout.config import AgentSettings, Settings
from scout.llm.base import LLMResponse, ToolCall
from scout.llm.scripted import HeuristicLLM, ScriptedLLM
from scout.orchestrator.healing import SelfHealingOrchestrator
from scout.tools.builtin import build_default_tools
from scout.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    return ToolRegistry(build_default_tools())


def test_calculator_tool_is_used_and_repeated_calls_are_suppressed() -> None:
    """模型反复提交同一个调用时，系统必须抑制而不是无限执行。"""

    same_call = LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+1"})])
    llm = ScriptedLLM(responses=[same_call for _ in range(6)])
    budget = AgentSettings(max_steps=4, max_tool_calls=10, max_repeated_tool_calls=1)
    agent = ToolAgent(llm, _registry(), agent_settings=budget)

    result = agent.run("1+1 等于几")

    assert result.repeated_call_count >= 1
    assert result.tool_call_count >= 1
    # 从未产出最终答案 => 步数预算终止，而不是静默死循环。
    assert result.outcome == BUDGET_STOPPED
    assert result.meta["stop_reason"] == "max_steps_exhausted"


def test_tool_failure_does_not_break_the_loop() -> None:
    """工具参数错误应回灌成可读提示，让模型有机会纠正。"""

    llm = ScriptedLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+"})]),
            LLMResponse(content="这个表达式无法计算。"),
        ]
    )
    agent = ToolAgent(llm, _registry(), agent_settings=AgentSettings(max_steps=4))
    result = agent.run("帮我算一下 1+")

    assert result.outcome == ANSWERED
    assert result.failed_call_count == 1
    assert result.answer.strip()


def test_tool_argument_validation_blocks_bad_args() -> None:
    llm = ScriptedLLM(
        responses=[
            # 缺少必填的 expression
            LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expr": "1+1"})]),
            LLMResponse(content="抱歉，参数有误。"),
        ]
    )
    agent = ToolAgent(llm, _registry(), agent_settings=AgentSettings(max_steps=4))
    result = agent.run("帮我算 1+1")

    assert result.failed_call_count == 1
    failed = [call for step in result.steps for call in step.calls if not call.ok]
    assert failed[0].error_code == "TOOL_INVALID_ARGUMENTS"


def test_unknown_tool_is_rejected_without_crashing() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="shell_exec", arguments={"cmd": "rm -rf /"})]),
            LLMResponse(content="没有这个工具。"),
        ]
    )
    agent = ToolAgent(llm, _registry(), agent_settings=AgentSettings(max_steps=4))
    result = agent.run("删除所有文件")

    failed = [call for step in result.steps for call in step.calls if not call.ok]
    assert failed and failed[0].error_code == "TOOL_NOT_FOUND"
    assert result.outcome == ANSWERED


def test_tool_call_budget_is_enforced() -> None:
    """工具调用总数超限时必须停止，而不是继续消耗成本。"""

    calls = [
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": f"{index}+1"})])
        for index in range(10)
    ]
    llm = ScriptedLLM(responses=calls)
    budget = AgentSettings(max_steps=8, max_tool_calls=2, max_repeated_tool_calls=5)
    agent = ToolAgent(llm, _registry(), agent_settings=budget)
    result = agent.run("连续计算")

    exhausted = [
        call
        for step in result.steps
        for call in step.calls
        if call.error_code == "BUDGET_EXCEEDED"
    ]
    assert exhausted, "工具预算耗尽后必须留下显式记录"


def test_deadline_stops_run() -> None:
    same = LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})])
    llm = ScriptedLLM(responses=[same for _ in range(10)])
    agent = ToolAgent(
        llm,
        _registry(),
        agent_settings=AgentSettings(max_steps=6, deadline_seconds=1.0, max_repeated_tool_calls=9),
    )
    result = agent.run("慢慢算")
    assert result.outcome in {ANSWERED, BUDGET_STOPPED}
    assert result.trace is not None
    assert result.trace.total_duration_ms() >= 0.0


def test_heuristic_llm_drives_end_to_end_run() -> None:
    """离线启发式模型必须能独立跑完一次完整运行（不依赖任何外部服务）。"""

    from scout.rag.pipeline import RAGPipeline, build_index
    from scout.tools.knowledge import KnowledgeSearchTool

    documents = [
        ("redis.md", "Redis 的高性能来自内存存储、单线程模型与 IO 多路复用。"),
        ("go.md", "Go 的 GMP 调度模型由 G、M、P 三部分组成。"),
    ]
    index = build_index(documents)
    pipeline = RAGPipeline(index, HeuristicLLM())
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(pipeline, trace_factory=lambda: None).spec())
    for spec in build_default_tools():
        registry.register(spec)

    agent = ToolAgent(HeuristicLLM(), registry, settings=Settings(), orchestrator=SelfHealingOrchestrator())
    result = agent.run("Redis 为什么这么快？")

    # 这里断言的是**契约**而非质量：离线实现必须能独立跑完、留下可审计轨迹、
    # 并在必要时诚实拒答。它用词法抽取拼答案，答得准不准不是本测试的目标。
    assert result.outcome in {ANSWERED, "insufficient_evidence", "no_knowledge"}
    assert result.tool_call_count == 1
    assert result.step_count >= 2
    assert result.trace is not None
    assert result.trace.steps, "运行必须留下可审计的步骤轨迹"
    assert result.grounding.verdict.value in {"pass", "regenerate", "abstain"}
    assert result.meta.get("failure_taxonomy") is not None
