"""阶段钩子测试：流水线与多智能体阶段事件序列。"""

from __future__ import annotations

from scout.llm.scripted import HeuristicLLM
from scout.multiagent import MultiAgentOrchestrator
from scout.rag.pipeline import RAGPipeline, build_index

DOCS = [
    ("a.md", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"),
    ("b.md", "低空经济标准体系涵盖低空航空器、起降设施与运行服务等领域。"),
]


def test_pipeline_emits_real_stage_hooks() -> None:
    """阶段钩子必须落在真实断点上，且顺序与主流程一致：

    retrieve → grade → generate → grounding → done。
    伪造的阶段比没有更糟——所以测试要断言钩子真的存在且语义正确。"""
    pipeline = RAGPipeline(build_index(DOCS), HeuristicLLM())
    stages: list[tuple[str, dict]] = []
    result = pipeline.answer(
        "云计算标准体系结构包括哪几个部分？",
        on_stage=lambda name, payload: stages.append((name, payload)),
    )
    names = [name for name, _ in stages]

    # 阶段必须以问题为中心的真实顺序发生
    assert names[0] == "retrieve"
    assert names[1] == "grade"
    assert "generate" in names
    assert "grounding" in names
    assert names[-1] == "done"

    # 每个阶段都带上真实的负载信息（不是占位）
    assert stages[0][1]["units"] > 0
    done_payload = dict([p for n, p in stages if n == "done"][0])
    assert done_payload["outcome"] == result.outcome


def test_pipeline_without_on_stage_is_unchanged() -> None:
    """不传 on_stage 的行为必须和以前完全一样（不拆 API）。"""
    pipeline = RAGPipeline(build_index(DOCS), HeuristicLLM())
    result = pipeline.answer("云计算标准体系结构包括哪几个部分？")
    assert result.outcome == "answered"
    assert result.answer


def test_multiagent_emits_plan_fanout_grounding_done() -> None:
    pipeline = RAGPipeline(build_index(DOCS), HeuristicLLM())
    orch = MultiAgentOrchestrator(pipeline, HeuristicLLM())
    stages: list[str] = []
    orch.answer(
        "云计算和低空经济两份标准体系在划分方式上有什么不同？",
        on_stage=lambda name, _payload: stages.append(name),
    )
    assert "plan" in stages
    assert "fanout" in stages
    assert "synthesize" in stages
    assert "grounding" in stages
    assert "done" in stages
