"""评测运行器。

负责：从语料目录构建索引 → 逐条跑样本 → 收集观测 → 产出报告。

两个刻意的设计：

1. **索引只建一次，全流程复用**。每条样本重建索引既慢又会让不同样本
   处在不同索引状态上，报告之间失去可比性。
2. **报告绑定三样身份**：数据集指纹、语料指纹、配置标签。
   没有它们，两次报告的分数差异永远无法归因到是"代码改了""语料变了"
   还是"配置不同"——这正是很多项目评测做不下去的原因。
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import scout

from ..agent.loop import ToolAgent
from ..config import Settings, get_settings
from ..llm.base import LLMClient
from ..llm.scripted import default_client
from ..multiagent import MultiAgentOrchestrator
from ..orchestrator.healing import SelfHealingOrchestrator
from ..rag.index import HybridIndex, RetrievalMode
from ..rag.merge import MergeMode
from ..rag.pipeline import RAGPipeline, build_index
from ..rag.pipeline import PipelineConfig
from ..tools.builtin import build_default_tools
from ..tools.knowledge import KnowledgeSearchTool
from ..tools.registry import ToolRegistry
from ..verify.grounding import abstained_from
from .dataset import EvalCase, EvalDataset, corpus_fingerprint
from .metrics import (
    DEFAULT_K_VALUES,
    CaseMetrics,
    CaseObservation,
    MetricValue,
    aggregate,
    score_case,
    tag_distribution,
    tag_slices,
)

SUPPORTED_SUFFIXES = (".md", ".txt")


def load_corpus(directory: str | Path, *, max_files: int | None = None) -> list[tuple[str, str]]:
    """从目录读取语料。返回 ``[(filename, text), ...]``，按文件名排序保证确定性。"""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在：{root}")
    documents: list[tuple[str, str]] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        # 语料目录里通常会放一份 README 说明来源，它不是待检索的文档。
        if path.stem.lower() in {"readme", "index", "说明"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        documents.append((path.name, text))
        if max_files is not None and len(documents) >= max_files:
            break
    return documents


@dataclass(slots=True)
class RunConfig:
    """一次评测运行的配置。"""

    mode: str = "pipeline"
    """``pipeline`` 跑完整 RAG 流水线；``agent`` 跑 ReAct Agent。"""

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    label: str = ""
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    limit: int | None = None
    skip: int = 0
    """跳过前 N 条样本。

    它的用途不是"少跑一点"，而是**跑评测集之外的那一部分**：
    数据飞轮要挖的是"还没进评测集"的问题，
    而扩展集（v2 = v1 + 新增）的前 N 条恰好就是 v1，
    不跳过就永远挖不到增量——这个坑很隐蔽，结果看起来像"飞轮没用"。
    """

    repeat: int = 1

    def effective_label(self) -> str:
        return self.label or f"{self.mode}:{self.pipeline_config.label()}"


@dataclass(slots=True)
class EvalReport:
    """评测报告。"""

    dataset_name: str
    dataset_fingerprint: str
    corpus_fingerprint: str
    config_label: str
    mode: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    tag_slices: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    tag_distribution: dict[str, int] = field(default_factory=dict)
    case_metrics: list[CaseMetrics] = field(default_factory=list)
    observations: list[CaseObservation] = field(default_factory=list)
    corpus_stats: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    started_at: float = 0.0
    environment: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float | None:
        entry = self.metrics.get(name)
        return entry.value if entry else None

    def to_dict(self, *, include_observations: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scout_version": scout.__version__,
            "dataset": {"name": self.dataset_name, "fingerprint": self.dataset_fingerprint},
            "corpus": {"fingerprint": self.corpus_fingerprint, **self.corpus_stats},
            "config": {"label": self.config_label, "mode": self.mode},
            "environment": self.environment,
            "duration_seconds": round(self.duration_seconds, 3),
            "started_at": self.started_at,
            "metrics": {name: value.to_dict() for name, value in self.metrics.items()},
            "tag_distribution": self.tag_distribution,
            "tag_slices": {
                tag: {name: value.to_dict() for name, value in values.items()}
                for tag, values in self.tag_slices.items()
            },
            "cases": [item.to_dict() for item in self.case_metrics],
        }
        if include_observations:
            payload["observations"] = [obs.to_dict() for obs in self.observations]
        return payload


def _observe_pipeline(
    pipeline: RAGPipeline,
    case: EvalCase,
) -> tuple[CaseObservation, float]:
    started = time.perf_counter()
    result = pipeline.answer(case.question)
    latency_ms = (time.perf_counter() - started) * 1000.0

    observation = CaseObservation(
        case_id=case.case_id,
        question=case.question,
        outcome=result.outcome,
        answer=result.answer,
        retrieved_chunk_ids=[unit.chunk.chunk_id for unit in result.units],
        retrieved_sources=[unit.chunk.filename for unit in result.units],
        # 评测匹配必须用**原始叶子块文本**，而不是 unit.context_text：
        # 后者可能已被消毒（neutralize / NFKC / 截断），
        # 用它来匹配会把"安全措施改变了文本"误报成"检索没命中"。
        retrieved_texts=[unit.chunk.text for unit in result.units],
        latency_ms=latency_ms,
        step_count=len(result.trace.steps) if result.trace else 0,
        tool_call_count=0,
        answer_support_rate=result.grounding.support_rate,
        answer_coverage=result.grounding.coverage,
        abstained=abstained_from(result.answer, result.outcome),
        error_code=result.error_code,
        meta={
            key: value
            for key, value in result.meta.items()
            if key
            in {
                "config_label",
                "rewrite_triggered",
                "rewrite_method",
                "auto_merge_mode",
                "auto_merge_applied",
                "rerank_applied",
                "grade_route",
                "retrieval_mode",
                "retrieval_unique_candidates",
                "grounding_verdict",
            }
        },
    )
    return observation, latency_ms


def _observe_agent(
    agent: ToolAgent,
    pipeline: RAGPipeline,
    case: EvalCase,
) -> tuple[CaseObservation, float]:
    started = time.perf_counter()
    run = agent.run(case.question)
    latency_ms = (time.perf_counter() - started) * 1000.0

    evidence = agent._collect_evidence()  # noqa: SLF001 - 运行器与 Agent 同属本包
    observation = CaseObservation(
        case_id=case.case_id,
        question=case.question,
        outcome=run.outcome,
        answer=run.answer,
        retrieved_chunk_ids=[unit.chunk.chunk_id for unit in evidence],
        retrieved_sources=[unit.chunk.filename for unit in evidence],
        # 见 _observe_pipeline：匹配用原始叶子块文本，不用可能被消毒改写的 context_text。
        retrieved_texts=[unit.chunk.text for unit in evidence],
        latency_ms=latency_ms,
        step_count=run.step_count,
        tool_call_count=run.tool_call_count,
        failed_call_count=run.failed_call_count,
        repeated_call_count=run.repeated_call_count,
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        answer_support_rate=run.grounding.support_rate,
        answer_coverage=run.grounding.coverage,
        abstained=abstained_from(run.answer, run.outcome),
        error_code=run.error_code,
        meta={
            "stop_reason": run.meta.get("stop_reason"),
            "grounding_verdict": run.grounding.verdict.value,
            "failure_taxonomy": run.meta.get("failure_taxonomy"),
        },
    )
    _ = pipeline
    return observation, latency_ms


def _observe_multiagent(
    orchestrator: MultiAgentOrchestrator,
    case: EvalCase,
) -> tuple[CaseObservation, float]:
    """以多智能体模式跑一条样本。

    检索证据取**所有成功子 Agent 的压缩证据并集**——这是"上下文隔离"的价值所在：
    每个子 Agent 独立检索自己的那片，合并后覆盖单个 Top-K 覆盖不到的多文档场景。
    """

    started = time.perf_counter()
    result = orchestrator.answer(case.question)
    latency_ms = (time.perf_counter() - started) * 1000.0

    # 证据统一从 result.units 取：多智能体路径是各子 Agent 压缩证据的并集，
    # 单路回退路径是回退到的那次检索的证据。两条路径都不能漏记。
    merged_units = list(result.units)
    # 用原始叶子块文本做 gold 匹配（见 _observe_pipeline 的说明）。
    observation = CaseObservation(
        case_id=case.case_id,
        question=case.question,
        outcome=result.outcome,
        answer=result.answer,
        retrieved_chunk_ids=[unit.chunk.chunk_id for unit in merged_units],
        retrieved_sources=[unit.chunk.filename for unit in merged_units],
        retrieved_texts=[unit.chunk.text for unit in merged_units],
        latency_ms=latency_ms,
        step_count=len(result.trace.steps) if result.trace else 0,
        tool_call_count=0,
        answer_support_rate=(result.grounding.support_rate if result.grounding else 0.0),
        answer_coverage=(result.grounding.coverage if result.grounding else 0.0),
        abstained=abstained_from(result.answer, result.outcome),
        error_code="",
        meta={
            "mode": result.meta.get("mode"),
            "subagents_total": result.meta.get("subagents_total", 0),
            "subagents_completed": result.meta.get("subagents_completed", 0),
            "coverage_gaps": len(result.coverage_gaps),
            "plan": result.plan,
            "grounding_verdict": (result.grounding.verdict.value if result.grounding else None),
        },
    )
    return observation, latency_ms


def run_evaluation(
    dataset: EvalDataset,
    documents: Sequence[tuple[str, str]],
    *,
    run_config: RunConfig | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    index: HybridIndex | None = None,
) -> EvalReport:
    """执行一次评测。"""

    config = run_config or RunConfig()
    effective = settings or get_settings()
    client = llm or default_client()

    active_index = index or build_index(documents, settings=effective)
    pipeline = RAGPipeline(active_index, client, settings=effective, config=config.pipeline_config)

    cases = list(dataset.cases)
    if config.skip:
        cases = cases[config.skip :]
    if config.limit is not None:
        cases = cases[: config.limit]

    agent: ToolAgent | None = None
    if config.mode == "agent":
        registry = ToolRegistry()
        knowledge = KnowledgeSearchTool(pipeline, trace_factory=lambda: None)
        registry.register(knowledge.spec())
        for spec in build_default_tools():
            registry.register(spec)
        agent = ToolAgent(client, registry, settings=effective, orchestrator=SelfHealingOrchestrator())

    orchestrator: MultiAgentOrchestrator | None = None
    if config.mode == "multiagent":
        orchestrator = MultiAgentOrchestrator(pipeline, client, settings=effective, config=config.pipeline_config)

    observations: list[CaseObservation] = []
    case_metrics: list[CaseMetrics] = []
    started_at = time.time()

    for case in cases:
        if config.mode == "agent" and agent is not None:
            observation, _latency = _observe_agent(agent, pipeline, case)
        elif config.mode == "multiagent" and orchestrator is not None:
            observation, _latency = _observe_multiagent(orchestrator, case)
        else:
            observation, _latency = _observe_pipeline(pipeline, case)
        observations.append(observation)
        case_metrics.append(
            score_case(case, observation, k_values=config.k_values)
        )

    duration = time.time() - started_at
    metrics = aggregate(cases, observations, case_metrics, k_values=config.k_values)
    return EvalReport(
        dataset_name=dataset.name,
        dataset_fingerprint=dataset.fingerprint(),
        corpus_fingerprint=corpus_fingerprint(documents),
        config_label=config.effective_label(),
        mode=config.mode,
        metrics=metrics,
        tag_slices=tag_slices(case_metrics, observations, k=config.k_values[2] if len(config.k_values) > 2 else 5),
        tag_distribution=tag_distribution(cases),
        case_metrics=case_metrics,
        observations=observations,
        corpus_stats=active_index.corpus_stats(),
        duration_seconds=duration,
        started_at=started_at,
        environment={
            "python": platform.python_version(),
            "llm": client.model_name,
            "embedder": active_index.embedder.name,
            "auto_merge_mode": config.pipeline_config.merge_mode.value,
            "retrieval_mode": config.pipeline_config.retrieval_mode.value,
        },
    )


def default_ablation_configs() -> list[RunConfig]:
    """默认消融矩阵。

    每个配置只改一个维度，这样才可以把指标差异归因到那个维度上。
    第一行是全量基线，最后一行是"几乎什么都不开"的对照。
    """

    base = PipelineConfig()
    return [
        RunConfig(label="full", pipeline_config=base),
        RunConfig(label="-rerank", pipeline_config=base.with_overrides(rerank_enabled=False)),
        RunConfig(label="-rewrite", pipeline_config=base.with_overrides(rewrite_enabled=False)),
        RunConfig(label="-merge", pipeline_config=base.with_overrides(merge_mode=MergeMode.OFF)),
        RunConfig(
            label="merge-replace",
            pipeline_config=base.with_overrides(merge_mode=MergeMode.REPLACE),
        ),
        RunConfig(label="-route", pipeline_config=base.with_overrides(complexity_routing=False)),
        RunConfig(
            label="-gate",
            pipeline_config=base.with_overrides(sufficiency_gate=False, injection_defense=False),
        ),
        RunConfig(
            label="baseline-dense-only",
            pipeline_config=base.with_overrides(
                retrieval_mode=RetrievalMode.DENSE_ONLY,
                merge_mode=MergeMode.OFF,
                rerank_enabled=False,
                rewrite_enabled=False,
                complexity_routing=False,
                sufficiency_gate=False,
                injection_defense=False,
            ),
        ),
    ]


def run_ablation(
    dataset: EvalDataset,
    documents: Sequence[tuple[str, str]],
    *,
    configs: Sequence[RunConfig] | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> list[EvalReport]:
    """跑完整个消融矩阵。

    **索引只构建一次**并复用给所有配置——这是让消融结果可比的前提：
    如果每个配置各建一次索引，向量缓存状态不同，分数差异里就混进了索引噪声。
    """

    effective = settings or get_settings()
    client = llm or default_client()
    shared_index = build_index(documents, settings=effective)
    reports: list[EvalReport] = []
    for config in configs or default_ablation_configs():
        reports.append(
            run_evaluation(
                dataset,
                documents,
                run_config=config,
                llm=client,
                settings=effective,
                index=shared_index,
            )
        )
    return reports


__all__ = [
    "EvalReport",
    "RunConfig",
    "default_ablation_configs",
    "load_corpus",
    "run_ablation",
    "run_evaluation",
]
