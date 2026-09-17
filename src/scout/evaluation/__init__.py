"""评测框架：标签体系、数据集、三层指标、运行器与报告。"""

from __future__ import annotations

from .dataset import (
    SCHEMA_VERSION,
    EvalCase,
    EvalDataset,
    corpus_fingerprint,
    load_dataset,
    save_dataset,
)
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
from .report import (
    ABLATION_COLUMNS,
    METRIC_LAYERS,
    format_metric,
    render_ablation_table,
    render_failure_taxonomy,
    render_report,
)
from .runner import (
    EvalReport,
    RunConfig,
    default_ablation_configs,
    load_corpus,
    run_ablation,
    run_evaluation,
)
from .taxonomy import (
    MUST_ANSWER_TAGS,
    REFUSAL_EXPECTED_TAGS,
    TAG_DESCRIPTIONS,
    QueryTag,
    describe,
)

__all__ = [
    "ABLATION_COLUMNS",
    "DEFAULT_K_VALUES",
    "METRIC_LAYERS",
    "MUST_ANSWER_TAGS",
    "REFUSAL_EXPECTED_TAGS",
    "SCHEMA_VERSION",
    "TAG_DESCRIPTIONS",
    "CaseMetrics",
    "CaseObservation",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "MetricValue",
    "QueryTag",
    "RunConfig",
    "aggregate",
    "corpus_fingerprint",
    "default_ablation_configs",
    "describe",
    "format_metric",
    "load_corpus",
    "load_dataset",
    "render_ablation_table",
    "render_failure_taxonomy",
    "render_report",
    "run_ablation",
    "run_evaluation",
    "save_dataset",
    "score_case",
    "tag_distribution",
    "tag_slices",
]
