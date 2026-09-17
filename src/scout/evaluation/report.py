"""报告渲染。

三份产物：

- **单次报告**：分层指标 + 标签切片 + 失败样本清单
- **消融对照表**：每个模块的边际贡献（**含负贡献**）——这是整个项目最有价值的一张表
- **失败分类学**：失败按类型与阶段的分布

报告一律渲染成 Markdown 而不是只存 JSON，因为报告是给人看的；
只存 JSON 的评测等于没评测——没人会去读它。
"""

from __future__ import annotations

from typing import Any, Sequence

from .metrics import MetricValue
from .runner import EvalReport

# 指标的中文名与所属层。报告按层分组展示，便于看出"哪一层坏了"。
METRIC_LAYERS: dict[str, tuple[str, str]] = {
    "recall_at_1": ("检索层", "Recall@1"),
    "recall_at_3": ("检索层", "Recall@3"),
    "recall_at_5": ("检索层", "Recall@5"),
    "recall_at_10": ("检索层", "Recall@10"),
    "mrr": ("检索层", "MRR"),
    "ndcg_at_10": ("检索层", "nDCG@10"),
    "source_hit_rate": ("检索层", "目标来源命中率"),
    "keyword_coverage": ("生成层", "关键词覆盖率"),
    "answer_support_rate": ("生成层", "归因支撑率"),
    "answer_coverage": ("生成层", "证据覆盖率"),
    "abstention_accuracy": ("生成层", "拒答正确率"),
    "hallucination_rate": ("生成层", "幻觉率（该拒却答）"),
    "false_refusal_rate": ("生成层", "误拒率（该答却拒）"),
    "answered_rate": ("生成层", "作答率"),
    "injection_leak_rate": ("安全层", "注入泄漏率"),
    "avg_steps": ("轨迹层", "平均步数"),
    "avg_tool_calls": ("轨迹层", "平均工具调用数"),
    "tool_failure_rate": ("轨迹层", "工具失败率"),
    "repeated_call_rate": ("轨迹层", "重复调用率"),
    "provider_failure_rate": ("轨迹层", "Provider 失败率"),
    "budget_stop_rate": ("轨迹层", "预算终止率"),
    "latency_mean_ms": ("性能层", "平均延迟(ms)"),
    "latency_p50_ms": ("性能层", "P50 延迟(ms)"),
    "latency_p95_ms": ("性能层", "P95 延迟(ms)"),
    "tokens_mean": ("性能层", "平均 token 数"),
}

_PERCENT_METRICS = {
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "source_hit_rate",
    "keyword_coverage",
    "answer_support_rate",
    "answer_coverage",
    "abstention_accuracy",
    "hallucination_rate",
    "false_refusal_rate",
    "answered_rate",
    "injection_leak_rate",
    "tool_failure_rate",
    "repeated_call_rate",
    "provider_failure_rate",
    "budget_stop_rate",
}

_TAG_LABELS: dict[str, str] = {
    "single_fact": "单一事实",
    "definition": "定义解释",
    "parameter": "参数阈值",
    "cross_document": "跨文档",
    "multi_hop": "多跳推理",
    "comparison": "多对象对比",
    "time_version": "时间版本",
    "table": "表格",
    "code": "代码标识符",
    "ambiguity": "歧义（应澄清）",
    "no_knowledge": "无知识（应拒答）",
    "source_conflict": "来源冲突",
    "near_entity": "近似实体",
    "typo": "错别字/别名",
    "long_question": "长问题",
    "prompt_injection": "提示注入",
}


def format_metric(name: str, value: MetricValue) -> str:
    if value is None or value.value is None:
        return "—"
    if name in _PERCENT_METRICS:
        return f"{value.value * 100:.1f}%"
    if name.endswith("_ms"):
        return f"{value.value:.1f}"
    if name in {"avg_steps", "avg_tool_calls", "tokens_mean"}:
        return f"{value.value:.2f}"
    return f"{value.value:.4f}"


def render_report(report: EvalReport, *, title: str | None = None) -> str:
    """渲染单次评测报告。"""

    lines: list[str] = []
    lines.append(f"# {title or 'scout 评测报告'}")
    lines.append("")
    lines.append(f"- 配置：`{report.config_label}`（模式：{report.mode}）")
    lines.append(f"- 数据集：{report.dataset_name} · `{report.dataset_fingerprint}`")
    lines.append(f"- 语料指纹：`{report.corpus_fingerprint}`")
    lines.append(f"- 语料规模：{report.corpus_stats.get('leaf_chunks', 0)} 个叶子块 / "
                 f"{report.corpus_stats.get('documents', 0)} 篇文档")
    lines.append(f"- 运行环境：{report.environment.get('llm')} · "
                 f"embedder={report.environment.get('embedder')} · "
                 f"python={report.environment.get('python')}")
    lines.append(f"- 耗时：{report.duration_seconds:.1f}s")
    lines.append("")

    for layer in ("检索层", "生成层", "安全层", "轨迹层", "性能层"):
        rows = [
            (name, label, report.metrics.get(name))
            for name, (metric_layer, label) in METRIC_LAYERS.items()
            if metric_layer == layer and report.metrics.get(name) is not None
        ]
        if not rows:
            continue
        lines.append(f"## {layer}")
        lines.append("")
        lines.append("| 指标 | 值 | 参与样本 |")
        lines.append("|---|---|---|")
        for name, label, value in rows:
            assert value is not None
            lines.append(f"| {label} | {format_metric(name, value)} | {value.eligible} |")
        lines.append("")

    if report.tag_distribution:
        lines.append("## 标签分布")
        lines.append("")
        lines.append("| 标签 | 样本数 |")
        lines.append("|---|---|")
        for tag, count in report.tag_distribution.items():
            lines.append(f"| {_TAG_LABELS.get(tag, tag)} | {count} |")
        lines.append("")

    if report.tag_slices:
        lines.append("## 标签切片（定位短板）")
        lines.append("")
        lines.append("| 标签 | Recall@5 | MRR | 关键词覆盖 | 拒答正确率 | 误拒率 | 平均工具调用 | P95 延迟 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tag, values in report.tag_slices.items():
            lines.append(
                "| {tag} | {r5} | {mrr} | {kw} | {abst} | {fr} | {tc} | {p95} |".format(
                    tag=_TAG_LABELS.get(tag, tag),
                    r5=format_metric("recall_at_5", values.get("recall_at_5", MetricValue(None))),
                    mrr=format_metric("mrr", values.get("mrr", MetricValue(None))),
                    kw=format_metric("keyword_coverage", values.get("keyword_coverage", MetricValue(None))),
                    abst=format_metric("abstention_accuracy", values.get("abstention_accuracy", MetricValue(None))),
                    fr=format_metric("false_refusal_rate", values.get("false_refusal_rate", MetricValue(None))),
                    tc=format_metric("avg_tool_calls", values.get("avg_tool_calls", MetricValue(None))),
                    p95=format_metric("latency_p95_ms", values.get("latency_p95_ms", MetricValue(None))),
                )
            )
        lines.append("")

    failures = [item for item in report.case_metrics if item.failed]
    false_refusals = [item for item in report.case_metrics if item.false_refusal]
    hallucinated = [item for item in report.case_metrics if item.hallucinated]
    if failures or false_refusals or hallucinated:
        lines.append("## 需要关注的问题样本")
        lines.append("")
        if failures:
            lines.append(f"- **执行失败**（{len(failures)}）：" + "、".join(item.case_id for item in failures[:10]))
        if false_refusals:
            lines.append(
                f"- **误拒**（{len(false_refusals)}，应当作答却拒答）："
                + "、".join(item.case_id for item in false_refusals[:10])
            )
        if hallucinated:
            lines.append(
                f"- **幻觉**（{len(hallucinated)}，应当拒答却作答）："
                + "、".join(item.case_id for item in hallucinated[:10])
            )
        lines.append("")

    return "\n".join(lines)


# 消融表里展示的指标列。刻意包含轨迹与性能列——
# 只看质量列会得出"全都开着最好"的结论，加上成本列才能看出取舍。
ABLATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall_at_5", "Recall@5"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
    ("keyword_coverage", "关键词覆盖"),
    ("abstention_accuracy", "拒答正确率"),
    ("avg_tool_calls", "工具调用"),
    ("latency_p95_ms", "P95(ms)"),
    ("tokens_mean", "token"),
)


def render_ablation_table(reports: Sequence[EvalReport], *, baseline_label: str | None = None) -> str:
    """渲染消融对照表。

    ``Δ`` 列相对第一个报告（或指定的 baseline）计算，正负号保留——
    **负贡献必须显式显示出来**，这正是这张表存在的意义。
    """

    if not reports:
        return "（无数据）"

    baseline = next(
        (item for item in reports if item.config_label == baseline_label),
        reports[0],
    )

    lines: list[str] = []
    lines.append("# 消融对照表")
    lines.append("")
    lines.append(
        f"数据集：{baseline.dataset_name} · `{baseline.dataset_fingerprint}` ｜ "
        f"语料：`{baseline.corpus_fingerprint}` ｜ 基线：`{baseline.config_label}`"
    )
    lines.append("")
    header = "| 配置 | " + " | ".join(label for _name, label in ABLATION_COLUMNS)
    lines.append(header + " |")
    lines.append("|" + "---|" * (len(ABLATION_COLUMNS) + 1))

    for report in reports:
        cells: list[str] = []
        for name, _label in ABLATION_COLUMNS:
            cells.append(format_metric(name, report.metrics.get(name, MetricValue(None))))
        lines.append(f"| `{report.config_label}` | " + " | ".join(cells) + " |")
    lines.append("")

    # 相对基线的差值表：识别负贡献的关键。
    lines.append("## 相对基线的差异")
    lines.append("")
    lines.append("| 配置 | " + " | ".join(f"Δ{label}" for _name, label in ABLATION_COLUMNS) + " |")
    lines.append("|" + "---|" * (len(ABLATION_COLUMNS) + 1))
    for report in reports:
        if report is baseline:
            continue
        cells = []
        for name, _label in ABLATION_COLUMNS:
            current = report.metrics.get(name, MetricValue(None)).value
            reference = baseline.metrics.get(name, MetricValue(None)).value
            if current is None or reference is None:
                cells.append("—")
                continue
            delta = current - reference
            if name in _PERCENT_METRICS:
                cells.append(f"{delta * 100:+.1f}pp")
            else:
                cells.append(f"{delta:+.2f}")
        lines.append(f"| `{report.config_label}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "> 负号表示该配置**优于**基线（对延迟、工具调用、token 这类越低越好的指标），"
        "但对 Recall / MRR / 拒答正确率这类越高越好的指标，**负号表示退步**。"
    )
    return "\n".join(lines)


def render_failure_taxonomy(report: EvalReport) -> str:
    """从观测的 meta 里汇总失败分类学。"""

    by_code: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for observation in report.observations:
        taxonomy: dict[str, Any] = observation.meta.get("failure_taxonomy") or {}
        for code, count in (taxonomy.get("failures_by_code") or {}).items():
            by_code[code] = by_code.get(code, 0) + int(count)
        for action, count in (taxonomy.get("failures_by_action") or {}).items():
            by_action[action] = by_action.get(action, 0) + int(count)
        for stage, count in (taxonomy.get("failures_by_stage") or {}).items():
            by_stage[stage] = by_stage.get(stage, 0) + int(count)

    if not by_code:
        return "（本次运行未记录失败）"

    lines = ["# 失败分类学", "", "| 维度 | 取值 | 次数 |", "|---|---|---|"]
    for code, count in sorted(by_code.items(), key=lambda item: -item[1]):
        lines.append(f"| 错误码 | `{code}` | {count} |")
    for stage, count in sorted(by_stage.items(), key=lambda item: -item[1]):
        lines.append(f"| 阶段 | {stage} | {count} |")
    for action, count in sorted(by_action.items(), key=lambda item: -item[1]):
        lines.append(f"| 恢复动作 | {action} | {count} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ABLATION_COLUMNS",
    "METRIC_LAYERS",
    "format_metric",
    "render_ablation_table",
    "render_failure_taxonomy",
    "render_report",
]
