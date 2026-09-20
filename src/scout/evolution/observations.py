"""观测留痕：把每次问答的结构化结果落成可回放的记录。

**为什么飞轮的第一环是"落盘"而不是"挖掘"。**
挖掘逻辑（:mod:`scout.evolution.flywheel`）早就写好了，但它一直没真正转起来——
原因不是算法，而是**没有数据喂它**：运行时把结果返给调用方就结束了，
trace 是内存对象，进程一退就没了。
所以飞轮的第一个真实卡点是"把观测留下来"，而不是"把挖掘写得更聪明"。

落盘格式选 **JSONL 追加**，与检查点同一个理由：追加是 O(1)、
崩溃最多丢最后一行、历史可审计、人可以直接看。
**不选数据库**：这一层的量级在 10^3~10^5，而且它是离线批处理，
引入一个需要运维的依赖换不到任何东西。

记录里刻意保留 ``retrieved_sources`` 与 ``answer_support_rate``：
挖掘规则要靠它们区分"该答没答""答了但没依据""检索为空"——
**没有这些字段，飞轮只能挖出"失败了"，挖不出"为什么失败"。**
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..evaluation.metrics import CaseObservation


@dataclass(slots=True)
class FlywheelReport:
    """飞轮跑一圈的报告。"""

    observations: int = 0
    mined: int = 0
    approved: int = 0
    rejected: int = 0
    dataset_before: int = 0
    dataset_after: int = 0
    new_cases: int = 0
    skipped_existing: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "mined": self.mined,
            "approved": self.approved,
            "rejected": self.rejected,
            "dataset_before": self.dataset_before,
            "dataset_after": self.dataset_after,
            "new_cases": self.new_cases,
            "skipped_existing": self.skipped_existing,
            "growth": self.dataset_after - self.dataset_before,
            "by_reason": dict(self.by_reason),
            "output_path": self.output_path,
        }

    def render(self) -> str:
        lines = [
            "| 环节 | 数量 |",
            "|---|---|",
            f"| 读取观测 | {self.observations} |",
            f"| 挖出候选 | {self.mined} |",
            f"| 复核通过 | {self.approved} |",
            f"| 复核拒绝/待定 | {self.rejected} |",
            f"| 已在评测集内（跳过） | {self.skipped_existing} |",
            f"| 新增样本 | **{self.new_cases}** |",
            f"| 评测集 | {self.dataset_before} → **{self.dataset_after}** |",
        ]
        if self.by_reason:
            lines.append("")
            lines.append("候选来源分布：" + "，".join(f"{k}={v}" for k, v in sorted(self.by_reason.items())))
        if self.new_cases == 0 and self.skipped_existing:
            lines.append("")
            lines.append(
                "> 新增为 0：挖出的问题**已经在评测集里**。这不是 bug——"
                "从评测集自己的观测里挖，本来就不可能挖出新东西。"
                "飞轮的增量来自**评测集之外的真实流量**。"
            )
        return "\n".join(lines) + "\n"


class ObservationLog:
    """观测日志：append-only JSONL。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, observation: CaseObservation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **observation.to_dict(),
            "recorded_at": time.time(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def extend(self, observations: Iterable[CaseObservation]) -> int:
        count = 0
        for observation in observations:
            self.append(observation)
            count += 1
        return count

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 最后一行可能被崩溃写坏，跳过即可——append-only 的语义是"前面写成功了"
                continue
        return records

    def __len__(self) -> int:
        return len(self.read())


def observation_from_pipeline(result: Any, *, case_id: str = "") -> CaseObservation:
    """把一个流水线/多智能体结果转成观测记录。

    这里是"运行时 → 飞轮"的唯一接口。它必须包含挖掘所需的全部判据，
    否则挖掘规则会退化成一堆特判。
    """

    units = list(getattr(result, "units", []) or [])
    grounding = getattr(result, "grounding", None)
    meta = dict(getattr(result, "meta", {}) or {})
    trace = getattr(result, "trace", None)
    return CaseObservation(
        case_id=case_id or meta.get("case_id", "") or "",
        question=str(getattr(result, "question", "") or ""),
        outcome=str(getattr(result, "outcome", "") or ""),
        answer=str(getattr(result, "answer", "") or ""),
        retrieved_chunk_ids=[unit.chunk.chunk_id for unit in units],
        retrieved_sources=[unit.chunk.filename for unit in units],
        retrieved_texts=[],
        latency_ms=float(meta.get("total_duration_ms") or 0.0),
        step_count=len(trace.steps) if trace is not None else 0,
        answer_support_rate=float(getattr(grounding, "support_rate", 0.0) or 0.0),
        answer_coverage=float(getattr(grounding, "coverage", 0.0) or 0.0),
        abstained=str(getattr(result, "outcome", "")) != "answered",
        error_code=str(getattr(result, "error_code", "") or ""),
        meta={
            "config_label": meta.get("config_label", ""),
            "grade_route": meta.get("grade_route", ""),
            "grounding_verdict": getattr(getattr(grounding, "verdict", None), "value", ""),
            "regenerations": int(meta.get("regenerations") or 0),
            "intent": meta.get("intent", {}),
            # 诚实标注：这些是从离线语料跑出来的观测，不是线上流量。
            # 飞轮本身不区分来源，但报告要区分——否则"评测集增长"会被误当成线上改进。
            "source": "offline_eval",
        },
    )


def mine_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """只看不写：给观测一份挖掘摘要（用于先看有没有值得挖的东西）。"""

    questions = [str(item.get("question") or "") for item in records if item.get("question")]
    abstained = sum(1 for item in records if item.get("abstained"))
    low_support = sum(
        1
        for item in records
        if item.get("answer_support_rate") and float(item["answer_support_rate"]) < 0.6
    )
    empty = sum(1 for item in records if not item.get("retrieved_sources"))
    return {
        "total": len(records),
        "unique_questions": len(set(questions)),
        "abstained": abstained,
        "low_support": low_support,
        "empty_retrieval": empty,
    }


__all__ = [
    "FlywheelReport",
    "ObservationLog",
    "mine_report",
    "observation_from_pipeline",
]
