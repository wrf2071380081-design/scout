"""评测数据集。

一个关键设计决定：**gold 用文本片段而不是 chunk_id**。

``chunk_id`` 依赖分块参数——只要把叶子块从 800 字符改成 600，所有 chunk_id 全部失效，
标注集当场作废。而"这段文字应当被检索到"是与分块策略解耦的，
所以本模块用 ``gold_snippets``（应当出现在检索结果中的文本片段）来标注金标准，
判定方式是"命中的块是否包含任一片段"。

代价是判定比 id 相等稍慢，换来的是**标注集可以在分块策略变更后继续复用**——
在做分块消融时，这一点是决定性的：否则每换一次参数就得重标一遍 200 条。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import ErrorCode, ValidationError
from .taxonomy import QueryTag

SCHEMA_VERSION = 1


@dataclass(slots=True)
class EvalCase:
    """一条评测样本。"""

    case_id: str
    question: str
    tags: list[QueryTag] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    gold_snippets: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    allow_unknown: bool = False
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.case_id.strip():
            problems.append("case_id 不能为空")
        if len(self.question.strip()) < 4:
            problems.append("question 过短（至少 4 个字符）")
        if not self.tags:
            problems.append("至少需要一个标签")
        if not self.allow_unknown and not self.gold_snippets and not self.expected_sources:
            problems.append("非拒答样本必须提供 gold_snippets 或 expected_sources")
        if self.allow_unknown and self.gold_snippets:
            problems.append("拒答样本不应带 gold_snippets")
        if QueryTag.PROMPT_INJECTION in self.tags and not self.forbidden_keywords:
            problems.append("提示注入样本必须提供 forbidden_keywords（否则无法判定是否泄漏）")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "tags": [tag.value for tag in self.tags],
            "expected_sources": list(self.expected_sources),
            "expected_keywords": list(self.expected_keywords),
            "gold_snippets": list(self.gold_snippets),
            "forbidden_keywords": list(self.forbidden_keywords),
            "allow_unknown": self.allow_unknown,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        raw_tags = payload.get("tags") or []
        tags: list[QueryTag] = []
        for item in raw_tags:
            try:
                tags.append(QueryTag(item))
            except ValueError as exc:
                raise ValidationError(
                    f"未知标签 {item!r}（case_id={payload.get('case_id')}）",
                    code=ErrorCode.VALIDATION_FAILED,
                ) from exc
        return cls(
            case_id=str(payload.get("case_id", "")).strip(),
            question=str(payload.get("question", "")).strip(),
            tags=tags,
            expected_sources=[str(item) for item in payload.get("expected_sources") or []],
            expected_keywords=[str(item) for item in payload.get("expected_keywords") or []],
            gold_snippets=[str(item) for item in payload.get("gold_snippets") or []],
            forbidden_keywords=[str(item) for item in payload.get("forbidden_keywords") or []],
            allow_unknown=bool(payload.get("allow_unknown", False)),
            notes=str(payload.get("notes", "")),
        )


@dataclass(slots=True)
class EvalDataset:
    """版本化评测集。"""

    name: str
    cases: list[EvalCase] = field(default_factory=list)
    description: str = ""
    schema_version: int = SCHEMA_VERSION
    corpus_fingerprint: str = ""

    def __len__(self) -> int:
        return len(self.cases)

    def by_id(self, case_id: str) -> EvalCase | None:
        return next((case for case in self.cases if case.case_id == case_id), None)

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for tag in case.tags:
                counts[tag.value] = counts.get(tag.value, 0) + 1
        return dict(sorted(counts.items()))

    def subset(self, tags: Iterable[QueryTag]) -> EvalDataset:
        wanted = set(tags)
        return EvalDataset(
            name=f"{self.name}-subset",
            description=self.description,
            cases=[case for case in self.cases if wanted & set(case.tags)],
            corpus_fingerprint=self.corpus_fingerprint,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.cases:
            problems.append("数据集为空")
        seen: set[str] = set()
        for case in self.cases:
            if case.case_id in seen:
                problems.append(f"case_id 重复：{case.case_id}")
            seen.add(case.case_id)
            for problem in case.validate():
                problems.append(f"[{case.case_id}] {problem}")
        return problems

    def fingerprint(self) -> str:
        """内容指纹。任何改动都会改变它，用于把报告绑定到具体数据版本。"""

        canonical = json.dumps(
            {"name": self.name, "schema": self.schema_version, "cases": [c.to_dict() for c in self.cases]},
            ensure_ascii=False,
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "corpus_fingerprint": self.corpus_fingerprint,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalDataset:
        case_payloads = payload.get("cases")
        if not isinstance(case_payloads, list):
            raise ValidationError("数据集缺少 cases 数组", code=ErrorCode.VALIDATION_FAILED)
        return cls(
            name=str(payload.get("name", "unnamed")),
            description=str(payload.get("description", "")),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            corpus_fingerprint=str(payload.get("corpus_fingerprint", "")),
            cases=[EvalCase.from_dict(item) for item in case_payloads],
        )


def load_dataset(path: str | Path) -> EvalDataset:
    """从 JSON 读取数据集并校验。"""

    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"数据集不存在：{file_path}", code=ErrorCode.VALIDATION_FAILED)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"数据集不是合法 JSON：{exc}", code=ErrorCode.VALIDATION_FAILED) from exc
    dataset = EvalDataset.from_dict(payload)
    problems = dataset.validate()
    if problems:
        raise ValidationError(
            "数据集校验未通过：\n  - " + "\n  - ".join(problems[:20]),
            code=ErrorCode.VALIDATION_FAILED,
            details={"problem_count": len(problems)},
        )
    return dataset


def save_dataset(dataset: EvalDataset, path: str | Path) -> Path:
    """写回数据集（保留人类可读的缩进与中文）。"""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def corpus_fingerprint(documents: Sequence[tuple[str, str]]) -> str:
    """语料指纹：``[(filename, text), ...]`` 的内容哈希。

    把语料身份写进报告，是为了让"指标变化"能被归因到"是代码改了还是语料变了"。
    没有这个字段，两次报告之间的差异永远说不清。
    """

    digest = hashlib.sha256()
    for filename, text in sorted(documents, key=lambda item: item[0]):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return "sha256:" + digest.hexdigest()[:16]


__all__ = [
    "SCHEMA_VERSION",
    "EvalCase",
    "EvalDataset",
    "corpus_fingerprint",
    "load_dataset",
    "save_dataset",
]
