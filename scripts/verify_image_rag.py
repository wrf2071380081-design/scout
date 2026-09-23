"""端到端验证：图片能不能走完 RAG 全链路（抽取 → 切分 → 向量化 → 入 Milvus → 检索）。

**为什么要专门写这个脚本。**
"支持图片入库"这句话有两层含义，差得很远：

1. 图片能被读成文本（抽取层）——这一步早就有了；
2. **图片里的内容能真的被检索到**（全链路）——这一步才叫"入库"。

第二层才是使用者真正关心的。而它坏掉的方式很隐蔽：
抽取成功、切分成功、报表正常，**但建索引那条路根本不读图片**，
于是"我放了几张扫描件进去"变成"检索里什么都没有"。

本脚本用**只有图里才有的内容**当查询，检索到才算通过——
关键词必须在检索证据里出现，而不是"看起来有结果"。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import get_settings  # noqa: E402
from scout.data import build_extractor_from_settings  # noqa: E402
from scout.evaluation.runner import load_corpus  # noqa: E402
from scout.rag.pipeline import RAGPipeline, build_index  # noqa: E402
from scout.trace import Trace  # noqa: E402

CORPUS = Path(os.environ["IMAGE_CORPUS"])
# 这些事实**只存在于图片里**——语料目录里没有任何文本文件提到它们
PROBES = [
    ("milvus-minio 的端口映射是什么？", ["9000"]),
    ("milvus-attu 用的是哪个镜像？", ["zilliz/attu"]),
    ("redis 容器的镜像版本是什么？", ["7-alpine"]),
    ("milvus-standalone 的端口是多少？", ["19530"]),
]


def main() -> int:
    settings = get_settings()
    use_milvus = settings.milvus.enabled
    extractor = build_extractor_from_settings(settings)

    lines = ["# 端到端验证：图片 → OCR → 检索", ""]
    lines.append(f"- 语料目录：`{CORPUS}`")
    lines.append(f"- 抽取器：`{type(extractor).__name__ if extractor else '（未配置）'}`")
    lines.append(f"- 向量后端：**{'Milvus' if use_milvus else '内存'}**")
    lines.append("")

    if extractor is None:
        lines.append("## 结论：抽取器未配置 —— 无法验证")
        lines.append("")
        lines.append("图片不会被读进来。设置 SCOUT_VISION_ENABLED=1 与模型后重试。")
        _write(lines)
        return 1

    # —— 1) 语料加载（图片走抽取）——
    skipped: list[str] = []
    documents = load_corpus(CORPUS, image_extractor=extractor, skip_log=skipped)
    lines.append("## 1. 语料加载")
    lines.append("")
    lines.append(f"- 成功读入：**{len(documents)}** 篇")
    for name, text in documents:
        lines.append(f"  - `{name}`（{len(text)} 字符）")
    if skipped:
        lines.append(f"- ⚠️ 跳过 {len(skipped)} 个：{skipped[:3]}")
    lines.append("")
    if not documents:
        lines.append("## 结论：语料为空 —— 图片没有被读进来")
        _write(lines)
        return 1

    # —— 2) 建索引（可选 Milvus）——
    lines.append("## 2. 建索引")
    lines.append("")
    index = build_index(documents, settings=settings, milvus=use_milvus, strict=use_milvus)
    lines.append(f"- 索引类型：`{type(index).__name__}`")
    lines.append(f"- 分块数：**{len(index.chunks)}**")
    if use_milvus:
        status = index.milvus_status()
        lines.append(f"- Milvus 集合：`{status.get('collection')}`｜实体 {status.get('entities')}")
        lines.append(
            f"- 降级码：{status.get('degraded_code') or '（无，说明真的走了 Milvus）'}"
        )
    lines.append("")

    # —— 3) 检索验证：只用图里才有的内容提问 ——
    pipeline = RAGPipeline(index, settings=settings)
    lines.append("## 3. 检索验证（查询只包含图中才有的事实）")
    lines.append("")
    lines.append("| 查询 | 期望命中 | 实际 | 判定 |")
    lines.append("|---|---|---|---|")
    passed = 0
    for question, expected in PROBES:
        trace = Trace(question=question)
        units, _meta, _grade = pipeline.collect(question, trace)
        evidence = "\n".join(unit.context_text for unit in units)
        hits = [token for token in expected if token in evidence]
        ok = len(hits) == len(expected)
        passed += int(ok)
        lines.append(
            f"| {question} | {'、'.join(expected)} | {'、'.join(hits) or '（无）'} "
            f"| {'✅' if ok else '❌'} |"
        )
    lines.append(f"\n**通过 {passed}/{len(PROBES)}**\n")

    # —— 4) 生成验证：真的要答出来 ——
    lines.append("## 4. 端到端问答（含生成）")
    lines.append("")
    question = PROBES[0][0]
    result = pipeline.answer(question)
    lines.append(f"- 问题：{question}")
    lines.append(f"- 结果：`{result.outcome}`｜证据 {len(result.units)} 条")
    lines.append(f"- 回答：{result.answer[:300]}")
    lines.append("")

    ok_all = passed == len(PROBES)
    lines.append("## 结论")
    lines.append("")
    if ok_all:
        lines.append(
            "✅ **图片走完了 RAG 全链路**：抽取 → 切分 → 向量化 → 建索引"
            f"（{'Milvus' if use_milvus else '内存'}）→ 检索，"
            "且只有图中才存在的事实能被检索到。"
        )
    else:
        lines.append(
            f"❌ 有 {len(PROBES) - passed} 条查询没能从图里检索到目标内容——全链路未打通。"
        )
    _write(lines)
    return 0 if ok_all else 1


def _write(lines: list[str]) -> None:
    import json

    out = Path(__file__).resolve().parents[1] / "reports" / "image_rag_e2e.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
