"""Milvus 后端联机冒烟：连通性、往返、以及**与内存精确检索的重合度**。

**为什么这个脚本值得单独写。**
内存索引是**精确**的（暴力算余弦），Milvus 默认用 HNSW——它是**近似**的。
两者结果不会完全一致，这不是 bug，是 ANN 的固有代价。

所以"接上向量库之后指标会不会变"这个问题的答案是：
**会，而且变化量本身就是一个必须被测量的指标**（ANN 召回率）。
不测它，就不知道上线后的召回损失是多少；
测了它，就能回答"为了这个延迟收益，我们付出了几个百分点的召回"。

脚本产出三件事：
1. **连通性与往返**：集合建了没、多少实体、写入与检索是否成功；
2. **重合度**：同一批查询下，Milvus top-k 与内存精确 top-k 的重合比例（即 ANN 召回估计）；
3. **延迟对照**：两种后端各自的检索耗时（含 p50/p95）。

用法（需先启动容器）::

    docker start milvus-etcd milvus-minio milvus-standalone
    python scripts/milvus_smoke.py

环境变量：``SCOUT_MILVUS_URI``（默认 http://127.0.0.1:19530）、``SMOKE_DOCS``、``SMOKE_QUERIES``。
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import get_settings  # noqa: E402
from scout.evaluation.runner import load_corpus  # noqa: E402
from scout.rag.embed import default_embedder  # noqa: E402
from scout.rag.pipeline import build_index  # noqa: E402

CORPUS = Path(os.environ.get("SMOKE_CORPUS", r"C:\agent-project\scout\datasets\longdoc-gold"))
OUT_MD = Path(os.environ.get("SMOKE_OUT", r"C:\agent-project\scout\evals\results\milvus_smoke.md"))
MAX_DOCS = int(os.environ.get("SMOKE_DOCS", "8"))
TOP_K = int(os.environ.get("SMOKE_TOP_K", "10"))

QUERIES = [
    "云计算标准体系结构包括哪几个部分？",
    "到2027年要新制定多少项标准？",
    "低空经济标准体系的重点方向是什么？",
    "电力中长期市场的基本规则有哪些内容？",
    "工业产品缺陷检测的流程是什么？",
    "数字化转型实施方案提出了哪些举措？",
    "氢能综合应用试点的申报要求是什么？",
    "算力互联互通行动计划的重点任务是什么？",
]


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[index]


def main() -> int:
    settings = get_settings()
    documents = load_corpus(CORPUS, max_files=MAX_DOCS)
    if not documents:
        print(f"语料为空：{CORPUS}")
        return 1

    lines: list[str] = ["# Milvus 后端联机冒烟报告", ""]
    lines.append(f"- Milvus：`{settings.milvus.uri}`（索引 {settings.milvus.index_type} / {settings.milvus.metric_type}）")
    lines.append(f"- 语料：{len(documents)} 篇（{CORPUS.name}）")
    lines.append(f"- 检索深度：top-{TOP_K}｜查询：{len(QUERIES)} 条")
    lines.append("")

    embedder = default_embedder(settings, backend="auto")
    lines.append(f"- 向量器：{embedder.name}（dim={embedder.dim}）")
    lines.append("")

    # —— 内存精确索引 ——
    t0 = time.perf_counter()
    memory_index = build_index(documents, settings=settings, embedder=embedder, milvus=False)
    memory_build = time.perf_counter() - t0

    # —— Milvus 索引（strict：拒绝降级，否则这份报告可能其实跑在内存上）——
    strict_settings = replace(settings, milvus=replace(settings.milvus, require_sync=True))
    t0 = time.perf_counter()
    try:
        milvus_index = build_index(
            documents,
            settings=strict_settings,
            embedder=embedder,
            milvus=True,
            strict=True,
        )
    except Exception as exc:  # noqa: BLE001 - 连不上就是这份报告的核心结论
        from scout.rag.milvus_store import MilvusDenseStore

        detail = f"{type(exc).__name__}: {exc}"
        lines.append("## 结论：Milvus 不可用")
        lines.append("")
        lines.append(f"```\n{detail}\n```")
        lines.append("")
        # 区分"连不上"与"连上了但用法有错"——修复动作完全不同，
        # 把 schema 冲突也报成"先把容器起来"会让人白折腾半天。
        if MilvusDenseStore.looks_like_connection_error(detail):
            lines.append("**诊断：连不上服务。** 先把容器起来：")
            lines.append("")
            lines.append("```")
            lines.append("docker start milvus-etcd milvus-minio milvus-standalone")
            lines.append("```")
            lines.append("")
            lines.append("（Milvus standalone 依赖 etcd 与 MinIO，缺一不可。）")
        else:
            lines.append(
                "**诊断：服务连上了，但这次调用失败——是用法/数据层面的问题，不是容器没起。**"
            )
            lines.append("请按上面的异常信息排查（集合 schema、字段类型、索引参数等）。")
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 1
    milvus_build = time.perf_counter() - t0

    status = milvus_index.milvus_status()
    lines.append("## 连通性与往返")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 集合 | `{status.get('collection')}` |")
    lines.append(f"| 实体数 | {status.get('entities')} |")
    lines.append(f"| 维度 | {status.get('dimension')} |")
    lines.append(f"| 降级码 | {status.get('degraded_code') or '（无，说明真的走了 Milvus）'} |")
    lines.append(f"| 建索引耗时 | 内存 {memory_build:.2f}s ｜ Milvus {milvus_build:.2f}s |")
    lines.append("")

    if status.get("degraded_code"):
        lines.append("> ⚠️ 出现降级码说明这次**没有真正用上 Milvus**，下面的重合度没有意义。")
        lines.append("")

    # —— 重合度与延迟 ——
    lines.append("## 与内存精确检索的重合度（ANN 召回估计）")
    lines.append("")
    lines.append("| 查询 | 重合 | 内存(ms) | Milvus(ms) |")
    lines.append("|---|---|---|---|")

    overlaps: list[float] = []
    memory_latency: list[float] = []
    milvus_latency: list[float] = []
    for query in QUERIES:
        t0 = time.perf_counter()
        exact = memory_index.search(query, top_k=TOP_K)
        memory_latency.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        approx = milvus_index.search(query, top_k=TOP_K)
        milvus_latency.append((time.perf_counter() - t0) * 1000.0)

        exact_ids = [hit.chunk.chunk_id for hit in exact.hits]
        approx_ids = [hit.chunk.chunk_id for hit in approx.hits]
        if not exact_ids:
            continue
        common = len(set(exact_ids) & set(approx_ids))
        overlap = common / len(exact_ids)
        overlaps.append(overlap)
        lines.append(
            f"| {query[:22]} | {overlap:.0%}（{common}/{len(exact_ids)}） "
            f"| {memory_latency[-1]:.1f} | {milvus_latency[-1]:.1f} |"
        )

    lines.append("")
    if overlaps:
        mean_overlap = sum(overlaps) / len(overlaps)
        lines.append(f"- **平均重合度：{mean_overlap:.1%}**（即 top-{TOP_K} 的 ANN 召回估计）")
        lines.append(
            f"- 内存检索延迟：p50 {_percentile(memory_latency, 0.5):.1f}ms ｜ "
            f"p95 {_percentile(memory_latency, 0.95):.1f}ms ｜ "
            f"均值 {statistics.mean(memory_latency):.1f}ms"
        )
        lines.append(
            f"- Milvus 检索延迟：p50 {_percentile(milvus_latency, 0.5):.1f}ms ｜ "
            f"p95 {_percentile(milvus_latency, 0.95):.1f}ms ｜ "
            f"均值 {statistics.mean(milvus_latency):.1f}ms"
        )
        lines.append("")
        lines.append("## 怎么读这份报告")
        lines.append("")
        lines.append(
            "- 内存索引是**精确**检索（暴力余弦），Milvus/HNSW 是**近似**检索。"
            "两者不会 100% 一致，这不是 bug，是 ANN 的固有取舍。"
        )
        lines.append(
            "- **重合度就是 ANN 召回**：它回答「为了这个延迟，我们付出了几个百分点的召回」。"
            "重合度高说明参数（M / efConstruction / ef）够用；偏低就该调参或换 IVF_FLAT。"
        )

        # 延迟结论**由数据算出来**，不写死。
        # 这条是踩过的坑：脚本初版写着「小语料下内存往往更快（省一次网络往返）」，
        # 而实测在 156 个向量时 Milvus 就已经更快（9.3ms vs 15.7ms）——
        # 因为内存实现是纯 Python 逐条算余弦，瓶颈在解释器开销，
        # 一次网络往返（几毫秒）跟它比可以忽略。
        # **报告里不该出现没有数据支持的解读**，所以这里改成按实测下结论。
        mem_p50 = _percentile(memory_latency, 0.5)
        mil_p50 = _percentile(milvus_latency, 0.5)
        if mil_p50 < mem_p50:
            ratio = mem_p50 / mil_p50 if mil_p50 else 0.0
            lines.append(
                f"- **延迟：Milvus 更快（p50 {mil_p50:.1f}ms vs 内存 {mem_p50:.1f}ms，约 {ratio:.1f}×）。**"
                " 原因不是向量库更快，而是**我们的内存实现是纯 Python 逐条算余弦**——"
                "它的定位是「精确、可复现的召回基线」，不是性能实现。"
                "一次网络往返（几毫秒）相对解释器开销可以忽略，所以这个反直觉的结果是合理的。"
            )
            lines.append(
                "  **要真比性能，应该把内存版换成 numpy 批量矩阵乘**——"
                "那才能把「算法差异」和「实现语言差异」分开。"
            )
        else:
            lines.append(
                f"- 延迟：内存更快（p50 {mem_p50:.1f}ms vs Milvus {mil_p50:.1f}ms）——"
                "小规模下省掉一次网络往返的收益超过了索引结构的优势。"
            )
        lines.append(
            "- **重合度 100% 在小规模上是必然的**：几百个向量时 HNSW 几乎等价于暴力检索。"
            "所以这个数字此刻的意义是「参数没配错」，不是「我们的 ANN 很好」——"
            "**要得到有信息量的 ANN 召回，必须把规模拉到万级以上再测。**"
        )
    else:
        lines.append("- 没有可比较的检索结果（语料或查询为空）。")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告 -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
