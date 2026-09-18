"""在公开基准 MuSiQue-Ans 上评测 scout 的检索与作答。

**为什么要有这个脚本。**

自建评测集永远有一个无法回避的质疑：**题是你自己出的，分是你自己算的。**
在公开、冻结的基准上评一次，是把"我觉得它好"换成"在别人定的卷子上它拿了多少分"。

MuSiQue-Ans 是公开的多跳问答基准，每条样本含一个问题、一个标准答案（含别名），
以及 20 个段落，其中若干被标注为**支撑段落（supporting）**。这正好对应 RAG 的两件事：

- **检索**：能不能把支撑段落找回来（Recall@k）
- **作答**：能不能给出接近标准答案的答案（token 级 F1）

**为什么报 bootstrap 95% 置信区间，而不是只报一个点估计。**

"Recall@5 = 41.0%" 这种数字在 200 条样本上其实是有噪声的。
置信区间回答的是"这个结论有多稳"——
尤其是做**消融对比**时，+1.8pp 这种小提升到底是真的还是噪声，
必须靠配对 bootstrap 才能说清。这也是本项目最想强调的一条：
**结论要能被验证，而不只是被汇报。**

用法：

```bash
python scripts/bench_musique.py --limit 200 --configs full,dense
```

数据文件：``datasets/musique_ans_dev.jsonl``（来自 HF ``dgslibisey/MuSiQue`` 的
``musique_ans_v1.0_dev.jsonl``，公开冻结的 dev 集）。
缺失时脚本会给出下载提示，不会静默使用别的语料。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import Settings  # noqa: E402
from scout.llm.scripted import HeuristicLLM  # noqa: E402
from scout.rag.index import RetrievalMode  # noqa: E402
from scout.rag.merge import MergeMode  # noqa: E402
from scout.rag.pipeline import PipelineConfig, RAGPipeline, build_index  # noqa: E402
from scout.trace import Trace  # noqa: E402

DATA_FILE = Path(__file__).resolve().parents[1] / "datasets" / "musique_ans_dev.jsonl"
SOURCE_URL = "https://huggingface.co/datasets/dgslibisey/MuSiQue"
DEFAULT_LIMIT = 200
BOOTSTRAP_ROUNDS = 1000


# —— 数据 ——


def load_musique(limit: int) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """读取 MuSiQue，返回 (样本列表, 语料文档列表)。

    语料是**所有样本段落的并集**——也就是说，每个问题都要在包含其他问题段落的
    大池子里检索。这是有意为之的更严格设定：如果每个问题只在自己的 20 段里选，
    检索任务会简单得多，得到的分数也不能说明真实场景下的表现。
    """

    if not DATA_FILE.exists():
        raise SystemExit(
            f"缺少数据文件：{DATA_FILE}\n"
            f"请从 {SOURCE_URL} 下载 musique_ans_v1.0_dev.jsonl 放到该路径。"
        )
    samples: list[dict[str, Any]] = []
    documents: list[tuple[str, str]] = []
    with DATA_FILE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            paragraphs = payload.get("paragraphs") or []
            if not paragraphs:
                continue
            qid = str(payload.get("id") or f"q{index}")
            gold_docs: list[str] = []
            for position, para in enumerate(paragraphs):
                title = str(para.get("title") or f"p{position}")
                text = str(para.get("paragraph_text") or "").strip()
                if not text:
                    continue
                name = f"{qid}|{title}"
                documents.append((name, text))
                if para.get("is_supporting"):
                    gold_docs.append(name)
            if not gold_docs:
                continue
            samples.append(
                {
                    "case_id": qid,
                    "question": str(payload.get("question") or "").strip(),
                    "answers": [str(payload.get("answer") or "")]
                    + [str(item) for item in (payload.get("answer_aliases") or [])],
                    "gold_docs": gold_docs,
                }
            )
    return samples, documents


# —— 指标 ——


_TOKEN = re.compile(r"[\w]+|[^\s\w]", re.UNICODE)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """token 级 F1，取所有标准答案别名中的最高值。"""

    pred_tokens = _TOKEN.findall((prediction or "").lower())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for reference in references:
        gold_tokens = _TOKEN.findall((reference or "").lower())
        if not gold_tokens:
            continue
        common: dict[str, int] = {}
        for token in pred_tokens:
            common[token] = common.get(token, 0) + 1
        overlap = 0
        gold_counts: dict[str, int] = {}
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1
        for token, count in common.items():
            overlap += min(count, gold_counts.get(token, 0))
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """检索召回率：gold 文档中有多少出现在前 k 个去重结果里。"""

    if not gold:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    hits = sum(1 for name in gold if name in top)
    return hits / len(gold)


# —— 统计 ——


def bootstrap_ci(values: Sequence[float], *, rounds: int = BOOTSTRAP_ROUNDS, seed: int = 42) -> tuple[float, float]:
    """单样本的 bootstrap 95% 置信区间（百分位法）。"""

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    low = means[int(0.025 * rounds)]
    high = means[min(int(0.975 * rounds), rounds - 1)]
    return (low, high)


def paired_bootstrap_delta(
    left: Sequence[float], right: Sequence[float], *, rounds: int = BOOTSTRAP_ROUNDS, seed: int = 42
) -> tuple[float, tuple[float, float]]:
    """配对 bootstrap：估计 (left − right) 的均值与置信区间。

    配对是关键：**同一条样本上的两次运行相减**，把"这条题难不难"这个
    共同因素消掉，剩下的才是配置差异。非配对比较会被题目难度差异淹没。
    """

    if len(left) != len(right) or not left:
        return (0.0, (0.0, 0.0))
    size = len(left)
    deltas = [a - b for a, b in zip(left, right)]
    point = sum(deltas) / size
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        picks = [rng.randrange(size) for _ in range(size)]
        means.append(sum(deltas[i] for i in picks) / size)
    means.sort()
    return (point, (means[int(0.025 * rounds)], means[min(int(0.975 * rounds), rounds - 1)]))


# —— 配置 ——


def make_config(mode: str) -> PipelineConfig:
    """按模式构造流水线配置。"""

    if mode == "dense":
        # 稠密基线：关掉稀疏通道、重排与父块合并——这是"只留语义向量"的对照组。
        return PipelineConfig(
            retrieval_mode=RetrievalMode.DENSE_ONLY,
            rerank_enabled=False,
            merge_mode=MergeMode.OFF,
        )
    if mode == "hybrid_norerank":
        return PipelineConfig(retrieval_mode=RetrievalMode.HYBRID, rerank_enabled=False)
    return PipelineConfig()  # 全量默认配置


CONFIGS: dict[str, Callable[[], PipelineConfig]] = {
    "full": lambda: make_config("full"),
    "dense": lambda: make_config("dense"),
    "hybrid_norerank": lambda: make_config("hybrid_norerank"),
}


def run_config(
    samples: list[dict[str, Any]],
    documents: list[tuple[str, str]],
    mode: str,
    settings: Settings,
    embed_backend: str = "auto",
) -> dict[str, Any]:
    """跑一种配置，返回每条样本的指标。"""

    index = build_index(documents, settings=settings, embed_backend=embed_backend)
    pipeline = RAGPipeline(index, HeuristicLLM(), settings=settings, config=CONFIGS[mode]())
    per_case: list[dict[str, Any]] = []
    started = time.time()
    for sample in samples:
        trace = Trace(question=sample["question"])
        units, _meta, _grade = pipeline.collect(sample["question"], trace)
        retrieved = [unit.chunk.filename for unit in units]
        result = pipeline.answer(sample["question"])
        per_case.append(
            {
                "case_id": sample["case_id"],
                "recall@1": recall_at_k(retrieved, sample["gold_docs"], 1),
                "recall@3": recall_at_k(retrieved, sample["gold_docs"], 3),
                "recall@5": recall_at_k(retrieved, sample["gold_docs"], 5),
                "recall@10": recall_at_k(retrieved, sample["gold_docs"], 10),
                "answer_f1": token_f1(result.answer, sample["answers"]),
                "outcome": result.outcome,
            }
        )
    return {
        "mode": mode,
        "per_case": per_case,
        "duration_s": time.time() - started,
        "corpus_docs": len(documents),
        "chunks": len(index.chunks),
        "embedder": index.embedder.name,
    }


def summarise(run: dict[str, Any]) -> dict[str, Any]:
    cases = run["per_case"]
    summary: dict[str, Any] = {"mode": run["mode"], "n": len(cases)}
    for key in ("recall@1", "recall@3", "recall@5", "recall@10", "answer_f1"):
        values = [float(item[key]) for item in cases]
        mean = sum(values) / len(values) if values else 0.0
        low, high = bootstrap_ci(values)
        summary[key] = {
            "mean": round(mean * 100, 2),
            "ci95": [round(low * 100, 2), round(high * 100, 2)],
        }
    summary["answered_rate"] = round(
        100.0 * sum(1 for item in cases if item["outcome"] == "answered") / max(1, len(cases)), 2
    )
    summary["duration_s"] = round(run["duration_s"], 1)
    return summary


# —— 报告 ——


def render(summaries: list[dict[str, Any]], runs: dict[str, dict[str, Any]], limit: int) -> str:
    lines = [
        "# MuSiQue-Ans 公开基准评测报告",
        "",
        f"- 数据集：**MuSiQue-Ans dev**（公开冻结样本）｜来源 `{SOURCE_URL}`",
        f"- 样本数：**{summaries[0]['n']}**（前 {limit} 条含支撑段落的样本）",
        f"- 语料规模：{runs[summaries[0]['mode']]['corpus_docs']} 个段落（所有样本段落的并集）"
        f" → {runs[summaries[0]['mode']]['chunks']} 个块",
        f"- 向量器：**{runs[summaries[0]['mode']]['embedder']}**"
        f"{'（语义向量）' if 'hashing' not in runs[summaries[0]['mode']]['embedder'] else '（词法哈希，非语义基线）'}",
        "- 统计方法：**bootstrap 百分位法**，1000 次重采样，95% 置信区间",
        "- 检索设定：每个问题都要在**包含其他问题段落的大池子**里检索（更严格的设定）",
        "",
        "## 各配置结果",
        "",
        "| 配置 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Answer F1 | 作答率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        def cell(key: str) -> str:
            item = summary[key]
            return f"{item['mean']:.1f}% [{item['ci95'][0]:.1f}, {item['ci95'][1]:.1f}]"

        lines.append(
            f"| `{summary['mode']}` | {cell('recall@1')} | {cell('recall@3')} | "
            f"{cell('recall@5')} | {cell('recall@10')} | {cell('answer_f1')} | "
            f"{summary['answered_rate']:.0f}% |"
        )

    if len(summaries) >= 2:
        base = summaries[0]
        lines += ["", "## 配对消融（相对首个配置的差量，pp）", ""]
        lines.append("| 对比 | 指标 | 差量 | 95% CI | 是否跨零 |")
        lines.append("|---|---|---|---|---|")
        left_cases = runs[base["mode"]]["per_case"]
        for other in summaries[1:]:
            right_cases = runs[other["mode"]]["per_case"]
            for key in ("recall@5", "recall@10", "answer_f1"):
                delta, (low, high) = paired_bootstrap_delta(
                    [float(a[key]) for a in left_cases],
                    [float(b[key]) for b in right_cases],
                )
                crosses = "是（不显著）" if low <= 0 <= high else "否"
                lines.append(
                    f"| `{base['mode']}` − `{other['mode']}` | {key} | "
                    f"{delta * 100:+.2f} | [{low * 100:.2f}, {high * 100:.2f}] | {crosses} |"
                )

    lines += [
        "",
        "## 读这份报告要注意的两点",
        "",
        "1. **Answer F1 的天花板由离线生成器决定。** `HeuristicLLM` 是词法抽取式实现，",
        "   它的作用是让流程可复现，不是打榜。检索指标（Recall@k）不受此限制，",
        "   因此本报告的**主要结论看检索列**。",
        "2. **置信区间比点估计重要。** 200 条样本上的 ±2pp 差异很可能跨零；",
        "   凡 CI 跨零的结论都不该被当成'这个模块有效'。",
        "",
        "> 本报告由 `scripts/bench_musique.py` 生成，可离线复现（不需要任何 API key）。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="在 MuSiQue-Ans 上评测 scout")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--configs", default="full,dense")
    parser.add_argument(
        "--embed",
        choices=("auto", "local", "hashing", "openai"),
        default="auto",
        help="向量器后端；local=本地语义模型（BAAI/bge-small-zh-v1.5）",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    # Settings 是 frozen 的（配置不该被运行时改写），所以用 replace 构造。
    settings = dataclasses.replace(
        Settings(),
        retrieval=dataclasses.replace(Settings().retrieval, top_k=20, evidence_budget_chars=16000),
    )

    samples, documents = load_musique(args.limit)
    print(f"样本 {len(samples)} 条 / 段落 {len(documents)} 个，向量器={args.embed}，开始评测…", flush=True)

    runs: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for mode in [item.strip() for item in args.configs.split(",") if item.strip()]:
        if mode not in CONFIGS:
            raise SystemExit(f"未知配置：{mode}（可选 {', '.join(CONFIGS)}）")
        run = run_config(samples, documents, mode, settings, args.embed)
        runs[mode] = run
        summary = summarise(run)
        summaries.append(summary)
        print(
            f"  {mode}: Recall@5 {summary['recall@5']['mean']:.1f}% "
            f"CI[{summary['recall@5']['ci95'][0]:.1f}, {summary['recall@5']['ci95'][1]:.1f}] "
            f"| F1 {summary['answer_f1']['mean']:.1f}% | {run['duration_s']:.1f}s"
            f" | embedder={run['embedder']}",
            flush=True,
        )

    report = render(summaries, runs, args.limit)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "evals" / "results" / "musique_bench.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"已生成 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
