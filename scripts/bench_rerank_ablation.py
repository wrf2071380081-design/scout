"""重排消融 @ 规模：**只测检索指标，不调 LLM**。

**为什么这个脚本存在（一个省下 8 小时的观察）。**
重排器对比的核心指标是 Recall@k——它是**检索指标，与生成模型无关**。
之前的对照因为要拿端到端作答率，被迫每条走真实 LLM：
30 条 × 2 配置 ≈ 45 分钟，扩到 300 条就是 8 小时以上。

但只要目标是 Recall@k，就完全不需要 LLM：
索引建一次、两个配置共用，本脚本 300 条 × 2 配置约十几分钟跑完。
**端到端作答率仍然需要真实 LLM——那是另一件事，不该混在一次实验里。**
把"检索质量"和"生成质量"分开测，既省成本，也让两个结论各自干净。

输出：两个配置的 Recall@k + **配对 bootstrap 置信区间**（同一批样本上做差，
消掉"这题难不难"这个共同因素）。
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SCOUT_EMBED_BACKEND", "auto")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import get_settings  # noqa: E402
from scout.llm.scripted import HeuristicLLM  # noqa: E402
from scout.rag.pipeline import RAGPipeline, build_index  # noqa: E402
from scout.trace import Trace  # noqa: E402

DATA_FILE = Path(r"C:\agent-project\scout\datasets\musique_ans_dev.jsonl")
OUT_MD = Path(
    os.environ.get("BENCH_OUT", r"C:\agent-project\scout\evals\results\musique_rerank_ablation.md")
)
SAMPLE_LIMIT = int(os.environ.get("BENCH_LIMIT", "300"))
SEED = 42
CONFIGS = (("lexical", "词法重排"), ("cross", "cross-encoder"))


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    return sum(1 for name in gold if name in top) / len(gold)


def bootstrap_ci(values: list[float], rounds: int = 1000, seed: int = SEED) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(rounds)
    )
    return (means[int(0.025 * rounds)], means[min(int(0.975 * rounds), rounds - 1)])


def load_samples(limit: int) -> tuple[list[dict], list[tuple[str, str]]]:
    samples: list[dict] = []
    documents: list[tuple[str, str]] = []
    seen: set[str] = set()
    with DATA_FILE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line.strip()) if line.strip() else None
            if not payload:
                continue
            paragraphs = payload.get("paragraphs") or []
            if not paragraphs:
                continue
            qid = str(payload.get("id") or f"q{index}")
            gold: list[str] = []
            for position, para in enumerate(paragraphs):
                title = str(para.get("title") or f"p{position}")
                text = str(para.get("paragraph_text") or "").strip()
                if not text:
                    continue
                name = f"{qid}|{title}"
                if name not in seen:
                    seen.add(name)
                    documents.append((name, text))
                if para.get("is_supporting"):
                    gold.append(name)
            if gold:
                samples.append({"case_id": qid, "question": str(payload.get("question") or ""), "gold": gold})
            if len(samples) >= limit:
                break
    return samples, documents


def main() -> int:
    settings = get_settings()
    samples, documents = load_samples(SAMPLE_LIMIT)

    lines: list[str] = []
    lines.append("# 重排消融（检索指标，不调 LLM）\n")
    lines.append(f"- 样本：**{len(samples)}** 条（MuSiQue-Ans dev 前 {SAMPLE_LIMIT} 条含支撑段落的样本）")
    lines.append(f"- 语料：{len(documents)} 个段落的联合检索池")
    lines.append("- 检索：稠密(语义向量) + BM25，RRF 融合；唯一变量＝**重排器后端**")
    lines.append("- 生成：**离线启发式**（检索指标与生成模型无关，故不消耗任何 LLM 调用）")
    lines.append(f"- 统计：配对 bootstrap，1000 次重采样，95% 置信区间，seed={SEED}\n")

    t0 = time.time()
    index = build_index(documents, settings=settings, embed_backend="auto")
    lines.append(f"索引：{len(index.chunks)} 块（{time.time() - t0:.0f}s，向量器 {index.embedder.name}）\n")

    per_config: dict[str, dict[str, list[float]]] = {}
    for backend, label in CONFIGS:
        tuned = replace(settings, retrieval=replace(settings.retrieval, rerank_backend=backend))
        pipeline = RAGPipeline(index, HeuristicLLM(), settings=tuned)
        r5: list[float] = []
        r10: list[float] = []
        started = time.time()
        for i, sample in enumerate(samples, start=1):
            trace = Trace(question=sample["question"])
            try:
                units, _meta, _grade = pipeline.collect(sample["question"], trace)
                retrieved = [unit.chunk.filename for unit in units]
            except Exception:  # noqa: BLE001 - 单条失败不拖垮整批
                retrieved = []
            r5.append(recall_at_k(retrieved, sample["gold"], 5))
            r10.append(recall_at_k(retrieved, sample["gold"], 10))
            if i % 50 == 0:
                print(f"  [{label}] {i}/{len(samples)} … {time.time() - started:.0f}s", flush=True)
        per_config[backend] = {"r5": r5, "r10": r10}
        lines.append(
            f"- `{label}` 重排完成：{time.time() - started:.0f}s"
        )
    lines.append("")

    lines.append("## 绝对值\n")
    lines.append("| 重排器 | Recall@5（95% CI） | Recall@10（95% CI） |")
    lines.append("|---|---|---|")
    for backend, label in CONFIGS:
        data = per_config[backend]
        m5 = sum(data["r5"]) / len(data["r5"])
        m10 = sum(data["r10"]) / len(data["r10"])
        lo5, hi5 = bootstrap_ci(data["r5"])
        lo10, hi10 = bootstrap_ci(data["r10"])
        lines.append(
            f"| {label} | **{m5 * 100:.1f}%** [{lo5 * 100:.1f}, {hi5 * 100:.1f}] "
            f"| **{m10 * 100:.1f}%** [{lo10 * 100:.1f}, {hi10 * 100:.1f}] |"
        )

    lines.append("\n## 配对差量（cross-encoder − 词法重排）\n")
    lines.append("| 指标 | 差量（pp） | 配对 bootstrap 95% CI | 跨零 |")
    lines.append("|---|---|---|---|")
    for key, name in (("r5", "Recall@5"), ("r10", "Recall@10")):
        base = per_config["lexical"][key]
        target = per_config["cross"][key]
        deltas = [b - a for a, b in zip(base, target)]
        mean = sum(deltas) / len(deltas)
        low, high = bootstrap_ci(deltas)
        crosses = low <= 0 <= high
        lines.append(
            f"| {name} | **{mean * 100:+.2f}** | [{low * 100:+.2f}, {high * 100:+.2f}] "
            f"| {'是' if crosses else '否'} |"
        )

    lines.append("\n## 怎么读\n")
    lines.append(
        "- 这是**同一批样本上的配对比较**：每条的差量先算出来再统计，"
        "消掉了'这题难不难'这个共同因素，比两组独立均值相减更灵敏。\n"
        "- **置信区间不跨零**才说明差异不是噪声。跨零的数字不能拿去汇报。\n"
        "- 本表不含作答率——作答率需要真实 LLM，属于另一次实验。"
        "把检索与生成分开测，两个结论各自干净。\n"
    )

    # —— 难度随前缀的变化：解释"小样本时数字为什么更高" ——
    # MuSiQue dev 是按顺序取的，前 N 条的难度并不代表全集。
    # 早期用一个更小的前缀（如 50 条）报过更高的数字，这不是造假，是抽样偏差；
    # 把这条曲线画出来，才能解释清楚"为什么扩到 300 条之后数字降了"。
    lines.append("\n## 前缀难度曲线（为什么小样本上的数字更高）\n")
    lines.append("| 前缀样本数 | 词法重排 Recall@5 | cross-encoder Recall@5 | 差量（pp） |")
    lines.append("|---|---|---|---|")
    for prefix in (50, 100, 200, len(samples)):
        if prefix > len(samples):
            continue
        lex = per_config["lexical"]["r5"][:prefix]
        cross = per_config["cross"]["r5"][:prefix]
        m_lex = sum(lex) / len(lex)
        m_cross = sum(cross) / len(cross)
        lines.append(
            f"| {prefix} | {m_lex * 100:.1f}% | {m_cross * 100:.1f}% | {(m_cross - m_lex) * 100:+.2f} |"
        )
    lines.append("")
    lines.append(
        "> 这条曲线的用途是**把抽样偏差摊开**：如果前 50 条明显比全集更容易，"
        "那么基于前 50 条报出的绝对指标就会偏高。"
        "**报绝对指标必须用它自己的样本量对应的行，不能拿小样本的行去代表全集。**\n"
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-24:]))
    print(f"\ndone -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
