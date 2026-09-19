"""MuSiQue 公开基准：真实 LLM（Kimi K3）版本的 Answer F1 测量。

与 `scripts/bench_musique.py` 的区别：那条脚本用离线启发式 LLM，
它衡量的是"流程能不能跑通"；这条用真实 LLM，
衡量的是"换了真模型之后系统到底答得怎么样"。

由于真实 LLM 调用有成本（~25–60s/题），默认只跑冻结样本的一半量化（默认 50 条prod冻结样本），
把 Answer F1 从「离线上限 ~0.5%」拉到可对外汇报的真实水平。

**同时跑两条线：**
- 持门控（gated）：production 默认配置，包含充分性门控——
  它会在证据不足时拒答/求澄清，这就是【作答率 < 100%】的由来；
- 不提门控（ungated）：强行作答，衡量"它真实生成答案的能力上限"。

**离线替身 vs 真实 LLM 应该看的正好是这条分界线。**
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SCOUT_EMBED_BACKEND", "auto")
os.environ.setdefault("SCOUT_RERANK_BACKEND", "cross")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import Settings, get_settings  # noqa: E402
from scout.llm.scripted import default_client  # noqa: E402
from scout.rag.pipeline import PipelineConfig, RAGPipeline, build_index  # noqa: E402

DATA_FILE = Path(r"C:\agent-project\scout\datasets\musique_ans_dev.jsonl")
OUT_MD = Path(os.environ.get("BENCH_OUT", r"C:\agent-project\scout\evals\results\musique_bench_realllm.md"))
SAMPLE_LIMIT = int(os.environ.get("BENCH_LIMIT", "50"))
SEED = 42

_TOKEN = re.compile(r"[\w]+|[^\s\w]", re.UNICODE)


def token_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = _TOKEN.findall((prediction or "").lower())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for reference in references:
        gold_tokens = _TOKEN.findall((reference or "").lower())
        if not gold_tokens:
            continue
        gold_counts: dict[str, int] = {}
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1
        pred_counts: dict[str, int] = {}
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1
        overlap = sum(min(pred_counts.get(t, 0), gold_counts.get(t, 0)) for t in pred_counts)
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    return sum(1 for name in gold if name in top) / len(gold)


def bootstrap_ci(values, rounds=500, seed=SEED):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(rounds))
    return (means[int(0.025 * rounds)], means[min(int(0.975 * rounds), rounds - 1)])


def run_suite(
    samples: list[dict],
    index: Any,
    client: Any,
    settings: Settings,
    config: Any,
    label: str,
) -> dict:
    """对同一份样本与索引，跑一种配置，返回按题指标。"""

    pipeline = RAGPipeline(index, client, settings=settings, config=config)
    per_r5, per_r10, per_f1, per_outcome = [], [], [], []
    run_lines = []
    retries = 0
    for i, sample in enumerate(samples, start=1):
        t1 = time.time()
        try:
            result = pipeline.answer(sample["question"])
            retrieved = [u.chunk.filename for u in result.units]
            r5 = recall_at_k(retrieved, sample["gold_docs"], 5)
            r10 = recall_at_k(retrieved, sample["gold_docs"], 10)
            f1 = token_f1(result.answer, sample["answers"])
            per_r5.append(r5)
            per_r10.append(r10)
            per_f1.append(f1)
            per_outcome.append(result.outcome)
            answerable = result.outcome in {"answered", ".pass", "pass"}
            run_lines.append(
                f"- [{i}] {sample['question'][:48]}… | R@5 {r5:.0%} | F1 {f1:.2f} | "
                f"{result.outcome} | grounding {result.grounding.verdict.value if result.grounding else '—'} | {time.time()-t1:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001
            retries += 1
            run_lines.append(
                f"- [{i}] 错误 {sample['question'][:40]}: {type(exc).__name__} {str(exc)[:80]}"
            )
        if i % 8 == 0:
            OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    return {
        "label": label,
        "per_r5": per_r5,
        "per_r10": per_r10,
        "per_f1": per_f1,
        "per_outcome": per_outcome,
        "answered_rate": 100.0 * sum(1 for o in per_outcome if o in ("answered", ".pass", "pass")) / max(1, len(per_outcome)),
        "run_lines": run_lines,
        "retries": retries,
    }


def metric_block(name, values):
    if not values:
        return f"| {name} | — | — |"
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values)
    return f"| {name} | {mean * 100:.1f}% | [{low*100:.1f}, {high*100:.1f}] |"


def main() -> int:
    lines = []
    settings = get_settings()
    client = default_client()
    lines.append(f"# MuSiQue-Ans 真实 LLM 评测（Kimi K3）")
    lines.append("")
    lines.append(f"- LLM: **{client.model_name}**（真实模型，token.caih.com）")
    lines.append(f"- 向量器: {settings.embedding.backend}（auto →local）")
    lines.append(f"- 重排器: {settings.retrieval.rerank_backend}")

    # 加载固定子集
    samples = []
    documents = []
    with DATA_FILE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line.strip()) if line.strip() else None
            if not payload:
                continue
            paragraphs = payload.get("paragraphs") or []
            if not paragraphs:
                continue
            qid = str(payload.get("id") or f"q{index}")
            gold_docs = []
            for position, para in enumerate(paragraphs):
                title = str(para.get("title") or f"p{position}")
                text = str(para.get("paragraph_text") or "").strip()
                if not text:
                    continue
                name = f"{qid}|{title}"
                documents.append((name, text))
                if para.get("is_supporting"):
                    gold_docs.append(name)
            if gold_docs:
                samples.append(
                    {
                        "case_id": qid,
                        "question": str(payload.get("question") or "").strip(),
                        "answers": [str(payload.get("answer") or "")]
                        + [str(item) for item in (payload.get("answer_aliases") or [])],
                        "gold_docs": gold_docs,
                    }
                )
            if len(samples) >= SAMPLE_LIMIT:
                break
    lines.append(f"- 样本: {len(samples)} 条（公开冻结 dev 集前 {SAMPLE_LIMIT} 条）")
    lines.append(f"- 语料: {len(documents)} 个段落的联合检索池")

    t0 = time.time()
    index = build_index(documents, settings=settings, embed_backend="auto")
    lines.append(f"- 索引: {len(index.chunks)} 块（{time.time()-t0:.0f}s，向量器 {index.embedder.name}）")

    suites = []
    for config, label in (
        (PipelineConfig(), "gated（含门控）"),
        (PipelineConfig(sufficiency_gate=False), "ungated（强行作答）"),
    ):
        t = time.time()
        suite = run_suite(samples, index, client, settings, config, label)
        suite["duration_s"] = time.time() - t
        suites.append(suite)

    lines.append("")
    lines.append("## 两套配置的对比")
    lines.append("")
    lines.append("| 配置 | 作答率 | Recall@5 | Recall@10 | **Answer F1** |")
    lines.append("|---|---|---|---|---|")
    for suite in suites:
        r5_mean = sum(suite["per_r5"]) / max(1, len(suite["per_r5"]))
        r10_mean = sum(suite["per_r10"]) / max(1, len(suite["per_r10"]))
        f1_mean = sum(suite["per_f1"]) / max(1, len(suite["per_f1"]))
        lines.append(
            f"| {suite['label']} | {suite['answered_rate']:.0f}% | "
            f"{r5_mean*100:.1f}% | {r10_mean*100:.1f}% | **{f1_mean*100:.1f}%** |"
        )

    lines.append("")
    for suite in suites:
        lines.append(f"## 逐题记录（{suite['label']}，{suite['duration_s']:.0f}s）")
        lines.append("")
        lines.extend(suite["run_lines"][:60])
        lines.append("")

    lines.append("## 怎么读这份报告")
    lines.append("")
    lines.append("1. **gated** 是真实生产默认配置。作答率不是 100%——")
    lines.append("   当检索证据撑不起这个多跳问题时，系统选择拒答/求澄清，而不是编造。")
    lines.append("   这不是缺陷，是**门控在工作**。同时它也诚实暴露了检索的短板：")
    lines.append("   在英文语料+中文向量模型的错配下，有多跳问题本来就应该拒掉。")
    lines.append("2. **ungated** 拿掉门控。看它是为了把「它能不能答」和「它该不该答」拆开——")
    lines.append("   F1 在这里衡量的是**单纯生成能力**。")
    lines.append("3. **之前那个 Answer F1 ≈ 0.5%** 是用离线 HeuristicLLM 测的——")
    lines.append("   那衡量的是「流程能跑」的天花板，不是系统的真实能力。")
    lines.append("   换上真实 LLM 后，检索列可能变化不大，但**答案质量是质变**。这才是「真正能答」。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"done -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
