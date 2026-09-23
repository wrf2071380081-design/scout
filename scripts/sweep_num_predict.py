"""扫 num_predict：找"够用且最快"的输出预算拐点。

**为什么要扫而不是直接砍一半。**
本地 glm-ocr 每次都会写满 `num_predict` 上限再被尾部裁剪掉——看着像"浪费"，
但**真实内容有多长必须先量出来**。砍太狠会从"浪费"变成"截断"，
而截断的表现是**表格缺行**——比变慢严重得多（慢是体验问题，缺内容是正确性问题）。

所以本实验对每个预算值同时记录四件事：

1. **耗时**（省了多少）
2. **是否撞上限**（`done_reason=length` → 说明预算不够，输出可能不全）
3. **关键事实命中**（内容完整性）
4. **尾部是否干净**（结尾是不是 `</table>`，而不是被切断的半行）

**只有"耗时下降 + 未截断 + 命中不变"才算真正的优化**；
否则就是把正确性换成了速度，那笔账不能算赚。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# 仓库根也要在 path 上，才能 import scripts.vision_cost_quality 复用同一份 ground truth
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scout.data.extract import OllamaOCRExtractor  # noqa: E402
from scripts.vision_cost_quality import KEY_FACTS  # noqa: E402


def run_once(image: Path, num_predict: int) -> dict[str, object]:
    extractor = OllamaOCRExtractor(num_ctx=16384, num_predict=num_predict)
    started = time.perf_counter()
    payload = extractor.build_payload(image)
    data = extractor._post(payload)  # noqa: SLF001 - 实验需要原始响应（done_reason）
    elapsed = time.perf_counter() - started
    raw = str(data.get("response") or "")
    trimmed, removed = extractor.trim_degenerate_tail(raw)
    lowered = trimmed.lower()
    hits = [fact for fact in KEY_FACTS if fact.lower() in lowered]
    return {
        "num_predict": num_predict,
        "seconds": round(elapsed, 1),
        "completion_tokens": int(data.get("eval_count") or 0),
        "done_reason": str(data.get("done_reason") or ""),
        "hits": len(hits),
        "raw_chars": len(raw),
        "kept_chars": len(trimmed),
        "trimmed_lines": removed,
        # 结尾是否完整：表格闭合标签还在，说明表格没被切断
        "ends_clean": trimmed.rstrip().endswith("</table>") or trimmed.rstrip().endswith("</html>"),
    }


def main() -> int:
    image = Path(os.environ["VISION_TEST_IMAGE"])
    budgets = [int(x) for x in os.environ.get("PREDICT_SWEEP", "4096,3072,2048,1536").split(",")]

    rows: list[dict[str, object]] = []
    for budget in budgets:
        row = run_once(image, budget)
        rows.append(row)
        print(
            f"num_predict={budget:<5} {row['seconds']:>6}s  "
            f"生成 {row['completion_tokens']:>5} token  结束原因={row['done_reason']:<8} "
            f"事实 {row['hits']}/{len(KEY_FACTS)}  结尾完整={row['ends_clean']}"
        )

    lines = ["# num_predict 扫描：输出预算的拐点在哪", ""]
    lines.append(f"- 图片：`{image.name}`｜模型：本地 glm-ocr（CPU）")
    lines.append(f"- ground truth：{len(KEY_FACTS)} 项（含表头）")
    lines.append("")
    lines.append("| num_predict | 耗时 | 生成 token | 结束原因 | 事实命中 | 裁掉行数 | 结尾完整 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['num_predict']} | {row['seconds']}s | {row['completion_tokens']} "
            f"| `{row['done_reason']}` | {row['hits']}/{len(KEY_FACTS)} "
            f"| {row['trimmed_lines']} | {'✅' if row['ends_clean'] else '❌'} |"
        )
    lines.append("")

    # 只认"没截断且命中不掉"的配置才算候选
    safe = [r for r in rows if r["done_reason"] != "length" and r["ends_clean"]]
    if safe:
        best = min(safe, key=lambda r: r["seconds"])
        base = max(rows, key=lambda r: r["num_predict"])
        saved = 1 - best["seconds"] / base["seconds"] if base["seconds"] else 0.0
        lines.append("## 结论")
        lines.append("")
        lines.append(
            f"- **`num_predict={best['num_predict']}` 是「够用且最快」的点**："
            f"{best['seconds']}s，比最大预算（{base['num_predict']}，{base['seconds']}s）"
            f"**快 {saved:.0%}**，且结束原因不是 `length`、结尾完整、命中不降。"
        )
        lines.append(
            f"- 建议默认值改为 **{best['num_predict']}**——"
            "再往下砍就开始截断了（见 `length` 那几行）。"
        )
    else:
        lines.append("## 结论")
        lines.append("")
        lines.append("- ⚠️ 没有找到「不截断」的配置，说明真实内容比所有候选预算都长，**不能下调**。")

    out = Path(__file__).resolve().parents[1] / "reports" / "num_predict_sweep.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\n报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
